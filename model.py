import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torchvision.ops import DeformConv2d
from pytorch_wavelets import DWTForward, DWTInverse

from vim.models_mamba import Mamba, RMSNorm


class DeformablePConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super().__init__()
        inter_channels = out_channels // 4
       
        self.offset_conv1 = nn.Conv2d(in_channels, 2 * 1 * kernel_size, kernel_size=3, padding=1)
        self.deform_conv1 = DeformConv2d(in_channels, inter_channels, kernel_size=(1, kernel_size),
                                         stride=stride, padding=(0, kernel_size//2))
        
        self.offset_conv2 = nn.Conv2d(in_channels, 2 * kernel_size * 1, kernel_size=3, padding=1)
        self.deform_conv2 = DeformConv2d(in_channels, inter_channels, kernel_size=(kernel_size, 1),
                                         stride=stride, padding=(kernel_size//2, 0))
        
        self.offset_conv3 = nn.Conv2d(in_channels, 2 * 1 * kernel_size, kernel_size=3, padding=1)
        self.deform_conv3 = DeformConv2d(in_channels, inter_channels, kernel_size=(1, kernel_size),
                                         stride=stride, padding=(0, kernel_size//2))
        
        self.offset_conv4 = nn.Conv2d(in_channels, 2 * kernel_size * 1, kernel_size=3, padding=1)
        self.deform_conv4 = DeformConv2d(in_channels, inter_channels, kernel_size=(kernel_size, 1),
                                         stride=stride, padding=(kernel_size//2, 0))
        
        self.final_conv = nn.Conv2d(inter_channels * 4, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)
        
    def forward(self, x):
        offset1 = self.offset_conv1(x)
        x1 = self.deform_conv1(x, offset1)
        offset2 = self.offset_conv2(x)
        x2 = self.deform_conv2(x, offset2)
        offset3 = self.offset_conv3(x)
        x3 = self.deform_conv3(x, offset3)
        offset4 = self.offset_conv4(x)
        x4 = self.deform_conv4(x, offset4)
        
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        x_out = self.final_conv(x_cat)
        x_out = self.silu(self.bn(x_out))
        return x_out


class WaveletDecompose(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.dwt = None  
        self.current_device = None
    
    def _get_dwt(self, device):
        if self.dwt is None or self.current_device != device:
            self.dwt = DWTForward(J=1, wave='haar', mode='zero').to(device)
            self.current_device = device
        return self.dwt
    
    def forward(self, x):
        device = x.device
        dwt = self._get_dwt(device)
        
        with torch.cuda.amp.autocast(enabled=False):
            x_float32 = x.to(torch.float32)
            x_ll, Yh = dwt(x_float32)
            x_h = Yh[0]  # [B, C, 3, H/2, W/2]
            B, C, _, H_half, W_half = x_h.shape
            x_high = x_h.view(B, C * 3, H_half, W_half)  # [B, 3C, H/2, W/2]
            return x_ll.to(x.dtype), x_high.to(x.dtype)


class DetailStreamEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.edge_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 8, out_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_high):
        x = self.conv1(x_high)
        x = self.conv2(x)
        att = self.edge_attention(x)
        x_enhanced = x * att
        return x_enhanced


class GlobalStreamEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x_ll):
        x = self.conv1(x_ll)
        x = self.conv2(x)
        return x


class DeltaVisionMambaBlock(nn.Module):
    def __init__(self, embed_dim): 
        super().__init__()
        self.linear_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = RMSNorm(embed_dim)
        self.mixer = Mamba(d_model=embed_dim, d_state=16, d_conv=4)
        self.register_buffer('prev_token', torch.zeros(1, 1, embed_dim))
    
    def forward(self, x): 
        B, N, D = x.shape
        skip_connection = x
        prev_x = torch.cat([self.prev_token.expand(B, -1, -1), x[:, :-1, :]], dim=1)
        diff_info = x - prev_x
        combined_info = torch.cat([x, diff_info], dim=-1)
        projected_info = self.linear_proj(combined_info)
        normed_info = self.norm(projected_info)
        mamba_out = self.mixer(normed_info)
        return mamba_out + skip_connection


class DiVimEncoder(nn.Module):
    def __init__(self, pretrained_weights_path, in_channels, embed_dim, depth=12):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, 1, 1)
        self.layers = nn.ModuleList([DeltaVisionMambaBlock(embed_dim) for _ in range(depth)])
        self.norm_f = RMSNorm(embed_dim)
        
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            try:
                checkpoint = torch.load(pretrained_weights_path, map_location='cpu')
                state_dict = checkpoint.get('model', checkpoint)
                loaded_count = 0
                for i in range(depth):
                    mamba_block_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith(f'layers.{i}.'):
                            new_key = k[len(f'layers.{i}.'):]
                            mamba_block_dict[new_key] = v
                    if mamba_block_dict:
                        self.layers[i].mixer.load_state_dict(mamba_block_dict, strict=False)
                        loaded_count += 1
                if loaded_count > 0:
                    print(f" 成功加载 {loaded_count}/{depth} 个Mamba层预训练权重")
            except Exception as e:
                print(f" 加载预训练权重时出错: {e}")
    
    def forward(self, x): 
        x = self.proj(x)
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2)
        for blk in self.layers: 
            x_seq = blk(x_seq)
        x_seq = self.norm_f(x_seq)
        x_out = x_seq.transpose(1, 2).view(B, C, H, W)
        return x_out


class FourierTransformBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.amplitude_processor = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        with torch.cuda.amp.autocast(enabled=False):
            x_float32 = x.to(torch.float32)
            fft_x = torch.fft.fft2(x_float32, norm='ortho')
            amplitude = fft_x.abs()
            phase = fft_x.angle()
            enhanced_amplitude = self.amplitude_processor(amplitude)
            real_part = enhanced_amplitude * torch.cos(phase)
            imag_part = enhanced_amplitude * torch.sin(phase)
            ifft_input = torch.complex(real_part, imag_part)
            restored_x = torch.fft.ifft2(ifft_input, s=(H, W), norm='ortho')
            restored_x_real = restored_x.real
        return x + self.alpha * restored_x_real.to(x.dtype)


class CrossFusionBlock(nn.Module):
    def __init__(self, global_channels, detail_channels, out_channels):
        super().__init__()
        
        self.global_to_detail = nn.Sequential(
            nn.Conv2d(global_channels, detail_channels, 1),
            nn.Sigmoid()
        )
        
        self.detail_to_global = nn.Sequential(
            nn.Conv2d(detail_channels, global_channels, 1),
            nn.Sigmoid()
        )
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(global_channels + detail_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, global_feat, detail_feat):
        g2d_att = self.global_to_detail(global_feat)
        detail_enhanced = detail_feat * g2d_att
        
        d2g_att = self.detail_to_global(detail_feat)
        global_enhanced = global_feat * d2g_att
        
        fused = torch.cat([global_enhanced, detail_enhanced], dim=1)
        out = self.fusion_conv(fused)
        
        return out


class GrayFeatureExtractor(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.rgb_to_gray = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        
        self.contrast_enhance = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
    def forward(self, rgb):
        gray = self.rgb_to_gray(rgb)
        enhanced = self.contrast_enhance(gray)
        return enhanced


class DualPathFusion(nn.Module):

    def __init__(self, wavelet_channels, mamba_channels, out_channels):
        super().__init__()
        
        self.wavelet_proj = nn.Conv2d(wavelet_channels, out_channels, 1)
        self.mamba_proj = nn.Conv2d(mamba_channels, out_channels, 1)
        
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 2, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels * 2, 1),
            nn.Sigmoid()
        )
        
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.Sigmoid()
        )
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, wavelet_feat, mamba_feat):
        B, _, H, W = wavelet_feat.shape
        
        w_proj = self.wavelet_proj(wavelet_feat)
        m_proj = self.mamba_proj(mamba_feat)
        
        w_max = torch.max(w_proj, dim=1, keepdim=True)[0]
        w_avg = torch.mean(w_proj, dim=1, keepdim=True)
        w_spatial = self.spatial_attn(torch.cat([w_max, w_avg], dim=1))
        w_enhanced = w_proj * w_spatial
        
        concat_feat = torch.cat([w_enhanced, m_proj], dim=1)
        channel_weights = self.channel_attn(concat_feat)
        w_weight, m_weight = torch.chunk(channel_weights, 2, dim=1)
        
        w_weighted = w_enhanced * w_weight
        m_weighted = m_proj * m_weight
        
        gate = self.gate(torch.cat([w_weighted, m_weighted], dim=1))
        gated_feat = gate * w_weighted + (1 - gate) * m_weighted
        
        fused = self.fusion_conv(torch.cat([gated_feat, m_proj], dim=1))
        output = self.relu(fused + m_proj)
        
        return output


class WaveletReconstructUpsample(nn.Module):
    
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.idwt = None
        self.current_device = None
        
        self.sub_channels = in_channels // 4
        assert in_channels % 4 == 0, f"in_channels ({in_channels})"
        
        self.to_ll = nn.Sequential(
            nn.Conv2d(in_channels, self.sub_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, self.sub_channels), self.sub_channels),
            nn.Tanh()  # [-1, 1]
        )
        
        self.to_high = nn.Sequential(
            nn.Conv2d(in_channels, self.sub_channels * 3, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, self.sub_channels * 3), self.sub_channels * 3),
            nn.Tanh()  # [-1, 1]
        )
        
        self.scale_factor = nn.Parameter(torch.tensor(3.0))
        
        if self.sub_channels != out_channels:
            self.channel_adjust = nn.Sequential(
                nn.Conv2d(self.sub_channels, out_channels, 1, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.channel_adjust = nn.Identity()
    
    def _get_idwt(self, device):
        if self.idwt is None or self.current_device != device:
            self.idwt = DWTInverse(wave='haar', mode='zero').to(device)
            self.current_device = device
        return self.idwt
    
    def forward(self, fused_feat):
        device = fused_feat.device
        B, C, H, W = fused_feat.shape

        with torch.cuda.amp.autocast(enabled=False):
            fused_feat_f32 = fused_feat.float()
            
            x_ll = self.to_ll(fused_feat_f32)
            x_high = self.to_high(fused_feat_f32)
            
        
            scale = torch.clamp(self.scale_factor, 1.0, 5.0)
            x_ll = x_ll * scale
            x_high = x_high * scale
        
            x_lh = x_high[:, :self.sub_channels, :, :]
            x_hl = x_high[:, self.sub_channels:2*self.sub_channels, :, :]
            x_hh = x_high[:, 2*self.sub_channels:, :, :]
            
            x_high_stack = torch.stack([x_lh, x_hl, x_hh], dim=2)
            Yh = [x_high_stack]
         
            idwt = self._get_idwt(device)
            x_reconstructed = idwt((x_ll, Yh))
          
            x_reconstructed = x_reconstructed.to(fused_feat.dtype)
        
        out = self.channel_adjust(x_reconstructed)
        
        return out


class TriPathMultiScaleCrossAttention(nn.Module):
    def __init__(self, decoder_channels, encoder_channels_list, attention_dim=128, num_heads=4):
        super().__init__()
        assert len(encoder_channels_list) == 3, " [detail, structure, global]"
        
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.q_conv1 = nn.Conv2d(decoder_channels, attention_dim, 1)
        self.k_conv1 = nn.Conv2d(encoder_channels_list[0], attention_dim, 1)
        self.v_conv1 = nn.Conv2d(encoder_channels_list[0], attention_dim, 1)
       
        self.q_conv2 = nn.Conv2d(decoder_channels, attention_dim, 1)
        self.k_conv2 = nn.Conv2d(encoder_channels_list[1], attention_dim, 1)
        self.v_conv2 = nn.Conv2d(encoder_channels_list[1], attention_dim, 1)
      
        self.q_conv3 = nn.Conv2d(decoder_channels, attention_dim, 1)
        self.k_conv3 = nn.Conv2d(encoder_channels_list[2], attention_dim, 1)
        self.v_conv3 = nn.Conv2d(encoder_channels_list[2], attention_dim, 1)
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(attention_dim * 3, decoder_channels, 1),
            nn.GroupNorm(8, decoder_channels),
            nn.SiLU(inplace=True)
        )
        self.norm = nn.GroupNorm(min(8, decoder_channels), decoder_channels)

    def calculate_attention(self, q, k, v, B, H, W):
        q = q.flatten(2).view(B, self.num_heads, self.head_dim, -1)
        k = k.flatten(2).view(B, self.num_heads, self.head_dim, -1)
        v = v.flatten(2).view(B, self.num_heads, self.head_dim, -1)
        
        with torch.cuda.amp.autocast(enabled=False): 
            q_f32 = q.float()
            k_f32 = k.float()
            attn = (q_f32 @ k_f32.transpose(-2, -1)) * (1.0 / self.scale)
            attn = attn.softmax(dim=-1)
            attn = attn.to(v.dtype)
        out = (attn @ v)
        out = out.reshape(B, -1, H, W)
        return out

    def forward(self, decoder_feature, encoder_features_list):
        B, C, H, W = decoder_feature.shape
        
        aligned_features = []
        for feat in encoder_features_list:
            if feat.shape[2:] != (H, W):
                feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=True)
            aligned_features.append(feat)
        
        feat_detail, feat_struct, feat_global = aligned_features
        
        q1, k1, v1 = self.q_conv1(decoder_feature), self.k_conv1(feat_detail), self.v_conv1(feat_detail)
        out1 = self.calculate_attention(q1, k1, v1, B, H, W)
        
        q2, k2, v2 = self.q_conv2(decoder_feature), self.k_conv2(feat_struct), self.v_conv2(feat_struct)
        out2 = self.calculate_attention(q2, k2, v2, B, H, W)
        
        q3, k3, v3 = self.q_conv3(decoder_feature), self.k_conv3(feat_global), self.v_conv3(feat_global)
        out3 = self.calculate_attention(q3, k3, v3, B, H, W)
        
        tri_path_out = torch.cat([out1, out2, out3], dim=1)
        fused = self.fusion_conv(tri_path_out)
        
        return self.norm(decoder_feature + fused)


class TriPathDecoderBlock(nn.Module):
   
    
    def __init__(self, 
                 in_channels,
                 skip_detail_channels,
                 skip_structure_channels,
                 skip_global_channels,
                 out_channels,
                 attention_dim=128,
                 num_heads=4,
                 dropout_p=0.3,
                 use_wavelet_upsample=True):
        super().__init__()
        
        self.use_wavelet_upsample = use_wavelet_upsample
        
        if use_wavelet_upsample:
            self.wavelet_upsample = WaveletReconstructUpsample(
                in_channels=skip_detail_channels,
                out_channels=out_channels
            )
        
        self.bilinear_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.tri_path_cross_attention = TriPathMultiScaleCrossAttention(
            decoder_channels=out_channels,
            encoder_channels_list=[
                skip_detail_channels,
                skip_structure_channels,
                skip_global_channels
            ],
            attention_dim=attention_dim,
            num_heads=num_heads
        )
        
        self.refine_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip_detail, skip_structure, skip_global):
      
        if self.use_wavelet_upsample:
            if skip_detail.shape[2] == x.shape[2]:
                skip_upsampled = self.wavelet_upsample(skip_detail)  # [B, out_C, 2H, 2W]
            else: 
                skip_upsampled = skip_detail
             
                if skip_upsampled.shape[1] != self.wavelet_upsample.out_channels:
                    skip_upsampled = nn.Conv2d(skip_upsampled.shape[1], 
                                               self.wavelet_upsample.out_channels, 
                                               1).to(skip_detail.device)(skip_upsampled)
        else:
            if skip_detail.shape[2] == x.shape[2]:
                skip_upsampled = F.interpolate(skip_detail, scale_factor=2, 
                                              mode='bilinear', align_corners=True)
            else:
                skip_upsampled = skip_detail
        
        x_upsampled = self.bilinear_upsample(x)  # [B, out_C, 2H, 2W]
        
        merged = x_upsampled + skip_upsampled  # [B, out_C, 2H, 2W]
        
        merged = self.tri_path_cross_attention(
            merged, 
            [skip_upsampled, skip_structure, skip_global]
        )
        
        out = self.refine_conv(merged)
        
        return out


class TriPathRefinementBlock(nn.Module):
    
    def __init__(self, 
                 in_channels,
                 skip_channels_list,
                 out_channels,
                 attention_dim=32,
                 num_heads=4,
                 dropout_p=0.3):
        super().__init__()
        
        self.channel_align = nn.Conv2d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
        
        self.tri_path_cross_attention = TriPathMultiScaleCrossAttention(
            decoder_channels=out_channels,
            encoder_channels_list=skip_channels_list,
            attention_dim=attention_dim,
            num_heads=num_heads
        )
        
        self.refine_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip_connections_list):
        x = self.channel_align(x)
        x = self.tri_path_cross_attention(x, skip_connections_list)
        x = self.refine_conv(x)
        return x


class WaveletDualStreamVimUNet(nn.Module):
    
    
    def __init__(self, pretrained_weights_path=None, patch_size=48, 
                 use_wavelet_upsample=True):
        super().__init__()
        
        self.patch_size = patch_size
        self.use_wavelet_upsample = use_wavelet_upsample
        
        
        self.stem = DeformablePConv(3, 64, stride=1)
        
        self.wavelet_decompose1 = WaveletDecompose()  # 48 → 24
        self.wavelet_decompose2 = WaveletDecompose()  # 24 → 12
        
        self.global_encoder1 = GlobalStreamEncoder(64, 128)
        self.global_encoder2 = GlobalStreamEncoder(128, 256)
     
        self.detail_encoder1 = DetailStreamEncoder(64 * 3, 128)
        self.detail_encoder2 = DetailStreamEncoder(128 * 3, 256)
        
        self.cross_fusion1 = CrossFusionBlock(128, 128, 128)
        self.cross_fusion2 = CrossFusionBlock(256, 256, 256)
        
        
        self.gray_feature_extractor = GrayFeatureExtractor()
        self.mamba_stem = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.mamba_downsample1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        self.mamba_downsample2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU()
        )
        self.mamba_bottleneck = DiVimEncoder(
            pretrained_weights_path,
            in_channels=256,
            embed_dim=384,
            depth=12
        )
        
        
        self.dual_path_fusion = DualPathFusion(
            wavelet_channels=256,
            mamba_channels=384,
            out_channels=384
        )
        self.fourier_bottleneck = FourierTransformBlock(channels=384)
    
        
        # Decoder 1: 12 → 24 (H/4 → H/2)
        self.decoder1 = TriPathDecoderBlock(
            in_channels=384,
            skip_detail_channels=128,     # fused_feat1 
            skip_structure_channels=256,  # fused_feat2
            skip_global_channels=384,     # fourier_out
            out_channels=128,
            attention_dim=128,
            num_heads=4,
            use_wavelet_upsample=use_wavelet_upsample
        )
        
        # Decoder 2: 24 → 48 (H/2 → H)
        self.decoder2 = TriPathDecoderBlock(
            in_channels=128,
            skip_detail_channels=64,      # s0 ← 小波重构
            skip_structure_channels=128,  # fused_feat1
            skip_global_channels=256,     # fused_feat2
            out_channels=64,
            attention_dim=64,
            num_heads=4,
            use_wavelet_upsample=use_wavelet_upsample
        )
        
        # Decoder 3: 48 → 48 
        self.decoder3 = TriPathRefinementBlock(
            in_channels=64,
            skip_channels_list=[64, 128, 256],  # [s0, fused_feat1, fused_feat2]
            out_channels=32,
            attention_dim=32,
            num_heads=4,
            dropout_p=0.3
        )
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
    def forward(self, x):
        
        
            s0 = self.stem(x)  # [B, 64, 48, 48]
            
            
            x_ll1, x_high1 = self.wavelet_decompose1(s0)
            # x_ll1: [B, 64, 24, 24] - LL
            # x_high1: [B, 192, 24, 24] - LH+HL+HH
            
            global_feat1 = self.global_encoder1(x_ll1)      # [B, 128, 24, 24]
            detail_feat1 = self.detail_encoder1(x_high1)    # [B, 128, 24, 24]
            fused_feat1 = self.cross_fusion1(global_feat1, detail_feat1)  # [B, 128, 24, 24]
            
            # Level 2:  (24 → 12)
            x_ll2, x_high2 = self.wavelet_decompose2(fused_feat1)
            # x_ll2: [B, 128, 12, 12]
            # x_high2: [B, 384, 12, 12]
            
            global_feat2 = self.global_encoder2(x_ll2)      # [B, 256, 12, 12]
            detail_feat2 = self.detail_encoder2(x_high2)    # [B, 256, 12, 12]
            fused_feat2 = self.cross_fusion2(global_feat2, detail_feat2)  # [B, 256, 12, 12]
            
            gray_feat = self.gray_feature_extractor(x)      # [B, 1, 48, 48]
            x_with_gray = torch.cat([x, gray_feat], dim=1)  # [B, 4, 48, 48]
            
            m0 = self.mamba_stem(x_with_gray)               # [B, 64, 48, 48]
            m1 = self.mamba_downsample1(m0)                 # [B, 128, 24, 24]
            m2 = self.mamba_downsample2(m1)                 # [B, 256, 12, 12]
            mamba_out = self.mamba_bottleneck(m2)           # [B, 384, 12, 12]
            
            fused_bottleneck = self.dual_path_fusion(fused_feat2, mamba_out)  # [B, 384, 12, 12]
            fourier_out = self.fourier_bottleneck(fused_bottleneck)           # [B, 384, 12, 12]
            
            # Decoder 1: 12 → 24 (H/4 → H/2)
            d1 = self.decoder1(
                x=fourier_out,              # [B, 384, 12, 12]
                skip_detail=fused_feat1,    # [B, 128, 24, 24] 
                skip_structure=fused_feat2, # [B, 256, 12, 12]
                skip_global=fourier_out     # [B, 384, 12, 12]
            )  # → [B, 128, 24, 24]
            
            # Decoder 2: 24 → 48 (H/2 → H)
            d2 = self.decoder2(
                x=d1,                       # [B, 128, 24, 24]
                skip_detail=s0,             # [B, 64, 48, 48] 
                skip_structure=fused_feat1, # [B, 128, 24, 24]
                skip_global=fused_feat2     # [B, 256, 12, 12]
            )  # → [B, 64, 48, 48]
            
            skip_list_3 = [s0, fused_feat1, fused_feat2]
            d3 = self.decoder3(d2, skip_list_3)  # [B, 32, 48, 48]
            
            logits = self.final_conv(d3)  # [B, 1, 48, 48]
            
            return logits
