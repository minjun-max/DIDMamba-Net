# configs.py

DRIVE_CONFIG = {
    'image_size': 48,
    'batch_size': 16,
    'lr': 1e-5,
    'lr_mamba': 1e-6,
    'epochs': 80,
    'patience': 50,
    'loss': {
        'w_bce': 1.0,
        'w_dice': 1.5,
        'w_tversky': 1.5,
        'w_focal': 0.8,
        'tversky_alpha': 0.7,
        'tversky_beta': 0.3,
        'focal_alpha': 0.25,
        'focal_gamma': 1.5,
        'wavelet_type': 'multiscale',
        'high_freq_weight': 2.0
    }
}

STARE_CONFIG = {
    'image_size': 48,
    'batch_size': 16,
    'lr': 1.5e-5,
    'lr_mamba': 1.5e-6,
    'epochs': 100,
    'patience': 30,
    'loss': {
        'w_bce': 1.0,
        'w_dice': 2.0,
        'w_tversky': 1.8,
        'w_focal': 1.2,
        'tversky_alpha': 0.8,
        'tversky_beta': 0.25,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'wavelet_type': 'frequency_weighted',
        'high_freq_weight': 2.5
    }
}

CHASE_CONFIG = {
    'image_size': 48,
    'batch_size': 12,
    'lr': 2e-5,
    'lr_mamba': 2e-6,
    'epochs': 100,
    'patience': 50,
    'loss': {
        'w_bce': 1.0,
        'w_dice': 2.0,
        'w_tversky': 1.5,
        'w_focal': 1.0,
        'tversky_alpha': 0.7,
        'tversky_beta': 0.3,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'wavelet_type': 'multiscale',
        'high_freq_weight': 3.0
    }
}

HRF_CONFIG = {
    'image_size': 48,
    'batch_size': 8,
    'lr': 8e-6,
    'lr_mamba': 8e-7,
    'epochs': 50,
    'patience': 40,
    'loss': {
        'w_bce': 1.0,
        'w_dice': 2.0,
        'w_tversky': 0.5,
        'w_focal': 1.0,
        'tversky_alpha': 0.5,
        'tversky_beta': 0.5,
        'focal_alpha': 0.25,
        'focal_gamma': 1.5,
        'wavelet_type': 'multiscale',
        'high_freq_weight': 1.5
    }
}

def get_config(dataset):
    configs = {
        'DRIVE': DRIVE_CONFIG,
        'STARE': STARE_CONFIG,
        'CHASE': CHASE_CONFIG,
        'HRF': HRF_CONFIG
    }
    return configs.get(dataset.upper())