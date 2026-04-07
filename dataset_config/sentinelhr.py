classes = {
    (0, 0, 0): "Unidentifiable",
    (0, 0, 255): "Unidentifiable",
    (255, 255, 255): "Unidentifiable",
    (0, 255, 0): "Forest",
    (255, 0, 0): "Rice field",
    (0, 255, 255): "Water",
    (255, 255, 0): "Residential"
}

weights = [0.5, 1.0, 1.5, 1.0, 1.0]
base_path = 'sentinel_hr_lr_dataset_cut'
n_classes = 5
n_channels = 13
num_tiles = 1
