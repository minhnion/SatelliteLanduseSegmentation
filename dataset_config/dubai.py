classes = {
    (60, 16, 152): "Building",
    (132, 41, 246): "Land",
    (110, 193, 228): "Road",
    (254, 221, 58): "Vegetation",
    (226, 169, 41): "Water",
    (155, 155, 155): "Unlabeled"
}

weights = [1.0, 0.8, 0.8, 1.0, 1.0, 0.8]
base_path = 'dubai_dataset_splitted'
n_channels = 3
n_classes = len(classes)
num_tiles = 3
