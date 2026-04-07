classes = {
    (0, 255, 255): "Urban land",
    (255, 255, 0): "Agriculture land",
    (255, 0, 255): "Rangeland",
    (0, 255, 0): "Forest land",
    (0, 0, 255): "Water",
    (255, 255, 255): "Barren land",
    (0, 0, 0): "Unknown"
}

weights = [1.0, 0.8, 0.8, 1.0, 1.0, 0.8, 1.0]
base_path = 'deepglobe-classification'
n_classes = len(classes)
n_channels = 3
num_tiles = 10
