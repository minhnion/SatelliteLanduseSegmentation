RGB_TO_CLASS = {
    (0, 0, 0): "Unidentifiable",
    (0, 0, 255): "Unidentifiable",
    (255, 255, 255): "Unidentifiable",
    (0, 255, 0): "Forest",
    (255, 0, 0): "Rice field",
    (0, 255, 255): "Water",
    (255, 255, 0): "Residential",
}


def _ordered_class_names():
    seen = []
    for class_name in RGB_TO_CLASS.values():
        if class_name not in seen:
            seen.append(class_name)
    return seen


CLASS_NAMES = _ordered_class_names()


SEASON_BY_DATE_TAG = {
    "0304_2023": {"key": "dong_xuan", "label": "Dong Xuan"},
    "0809_2023": {"key": "he_thu", "label": "He Thu"},
}


AUTHALIC_RADIUS_M = 6_371_008.8


def slugify(name):
    return name.lower().replace(" ", "_")
