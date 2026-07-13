import os


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def map_vehicle_class(class_name: str) -> str:
    """
    Map raw model class names to canonical categories used by
    the counter/density/signal modules: 'car', 'bus', 'van', 'others'.
    Emergency vehicles are preserved as their own type.
    This prevents KeyError when models use unexpected class labels.
    """
    if not class_name:
        return "others"

    name = str(class_name)

    if _CLASS_MAP and name in _CLASS_MAP:
        return _CLASS_MAP[name]

    name_lower = name.lower()

    # Emergency vehicles - preserve as their own type
    if "ambulance" in name_lower:
        return "ambulance"

    if "fire" in name_lower and "truck" in name_lower:
        return "fire_truck"

    if "police" in name_lower:
        return "police"

    if name_lower in ("car", "sedan", "coupe", "hatchback"):
        return "car"

    if name_lower in ("bus",):
        return "bus"

    if name_lower in ("van", "truck", "lorry", "pickup"):
        return "van"

    # common two-wheelers and unknown types map to others
    if name_lower in ("motorbike", "motorcycle", "bike", "bicycle", "scooter"):
        return "others"

    return "others"


# optional explicit mapping populated from model class names
_CLASS_MAP = {}


def set_class_map(mapping: dict):
    """Set a mapping from model class name -> canonical category."""
    global _CLASS_MAP
    _CLASS_MAP = dict(mapping)


def build_class_map(class_names: dict) -> dict:
    """Build a conservative mapping from model class names to canonical keys.

    class_names may be a dict of {idx: name} or list-like. We map any
    name containing keywords to the canonical categories.
    """
    mapping = {}

    for key, name in class_names.items() if isinstance(class_names, dict) else enumerate(class_names):

        label = str(name).lower()

        if "ambulance" in label or "police" in label or "fire" in label:
            # keep emergency labels mapped to themselves for detection
            mapping[name] = name
            continue

        if "car" in label or "sedan" in label:
            mapping[name] = "car"
            continue

        if "bus" in label:
            mapping[name] = "bus"
            continue

        if "van" in label or "truck" in label or "lorry" in label or "pickup" in label:
            mapping[name] = "van"
            continue

        mapping[name] = "others"

    return mapping
