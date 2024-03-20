from torchvision import transforms 

from .config import config


def get_resize_transform():
    return transforms.Resize(
        size=config.input_xy,
        interpolation=transforms.InterpolationMode.BILINEAR
    )


def compose_transforms(transform_instances):
    return transforms.Compose(
        transforms=transform_instances
    )