from typing import Tuple, Optional
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms
from torchvision.models import ResNet18_Weights


def preprocess_fashion_image(image: Image.Image, auto_invert: bool = True) -> Tuple[Optional[object], Optional[np.ndarray], bool]:
    """Preprocess uploaded image to match FashionMNIST training format.

    - Handle alpha channel by compositing on white
    - Convert to grayscale
    - Autocontrast to stretch dynamic range
    - Detect background brightness from borders and invert if needed
    - Preserve aspect ratio by letterboxing/padding to 28x28
    - Normalize like training

    Returns: (tensor[1,28,28] or None, display_array[28,28] or None, inverted:boolean)
    """
    try:
        if image is None:
            return None, None, False

        if image.size[0] == 0 or image.size[1] == 0:
            return None, None, False

        # Handle alpha by compositing onto white
        if image.mode in ("RGBA", "LA"):
            try:
                bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
                image = Image.alpha_composite(bg, image.convert("RGBA")).convert("RGB")
            except Exception:
                image = image.convert("RGB")

        # Ensure RGB or L, then convert to grayscale
        if image.mode not in ("RGB", "L"):
            try:
                image = image.convert("RGB")
            except Exception:
                return None, None, False

        if image.mode != "L":
            try:
                image = image.convert("L")
            except Exception:
                return None, None, False

        # Autocontrast
        try:
            image = ImageOps.autocontrast(image)
        except Exception:
            pass

        # Estimate background from borders and optionally invert
        try:
            img_array = np.array(image, dtype=np.float32)
            h, w = img_array.shape
            if h < 3 or w < 3:
                border = 1
            else:
                border = max(1, int(min(h, w) * 0.1))

            border_pixels = []
            if h >= border:
                border_pixels.extend(img_array[:border, :].ravel())
                border_pixels.extend(img_array[-border:, :].ravel())
            if w >= border:
                border_pixels.extend(img_array[:, :border].ravel())
                border_pixels.extend(img_array[:, -border:].ravel())

            bg_median = np.median(border_pixels) if border_pixels else np.median(img_array)

            inverted = False
            if auto_invert and bg_median > 127:
                img_array = 255 - img_array
                inverted = True

            img_array = np.clip(img_array, 0, 255)
            image = Image.fromarray(img_array.astype(np.uint8), mode="L")
        except Exception:
            inverted = False

        # Resize with letterbox/pad to 28x28
        try:
            image = ImageOps.pad(image, (28, 28), method=Image.Resampling.LANCZOS, color=0, centering=(0.5, 0.5))
        except Exception:
            image = image.resize((28, 28), Image.Resampling.LANCZOS)

        # Display array before normalization
        try:
            display_array = np.array(image, dtype=np.float32)
        except Exception:
            return None, None, False

        # Normalize as during training
        try:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            img_tensor = transform(image)
            if img_tensor.shape != (1, 28, 28):
                return None, None, False
        except Exception:
            return None, None, False

        return img_tensor, display_array, inverted
    except Exception:
        return None, None, False


def preprocess_imagenet_image(image: Image.Image, weights: ResNet18_Weights):
    """Preprocess an image for ImageNet inference using the given weights.

    Returns: (tensor[3,224,224] or None, display_array[224,224,3] or None)
    """
    try:
        if image is None:
            return None, None

        if image.mode != "RGB":
            image = image.convert("RGB")

        transform = weights.transforms()
        img_tensor = transform(image)

        try:
            preview = ImageOps.fit(image, (224, 224), Image.Resampling.BILINEAR)
        except Exception:
            preview = image.resize((224, 224), Image.Resampling.BILINEAR)

        display_array = np.array(preview)
        return img_tensor, display_array
    except Exception:
        return None, None


