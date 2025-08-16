from typing import Tuple, Optional
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms
import streamlit as st


def preprocess_fashion_image(image: Image.Image, auto_invert: bool = True) -> Tuple[Optional[object], Optional[np.ndarray], bool]:
    """Preprocess uploaded image to match FashionMNIST training format.

    - Handle alpha channel by compositing on white
    - Convert to grayscale
    - Autocontrast to stretch dynamic range
    - Detect background brightness and invert if needed
    - Preserve aspect ratio by letterboxing/padding to 28x28
    - Normalize like training

    Returns: (tensor[1,28,28] or None, display_array[28,28] or None, inverted:boolean)
    """
    if image is None or image.size[0] == 0 or image.size[1] == 0:
        return None, None, False

    try:
        # Handle alpha by compositing onto white
        if image.mode in ("RGBA", "LA"):
            bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(bg, image.convert("RGBA")).convert("RGB")

        # Convert to grayscale
        if image.mode != "L":
            image = image.convert("L")

        # Autocontrast
        image = ImageOps.autocontrast(image)

        # Simple background detection and inversion
        img_array = np.array(image, dtype=np.float32)
        h, w = img_array.shape
        
        # Check if background is light (simple border check)
        border_size = max(1, min(h, w) // 10)
        border_pixels = []
        if h >= border_size:
            border_pixels.extend(img_array[:border_size, :].ravel())
            border_pixels.extend(img_array[-border_size:, :].ravel())
        if w >= border_size:
            border_pixels.extend(img_array[:, :border_size].ravel())
            border_pixels.extend(img_array[:, -border_size:].ravel())
        
        bg_median = np.median(border_pixels) if border_pixels else np.median(img_array)
        
        inverted = False
        if auto_invert and bg_median > 127:
            img_array = 255 - img_array
            inverted = True
        
        image = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8), mode="L")

        # Resize with letterbox/pad to 28x28
        image = ImageOps.pad(image, (28, 28), method=Image.Resampling.LANCZOS, color=0, centering=(0.5, 0.5))

        # Display array before normalization
        display_array = np.array(image, dtype=np.float32)

        # Normalize as during training
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        img_tensor = transform(image)
        
        if img_tensor.shape != (1, 28, 28):
            return None, None, False

        return img_tensor, display_array, inverted
        
    except Exception:
        return None, None, False

@st.cache_resource
def preprocess_imagent_weights(weights):
    if weights is None:
        return None
    return weights.transforms()

def preprocess_imagenet_image(image: Image.Image, weights):
    """Preprocess an image for ImageNet inference using the given weights.

    Returns: (tensor[3,224,224] or None, display_array[224,224,3] or None)
    """
    if image is None:
        return None, None

    try:
        if image.mode != "RGB":
            image = image.convert("RGB")

        transform = preprocess_imagent_weights(weights)
        img_tensor = transform(image)

        # Create preview for display
        preview = ImageOps.fit(image, (224, 224), Image.Resampling.BILINEAR)
        display_array = np.array(preview)
        
        return img_tensor, display_array
        
    except Exception:
        return None, None


