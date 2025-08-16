import streamlit as st
from implementation import ImageClassifier
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.models import resnet18, ResNet18_Weights
from preprocess import preprocess_fashion_image, preprocess_imagenet_image
import torch
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / 'model'
FASHION_WEIGHTS_PATH = MODEL_DIR / 'best_model_weights.pth'

# Device-agnostic setup
def setup_device():
    """Setup device-agnostic code for PyTorch"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        st.success(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.info("💻 Using CPU")
    return device

# --- UI ---
st.title("Image Classifier")

# Setup device
device = setup_device()

# Supported modes
MODEL_MODE = st.radio(
    "Choose model:",
    (
        "Fashion MNIST (10 classes)",
        "General (ImageNet, 1000 classes)"
    ),
    index=0
)

FASHION_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def predict_top1(model, img_tensor, categories, device):
    """Run a top-1 prediction and return (label, confidence)."""
    try:
        # Move model and input to device
        model = model.to(device)
        img_tensor = img_tensor.to(device)
        
        import time
        start_time = time.time()
        
        with torch.inference_mode():
            logits = model(img_tensor.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            top_prob, top_idx = torch.topk(probs[0], 1)
        
        inference_time = time.time() - start_time
        
        idx = int(top_idx.item())
        confidence = float(top_prob.item())
        label = categories[idx] if categories and idx < len(categories) else f"class_{idx}"
        
        # Show performance info
        if torch.cuda.is_available():
            st.success(f"⚡ GPU inference completed in {inference_time:.3f}s")
        else:
            st.info(f"⏱️ CPU inference completed in {inference_time:.3f}s")
            
        return label, confidence
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None, None

def open_image_safely(file):
    """Open an image and apply EXIF orientation safely with better error handling."""
    try:
        # Basic validation: size
        if hasattr(file, 'size') and file.size > 10 * 1024 * 1024:
            st.error("File size too large (max 10MB)")
            return None
        
        # Open and load image directly
        file.seek(0)
        img = Image.open(file)
        
        img.load()  # Force load to catch corruption errors

        # Apply EXIF orientation correction
        img = ImageOps.exif_transpose(img)
        
        # Ensure we have a valid image size
        if img.size[0] == 0 or img.size[1] == 0:
            st.error("Image has invalid dimensions")
            return None
            
        return img
    except Exception:
        st.error("Failed to open image")
        return None

@st.cache_resource
def load_models(mode: str, device):
    """Load models based on selected mode."""
    if mode.startswith("Fashion MNIST"):
        model = ImageClassifier(input_shape=1, hidden_shape=32, output_shape=len(FASHION_CLASSES))
        try:
            # Load weights relative to this file to be robust to different CWDs in deployment
            model.load_state_dict(torch.load(str(FASHION_WEIGHTS_PATH), map_location=device))
            model = model.to(device)  # Move to device
            model.eval()
            st.success(f"✅ Fashion MNIST model loaded on {device}")
            return ("fashion", model, FASHION_CLASSES)
        except FileNotFoundError:
            st.error("FashionMNIST model file is missing. Run `python implementation.py` to train it and ensure `model/best_model_weights.pth` is committed.")
            return None
        except Exception as e:
            st.error(f"Failed to load FashionMNIST model: {e}")
            return None
    else:
        try:
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
            model = model.to(device)  # Move to device
            model.eval()
            categories = weights.meta.get("categories", [])
            st.success(f"✅ ImageNet (ResNet18) model loaded on {device}")
            return ("imagenet", model, categories)
        except Exception:
            st.error("ImageNet model couldn't be loaded.")
            return None

# Load models (cache keyed by selected mode and device)
model_info = load_models(MODEL_MODE, device)

if model_info is not None:
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff']
    )

    if uploaded_file is not None:
        with st.spinner("Processing image..."):
            original_image = open_image_safely(uploaded_file)

            if original_image is not None:
                st.image(original_image, use_container_width=True)

                mode, model, categories = model_info
                
                if mode == "fashion":
                    processing_result = preprocess_fashion_image(original_image)
                    if processing_result[0] is not None:
                        img_tensor, processed_array, _ = processing_result
                        st.image(processed_array, width=200, clamp=True)                      
                        label, conf = predict_top1(model, img_tensor, categories, device)
                        if label is not None:
                            st.write(f"Prediction: {label} | Confidence: {conf:.2f}")
                    else:
                        st.error("Failed to process the image. Please try a different image.")
                else:  # ImageNet
                    img_tensor, display_array = preprocess_imagenet_image(original_image, ResNet18_Weights.DEFAULT)
                    if img_tensor is not None:
                        st.image(display_array, width=224, clamp=True)
                        
                        label, conf = predict_top1(model, img_tensor, categories, device)
                        if label is not None:
                            st.write(f"Prediction: {label} | Confidence: {conf:.2f}")
                    else:
                        st.error("Failed to process the image for ImageNet. Please try a different image.")
            else:
                st.error("Could not load the uploaded image. Please check the file format and try again.")
