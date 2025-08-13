import streamlit as st
from implementation import ImageClassifier
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.models import resnet18, ResNet18_Weights
from preprocess import preprocess_fashion_image, preprocess_imagenet_image
import torch

# --- UI ---
st.title("Image Classifier")

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

@st.cache_resource
def load_fashion_model():
    """Load the trained FashionMNIST model from disk."""
    model = ImageClassifier(input_shape=1, hidden_shape=32, output_shape=len(FASHION_CLASSES))
    try:
        model.load_state_dict(torch.load('model/best_model_weights.pth', map_location='cpu'))
        model.eval()
        st.success("✅ FashionMNIST model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Error loading FashionMNIST model: {e}")
        st.error("Please make sure you've trained the model using the training script.")
        return None

def validate_image(file):
    """Validate if the uploaded file is a valid image"""
    try:
        # Check file size (limit to 10MB)
        if hasattr(file, 'size') and file.size > 10 * 1024 * 1024:
            return False, "File size too large (max 10MB)"
        
        # Try to open and validate the image
        file.seek(0)  # Reset file pointer
        img = Image.open(file)
        img.verify()  # Verify it's a valid image
        
        # Reset file pointer after verification
        file.seek(0)
        return True, "Valid image"
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

def open_image_safely(file):
    """Open an image and apply EXIF orientation safely with better error handling."""
    try:
        # Validate the image first
        is_valid, message = validate_image(file)
        if not is_valid:
            st.error(message)
            return None
        
        # Reset file pointer and open image
        file.seek(0)
        img = Image.open(file)
        
        # Handle potential issues with image loading
        try:
            img.load()  # Force load the image data
        except Exception as load_error:
            st.warning(f"Image loading issue (continuing anyway): {load_error}")
        
        # Apply EXIF orientation correction
        img = ImageOps.exif_transpose(img)
        
        # Ensure we have a valid image size
        if img.size[0] == 0 or img.size[1] == 0:
            st.error("Image has invalid dimensions")
            return None
            
        return img
    except Exception as e:
        st.error(f"Failed to open image: {e}")
        return None

@st.cache_resource
def load_imagenet_model():
    """Load a pretrained ImageNet classifier (ResNet18) and its weights metadata."""
    try:
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.eval()
        categories = weights.meta.get("categories", [])
        st.success("✅ ImageNet (ResNet18) model loaded successfully")
        return model, weights, categories
    except Exception as e:
        st.error(f"❌ Error loading ImageNet pretrained model: {e}")
        st.info("If you're offline, the pretrained weights may fail to download. Connect to the internet and retry.")
        return None, None, []

if MODEL_MODE.startswith("Fashion MNIST"):
    st.caption("Mode: FashionMNIST (28x28 grayscale)")
    st.write("Classes: " + ', '.join(FASHION_CLASSES))
    model_fashion = load_fashion_model()
    active_model = ("fashion", model_fashion)
else:
    st.caption("Mode: General ImageNet (ResNet18, 1000 classes)")
    model_imagenet, imagenet_weights, imagenet_categories = load_imagenet_model()
    active_model = ("imagenet", (model_imagenet, imagenet_weights, imagenet_categories))

if (active_model[0] == "fashion" and active_model[1] is None) or (active_model[0] == "imagenet" and (active_model[1][0] is None)):
    # Model failed to load
    if active_model[0] == "fashion":
        st.error("FashionMNIST model file not found. Ensure 'model/best_model_weights.pth' exists.")
        st.info("💡 How to fix: Run `python implementation.py` to train and save the model, then refresh.")
    else:
        st.error("ImageNet pretrained model couldn't be loaded. Check internet connection and retry.")
else:
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff'],
        help="Supported formats: JPG, JPEG, PNG, BMP, GIF, TIFF (max 10MB)"
    )

    if uploaded_file is not None:
        with st.spinner("Processing image..."):
            original_image = open_image_safely(uploaded_file)

            if original_image is not None:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Original")
                    st.image(original_image, use_column_width=True)

                # Branch per model type
                if active_model[0] == "fashion":
                    processing_result = preprocess_fashion_image(original_image)
                    if processing_result[0] is not None:
                        img_tensor, processed_array, was_inverted = processing_result
                        with col2:
                            st.subheader("Processed (28x28)")
                            st.image(processed_array, width=200, clamp=True)
                            if was_inverted:
                                st.caption("🔄 Image was inverted for better recognition")

                        try:
                            with torch.inference_mode():
                                logits = active_model[1](img_tensor.unsqueeze(0))
                                probs = torch.softmax(logits, dim=1)
                                pred_idx = torch.argmax(probs, dim=1).item()
                                confidence = probs[0][pred_idx].item()

                            confidence_emoji = "🎯" if confidence > 0.7 else ("✅" if confidence > 0.5 else "⚠️")
                            st.success(f"{confidence_emoji} Prediction: **{FASHION_CLASSES[pred_idx]}**  |  Confidence: **{confidence:.2f}**")
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")
                            st.error("Please check that the model is properly loaded and the image was processed correctly.")
                    else:
                        st.error("Failed to process the image. Please try a different image.")

                else:  # ImageNet
                    model_imagenet, weights, categories = active_model[1]
                    img_tensor, display_array = preprocess_imagenet_image(original_image, weights)
                    if img_tensor is not None:
                        with col2:
                            st.subheader("Processed (224x224)")
                            st.image(display_array, width=224, clamp=True)

                        try:
                            with torch.inference_mode():
                                logits = model_imagenet(img_tensor.unsqueeze(0))
                                probs = torch.softmax(logits, dim=1)
                                top5_probs, top5_indices = torch.topk(probs[0], 5)

                            top_prob, top_idx = torch.topk(probs[0], 1)
                            cls_name = categories[top_idx.item()] if top_idx.item() < len(categories) else f"class_{int(top_idx.item())}"
                            st.success(f"Prediction: **{cls_name}**  |  Confidence: **{top_prob.item():.2f}**")
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")
                            st.error("Please check that the model is properly loaded and the image was processed correctly.")
                    else:
                        st.error("Failed to process the image for ImageNet. Please try a different image.")
            else:
                st.error("Could not load the uploaded image. Please check the file format and try again.")
