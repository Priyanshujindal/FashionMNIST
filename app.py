import streamlit as st
from implementation import ImageClassifier,get_dataloaders
from PIL import Image
from torchvision import transforms
import torch
st.title("Fashion MNIST Classifier")
st.write('Classes Avaiable : '
    'T-shirt/top ,',
 'Trouser , ',
 'Pullover , ',
 'Dress , ',
 'Coat , ',
 'Sandal , ',
 'Shirt , ',
 'Sneaker , ',
 'Bag , ',
 'Ankle boot')
trainset, _, _, _ = get_dataloaders()
classes = trainset.classes
@st.cache_resource
def load_model():
    model = ImageClassifier(input_shape=1, hidden_shape=10, output_shape=10)  # Use len(classes) if available
    model.load_state_dict(torch.load('model/model_weights.pth', map_location='cpu'))
    model.eval()
    return model
uploaded_file=st.file_uploader("choose the image",type=['img','jpg','png'])
if uploaded_file is not None:
    image=Image.open(uploaded_file).convert('L') #to convert to the gray scale as currently it is in the rgb
    st.image(image,caption='uploaded image')
    transforms=transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,),(0.5,))
    ])
    image_tensor=transforms(image).unsqueeze(0)
    image_tensor = 1 - image_tensor 
    model=load_model()
    with torch.inference_mode():
            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=1)            
            top_prob, pred_idx = torch.max(probs, dim=1)

            st.write("Class probabilities:", probs.numpy())  

            if top_prob.item() < 0.6:  
                st.write("Cannot classify object confidently")
            else:
                st.write(f"Prediction: {classes[pred_idx.item()]} "
                        f"(Confidence: {top_prob.item():.2f})")