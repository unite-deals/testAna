import streamlit as st
import torch
import numpy as np
import cv2
from torchvision.transforms import Compose
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet

# Initialize the model and transformations
@st.cache_resource
def load_model():
    model_path = "model-small.pt"  # Replace with the path to your small MiDaS model
    model = MidasNet_small(
        model_path,
        features=64,
        backbone="efficientnet_lite3",
        exportable=True,
        non_negative=True,
        blocks={"expand": True},
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    net_w, net_h = 256, 256
    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )
    return model, transform, device

# Convert depth map to anaglyph
def create_anaglyph(image, depth_map):
    h, w = depth_map.shape
    shift = 10  # Shift value for the anaglyph effect
    anaglyph = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Normalize depth map to range 0-255
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Create red (left) and cyan (right) channels
    red_channel = cv2.addWeighted(image[:, :, 2], 1, depth_map, 0.5, 0)
    cyan_channel = cv2.addWeighted(image[:, :, 0], 1, depth_map, 0.5, 0)
    
    # Shift the cyan channel for the 3D effect
    cyan_channel_shifted = np.roll(cyan_channel, shift, axis=1)
    
    # Combine channels
    anaglyph[:, :, 2] = red_channel  # Red
    anaglyph[:, :, 0] = cyan_channel_shifted  # Cyan
    
    return anaglyph

# Process the uploaded image
def process_image(image, model, transform, device):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_input = transform({"image": img})["image"]
    img_input = torch.from_numpy(img_input).to(device).unsqueeze(0)

    with torch.no_grad():
        depth_map = model(img_input)
        depth_map = (
            torch.nn.functional.interpolate(
                depth_map.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
    
    return depth_map

# Streamlit App Interface
st.title("Image to Depth Map and Anaglyph Generator")

# File upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load the model
    with st.spinner("Loading model..."):
        model, transform, device = load_model()

    # Generate depth map
    with st.spinner("Generating depth map..."):
        depth_map = process_image(image, model, transform, device)
        
        # Display depth map
        st.image(depth_map, caption="Depth Map", use_column_width=True, clamp=True, channels="GRAY")

    # Generate anaglyph
    with st.spinner("Generating anaglyph..."):
        anaglyph_image = create_anaglyph(image, depth_map)
        st.image(anaglyph_image, caption="Anaglyph Image", use_column_width=True)

st.write("Developed with MiDaS and Streamlit.")
