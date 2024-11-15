import os
import torch
import cv2
import numpy as np
import streamlit as st
from torchvision.transforms import Compose
from PIL import Image
from io import BytesIO

# Constants
MAX_IMAGES = 5
MAX_RESOLUTION = (3840, 2160)  # 4K resolution

# MiDaS model setup
midas_model_type = "MiDaS_small"  # or "DPT_Hybrid" for a smaller model
midas = torch.hub.load("intel-isl/MiDaS", midas_model_type)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.default_transform

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Depth estimation function
def estimate_depth(image):
    input_batch = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return depth_map

# Generate stereo image
def generate_stereo(left_img, depth_map, ipd=6.5, monitor_width=38.5):
    h, w, _ = left_img.shape
    deviation_cm = ipd * 0.12
    deviation = deviation_cm * monitor_width * (w / 1920)
    right = np.zeros_like(left_img)

    for row in range(h):
        for col in range(w):
            col_r = col - int((1 - (depth_map[row, col] / 255) ** 2) * deviation)
            if col_r >= 0:
                right[row, col_r] = left_img[row, col]

    # Fill missing pixels
    gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    rows, cols = np.where(gray == 0)
    for row, col in zip(rows, cols):
        for offset in range(1, int(deviation)):
            r_offset = col + offset
            l_offset = col - offset
            if r_offset < w and not np.all(right[row, r_offset] == 0):
                right[row, col] = right[row, r_offset]
                break
            if l_offset >= 0 and not np.all(right[row, l_offset] == 0):
                right[row, col] = right[row, l_offset]
                break

    return right

# Combine images into anaglyph
def create_anaglyph(left_img, right_img):
    anaglyph = np.zeros_like(left_img)
    anaglyph[..., 2] = left_img[..., 2]  # Red channel
    anaglyph[..., :2] = right_img[..., :2]  # Green and Blue channels
    return anaglyph

# Streamlit app
st.title("Image to Anaglyph Converter")
st.write("Upload up to 5 images (4K resolution max) to convert to anaglyph images.")

uploaded_files = st.file_uploader(
    "Upload Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    if len(uploaded_files) > MAX_IMAGES:
        st.error(f"Please upload a maximum of {MAX_IMAGES} images.")
    else:
        for uploaded_file in uploaded_files:
            # Process each file
            image = Image.open(uploaded_file)
            image = image.convert("RGB")
            img_array = np.array(image)

            if img_array.shape[0] > MAX_RESOLUTION[1] or img_array.shape[1] > MAX_RESOLUTION[0]:
                st.warning(f"Image {uploaded_file.name} exceeds 4K resolution and will be resized.")
                img_array = cv2.resize(img_array, MAX_RESOLUTION[::-1])

            # Convert to depth map
            depth_map = estimate_depth(img_array)

            # Generate stereo and anaglyph
            right_img = generate_stereo(img_array, depth_map)
            anaglyph = create_anaglyph(img_array, right_img)

            # Display results
            st.image(anaglyph, caption=f"Anaglyph for {uploaded_file.name}", use_column_width=True)

            # Download link
            output_img = Image.fromarray(anaglyph)
            buf = BytesIO()
            output_img.save(buf, format="PNG")
            byte_data = buf.getvalue()
            st.download_button(
                label="Download Anaglyph",
                data=byte_data,
                file_name=f"anaglyph_{uploaded_file.name}",
                mime="image/png",
            )
