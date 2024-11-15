import streamlit as st
import os
import torch
import cv2
import numpy as np
from torchvision.transforms import Compose
from midas.midas_net import MidasNet
from midas.transforms import Resize, NormalizeImage, PrepareForNet

# Constants
IPD = 6.5
MONITOR_W = 38.5
MODEL_PATH = "midas_v21_small-70d6b9c8.pt"
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

# Create folders if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Functions
def write_depth(depth, bits=1, reverse=True):
    depth_min = depth.min()
    depth_max = depth.max()
    max_val = (2**(8*bits))-1
    out = max_val * (depth - depth_min) / (depth_max - depth_min) if depth_max - depth_min > np.finfo("float").eps else 0
    return (max_val - out if not reverse else out).astype("uint16" if bits == 2 else "uint8")

def generate_stereo(left_img, depth):
    h, w, _ = left_img.shape
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    right = np.zeros_like(left_img)
    deviation_cm = IPD * 0.12
    deviation = deviation_cm * MONITOR_W * (w / 1920)
    for row in range(h):
        for col in range(w):
            col_r = col - int((1 - depth[row][col] ** 2) * deviation)
            if col_r >= 0:
                right[row][col_r] = left_img[row][col]
    return right

def overlap(im1, im2):
    composite = np.zeros_like(im1)
    composite[..., 2] = im1[..., 2]
    composite[..., :2] = im2[..., :2]
    return composite

def run_model(image_path, model):
    left_img = cv2.imread(image_path)
    img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB) / 255.0

    midas_model_type = "DPT_Large"  # or use "DPT_Hybrid" for a smaller model
    midas = torch.hub.load("intel-isl/MiDaS", midas_model_type)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    transform = midas_transforms.default_transform
    #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #input_batch = transform(image).unsqueeze(0)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=left_img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    midas_depth = output
    midas_depth = cv2.resize(midas_depth, (left_img.shape[1], left_img.shape[0]))
    midas_depth=cv2.normalize(midas_depth, None, 0, 255, cv2.NORM_MINMAX)
    midas_depth = ((midas_depth - midas_depth.min()) / (midas_depth.max()-midas_depth.min())) * 255
    midas_depth= midas_depth.astype(np.uint8)
    right_img = generate_stereo(left_img, depth_map)
    anaglyph = overlap(left_img, right_img)

    return depth_map, right_img, anaglyph

# Streamlit UI
st.title("Image to Anaglyph Converter")
st.write("Upload up to 5 images (max 4K resolution) to generate depth maps, stereo images, and anaglyph images.")

uploaded_files = st.file_uploader(
    "Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="file_uploader"
)
if uploaded_files:
    if len(uploaded_files) > 5:
        st.error("You can upload up to 5 images only.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MidasNet(MODEL_PATH, non_negative=True).to(device).eval()

        for file in uploaded_files:
            file_path = os.path.join(UPLOAD_FOLDER, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())

            depth_map, right_img, anaglyph = run_model(file_path, model)

            # Save outputs
            depth_map_path = os.path.join(OUTPUT_FOLDER, f"{file.name}_depth.png")
            anaglyph_path = os.path.join(OUTPUT_FOLDER, f"{file.name}_anaglyph.png")
            cv2.imwrite(depth_map_path, depth_map)
            cv2.imwrite(anaglyph_path, anaglyph)

            # Display results
            st.image([file_path, anaglyph_path], caption=["Original", "Anaglyph"])
            st.download_button("Download Anaglyph Image", open(anaglyph_path, "rb"), file.name + "_anaglyph.png")
