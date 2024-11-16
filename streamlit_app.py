import streamlit as st
import torch
import numpy as np
import cv2
from torchvision.transforms import Compose
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet
IPD = 6.5
MONITOR_W = 38.5
# Initialize the model and transformations
@st.cache_resource
def load_model():
    model_path = "midas_v21_small-70d6b9c8.pt"  # Replace with the path to your small MiDaS model
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
def write_depth(depth, bits=1, reverse=True):
    depth_min = depth.min()
    depth_max = depth.max()
    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = 0
    if not reverse:
        out = max_val - out

    if bits == 2:
        depth_map = out.astype("uint16")
    else:
        depth_map = out.astype("uint8")

    return depth_map


def generate_stereo(left_img, depth):
    h, w, c = left_img.shape

    depth_min = depth.min()
    depth_max = depth.max()
    depth = (depth - depth_min) / (depth_max - depth_min)
    depth=depth.astype(np.uint8)
    depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    right = np.zeros_like(left_img)

    deviation_cm = IPD * 0.12
    deviation = deviation_cm * MONITOR_W * (w / 1920)

    print("\ndeviation:", deviation)

    for row in range(h):
        for col in range(w):
            col_r = col - int((1 - depth[row][col] ** 2) * deviation)
            # col_r = col - int((1 - depth[row][col]) * deviation)
            if col_r >= 0:
                right[row][col_r] = left_img[row][col]

    right_fix = np.array(right)
    gray = cv2.cvtColor(right_fix, cv2.COLOR_BGR2GRAY)
    rows, cols = np.where(gray == 0)
    for row, col in zip(rows, cols):
        for offset in range(1, int(deviation)):
            r_offset = col + offset
            l_offset = col - offset
            if r_offset < w and not np.all(right_fix[row][r_offset] == 0):
                right_fix[row][col] = right_fix[row][r_offset]
                break
            if l_offset >= 0 and not np.all(right_fix[row][l_offset] == 0):
                right_fix[row][col] = right_fix[row][l_offset]
                break

    return right_fix


def overlap(im1, im2):
    width1 = im1.shape[1]
    height1 = im1.shape[0]
    width2 = im2.shape[1]
    height2 = im2.shape[0]

    # final image
    composite = np.zeros((height2, width2, 3), np.uint8)

    # iterate through "left" image, filling in red values of final image
    for i in range(height1):
        for j in range(width1):
            try:
                composite[i, j, 2] = im1[i, j, 2]
            except IndexError:
                pass

    # iterate through "right" image, filling in blue/green values of final image
    for i in range(height2):
        for j in range(width2):
            try:
                composite[i, j, 1] = im2[i, j, 1]
                composite[i, j, 0] = im2[i, j, 0]
            except IndexError:
                pass

    return composite

# Process the uploaded image
def process_image(image, model, transform, device):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    img_input = transform({"image": img})["image"]
    #img_input = torch.from_numpy(img_input).to(device).unsqueeze(0)

    with torch.no_grad():
        image = torch.from_numpy(img_input).to(device).unsqueeze(0)
        depth = model.forward(image)
        depth = (
                torch.nn.functional.interpolate(
                    depth.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        depth = cv2.blur(depth, (3, 3))
    #depth_map = write_depth(depth, bits=2, reverse=False)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth_map = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    #depth_map = write_depth(depth, bits=2, reverse=False)
    return depth_map

# Streamlit App Interface
st.title("Image to Depth Map and Anaglyph Generator")

# File upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    file_bytes = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = file_bytes
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load the model
    with st.spinner("Loading model..."):
        model, transform, device = load_model()

    # Generate depth map
    with st.spinner("Generating depth map..."):
        depth_map = process_image(image, model, transform, device)
        #depth_map = write_depth(depth, bits=2, reverse=False)
        # Display depth map
        #depth_map=cv2.imencode('.png', depth_map)[1].tobytes()
        st.image(depth_map, caption="Depth Map", use_column_width=True, clamp=True, channels="GRAY")
    with st.spinner("Generating right..."):
        right_img = generate_stereo(image, depth_map)
        #right_img=cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        #depth_map = write_depth(depth, bits=2, reverse=False)
        # Display depth map
        #depth_map=cv2.imencode('.png', depth_map)[1].tobytes()
        st.image(right_img, caption="Depth Map", use_column_width=True, clamp=True, channels="GRAY")
    # Generate anaglyph
    with st.spinner("Generating anaglyph..."):
        right_img = generate_stereo(image, depth_map)
        right_img=cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        #stereo = np.hstack([image, right_img])
        left_red = image[:, :, 0]  # Red channel from the left image
        right_green = right_img[:, :, 1]  # Green channel from the right image
        right_blue = right_img[:, :, 2]  # Blue channel from the right image

        # Create anaglyph by combining channels
        anaglyph = np.zeros_like(image)
        anaglyph[:, :, 0] = left_red  # Red channel from left image
        anaglyph[:, :, 1] = right_green  # Green channel from right image
        anaglyph[:, :, 2] = right_blue  # Blue ch Green and Blue channels from the right image

        # Create anaglyph by combining channels
        #anaglyph = np.zeros_like(image)
        #anaglyph[:, :, 2] = left_red  # Red channel from left image
        #anaglyph[:, :, 0:2] = right_green_blue 
        #anaglyph=cv2.bitwise_not(anaglyph)
        #anaglyph=cv2.cvtColor(anaglyph, cv2.COLOR_BGR2RGB)
        #anaglyph = cv2.cvtColor(anaglyph, cv2.COLOR_RGB2BGR)
        st.image(anaglyph, caption="Anaglyph Image", use_column_width=True)

st.write("Developed with MiDaS and Streamlit.")
