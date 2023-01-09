import os

import cv2
import pathlib

import torch
import numpy as np
from image_restoration.HyperRes1 import HyperRes
# from models import HyperRes, NoiseNet
from image_restoration.NoiseNet import NoiseNet
# from utils.DataUtils.CommonTools import modcrop, ToTensor, calculate_psnr, postProcessForStats
from image_restoration.CommonTools import modcrop, ToTensor, calculate_psnr, postProcessForStats

#### --input test/Set5/ --checkpoint pre_trained/Dej/model_best.pth --data_type j
# Global variables
alpha_slider_max = 150  # Maximum value for the slider controlling noise level
title_window = 'MoNet LiveDemo'  # Title for the window displaying the image
min_v = 15  # Minimum value for random noise level
max_v = 75  # Maximum value for random noise level

device = "cuda"


# Function to add deblocking to an image
def addDeJPEG(src1, lvl=None):
    global title_window

    # If no specific deblocking level is specified, generate a random one
    if not lvl:
        lvl = np.random.randint(10, 100)
    # Update window title with deblocking level
    title_window = "SR Factor {}".format(lvl)

    # Write image to JPEG file with specified deblocking level, then read it back in
    cv2.imwrite('tmp.jpg', src1 * 255, [int(cv2.IMWRITE_JPEG_QUALITY), lvl])
    # ret_img = cv2.imread('tmp.jpg', 0) / 255
    ret_img = cv2.imread('tmp.jpg') / 255
    # Delete temporary file
    os.remove('tmp.jpg')
    # Return image and deblocking level
    return np.clip(ret_img, 0, 1), lvl


# Load NoiseNet model
def loadNoiseNet(device):
    if not os.path.exists('pre_trained/noise_net_latest.pth'):
        print("No weights for NoiseNet")
        return None

    noise_model = NoiseNet().to(device)  # Create NoiseNet model
    noise_model.load_state_dict(torch.load('pre_trained/noise_net_latest.pth', map_location=device),
                                strict=False)  # Load weights
    return noise_model


def image_restoration(img_path, filename):
    device = 'cpu'
    curr_dict = {"j": addDeJPEG}
    # Load specified image(s)
    if os.path.isdir(img_path):
        img_path = os.path.join(img_path, np.random.choice(os.listdir(img_path)))
    print(img_path)
    assert (os.path.exists(img_path))
    gt = cv2.imread(img_path)

    # Modify the image by cropping by a factor of 2
    gt = modcrop(gt, 2)

    # Set the maximum value for the alpha trackbar based on data type

    alpha_slider_max = 100

    # Select a function to corrupt the image based on the data type
    corrupt_fun = curr_dict["j"]
    # Corrupt the image and store the noise level
    src1, rnd_noise = corrupt_fun(gt / 255, 20)

    # Get the height and width of the image
    h, w = gt.shape[:2]
    # Create a canvas for displaying the original and processed images side by side
    canvas = np.zeros((h, 2 * w, 3)).astype(np.uint8)
    # Display the corrupted image on the left side of the canvas
    disp_src = src1

    d_h, d_w = disp_src.shape[:2]
    canvas[:d_h, :d_w] = disp_src * 255

    # Convert the corrupted image to a tensor and move it to the designated device
    src1 = ToTensor()(src1).to(device)

    # Create model
    checkpoint = "image_restoration/Dej/model_best.pth"

    checkpoint = torch.load(checkpoint, map_location=device)

    # Load model weights from checkpoint
    model = HyperRes(meta_blocks=16, level=[15], device=device,
                     bias=True,
                     gray=False,
                     norm_factor=255)
    # Load the model weights
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.to(device)

    noise_net = loadNoiseNet(device)

    init_noise = 45
    if noise_net:
        init_noise = noise_net(src1.unsqueeze(0))
        print("Noise in image:\t{}\nPredicted Noise:\t{:.2f}".format(rnd_noise, init_noise.item()))
        print("==================================\n")

    # Set the model's level to the trackbar value
    model.setLevel(50)
    # Process the image
    with torch.no_grad():
        dst = model([src1.unsqueeze(0)])[0]
    # Post-process the image and get the PSNR
    dst = postProcessForStats(dst)[0]
    canvas[:, w:] = dst
    picture_path = os.path.abspath('flask_web/static/res_pics')
    pathlib.Path(picture_path).mkdir(parents=True, exist_ok=True)  # create all folders in the given path.
    os.chdir(picture_path)
    cv2.imwrite(filename, canvas)
