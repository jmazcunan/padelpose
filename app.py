import streamlit as st

import streamlit.components.v1 as components

import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import requests

st.set_page_config(page_title="Padel Pose", page_icon="🎾", layout="centered", initial_sidebar_state="auto", menu_items=None)

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

from helper import draw_landmarks_on_image

def download(url, filename):
    if not os.path.exists(filename):
        r = requests.get(url, allow_redirects=True)
        open(filename, 'wb').write(r.content)


# url = 'https://cdn.pixabay.com/photo/2019/03/12/20/39/girl-4051811_960_720.jpg'
# r = requests.get(url, allow_redirects=True)

# open('image.jpg', 'wb').write(r.content)

download(url='https://cdn.pixabay.com/photo/2019/03/12/20/39/girl-4051811_960_720.jpg', filename='image.jpg')
download(url='https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task ', filename='pose_landmarker.task')


# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("image.jpg")

# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
# cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

st.image(annotated_image)