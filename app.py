import streamlit as st

import streamlit.components.v1 as components
# import wget

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
import tempfile

from helper import draw_landmarks_on_image, download


# url = 'https://cdn.pixabay.com/photo/2019/03/12/20/39/girl-4051811_960_720.jpg'
# r = requests.get(url, allow_redirects=True)

# open('image.jpg', 'wb').write(r.content)

download(url='https://cdn.pixabay.com/photo/2019/03/12/20/39/girl-4051811_960_720.jpg', filename='image.jpg')
download(url='https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task', filename='pose_landmarker.task')


# STEP 2: Create an PoseLandmarker object.
# base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
# options = vision.PoseLandmarkerOptions(
#     base_options=base_options,
#     output_segmentation_masks=True)

# detector = vision.PoseLandmarker.create_from_options(options)




BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO)

landmarker = PoseLandmarker.create_from_options(options)

#st.success("landmarker created")

#uploaded_video = st.file_uploader("Video upload, ",accept_multiple_files=False,   type=["mp4"])

## Get Video # https://github.com/mpolinowski/streamLit-cv-mediapipe
stframe = st.empty()
video_file_buffer = st.file_uploader("Video upload, ",accept_multiple_files=False,   type=["mp4"])
temp_file = tempfile.NamedTemporaryFile(delete=False)

if not video_file_buffer:
    pass
    # video = cv2.VideoCapture(DEMO_VIDEO)
    # temp_file.name = DEMO_VIDEO

else:
    temp_file.write(video_file_buffer.read())
    cap = cv2.VideoCapture(temp_file.name)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = int(cap.get(cv2.CAP_PROP_FPS))
st.info(str(width)+"-"+str(height)+" - " + str(fps_input))

## Recording
# codec = cv2.VideoWriter_fourcc('a','v','c','1')
#codec = cv2.VideoWriter_fourcc(*"mp4v")
# codec = cv2.VideoWriter_fourcc('M','J','P','G')
# codec = cv2.VideoWriter_fourcc(*'XVID')

codec = cv2.VideoWriter_fourcc(*'MJPG')

import io
import av

# output_filename = 'output1.mp4'+".tmp"
# output = cv2.VideoWriter(output_filename, codec, fps_input, (width,height))

output_memory_file = io.BytesIO()
output = av.open(output_memory_file, 'w', format="mp4")  # Open "in memory file" as MP4 video output
stream = output.add_stream('h264', str(fps_input))  # Add H.264 video stream to the MP4 container, with framerate = fps.
stream.width = width  # Set frame width
stream.height = height  # Set frame height
#stream.pix_fmt = 'yuv444p'   # Select yuv444p pixel format (better quality than default yuv420p).
stream.pix_fmt = 'yuv420p'   # Select yuv420p pixel format for wider compatibility.
stream.options = {'crf': '17'}  # Select low crf for high quality (the price is larger file size).



save_output = True

if st.button("Process"):
    fps = cap.get(cv2.CAP_PROP_FPS)
    st.write(fps)
    frame_timestamp_ms = []
    frame_timestamp_ms_cont = []
    mp_images = []

    # Check if camera opened successfully
    if (cap.isOpened()== False):
        st.error("Error opening video stream or file")

    last_ms = 0

    # Read until video is completed
    frame_count = cap.get(7)

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            numpy_frame_from_opencv = frame.copy()

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
            mp_images.append(mp_image)

            frame_timestamp_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            frame_timestamp_ms_cont.append(last_ms)
            last_ms = last_ms+33

            # current_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            # pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms=0)

            # # STEP 5: Process the detection result. In this case, visualize it.
            # annotated_image = draw_landmarks_on_image(image.numpy_view(), pose_landmarker_result)
            # cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            # a=input()

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    # Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
    # You’ll need it to calculate the timestamp for each frame.

    # Loop through each frame in the video using VideoCapture#read()

    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    frame_timestamp_ms_input = [int(ts) for ts in frame_timestamp_ms_cont]

    detection_results = []

    X = []
    Y = []
    Z = []
    VISIBILITY = []
    PRESENCE = []
    figs = []

    for index, timestamp in enumerate(frame_timestamp_ms_input):
        mp_image = mp_images[index]
        image = mp_image
        pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms=timestamp)

        detection_results.append(pose_landmarker_result)

        # STEP 5: Process the detection result. In this case, visualize it.
        annotated_image = draw_landmarks_on_image(image.numpy_view(), pose_landmarker_result, [12,13])
        #cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        # print(pose_landmarker_result)
        # a=input()

        ### !!! OJO, hay pose_landmarks y pose_world_landmarks
        ## https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python

        if len(pose_landmarker_result.pose_landmarks) == 0:
            continue

        # print(len(pose_landmarker_result.pose_landmarks))
        # print(len(pose_landmarker_result.pose_landmarks[0])) #33

        landmarks = pose_landmarker_result.pose_landmarks[0]


        xs = []
        ys = []
        zs = []
        vis = []
        pres = []
        for i in landmarks:
            # Acquire x, y but don't forget to convert to integer.
            # x = int(i.x * image.shape[1])
            # y = int(i.y * image.shape[0])
            # Annotate landmarks or do whatever you want.
            #cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            #keypoint_pos.append((x, y))
            vis.append(i.visibility)
            pres.append(i.presence)

            xs.append(i.x)
            ys.append(i.y)
            zs.append(i.z)


        X.append(xs)
        Y.append(ys)
        Z.append(zs)
        VISIBILITY.append(vis)
        PRESENCE.append(pres)

    #     if build_figs:
    #         figs.append(draw_3d_world_plotly(pose_landmarker_result))




        if save_output:
            # output.write(annotated_image)
            frame = av.VideoFrame.from_ndarray(annotated_image, format='bgr24')  # Convert image from NumPy Array to frame.
            packet = stream.encode(frame)  # Encode video frame
            output.mux(packet)

    if save_output:
        # output.release()
        
        
        # Flush the encoder
        packet = stream.encode(None)
        output.mux(packet)
        output.close()

        output_memory_file.seek(0)  # Seek to the beginning of the BytesIO.
        #video_bytes = output_memory_file.read()  # Convert BytesIO to bytes array
        #st.video(video_bytes)
        st.video(output_memory_file)  # Streamlit supports BytesIO object - we don't have to convert it to bytes array.




        # os.system('ffmpeg -i {} -vcodec libx264 {}'.format(output_filename, output_filename.replace('.tmp', '')))

        # video_file = open(output_filename.replace('.tmp', ''), 'rb')
        # video_bytes = video_file.read()

        # st.video(video_bytes)

    #     """