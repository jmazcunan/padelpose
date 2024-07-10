import streamlit as st
import io
import av
import cv2
import numpy as np
import mediapipe as mp
import os
import tempfile
import plotly.graph_objects as go
import plotly.express as px
import joblib
import time
from datetime import datetime

from inference import predict_with_coordinates
from helper import draw_landmarks_track, download, plot_landmark_trajectory, draw_3d_world_plotly
from landmark_ids import pose_landmark_id

st.set_page_config(page_title="Padel Pose", page_icon="🎾", layout="centered", initial_sidebar_state="auto", menu_items=None)

download(url='https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task', filename='pose_landmarker.task')

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO)

landmarker = PoseLandmarker.create_from_options(options)

sel_alpha = 0.6


if "processed" not in st.session_state:
    st.session_state.processed = False

if "cropped_temp_file" not in st.session_state:
    with st.expander("Video input", expanded = not(st.session_state.processed)):
        video_file_buffer = st.file_uploader("Video upload",accept_multiple_files=False, type=["mp4"])

        if not video_file_buffer:
            st.session_state.processed = False
            st.stop()


        else:
            if "cropped_temp_file" not in st.session_state:
                if "temp_file" not in st.session_state: #prevent creating several tempfiles for every interaction
                    st.session_state.temp_file = tempfile.NamedTemporaryFile(delete=False)
                    st.session_state.temp_file.write(video_file_buffer.read())

                # Cargar el vídeo usando OpenCV
                cap = cv2.VideoCapture(st.session_state.temp_file.name)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                output_memory_file_raw = io.BytesIO()
                output_raw = av.open(output_memory_file_raw, 'w', format="mp4")  # Open "in memory file" as MP4 video output
                
                # Verificar si el vídeo tiene al menos 60 frames
                if total_frames < 60:
                    st.error("El vídeo debe tener al menos 60 frames.")
                else:
                    # Seleccionar el frame inicial usando un deslizador
                    if total_frames == 60:
                        start_frame = 0
                    else:
                        start_frame = st.slider('Delay', 0, total_frames - 60, 0)
                    
                    # Ir al frame inicial seleccionado más 30 frames
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + 30)
                    ret, frame = cap.read()
                    
                    
                    if ret:
                        # Mostrar la imagen del frame 30
                        st.image(frame, caption='Adjust the delay until you see the player impacting the ball', channels = "BGR")

                    # Botón para confirmar y recortar el vídeo
                    # st.write(fps)
                    if st.button('Process video'):
                        # Crear un archivo temporal para el vídeo recortado
                        cropped_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        
                        # Crear el contenedor de vídeo
                        output_raw = av.open(cropped_temp_file.name, 'w', format='mp4')
                        stream_raw = output_raw.add_stream('h264', int(fps))
                        stream_raw.width = width
                        stream_raw.height = height
                        stream_raw.pix_fmt = 'yuv420p'
                        stream_raw.options = {'crf': '17'}

                        # Ir al frame inicial seleccionado
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                        # Leer y escribir los siguientes 60 frames
                        for i in range(60):
                            ret, frame = cap.read()
                            if not ret:
                                break

                            frame = av.VideoFrame.from_ndarray(frame, format='bgr24')  # Convertir imagen de NumPy Array a frame
                            packet = stream_raw.encode(frame)  # Codificar el frame de vídeo
                            output_raw.mux(packet)  # Multiplexar el packet al contenedor

                        # Liberar los recursos
                        cap.release()

                        # Flush the encoder
                        packet = stream_raw.encode(None)
                        output_raw.mux(packet)
                        output_raw.close()

                        st.session_state.cropped_temp_file = cropped_temp_file
                        st.session_state.cap = cv2.VideoCapture(st.session_state.cropped_temp_file.name)
                        st.rerun()

                        # # Mostrar el vídeo recortado
                        # st.video(cropped_temp_file.name)



if ("cropped_temp_file" in st.session_state) and (not st.session_state.processed):
    # st.write(temp_file.name)
    cap = cv2.VideoCapture(st.session_state.cropped_temp_file.name)

    # st.write(st.session_state)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(cap.get(cv2.CAP_PROP_FPS))

    if "feedback_done" in st.session_state:
        del st.session_state.feedback_done
    output_memory_file_raw = io.BytesIO()
    output_raw = av.open(output_memory_file_raw, 'w', format="mp4")  # Open "in memory file" as MP4 video output
    stream_raw = output_raw.add_stream('h264', str(fps_input))  # Add H.264 video stream to the MP4 container, with framerate = fps.
    stream_raw.width = width  # Set frame width
    stream_raw.height = height  # Set frame height
    #stream.pix_fmt = 'yuv444p'   # Select yuv444p pixel format (better quality than default yuv420p).
    stream_raw.pix_fmt = 'yuv420p'   # Select yuv420p pixel format for wider compatibility.
    stream_raw.options = {'crf': '17'}  # Select low crf for high quality (the price is larger file size).
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    # st.write(fps)
    frame_timestamp_ms = []
    frame_timestamp_ms_cont = []
    # mp_images = []
    figs = []

    # Check if camera opened successfully
    if (cap.isOpened()== False):
        st.error("Error opening video stream or file")

    last_ms = 0

    # Read until video is completed
    frame_count = cap.get(7)
    # st.write(frame_count)
    detection_results = []

    progress_bar = st.progress(0, text="Processing video")
    
    frame_idx = 0

    while(cap.isOpened()):
        formatted_percentage = f"{frame_idx/frame_count:.2%}"
        progress_bar.progress(frame_idx/frame_count, text=f"Processing video: {formatted_percentage}")

        # progress_bar.progress(frame_idx/frame_count, text=f"Processing video: {(index/detections_length).2%}")
        frame_idx+=1

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            numpy_frame_from_opencv = frame.copy()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
            # mp_images.append(mp_image)
            frame_timestamp_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            frame_timestamp_ms_cont.append(last_ms)
            
            pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms=last_ms)
            last_ms = last_ms+33

            detection_results.append(pose_landmarker_result)
            annotated_image = draw_landmarks_track(mp_image.numpy_view(), pose_landmarker_result, alpha = sel_alpha)

            frame = av.VideoFrame.from_ndarray(annotated_image, format='bgr24')  # Convert image from NumPy Array to frame.
            packet = stream_raw.encode(frame)  # Encode video frame
            output_raw.mux(packet)

            figs.append(draw_3d_world_plotly(pose_landmarker_result))


        # Break the loop
        else:
            break
 
        
    # Flush the encoder
    packet = stream_raw.encode(None)
    output_raw.mux(packet)
    output_raw.close()

    output_memory_file_raw.seek(0)  # Seek to the beginning of the BytesIO.
    
    st.session_state.raw_video = output_memory_file_raw

    # Release the video capture object
    cap.release()
    #st.session_state.mp_images = mp_images
    st.session_state.detection_results = detection_results
    st.session_state.processed = True
    st.session_state.figs = figs
    st.rerun()


if st.session_state.processed:

    if st.button("New video"):
        del st.session_state.temp_file
        del st.session_state.cropped_temp_file
        del st.session_state.processed
        st.rerun()

    st.video(st.session_state.raw_video) 
    pred_label = predict_with_coordinates(st.session_state.detection_results, show_details=True)

    full_names_en = {"der": "Forehand",
                  "rev": "Backhand",
                  "vde": "Forehand volley", 
                  "vre": "Backhand volley",
                  "ban": "Bandeja",
                  "rem": "Smash"}
    
    full_names_es = {"der": "Derecha",
                  "rev": "Revés",
                  "vde": "Volea de derecha", 
                  "vre": "Volea de revés",
                  "ban": "Bandeja",
                  "rem": "Remate"}
    
    full_names = full_names_en

    def save_feedback(feedback_type, stroke):
        

        # Get the current datetime
        now = datetime.now()

        # Format the datetime string
        datetime_string = now.strftime("%Y%m%d_%H%M%S")
        
        foldername = os.path.join("feedback_data",feedback_type)

        if not os.path.exists(foldername):
            os.makedirs(foldername)
        
        joblib.dump(st.session_state.detection_results, os.path.join(foldername,stroke+os.path.basename(datetime_string)+".pkl"))
        st.toast("Thank you for your feedback!")
        st.session_state.feedback_done = feedback_type
        st.rerun()
   

    with st.container(border=True):
        fdb_cols = st.columns((4, 1, 1, 1))
        fdb_cols[0].success(f'Predicted class: {full_names[pred_label]}')

        if "feedback_done" not in st.session_state:

            if fdb_cols[-2].button("✅"):
                save_feedback(feedback_type="positive", stroke = pred_label)
                
            with fdb_cols[-1].popover("❌"):
                st.caption("Help us improve the model by providing the right answer!")
                possible_strokes = full_names.pop(pred_label)
                # possible_strokes.pop(pred_label[0])

                sel_stroke = st.selectbox("Correct stroke", full_names, format_func=lambda x: full_names[x])
                if st.button("Submit feedback"):
                    save_feedback(feedback_type="negative", stroke = sel_stroke)
        else:
            if str(st.session_state.feedback_done)=="positive":
                saved_feedback = "✅"
                fdb_cols[-2].button(saved_feedback, disabled=True)
            else:
                saved_feedback = "❌"
                fdb_cols[-1].button(saved_feedback, disabled=True)
        
        st.empty()
    

    tabs = st.tabs(["Track landmark", "View 3D"])

    with tabs[0]:
        pose_landmark_names = pose_landmark_id.keys()
        sel_landmark_ids = st.multiselect("Landmark", pose_landmark_id)
        landmark_ids = [pose_landmark_id[sel_landmark_id] for sel_landmark_id in sel_landmark_ids]

        if landmark_ids == []:
            st.info("Select a landmark")


        if st.button("Track") and (landmark_ids != []):

            cap = cv2.VideoCapture(st.session_state.cropped_temp_file.name)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_input = int(cap.get(cv2.CAP_PROP_FPS))

            output_memory_file = io.BytesIO()
            output = av.open(output_memory_file, 'w', format="mp4")  # Open "in memory file" as MP4 video output
            stream = output.add_stream('h264', str(fps_input))  # Add H.264 video stream to the MP4 container, with framerate = fps.
            stream.width = width  # Set frame width
            stream.height = height  # Set frame height
            #stream.pix_fmt = 'yuv444p'   # Select yuv444p pixel format (better quality than default yuv420p).
            stream.pix_fmt = 'yuv420p'   # Select yuv420p pixel format for wider compatibility.
            stream.options = {'crf': '17'}  # Select low crf for high quality (the price is larger file size).

            save_output = True
            # build_figs = True

            progress_bar = st.progress(0, text="Generating output video")

            X = []
            Y = []
            Z = []
            VISIBILITY = []
            PRESENCE = []
            figs = []
            prev_landmarks = []
            detections_length = len(st.session_state.detection_results)

            for index, detection_result in enumerate(st.session_state.detection_results):

                #while(cap.isOpened()):
                formatted_percentage = f"{(index+1)/detections_length:.2%}"
                progress_bar.progress((index+1)/detections_length, text=f"Generating output video: {formatted_percentage}")
                # frame_idx+=1

                # Capture frame-by-frame
                ret, frame = cap.read()
                if ret == True:

                    numpy_frame_from_opencv = frame.copy()

                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
                    
                #mp_image = st.session_state.mp_images[index]
                image = mp_image


                # Process the detection result.
                annotated_image = draw_landmarks_track(image.numpy_view(), detection_result, landmark_ids, prev_landmarks, alpha = sel_alpha)
                ## https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python

                if len(detection_result.pose_landmarks) == 0:
                    continue

                ## USAMOS WORLD LANDMARKS PARA LOS PLOTS 3D POSTERIORES
                landmarks = detection_result.pose_world_landmarks[0]


                xs = []
                ys = []
                zs = []
                vis = []
                pres = []
                for i in landmarks:
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

                if save_output:
                    # output.write(annotated_image)
                    frame = av.VideoFrame.from_ndarray(annotated_image, format='bgr24')  # Convert image from NumPy Array to frame.
                    packet = stream.encode(frame)  # Encode video frame
                    output.mux(packet)

            if save_output:

                # Flush the encoder
                packet = stream.encode(None)
                output.mux(packet)
                output.close()

                output_memory_file.seek(0)  # Seek to the beginning of the BytesIO.
                st.video(output_memory_file)  # Streamlit supports BytesIO object - we don't have to convert it to bytes array.

            X_np = np.array(X)
            Y_np = np.array(Y)
            Z_np = np.array(Z)
            VIS_np = np.array(VISIBILITY)
                        
            for landmark_name in sel_landmark_ids:
                landmark_id = pose_landmark_id[landmark_name]
                st.subheader(landmark_name)

                st.plotly_chart(plot_landmark_trajectory(X,Y,Z,VISIBILITY,landmark_id))
                #st.write(landmark_id)
            
                fig = go.Figure()
                fig = px.scatter(x=Y_np[:,landmark_id], y=Z_np[:,landmark_id], color = VIS_np[:,landmark_id], size = VIS_np[:,landmark_id], color_continuous_scale=px.colors.sequential.Bluered_r)

                fig.add_trace(go.Scatter(x=Y_np[:,landmark_id], y=Z_np[:,landmark_id],
                                    mode='lines',
                                    name='yz'))

                fig.update_layout(
                    title="YZ "+landmark_name,
                    xaxis_title="Y",
                    yaxis_title="Z")
                st.plotly_chart(fig)

    with tabs[1]:
        sel_frame = st.slider("Select frame", 1, len(st.session_state.figs), 1)
        button_placeholder = st.empty()
        plot_placeholder = st.empty()
        plot_placeholder.plotly_chart(st.session_state.figs[sel_frame-1], width=50)

        if button_placeholder.button("Animate"):
            for frame in range(1, len(st.session_state.figs)):
                plot_placeholder.plotly_chart(st.session_state.figs[frame])
                time.sleep(0.1)