
import joblib
import streamlit as st
import numpy as np
import plotly_express as px
import pandas as pd

def predict_with_coordinates(detection_results, show_details = False):

    model = joblib.load("lr_classifier01.pkl")
    le = joblib.load('label_encoder.pkl')

    x_frames = []   
    y_frames = []
    z_frames = []

    # print(len(detection_results))

    for frame_detection_result in detection_results:

        landmarks = frame_detection_result.pose_world_landmarks[0]
            # print(landmarks)

        x_landmarks = []
        y_landmarks = []
        z_landmarks = []

        for i in landmarks:

            x_landmarks.append(i.x)
            y_landmarks.append(i.y)
            z_landmarks.append(i.z)

        x_frames.append(x_landmarks)
        y_frames.append(y_landmarks)
        z_frames.append(z_landmarks)

    X_np = np.array([[x_frames],[y_frames],[z_frames]])
    # print(np.shape(X_np))
    X_np = np.transpose(X_np, (1, 0, 3, 2))
    array_shape = np.shape(X_np)
    X_np = X_np.reshape(array_shape[0], array_shape[1]*array_shape[2]*array_shape[3])

    pred_class = model.predict(X_np)
    pred_label = le.inverse_transform(pred_class)

    if show_details:
        with st.expander("Prediction probability breakdown"):
            # Get the prediction probabilities
            probabilities = model.predict_proba(X_np)
            instance_index = 0
            prob_data = probabilities[instance_index] * 100  # Convert probabilities to percentages
            class_labels = le.classes_

            # Create a DataFrame for plotting
            df = pd.DataFrame({
                'Class': class_labels,
                'Probability (%)': prob_data
            })

            # Create the horizontal bar chart
            fig = px.bar(df, y='Class', x='Probability (%)', orientation='h', title='Prediction Probabilities')
            fig.update_layout(xaxis_title='Probability (%)', yaxis_title='Class')
            
            st.plotly_chart(fig)

    return pred_label[0]

def get_probability_fig(model, X_np, le):

    probabilities = model.predict_proba(X_np)
    instance_index = 0
    prob_data = probabilities[instance_index] * 100  # Convert probabilities to percentages
    class_labels = le.classes_

    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'Class': class_labels,
        'Probability (%)': prob_data
    })

    # Create the horizontal bar chart
    fig = px.bar(df, y='Class', x='Probability (%)', orientation='h', title='Prediction Probabilities')
    fig.update_layout(xaxis_title='Probability (%)', yaxis_title='Class')

    return fig