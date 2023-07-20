from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import dataclasses
from typing import List, Mapping, Optional, Tuple, Union
import requests
import os
import plotly.express as px
import math
import cv2
import plotly.graph_objects as go
import pandas as pd



_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_BGR_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)


@dataclasses.dataclass
class DrawingSpec:
  # Color for drawing the annotation. Default to the white color.
  color: Tuple[int, int, int] = WHITE_COLOR
  # Thickness for drawing the annotation. Default to 2 pixels.
  thickness: int = 2
  # Circle radius. Default to 2 pixels.
  circle_radius: int = 2

def download(url, filename):
    if not os.path.exists(filename):
        r = requests.get(url, allow_redirects=True)
        open(filename, 'wb').write(r.content)

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def draw_landmark_custom(
    image: np.ndarray, 
    landmark_ids,
    landmark_list: landmark_pb2.NormalizedLandmarkList,
    previous_landmark_px = [],
    connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: Union[DrawingSpec,
                                 Mapping[int, DrawingSpec]] = DrawingSpec(
                                     color=RED_COLOR),
    connection_drawing_spec: Union[DrawingSpec,
                                   Mapping[Tuple[int, int],
                                           DrawingSpec]] = DrawingSpec()):
  """Draws the landmarks and the connections on the image.

  Args:
    image: A three channel BGR image represented as numpy ndarray.
    landmark_list: A normalized landmark list proto message to be annotated on
      the image.
    connections: A list of landmark index tuples that specifies how landmarks to
      be connected in the drawing.
    landmark_drawing_spec: Either a DrawingSpec object or a mapping from hand
      landmarks to the DrawingSpecs that specifies the landmarks' drawing
      settings such as color, line thickness, and circle radius. If this
      argument is explicitly set to None, no landmarks will be drawn.
    connection_drawing_spec: Either a DrawingSpec object or a mapping from hand
      connections to the DrawingSpecs that specifies the connections' drawing
      settings such as color and line thickness. If this argument is explicitly
      set to None, no landmark connections will be drawn.

  Raises:
    ValueError: If one of the followings:
      a) If the input image is not three channel BGR.
      b) If any connetions contain invalid landmark index.
  """
  if not landmark_list:
    return
  if image.shape[2] != _BGR_CHANNELS:
    raise ValueError('Input image must contain three channel bgr data.')
  image_rows, image_cols, _ = image.shape

  idx_to_coordinates = {}

  new_landmarks = []

  for idx, landmark in enumerate(landmark_list.landmark):
    if ((landmark.HasField('visibility') and
         landmark.visibility < _VISIBILITY_THRESHOLD) or
        (landmark.HasField('presence') and
         landmark.presence < _PRESENCE_THRESHOLD)):
      continue
    landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                   image_cols, image_rows)
    if landmark_px:
      idx_to_coordinates[idx] = landmark_px
  # if connections:
  #   num_landmarks = len(landmark_list.landmark)
  #   # Draws the connections if the start and end landmarks are both visible.
  #   for connection in connections:
  #     start_idx = connection[0]
  #     end_idx = connection[1]
  #     if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
  #       raise ValueError(f'Landmark index is out of range. Invalid connection '
  #                        f'from landmark #{start_idx} to landmark #{end_idx}.')
  #     if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
  #       drawing_spec = connection_drawing_spec[connection] if isinstance(
  #           connection_drawing_spec, Mapping) else connection_drawing_spec
  #       cv2.line(image, idx_to_coordinates[start_idx],
  #                idx_to_coordinates[end_idx], drawing_spec.color,
  #                drawing_spec.thickness)
  # Draws landmark points after finishing the connection lines, which is
  # aesthetically better.
  if landmark_drawing_spec:
    for idx, landmark_px in idx_to_coordinates.items():

      if idx in landmark_ids:  ### ONLY DRAW SELECTED LANDMARKS
        drawing_spec = landmark_drawing_spec[idx] if isinstance(
            landmark_drawing_spec, Mapping) else landmark_drawing_spec
        # White circle border
        circle_border_radius = max(drawing_spec.circle_radius + 1,
                                  int(drawing_spec.circle_radius * 1.2))
        cv2.circle(image, landmark_px, circle_border_radius, WHITE_COLOR,
                  drawing_spec.thickness)
        # Fill color into the circle
        cv2.circle(image, landmark_px, drawing_spec.circle_radius,
                  drawing_spec.color, drawing_spec.thickness)

        for prev_landmark_px in previous_landmark_px:
          cv2.circle(image, prev_landmark_px, circle_border_radius, WHITE_COLOR, drawing_spec.thickness)


        new_landmarks.append(landmark_px)
        #print(previous_landmark_px)
        #a = input()

  return new_landmarks
  
      
def draw_landmarks_track(rgb_image, detection_result, landmark_id=[], previous_landmark_px = [], alpha = 1):
  pose_landmarks_list = detection_result.pose_landmarks

  # Create a black canvas of the same size as the image
  height, width, channels = rgb_image.shape
  black_canvas = np.zeros((height, width, channels), dtype=np.uint8)

  # Scale the image's pixel values by the alpha value to fade it
  faded_image = cv2.addWeighted(rgb_image, alpha, black_canvas, 1 - alpha, 0)
  rgb_image = np.copy(faded_image)

  annotated_image = np.copy(rgb_image)


  
  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])

    if landmark_id == []:
      solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
    else:
      #print(previous_landmark_px)
      previous_landmark_px.extend(draw_landmark_custom(
        annotated_image, 
        landmark_id,
        pose_landmarks_proto,
        previous_landmark_px,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style()))
      
  return annotated_image



def plotly_3d_landmarks(landmark_list: landmark_pb2.NormalizedLandmarkList,
                   connections: Optional[List[Tuple[int, int]]] = None,
                   landmark_drawing_spec: DrawingSpec = DrawingSpec(
                       color=RED_COLOR, thickness=5),
                   connection_drawing_spec: DrawingSpec = DrawingSpec(
                       color=BLACK_COLOR, thickness=5),
                   elevation: int = 10,
                   azimuth: int = 10):
  """Plot the landmarks and the connections in matplotlib 3d.

  Args:
    landmark_list: A normalized landmark list proto message to be plotted.
    connections: A list of landmark index tuples that specifies how landmarks to
      be connected.
    landmark_drawing_spec: A DrawingSpec object that specifies the landmarks'
      drawing settings such as color and line thickness.
    connection_drawing_spec: A DrawingSpec object that specifies the
      connections' drawing settings such as color and line thickness.
    elevation: The elevation from which to view the plot.
    azimuth: the azimuth angle to rotate the plot.

  Raises:
    ValueError: If any connection contains an invalid landmark index.
  """
  cn2 = []
  if not landmark_list:
    print("not landmark_list")
    return
#   plt.figure(figsize=(10, 10))
#   ax = plt.axes(projection='3d')
#   ax.view_init(elev=elevation, azim=azimuth)
  plotted_landmarks = {}
  for idx, landmark in enumerate(landmark_list.landmark):
    #print(landmark)
    if ((landmark.HasField('visibility') and
         landmark.visibility < _VISIBILITY_THRESHOLD) or
        (landmark.HasField('presence') and
         landmark.presence < _PRESENCE_THRESHOLD)):
      continue
    # ax.scatter3D(
    #     xs=[-landmark.z],
    #     ys=[landmark.x],
    #     zs=[-landmark.y],
    #     color=_normalize_color(landmark_drawing_spec.color[::-1]),
    #     linewidth=landmark_drawing_spec.thickness)
    plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)
    #print(plotted_landmarks)
  #print(connections)
  if connections:
    out_cn = []
    num_landmarks = len(landmark_list.landmark)
    #print("num_landmarks:",num_landmarks)
    # Draws the connections if the start and end landmarks are both visible.
    for connection in connections:
      #print(connection)
      start_idx = connection[0]
      end_idx = connection[1]
      if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
        raise ValueError(f'Landmark index is out of range. Invalid connection '
                         f'from landmark #{start_idx} to landmark #{end_idx}.')
      if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
        landmark_pair = [
            plotted_landmarks[start_idx], plotted_landmarks[end_idx]
        ]
        out_cn.append(
                    dict(
                        xs=[landmark_pair[0][0], landmark_pair[1][0]],
                        ys=[landmark_pair[0][1], landmark_pair[1][1]],
                        zs=[landmark_pair[0][2], landmark_pair[1][2]],
                    )
                )
        # ax.plot3D(
        #     xs=[landmark_pair[0][0], landmark_pair[1][0]],
        #     ys=[landmark_pair[0][1], landmark_pair[1][1]],
        #     zs=[landmark_pair[0][2], landmark_pair[1][2]],
        #     color=_normalize_color(connection_drawing_spec.color[::-1]),
        #     linewidth=connection_drawing_spec.thickness)
    #print("out_cn:", out_cn)
    cn2 = {"xs": [], "ys": [], "zs": []}
    for pair in out_cn:
        for k in pair.keys():
            cn2[k].append(pair[k][0])
            cn2[k].append(pair[k][1])
            cn2[k].append(None)


  df = pd.DataFrame(plotted_landmarks).T.rename(columns={0: "z", 1: "x", 2: "y"})
  #print(cn2)
  #print(df)
#df["lm"] = df.index.map(lambda s: mp_pose.PoseLandmark(s).name).values             ##OPTIONAL: ADD LM FOR HOVER NAMES
  fig = (
    px.scatter_3d(df, x="z", y="x", z="y")#, hover_name="lm")
        .update_traces(marker={"color": "red"})
        .update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            scene={"camera": {"eye": {"x": 2.1, "y": 0, "z": 0}}},
        )
    )
  fig.add_traces(
        [
            go.Scatter3d(
                x=cn2["xs"],
                y=cn2["ys"],
                z=cn2["zs"],
                mode="lines",
                line={"color": "black", "width": 5},
                name="connections",
            )
        ]
    )
  return fig, cn2


def draw_3d_plotly(detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  #print(pose_landmarks_list)

  if len(pose_landmarks_list) == 0:
    return None

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])

    fig, cn2 = plotly_3d_landmarks(pose_landmarks_proto, solutions.pose.POSE_CONNECTIONS)
    # solutions.drawing_utils.draw_landmarks(
    #   annotated_image,
    #   pose_landmarks_proto,
    #   solutions.pose.POSE_CONNECTIONS,
    #   solutions.drawing_styles.get_default_pose_landmarks_style())
  return fig

def draw_3d_world_plotly(detection_result):
  pose_landmarks_list = detection_result.pose_world_landmarks
  #print(pose_landmarks_list)

  if len(pose_landmarks_list) == 0:
    return None

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_world_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_world_landmarks
    ])

    fig, cn2 = plotly_3d_landmarks(pose_landmarks_proto, solutions.pose.POSE_CONNECTIONS)
    # solutions.drawing_utils.draw_landmarks(
    #   annotated_image,
    #   pose_landmarks_proto,
    #   solutions.pose.POSE_CONNECTIONS,
    #   solutions.drawing_styles.get_default_pose_landmarks_style())
  return fig

def plot_landmark_trajectory(X,Y,Z,VIS,landmark_id):
  X_np = np.array(X)
  Y_np = np.array(Y)
  Z_np = np.array(Z)
  VIS_np = np.array(VIS)

  X = X_np[:,landmark_id]
  Y = Y_np[:,landmark_id]
  Z = Z_np[:,landmark_id]

  print(X[:5])
  print(Y[:5])
  print(Z[:5])

  # Create a 3D scatter plot for the initial position
  fig = go.Figure(data=[go.Scatter3d(x=[X[0]], y=[Y[0]], z=[Z[0]], mode='markers', marker=dict(size=10))])

  # Add a trace for the trajectory
  fig.add_trace(go.Scatter3d(x=X, y=Y, z=Z, mode='lines', line=dict(color='blue', width=2)))

  # Set axis labels and plot title
  fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), title='Landmark Movement')

  # Create animation frames
  frames = [go.Frame(data=[go.Scatter3d(x=X[:k+1], y=Y[:k+1], z=Z[:k+1])], name=str(k)) for k in range(1, len(X))]

  # Add the frames to the figure
  fig.update(frames=frames)

  # Define animation settings
  animation_settings = dict(frame=dict(duration=50, redraw=True), fromcurrent=True)

  # Add play and pause buttons to the animation
  fig.update_layout(updatemenus=[dict(type='buttons', showactive=False, buttons=[dict(label='Play', method='animate', args=[None, animation_settings]),
                                                                            dict(label='Pause', method='animate', args=[[None], animation_settings])])])

  # Set initial frame
  fig.update_layout(updatemenus=[dict(type='buttons', showactive=False, buttons=[dict(label='Play', method='animate', args=[None, animation_settings]),
                                                                            dict(label='Pause', method='animate', args=[[None], animation_settings])])],
                    sliders=[dict(currentvalue={'prefix': 'Frame: '}, steps=[dict(args=[[k], animation_settings],
                                                                                  label=str(k),
                                                                                  method='animate') for k in range(1, len(X))])])

  # fig.update_layout(scene = dict(xaxis=dict(range=[-2,2], autorange=False),
  #                               yaxis=dict(range=[-1,1], autorange=False),
  #                               zaxis=dict(range=[-1,1], autorange=False),


  #                               camera = {"eye": {"x": 1, "y": 1, "z": 2}}
  #                               ))
  return fig