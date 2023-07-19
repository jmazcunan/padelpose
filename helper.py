from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import dataclasses
from typing import List, Mapping, Optional, Tuple, Union
import requests
import os

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


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
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
