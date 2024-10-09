"""
    Aggregate landmarks

Aggregate/Convert 33 landmarks into OpenPose 17 landmarks.
The dictionary below shows the mapping and aggregation instruction
{
    0: 0,
    1: 5,
    2: 8,
    3: 2,
    4: 7,
    5: (9 + 10)/2,
    6: 12,
    7: 14,
    8: (16 + 18 + 20)/3,
    9: 11,
    10: 13,
    11: (15 + 17 + 19)/3,
    12: 24,
    13: 26,
    14: (28 + 30 + 32)/3,
    15: 23,
    16: 25,
    17: (27 + 29 + 31)/3,
}
"""

from typing import Sequence, Tuple

import mediapipe as mp
import yaml
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw

from configs.factory import HRNetConfig
from src.datasets.ltcc import LTCC

CONFIG_PATH = "./configs/hrnet.yaml"
LTCC_PATH = "/media/jurgen/Documents/datasets/LTCC_ReID"
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="./pretrained/pose_landmarker_heavy.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
)

detector = vision.PoseLandmarker.create_from_options(options)


def load_config(path: str):
    with open(file=path, mode="r") as f:
        payload = yaml.load(stream=f, Loader=yaml.FullLoader)
        config = HRNetConfig(**payload)
    return config


def parse_position(
    image_size: Tuple[int, int],
    landmark: mp.tasks.components.containers.NormalizedLandmark,
):
    return [landmark.x * image_size[0], landmark.y * image_size[1]]


def parse_landmarks(image_size: Tuple[int, int], result: vision.PoseLandmarkerResult):
    pose_landmarks = result.pose_landmarks
    landmark = pose_landmarks[0]
    payload = {
        "nose": parse_position(image_size=image_size, landmark=landmark[0]),
        "left eye (inner)": parse_position(image_size=image_size, landmark=landmark[1]),
        "left eye": parse_position(image_size=image_size, landmark=landmark[2]),
        "left eye (outer)": parse_position(image_size=image_size, landmark=landmark[3]),
        "right eye (inner)": parse_position(
            image_size=image_size, landmark=landmark[4]
        ),
        "right eye": parse_position(image_size=image_size, landmark=landmark[5]),
        "right eye (outer)": parse_position(
            image_size=image_size, landmark=landmark[6]
        ),
        "left ear": parse_position(image_size=image_size, landmark=landmark[7]),
        "right ear": parse_position(image_size=image_size, landmark=landmark[8]),
        "mouth (left)": parse_position(image_size=image_size, landmark=landmark[9]),
        "mouth (right)": parse_position(image_size=image_size, landmark=landmark[10]),
        "left shoulder": parse_position(image_size=image_size, landmark=landmark[11]),
        "right shoulder": parse_position(image_size=image_size, landmark=landmark[12]),
        "left elbow": parse_position(image_size=image_size, landmark=landmark[13]),
        "right elbow": parse_position(image_size=image_size, landmark=landmark[14]),
        "left wrist": parse_position(image_size=image_size, landmark=landmark[15]),
        "right wrist": parse_position(image_size=image_size, landmark=landmark[16]),
        "left pinky": parse_position(image_size=image_size, landmark=landmark[17]),
        "right pinky": parse_position(image_size=image_size, landmark=landmark[18]),
        "left index": parse_position(image_size=image_size, landmark=landmark[19]),
        "right index": parse_position(image_size=image_size, landmark=landmark[20]),
        "left thumb": parse_position(image_size=image_size, landmark=landmark[21]),
        "right thumb": parse_position(image_size=image_size, landmark=landmark[22]),
        "left hip": parse_position(image_size=image_size, landmark=landmark[23]),
        "right hip": parse_position(image_size=image_size, landmark=landmark[24]),
        "left knee": parse_position(image_size=image_size, landmark=landmark[25]),
        "right knee": parse_position(image_size=image_size, landmark=landmark[26]),
        "left ankle": parse_position(image_size=image_size, landmark=landmark[27]),
        "right ankle": parse_position(image_size=image_size, landmark=landmark[28]),
        "left heel": parse_position(image_size=image_size, landmark=landmark[29]),
        "right heel": parse_position(image_size=image_size, landmark=landmark[30]),
        "left foot index": parse_position(image_size=image_size, landmark=landmark[31]),
        "right foot index": parse_position(
            image_size=image_size, landmark=landmark[32]
        ),
    }
    return payload


def draw_landmarks(image: Image.Image, landmarks: Dict[str, Sequence[int]]):
    draw = ImageDraw.Draw(im=image)
    for x, y in landmarks.values():
        draw.circle(xy=[x, y], radius=3, fill="white")
    return draw


if __name__ == "__main__":
    dataset = LTCC(root=LTCC_PATH)
    gallery = dataset.gallery
    pose_output = []
    for idx, (path, _, _, _) in enumerate(gallery[50:60]):
        pil_image = Image.open(path)
        image = mp.Image.create_from_file(path)
        image_size = pil_image.size
        detection_result = detector.detect(image)
        pose_landmarks = parse_landmarks(image_size=image_size, result=detection_result)
        draw_landmarks_image = draw_landmarks(image=pil_image, landmarks=pose_landmarks)
        pil_image.save(f"sample/out_{idx}.jpg")
        pose_output.append(pose_landmarks)
        # break
