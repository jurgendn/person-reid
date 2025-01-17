import torch
import glob
import json
import os.path as osp
import random
import re
from typing import List, Optional, Union

import numpy as np
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision import transforms as T

cloth_change_ids = [
    0,
    88,
    2,
    4,
    89,
    146,
    136,
    123,
    93,
    124,
    122,
    45,
    7,
    10,
    66,
    81,
    140,
    13,
    21,
    30,
    40,
    53,
    61,
    71,
    76,
    111,
    116,
    128,
    131,
    132,
    149,
    151,
    144,
    27,
    48,
    115,
    117,
    86,
    87,
    36,
    39,
    43,
    109,
    75,
    135,
    130,
]


def get_orientation(angle: float, num_segments: int):
    pi = 180
    # Normalize the angle to range 0 <= angle < 360
    angle_normalized = angle % (2 * pi)
    if angle_normalized < 0:
        angle_normalized += 2 * pi
    angle_normalized = angle_normalized - 90
    # Calculate the segment angle and adjust the starting angle
    segment_angle = 2 * pi / num_segments
    angle_adjusted = angle_normalized + (pi / num_segments)

    # Determine the segment index
    k = np.floor(angle_adjusted / segment_angle) % num_segments

    return int(k)


class TrainDatasetOrientation(Dataset):
    def __init__(self, json_path: str, transforms: T.Compose | nn.Module) -> None:
        super(TrainDatasetOrientation, self).__init__()
        self.json_path = json_path
        self.train_dir = get_img_list(self.json_path)
        self.transforms = transforms
        self.id_to_idx, self.num_classes = classes_to_idx(self.train_dir)
        self.num_segments = 4
        self.pid2clothes = torch.zeros(size=(len(self.train_dir),)).long()

    def __len__(self):
        return len(self.train_dir)

    def __getitem__(self, index: int):
        img = self.train_dir[index]
        a_img_path = img["img_path"]
        a_id = img["p_id"]
        a_orientation = img["orientation"]
        a_pose = img["pose_landmarks"]
        position_mask = self.pid2clothes[a_id]
        # If the dataset doesn't contain cloth_id, set cloth_id = 0
        a_cloth_id = img.get("cloth_id", 0)

        a_img_tensor, a_size = get_img_tensor(a_img_path, self.transforms)
        pose_tensor = get_pose_tensor(a_pose, a_size)
        a_id_index = self.id_to_idx[a_id]

        same_id, diff_id = [], []
        for item in self.train_dir:
            if item["p_id"] == a_id:
                same_id.append(item)
            else:
                diff_id.append(item)

        same_id_diff_ori, diff_id_same_ori = [], []
        a_segment = get_orientation(angle=a_orientation, num_segments=self.num_segments)

        for item in self.train_dir:
            item_id = item["p_id"]
            item_segment = get_orientation(
                angle=item["orientation"], num_segments=self.num_segments
            )
            item_cloth_id = item.get("cloth_id", 0)

            # Check if anchor and item are in front/back or left/right segment together
            is_same_opposite_segment = (item_segment % (self.num_segments / 2)) != (
                a_segment % (self.num_segments / 2)
            )
            is_same_id = a_id != item_id
            is_diff_cloth = item_cloth_id != a_cloth_id

            if is_same_opposite_segment is True:
                if is_same_id is True and (
                    a_id not in cloth_change_ids or is_diff_cloth
                ):
                    same_id_diff_ori.append(item)
            elif is_same_opposite_segment is False:
                if is_same_id is False and (
                    a_id not in cloth_change_ids or is_diff_cloth
                ):
                    diff_id_same_ori.append(item)

        if len(same_id_diff_ori) > 0:
            p_img = random.choice(same_id_diff_ori)
        else:
            p_img = random.choice(same_id)
        p_img_path = p_img["img_path"]
        p_pose = p_img["pose_landmarks"]

        if len(diff_id_same_ori) > 0:
            n_img = random.choice(diff_id_same_ori)
        else:
            n_img = random.choice(diff_id)
        n_img_path = n_img["img_path"]
        n_pose = n_img["pose_landmarks"]

        p_img_tensor, p_size = get_img_tensor(p_img_path, self.transforms)
        n_img_tensor, n_size = get_img_tensor(n_img_path, self.transforms)

        p_pose_tensor = get_pose_tensor(p_pose, p_size)
        n_pose_tensor = get_pose_tensor(n_pose, n_size)

        return (
            (a_img_tensor, p_img_tensor, n_img_tensor),
            (pose_tensor, p_pose_tensor, n_pose_tensor),
            a_id_index,
            position_mask,
        )


class TrainDataset(Dataset):
    def __init__(
        self, train_dir: str, json_path: str, transforms: Union[T.Compose, nn.Module]
    ) -> None:
        super(TrainDataset, self).__init__()
        self.img_list = get_img_list(json_path)
        self.transforms = transforms
        (
            self.dataset,
            self.num_classes,
            self.num_imgs,
            self.num_clothes,
            self.pid2clothes,
        ) = process_train_data(train_dir, self.img_list)

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index: int):
        img_path, pose, pid, camid, clothes_id = self.dataset[index]

        img_tensor, size = get_img_tensor(img_path, self.transforms)
        pose_tensor = get_pose_tensor(pose, size)

        return img_tensor, pose_tensor, pid, clothes_id


def process_train_data(train_dir: str, img_list: str):
    img_paths = glob.glob(osp.join(train_dir, "*.png"))
    img_paths.sort()
    pattern1 = re.compile(r"(\d+)_(\d+)_c(\d+)")
    pattern2 = re.compile(r"(\w+)_c")

    pid_container = set()
    clothes_container = set()
    for img_path in img_paths:
        pid, _, _ = map(int, pattern1.search(img_path).groups())
        clothes_id = pattern2.search(img_path).group(1)
        pid_container.add(pid)
        clothes_container.add(clothes_id)
    pid_container = sorted(pid_container)
    clothes_container = sorted(clothes_container)
    pid2label = {pid: label for label, pid in enumerate(pid_container)}
    clothes2label = {
        clothes_id: label for label, clothes_id in enumerate(clothes_container)
    }

    num_pids = len(pid_container)
    num_clothes = len(clothes_container)

    dataset = []
    pid2clothes = np.zeros((num_pids, num_clothes))
    for img_path in img_paths:
        key = "img_path"
        val = img_path
        pid, _, camid = map(int, pattern1.search(img_path).groups())
        clothes = pattern2.search(img_path).group(1)
        camid -= 1  # index starts from 0
        pid = pid2label[pid]
        clothes_id = clothes2label[clothes]
        for item in img_list:
            if key in item and item[key] == val:
                found_item = item
        pose = found_item["pose_landmarks"]
        dataset.append((img_path, pose, pid, camid, clothes_id))
        pid2clothes[pid, clothes_id] = 1

    num_imgs = len(dataset)

    return dataset, num_pids, num_imgs, num_clothes, pid2clothes


class TestDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        transforms: Optional[Union[T.Compose, nn.Module]] = None,
        cloth_changing: bool = False,
    ) -> None:
        super(TestDataset, self).__init__()
        self.img_list = get_img_list(json_path)
        self.transforms = transforms
        self.cloth_changing_mode = cloth_changing
        self.num_classes = self.get_num_classes()

    def __len__(self):
        return len(self.img_list)

    def get_num_classes(self):
        classes = []
        for item in self.img_list:
            p_id = item["p_id"]
            classes.append(p_id)

        return len(np.unique(classes))

    def __getitem__(self, index: int):
        sample = self.img_list[index]
        img_path = sample["img_path"]
        p_id = sample["p_id"]
        cam_id = sample["cam_id"]
        pose = sample["pose_landmarks"]
        img_tensor, _ = get_img_tensor(img_path, self.transforms)
        pose_tensor = Tensor(pose)

        if self.cloth_changing_mode:
            cloth_id = sample["cloth_id"]
            return img_tensor, pose_tensor, p_id, cam_id, cloth_id, img_path
        else:
            return img_tensor, pose_tensor, p_id, cam_id, img_path


def get_img_tensor(img_path: str, transform: nn.Module):
    img = Image.open(img_path)
    img_size = img.size
    img_tensor = transform(img)
    return img_tensor, img_size


def get_pose_tensor(pose: List[List[float]], size):
    width, height = size
    processed_pose = [[item[0], item[1]] for item in pose]
    for item in processed_pose:
        item.append(width / height)
    return Tensor(processed_pose)


def classes_to_idx(train_dir: str):
    id_to_index = {}
    index = 0
    for item in train_dir:
        a_id = item["p_id"]
        if a_id not in id_to_index:
            id_to_index[a_id] = index
            index += 1
    return id_to_index, len(id_to_index)


def get_img_list(json_path: str):
    with open(json_path, "rb") as f:
        img_list = json.load(f)
    return img_list
