import json
from argparse import ArgumentParser
from typing import Tuple

import numpy as np
import torch
import yaml
from more_itertools import chunked
from PIL import Image
from torchvision import transforms as T
from tqdm.auto import tqdm

from configs.factory import HRNetConfig
from src.models.hrnet.pose_hrnet import PoseHighResolutionNet


def parse():
    parser = ArgumentParser()
    parser.add_argument("--metadata", dest="metadata", type=str)
    parser.add_argument("--dataset-name", dest="dataset_name", type=str)
    parser.add_argument("--target-set", dest="target_set", type=str)
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--device", dest="device", type=str)
    parser.add_argument(
        "--config-file",
        dest="config_file",
        type=str,
        default="configs/pose_hrnet.yaml",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        type=str,
        default="pretrained/hrnet_w32_256x192.pth",
    )

    args = parser.parse_args()
    return args


def get_landmark_from_heatmap(
    heatmap: torch.Tensor,
    original_img_size: Tuple[int, int],
    heatmap_size: Tuple[int, int],
) -> Tuple[int, int, float]:
    # Find the index of the maximum value in the heatmap (flattened to 1D)
    flat_index = torch.argmax(heatmap)
    # Convert the flat index back into (x, y) coordinates
    y, x = np.unravel_index(flat_index, heatmap.shape)

    prob = heatmap[y, x].item()
    # Scale the coordinates back to the original image size
    scale_x = original_img_size[1] / heatmap_size[1]  # width scaling factor
    scale_y = original_img_size[0] / heatmap_size[0]  # height scaling factor

    x = int(x * scale_x)
    y = int(y * scale_y)

    return (x, y, prob)


if __name__ == "__main__":
    args = parse()
    with open(args.config_file, "r") as f:
        payload = yaml.load(stream=f, Loader=yaml.FullLoader)
    cfg = HRNetConfig(**payload)

    if torch.cuda.is_available() is True:
        device = "cuda"
    else:
        device = "cpu"

    net = PoseHighResolutionNet(cfg=cfg)
    weights = torch.load(f=args.pretrained, weights_only=True)
    net.load_state_dict(state_dict=weights)

    net = net.to(device=device)

    with open(args.metadata, "r") as f:
        img_set = json.load(f)

    processed_json = []
    for idx, chunk in tqdm(enumerate(chunked(iterable=img_set, n=args.batch_size))):
        img_batch = []
        for img in chunk:
            image = Image.open(img["img_path"])
            image = image.resize(size=(192, 256))
            _x = T.ToTensor()(image) * 255
            img_batch.append(_x)
        x = torch.stack(img_batch).to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device):
                hm = net(x)
        hms = hm.to("cpu")
        np_image = np.array(image)
        for hm in hms:
            pose_landmarks = []
            for heatmap in hm:
                x, y, prob = get_landmark_from_heatmap(
                    heatmap=heatmap, original_img_size=image.size, heatmap_size=(64, 48)
                )
                pose_landmarks.append([x, y, prob])
        for data in chunk:
            data["pose_landmarks"] = pose_landmarks
        processed_json.extend(chunk)
    with open(f"external_data/{args.dataset_name}/{args.target_set}.json", "w") as f:
        json.dump(obj=processed_json, fp=f)
