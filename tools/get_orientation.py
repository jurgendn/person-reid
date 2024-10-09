import json
import os
from argparse import ArgumentParser, Namespace
from typing import Dict, Iterable, List, Literal

import torch
import yaml
from more_itertools import chunked
from PIL import Image
from torchvision import transforms as T
from tqdm.auto import tqdm

from configs.factory import HRNetConfig
from src.datasets.ltcc import LTCC
from src.datasets.prcc import PRCC
from src.models.orientations.pose_hrnet import PoseHighResolutionNet


def parse():
    parser = ArgumentParser()
    parser.add_argument(
        "--config", dest="config", type=str, default="configs/hrnet.yaml"
    )
    parser.add_argument(
        "--pretrained", dest="pretrained", type=str, default="pretrained/model_hboe.pt"
    )
    parser.add_argument("--dataset", dest="dataset", type=str, default="ltcc")
    parser.add_argument("--dataset-name", dest="dataset_name", type=str, default="ltcc")
    parser.add_argument("--target-set", dest="target_set", type=str, default="train")
    parser.add_argument("--device", dest="device", type=str, default="cpu")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=1)
    args = parser.parse_args()
    return args


def make_batch(dataset: Iterable, batch_size: int = 32):
    transformations = T.Compose(
        [
            T.Resize(size=(256, 192)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    """Yields batches of images and metadata from the dataset."""
    for batch in tqdm(chunked(dataset, batch_size)):
        images = [
            transformations(Image.open(fp=path).resize((256, 192)))
            for path, *_ in batch
        ]
        yield torch.stack(tensors=images), batch  # type: ignore


def load_config(args: Namespace) -> HRNetConfig:
    with open(file=args.config, mode="r") as f:
        payload = yaml.load(stream=f, Loader=yaml.FullLoader)
    config = HRNetConfig(**payload)
    return config


def get_dataset(dataset: LTCC | PRCC, target_set: str) -> Iterable:
    """Returns the correct subset of the dataset (train, test, query)."""
    dataset_mapping = {
        "train": dataset.train,
        "test": dataset.gallery,
        "query": dataset.query if isinstance(dataset, LTCC) else None,
        "query_same": dataset.query_same if isinstance(dataset, PRCC) else None,
        "query_diff": dataset.query_diff if isinstance(dataset, PRCC) else None,
    }
    if dataset_mapping.get(target_set) is None:
        raise ValueError(f"Invalid target set: {target_set}")
    return dataset_mapping[target_set]


def load_model(config: HRNetConfig, pretrained_path: str, device: str | torch.device):
    with open(file=pretrained_path, mode="rb") as f:
        weights = torch.load(f, weights_only=True)
    model = PoseHighResolutionNet(cfg=config)
    model.load_state_dict(state_dict=weights, strict=True)
    model.eval().to(device)
    return model


def process_batch(
    model: torch.nn.Module, batch: torch.Tensor, device: str | torch.device
) -> torch.Tensor:
    """Runs inference on a batch of images and returns the orientation output."""
    batch = batch.to(device)
    _, hoe_output = model(batch)
    return torch.argmax(hoe_output.to("cpu"), dim=1, keepdim=True) * 5


def create_output(batch: List, orientation: torch.Tensor) -> List[Dict]:
    """Generates a list of outputs with orientation information."""
    return [
        {
            "img_path": b[0],
            "p_id": b[1],
            "cam_id": b[2],
            "clothes_id": b[3],
            "orientation": int(o.numpy()[0]),
        }
        for b, o in zip(batch, orientation)
    ]


def get_orientation(
    dataset_name: Literal["ltcc", "prcc"],
    dataset_path: str,
    target_set: str,
    config: HRNetConfig,
    pretrained_path: str,
    batch_size: int,
    device: str | torch.device,
) -> List[Dict]:
    """Main function to get orientation information for the dataset."""
    model = load_model(config=config, pretrained_path=pretrained_path, device=device)
    dataset_cls = LTCC if dataset_name == "ltcc" else PRCC
    dataset = dataset_cls(root=dataset_path)
    image_set = get_dataset(dataset=dataset, target_set=target_set)

    output = []
    for tensors, batch in make_batch(dataset=image_set, batch_size=batch_size):
        orientation = process_batch(model, tensors, device)
        output.extend(create_output(batch, orientation))
    return output


def save_output(output: List[Dict], output_path: str):
    """Saves the output to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w+") as f:
        json.dump(output, f)


if __name__ == "__main__":
    args = parse()
    if args.device == "cuda" and torch.cuda.is_available() is False:
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    config = load_config(args=args)
    output = get_orientation(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset,
        target_set=args.target_set,
        config=config,
        pretrained_path=args.pretrained,
        batch_size=args.batch_size,
        device=device,
    )
    output_path = os.path.join(
        "external_data", args.dataset_name, f"{args.target_set}.json"
    )
    save_output(output, output_path)
