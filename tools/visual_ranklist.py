from argparse import ArgumentParser

import yaml

from configs.factory import (
    FTNetConfig,
    MainConfig,
    MiscellaneusConfig,
    ShapeEmbeddingConfig,
)
from src.models.baseline import Baseline
from utils.visualize_ranklist import visualize_ranklist


def parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
    )
    parser.add_argument(
        "--model-path",
        dest="model_path",
        type=str,
        default="checkpoints/model.pth",
    )
    parser.add_argument("--device", dest="device", type=str, default="cpu")
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=1)
    args = parser.parse_args()
    return args


def visualize():
    args = parser()
    with open(args.config, "r") as f:
        payload = yaml.load(f, Loader=yaml.FullLoader)
    main_config = MainConfig(
        **payload["main"], device=args.device, num_workers=args.num_workers
    )
    shape_embedding_config = ShapeEmbeddingConfig(**payload["shape_embedding"])
    ftnet_config = FTNetConfig(**payload["ftnet"])
    miscellaneous_config = MiscellaneusConfig(**payload["miscellaneous"])

    net = Baseline(
        main_config=main_config,
        ftnet_config=ftnet_config,
        shape_embedding_config=shape_embedding_config,
        miscs_config=miscellaneous_config,
    )
    visualize_ranklist(
        model=net,
        model_path=args.model_path,
        topk=1,
        config=main_config,
        is_clothes_change=False,
    )


if __name__ == "__main__":
    visualize()
