import os
from typing import List, Literal, Tuple

from pydantic import BaseModel, ConfigDict
from torchvision import transforms as T

from utils.img_transforms import RandomCroping, RandomErasing


class DatasetConfig(BaseModel):
    data_path: str
    dataset_path: str
    dataset_name: str

    train_json_path: str
    query_json_path: str
    gallery_json_path: str

    batch_size: int
    pin_memory: bool
    num_workers: int


class TrainerConfig(BaseModel):
    epochs: int


class FTNetConfig(BaseModel):
    target_layer: str
    output_layer_name: str
    weights: str


class MiscellaneusConfig(BaseModel):
    fusion_net_apprearance_dim: int
    fusion_net_shape_dim: int
    fusion_net_output_dim: int
    classifier_num_classes: int
    classifier_num_clothes: int


class HRNetStageConfig(BaseModel):
    num_modules: int
    num_branches: int
    block: str
    num_blocks: List
    num_channels: List
    fuse_method: str


class HRNetModelExtraConfig(BaseModel):
    pretrained_layers: List[str]
    final_conv_kernel: int

    stage2: HRNetStageConfig
    stage3: HRNetStageConfig
    stage4: HRNetStageConfig


class HRNetModelConfig(BaseModel):
    init_weights: bool
    use_featuremap: bool
    name: str
    num_joints: int
    pretrained: str
    target_type: str
    image_size: Tuple[int, int]
    heatmap_size: Tuple[int, int]
    sigma: int
    return_hoe: bool
    extra: HRNetModelExtraConfig


class HRNetConfig(BaseModel):
    auto_resume: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    cudnn_enabled: bool
    data_dir: str
    gpus: tuple
    output_dir: str
    log_dir: str
    workers: str
    print_freq: int

    model: HRNetModelConfig
    model_config = ConfigDict(protected_namespaces=())


class MainConfig(BaseModel):
    output_features: int
    agg: str
    input_size: Tuple[int, int]
    lr: float
    device: str
    color_jitter: bool
    random_erasing: bool

    @property
    def train_transform(self):
        transform = [
            T.Resize(self.input_size),
            RandomCroping(p=0.5),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        if self.color_jitter is True:
            transform = [
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
            ] + transform
        if self.random_erasing is True:
            transform += [
                RandomErasing(
                    probability=0.5,
                    sl=0.02,
                    sh=0.4,
                    r1=0.3,
                    mean=(0.4914, 0.4822, 0.4465),
                )
            ]
        return T.Compose(transforms=transform)

    @property
    def test_transform(self):
        return T.Compose(
            [
                T.Resize(self.input_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    data_path: str
    dataset_path: str
    dataset_name: str

    @property
    def train_path(self):
        return os.path.join(self.dataset_path, self.dataset_name, "train")

    @property
    def cloth_changing_mode(self):
        if self.dataset_name in ["market1501", "cuhk03"]:
            return False
        return True

    train_json_path: str
    query_json_path: str
    gallery_json_path: str

    orientation_guided: bool
    sampler: bool
    optimizer: str
    weight_decay: float

    use_warm_epoch: bool
    warm_epoch: int
    warm_up: float

    # Loss function
    cla_loss: Literal["cross_entropy", "arcface", "cosface", "circle"]
    cla_s: float
    cla_m: float

    use_triplet_loss: bool
    triplet_loss: str
    triplet_m: float

    use_pairwise_loss: bool
    pair_loss: str
    pair_m: float
    pair_s: float
    weight_pair: float

    use_clothes_loss: bool
    clothes_cla_loss: str
    cal: str
    epsilon: float
    start_epoch_cc: int
    start_epoch_adv: int

    train_from_scratch: bool

    @property
    def train_from_ckpt(self):
        return not self.train_from_scratch

    ckpt_path: str

    train_shape: bool
    num_refine_layers: int
    gcn_layer_type: Literal["ResGCN", "GCNConv"]
    num_gcn_layers: int
    aggregation_type: Literal["mean", "max", "sum"]

    batch_size: int
    pin_memory: bool
    num_workers: int

    output_features: int
    norm_feature: bool

    test_with_pose: bool

    save_path: str
    log_path: str

    @property
    def name(self):
        name = f"model_{self.dataset_name}_{self.epochs}epochs_{self.lr}lr_{self.batch_size}bs"

        if self.orientation_guided:
            name += "_ori"

        if self.train_from_scratch:
            name += "_fromscratch"
        else:
            name += "_fromckpt"

        if self.sampler:
            name += "_sampler"

        if self.use_warm_epoch:
            name += f"_{self.warm_epoch}warmepoch"
        if self.norm_feature:
            name += "_norm"

        name += f"_{self.cla_loss}"

        if self.use_triplet_loss:
            name += f"_{self.triplet_loss}"
        if self.use_pairwise_loss:
            name += f"_{self.pair_loss}"

        if self.train_shape:
            name += f"_{self.num_refine_layers}_{self.gcn_layer_type}_{self.num_gcn_layers}GCN_{self.aggregation_type}AGG"
        return name

    @property
    def model_name(self):
        return self.name + ".pth"


class ShapeEmbeddingConfig(BaseModel):
    pose_n_features: int
    n_hidden: int
    out_features: int
    relation_layers: List[Tuple[int, int]]
    edge_index: Tuple[List[int], List[int]]
