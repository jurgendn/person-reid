from typing import Any, Dict, List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics import functional as FM

from configs.factory import (
    FTNetConfig,
    MainConfig,
    MiscellaneusConfig,
    ShapeEmbeddingConfig,
)
from src.datasets.get_loader import get_train_data
from src.losses import build_losses
from src.models import build_models
from src.models.modules.r50 import FTNet
from utils.utils import normalize_feature


class Baseline(LightningModule):
    def __init__(
        self,
        main_config: MainConfig,
        ftnet_config: FTNetConfig,
        shape_embedding_config: ShapeEmbeddingConfig,
        miscs_config: MiscellaneusConfig,
    ) -> None:
        super(Baseline, self).__init__()
        self.config = main_config
        self.shape_edge_index = torch.LongTensor(shape_embedding_config.edge_index)
        # self.register_buffer("shape_edge_index", shape_edge_index)

        self.train_shape = main_config.train_shape

        self.orientation_guided = main_config.orientation_guided
        self.lr = main_config.lr
        self.train_data, self.sampler = get_train_data(config=main_config) 
        self.class_num = self.train_data.num_classes
        self.num_clothes = self.train_data.num_clothes
        self.pid2clothes = torch.from_numpy(self.train_data.pid2clothes)
        self.dataset_size = len(self.train_data)

        self.losses = build_losses()
        self.id_loss = self.losses["cla"]
        if self.config.use_triplet_loss is True:
            self.triplet_loss = self.losses["triplet"]
        if self.config.use_pairwise_loss is True:
            self.pair_loss = self.losses["pair"]

        self.models = build_models(
            main_config=main_config,
            ftnet_config=ftnet_config,
            shape_embedding_config=shape_embedding_config,
            miscs_config=miscs_config,
        )
        self.ft_net = self.models["cnn"]
        self.shape_embedding = self.models["shape"]
        self.fusion = self.models["fusion"]
        self.id_classifier = self.models["id_clf"]

        if self.config.use_clothes_loss is True:
            self.clothes_loss = self.losses["clothes"]
            self.cal = self.losses["cal"]
            self.clothes_classifier = self.models["clothes_clf"]

        self.use_warm_epoch = self.config.use_warm_epoch
        self.warm_epoch = self.config.warm_epoch
        self.warm_up = self.config.warm_up

        self.automatic_optimization = False

        self.training_step_outputs = []
        self.training_acc_outputs = []
        # self.validation_batch_outputs: List = []
        self.save_hyperparameters()

    def train_dataloader(self):
        if self.config.sampler is True:
            train_loader = DataLoader(
                self.train_data,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                sampler=self.sampler,
            )
        else:
            train_loader = DataLoader(
                self.train_data,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
            )
        return train_loader

    def on_epoch_start(self):
        self.sampler.set_epoch(self.current_epoch)

    def forward(
        self,
        x_image: torch.Tensor,
        x_pose_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x_pose_features = x_pose_features.to(self.device)
        edge_index = edge_index.to(self.device)

        appearance_feature = self.ft_net(x=x_image)

        if self.train_shape:
            shape_feature = self.shape_embedding(
                pose=x_pose_features, edge_index=edge_index
            )

            fusion_feature = self.fusion(
                appearance_features=appearance_feature, shape_features=shape_feature
            )
            return fusion_feature
        else:
            return appearance_feature

    def configure_optimizers(
        self,
    ) -> Tuple[List[optim.Optimizer], List]:
        """
        Configures and returns the optimizers and learning rate schedulers.

        This function sets up the optimizers for the network components based on the
        specified optimizer type in the configuration. It also initializes the learning
        rate scheduler for the main optimizer.

        Returns:
            Tuple[List[optim.Optimizer], List[optim.lr_scheduler._LRScheduler]]: A tuple containing
            a list of optimizers and a list of learning rate schedulers.
        """

        # Collect parameters from the model components
        parameters = (
            list(self.ft_net.parameters())
            + list(self.shape_embedding.parameters())
            + list(self.fusion.parameters())
            + list(self.id_classifier.parameters())
        )

        def _create_optimizer(
            optimizer_type: str, params: List[Any]
        ) -> optim.Optimizer:
            """Creates an optimizer based on the specified type."""
            if optimizer_type == "adam":
                return optim.Adam(
                    params=params,
                    lr=self.lr,
                    weight_decay=self.config.weight_decay,
                )
            elif optimizer_type == "sgd":
                return optim.SGD(
                    params=params,
                    lr=self.lr,
                    weight_decay=self.config.weight_decay,
                    momentum=0.9,
                    nesterov=True,
                )
            else:
                raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        # Create main optimizer and clothes classifier optimizer
        optimizer = _create_optimizer(self.config.optimizer, parameters)
        optimizer_cc = _create_optimizer(
            self.config.optimizer, self.clothes_classifier.parameters()
        )

        # Create the learning rate scheduler
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=20, gamma=0.1
        )

        return [optimizer, optimizer_cc], [lr_scheduler]

    def training_step(self, batch, batch_idx) -> Dict:
        self.optimizer, self.optimizer_cc = self.optimizers()

        if self.orientation_guided:
            (a_img, p_img, n_img), (a_pose, p_pose, n_pose), a_id = batch
            p_feature = self.forward(
                x_image=p_img, x_pose_features=p_pose, edge_index=self.shape_edge_index
            )
            n_feature = self.forward(
                x_image=n_img, x_pose_features=n_pose, edge_index=self.shape_edge_index
            )
            if self.config.norm_feature:
                p_feature = normalize_feature(p_feature)
                n_feature = normalize_feature(n_feature)

        else:
            a_img, a_pose, a_id, a_cloth_id = batch

        a_feature = self.forward(
            x_image=a_img, x_pose_features=a_pose, edge_index=self.shape_edge_index
        )

        pos_mask = self.pid2clothes[a_id].float().to(self.device) # Positive Mask
    
        # Normalize features
        if self.config.norm_feature:
            a_feature = normalize_feature(a_feature)

        loss = 0.0
        if self.config.use_clothes_loss:
            pred_clothes = self.clothes_classifier(a_feature.detach())
            clothes_loss = self.clothes_loss(pred_clothes, a_cloth_id)
            if self.current_epoch >= self.config.start_epoch_cc:
                self.optimizer_cc.zero_grad()
                self.manual_backward(clothes_loss)  # , retain_graph=True)
                self.optimizer_cc.step()

        new_pred_clothes = self.clothes_classifier(a_feature.detach())

        if self.train_shape:
            logits = self.id_classifier(a_feature)
            id_loss = self.id_loss(logits, a_id)
        else:
            id_loss = self.id_loss(a_feature, a_id)

        loss += id_loss

        if self.config.use_triplet_loss:
            triplet_loss = self.triplet_loss(a_feature, p_feature, n_feature)
            loss += triplet_loss

        if self.config.use_pairwise_loss:
            pair_loss = self.pair_loss(a_feature, a_id)
            loss += pair_loss * self.config.weight_pair

        if self.config.use_clothes_loss:
            clothes_adv_loss = self.cal(new_pred_clothes, a_cloth_id, pos_mask)
            if self.current_epoch >= self.config.start_epoch_adv:
                loss += clothes_adv_loss

        if self.use_warm_epoch:
            # Warm Up
            warm_iteration = (
                round(self.dataset_size / self.config.batch_size) * self.warm_epoch
            )  # first 5 epoch
            if self.current_epoch < self.warm_epoch:
                warm_up = min(1.0, self.warm_up + 0.9 / warm_iteration)
                loss = loss * warm_up

        self.optimizer.zero_grad()
        self.manual_backward(loss)
        self.optimizer.step()

        acc = FM.accuracy(
            preds=logits,
            target=a_id,
            task="multiclass",
            average="macro",
            num_classes=self.class_num,
        )

        self.training_step_outputs.append(loss)
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def on_train_epoch_end(self):
        epoch_loss = sum(self.training_step_outputs) / len(self.training_step_outputs)
        self.log("epoch_loss", epoch_loss)
        self.training_step_outputs.clear()


class InferenceBaseline(LightningModule):
    def __init__(self, r50_stride: int) -> None:
        super(InferenceBaseline, self).__init__()
        self.ft_net = FTNet(
            config=FTNetConfig(
                class_num=751,
                target_layers="layer4",
                output_layers_name="out",
                pretrained="IMAGENET1K_V2",
            )
        )

    def forward(self, x_image: torch.Tensor) -> torch.Tensor:
        appearance_feature = self.ft_net(x=x_image)
        return appearance_feature
