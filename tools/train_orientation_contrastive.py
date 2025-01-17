from argparse import ArgumentParser
from itertools import chain

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from global_var import dataloader_config, loss_config, model_config
from src.datasets.base_dataset import TrainDatasetOrientation
from src.models.cvsl_reid import CVSLReID, build_losses

LTCC_JSON = "external_data/ltcc/pose_train.json"
LTCC_SET = "/media/jurgen/Documents/datasets/LTCC_ReID/train"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--epochs", dest="epochs", type=int, default=10)
    parser.add_argument(
        "--log-every-n-epochs", dest="log_every_n_epochs", type=int, default=1
    )
    parser.add_argument("--ckpt", action="store_true", dest="ckpt")
    parser.add_argument("--learning-rate", dest="lr", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_set = TrainDatasetOrientation(
        json_path=LTCC_JSON,
        transforms=dataloader_config.transforms.train_transform,
    )
    train_loader = DataLoader(dataset=train_set, batch_size=16)
    edge_index = torch.Tensor(model_config.shape_encoder_config.edge_index).long()
    cvsl_model = CVSLReID(model_config=model_config, loss_config=loss_config)

    if args.ckpt:
        with open("./checkpoints/model.ckpt", "rb") as f:
            weights = torch.load(f=f, weights_only=True)
            cvsl_model.load_state_dict(weights)
    cvsl_model = cvsl_model.to("cuda")
    initial_lr = args.lr

    optimizer = optim.AdamW(params=cvsl_model.parameters(), lr=initial_lr)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=0.5, last_epoch=-1
    )
    losses = build_losses(config=loss_config)

    epoch_loss = []
    predicted_ids = []
    gt_ids = []

    for epoch in tqdm(range(args.epochs), leave=False):
        epoch_loss.clear()
        predicted_ids.clear()
        gt_ids.clear()
        for batch in tqdm(train_loader, leave=False):
            optimizer.zero_grad()
            (
                (a_img_tensor, p_img_tensor, n_img_tensor),
                (pose_tensor, p_pose_tensor, n_pose_tensor),
                a_id_index,
                position_mask,
            ) = batch
            edge_index = edge_index.to("cuda")
            a_id_index = a_id_index.to("cuda")
            position_mask = position_mask.to("cuda")

            a_img_tensor = a_img_tensor.to("cuda")
            p_img_tensor = p_img_tensor.to("cuda")
            n_img_tensor = n_img_tensor.to("cuda")

            pose_tensor = pose_tensor.to("cuda")
            p_pose_tensor = p_pose_tensor.to("cuda")
            n_pose_tensor = n_pose_tensor.to("cuda")
            # with torch.autocast(device_type="cuda"):
            a_logits = cvsl_model.forward(
                x_image=a_img_tensor,
                x_pose_features=pose_tensor,
                edge_index=edge_index,
            )
            p_logits = cvsl_model.forward(
                x_image=p_img_tensor,
                x_pose_features=p_pose_tensor,
                edge_index=edge_index,
            )
            n_logits = cvsl_model.forward(
                x_image=n_img_tensor,
                x_pose_features=n_pose_tensor,
                edge_index=edge_index,
            )
            cla_loss = losses["cla"](a_logits, a_id_index)
            triplet_loss = losses["triplet"](a_logits, p_logits, n_logits)
            clothes_loss = losses["clothes"](a_logits, a_id_index)

            loss = cla_loss + triplet_loss + clothes_loss

            loss.backward()
            optimizer.step()

            if epoch % args.log_every_n_epochs == 0:
                lr_scheduler.step()

            predicted_id = torch.argmax(input=a_logits, dim=1)
            with torch.no_grad():
                epoch_loss.append(loss.cpu().numpy())
                predicted_ids.append(predicted_id.cpu().numpy())
                gt_ids.append(a_id_index.cpu().numpy())
        print(np.mean(epoch_loss))
        if epoch % args.log_every_n_epochs == 0:
            predicted_ids = list(chain.from_iterable(predicted_ids))
            gt_ids = list(chain.from_iterable(gt_ids))
            print(
                f"Precision: {precision_score(y_true=gt_ids, y_pred=predicted_ids, average="macro", zero_division=0.0)}"
            )
            print(
                f"Recall: {recall_score(y_true=gt_ids, y_pred=predicted_ids, average="macro", zero_division=0.0)}"
            )
            print(
                f"F1 Score: {f1_score(y_true=gt_ids, y_pred=predicted_ids, average="macro",zero_division=0.0)}"
            )
    with open("./checkpoints/model.ckpt", "wb") as f:
        torch.save(obj=cvsl_model.state_dict(), f=f)
# TODO:
# Add checkpoints, save the checkpoint with given formatted
# Evaluation metrics after some certain epochs
