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
from src.datasets.base_dataset import TrainDataset
from src.models.cvsl_reid import CVSLReID, build_losses

LTCC_JSON = "external_data/ltcc/pose_train.json"
LTCC_SET = "/media/jurgen/Documents/datasets/LTCC_ReID/train"


if __name__ == "__main__":
    train_set = TrainDataset(
        train_dir=dataloader_config.dataset.train_path,
        json_path=LTCC_JSON,
        transforms=dataloader_config.transforms.train_transform,
    )
    train_loader = DataLoader(dataset=train_set, batch_size=64)
    edge_index = torch.Tensor(model_config.shape_encoder_config.edge_index).long()
    cvsl_model = CVSLReID(model_config=model_config, loss_config=loss_config).to("cuda")
    optimizer = optim.AdamW(params=cvsl_model.parameters(), lr=5e-3)
    losses = build_losses(config=loss_config)

    epoch_loss = []
    predicted_ids = []
    gt_ids = []
    for epoch in tqdm(range(15), leave=False):
        epoch_loss.clear()
        predicted_ids.clear()
        gt_ids.clear()
        for batch in tqdm(train_loader, leave=False):
            optimizer.zero_grad()
            img_tensor, pose_tensor, pid, clothes_id = batch

            edge_index = edge_index.to("cuda")
            img_tensor = img_tensor.to("cuda")
            pose_tensor = pose_tensor.to("cuda")
            pid = pid.to("cuda")
            clothes_id = clothes_id.to("cuda")
            logits = cvsl_model.forward(
                x_image=img_tensor,
                x_pose_features=pose_tensor,
                edge_index=edge_index,
            )

            cla_loss = losses["cla"](logits, pid)
            clothes_loss = losses["clothes"](logits, clothes_id)
            predicted_id = torch.argmax(input=logits, dim=1)
            loss = cla_loss + clothes_loss
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                epoch_loss.append(loss.cpu().numpy())
                predicted_ids.append(predicted_id.cpu().numpy())
                gt_ids.append(pid.cpu().numpy())
        print(np.mean(epoch_loss))
        if epoch % 3 == 0:
            predicted_ids = list(chain.from_iterable(predicted_ids))
            gt_ids = list(chain.from_iterable(gt_ids))
            print(
                f"Precision: {precision_score(y_true=gt_ids, y_pred=predicted_ids, average="macro")}"
            )
            print(
                f"Recall: {recall_score(y_true=gt_ids, y_pred=predicted_ids, average="macro")}"
            )
            print(
                f"F1 Score: {f1_score(y_true=gt_ids, y_pred=predicted_ids, average="macro")}"
            )

# TODO:
# Add checkpoints, save the checkpoint with given formatted
# Evaluation metrics after some certain epochs
