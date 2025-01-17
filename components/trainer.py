from typing import Callable, Dict, List

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from components.loggers import Logger


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: Callable,
        logger: Logger | None = None,
        callbacks: List | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.logger = logger
        self.callbacks = callbacks or []
        self.stop_training = False
        self.metrics = {}

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        self._run_callback("on_train_begin")

        for epoch in range(epochs):
            if self.stop_training:
                break

            self._run_callback("on_epoch_begin", epoch)

            # Training loop with progress bar
            self.model.train()
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            train_metrics = self._train_epoch(train_pbar)

            # Validation loop
            val_metrics = self.validate(val_loader)

            # Log metrics
            print("Logging metrics")
            self.log_metrics(epoch, train_metrics, val_metrics)

            self._run_callback("on_epoch_end", epoch)

        self._run_callback("on_train_end")

    def _train_epoch(self, pbar: tqdm):
        epoch_metrics = {"train_loss": 0.0}

        for batch in pbar:
            self._run_callback("on_batch_begin", batch)

            batch_metrics = self._train_step(batch)

            # Update epoch metrics
            for key, value in batch_metrics.items():
                epoch_metrics[key] = epoch_metrics.get(key, 0) + value

            # Update progress bar
            pbar.set_postfix(
                {k: f"{v/(pbar.n+1):.4f}" for k, v in epoch_metrics.items()}
            )

            self._run_callback("on_batch_end", batch)

        # Compute average metrics for the epoch
        epoch_metrics = {k: v / len(pbar) for k, v in epoch_metrics.items()}
        return epoch_metrics

    def _train_step(self, batch):
        inputs, targets = batch
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()

        # You can add more metrics here
        metrics = {
            "train_loss": loss.item(),
            "train_accuracy": (outputs.argmax(1) == targets).float().mean().item(),
        }
        return metrics

    def validate(self, val_loader: DataLoader):
        self.model.eval()
        val_metrics = {"val_loss": 0.0, "val_accuracy": 0.0}

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for batch in val_pbar:
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                val_metrics["val_loss"] += loss.item()
                val_metrics["val_accuracy"] += (
                    (outputs.argmax(1) == targets).float().mean().item()
                )

                val_pbar.set_postfix(
                    {k: f"{v/(val_pbar.n+1):.4f}" for k, v in val_metrics.items()}
                )

        # Compute average metrics
        val_metrics = {k: v / len(val_loader) for k, v in val_metrics.items()}
        return val_metrics

    def log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        metrics = {**train_metrics, **val_metrics}
        self.metrics[epoch] = metrics

        if self.logger is not None:
            self.logger.log_metrics(metrics=metrics, step=epoch)

    def _run_callback(self, method: str, *args):
        for callback in self.callbacks:
            getattr(callback, method)(self, *args)
