"""
Module for TensorBoard logging.

This module contains the TensorBoardLogger class, which provides methods for logging text, images, models,
checkpoints, dictionaries, parameters, and artifacts to TensorBoard. It manages the setup and configuration
required to write logs to TensorBoard's event files.

Classes:
    TensorBoardLogger: Handles logging to TensorBoard.
"""

import json
import os
import shutil
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard.writer import SummaryWriter

from .base import Logger


class TensorBoardLogger(Logger):
    """
    A logger for TensorBoard.

    This class handles logging of text, images, models, checkpoints, dictionaries, parameters, and artifacts
    to TensorBoard. It provides methods to log various types of data to TensorBoard's event files.

    Attributes:
        log_dir (str): Directory where TensorBoard logs will be stored.
    """

    def __init__(
        self,
        log_dir: str = "logs/tensorboard_logs",
        connection_url: Optional[str] = None,
    ) -> None:
        """
        Initializes the TensorBoardLogger.

        Args:
            log_dir (str): Directory where logs will be stored.
            connection_url (Optional[str]): Optional base directory for logging.

        """
        self.log_dir = log_dir
        if connection_url:
            self.log_dir = os.path.join(connection_url, log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        print(f"TensorBoard logging directory: {self.log_dir}")

    def log_text(self, tag: str, text: str, step: int) -> None:
        """
        Logs a text message.

        Args:
            tag (str): Tag associated with the text message.
            text (str): The text to log.
            step (int): The training step at which this text is logged.

        """
        self.writer.add_text(tag, text, step)
        print(f"Logged text under tag '{tag}' at step {step}")

    def log_image(
        self, tag: str, image: str | Image.Image | np.ndarray | torch.Tensor, step: int
    ) -> None:
        """
        Logs an image.

        Args:
            tag (str): Tag associated with the image.
            image (str | Image.Image | np.ndarray | torch.Tensor): The image to log.
            step (int): The training step at which this image is logged.

        """
        if isinstance(image, str):
            if os.path.exists(image):
                image = Image.open(image).convert("RGB")
                image = np.array(image)
            else:
                print(f"Image path '{image}' does not exist.")
                return
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        elif isinstance(image, torch.Tensor):
            if image.ndim == 4 and image.shape[0] == 1:  # single batch
                image = image.squeeze(0)
            image = image.permute(1, 2, 0).numpy()
        elif isinstance(image, np.ndarray):
            pass  # already in the right format
        else:
            print("Unsupported image type.")
            return

        self.writer.add_image(tag, image, step, dataformats="HWC")
        print(f"Logged image under tag '{tag}' at step {step}")

    def log_model(self, model: torch.nn.Module, model_name: str = "model.pth") -> None:
        """
        Logs a model by saving its state_dict.

        Args:
            model (torch.nn.Module): The model to log.
            model_name (str): The name under which the model will be saved.

        """
        model_path = os.path.join(self.log_dir, model_name)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to '{model_path}'")

    def log_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        checkpoint_name: str = "checkpoint.pth",
    ) -> None:
        """
        Logs a checkpoint, including model and optimizer states.

        Args:
            model (torch.nn.Module): The model whose state is to be logged.
            optimizer (torch.optim.Optimizer): The optimizer whose state is to be logged.
            epoch (int): The epoch number for the checkpoint.
            checkpoint_name (str): The name under which the checkpoint will be saved.

        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint_path = os.path.join(self.log_dir, checkpoint_name)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to '{checkpoint_path}' at epoch {epoch}")

    def log_dict(self, data: Dict, file_name: str = "data.json") -> None:
        """
        Logs arbitrary dictionary data to a JSON file.

        Args:
            data (Dict): The dictionary to log.
            file_name (str): The file name to save the dictionary data.

        """
        file_path = os.path.join(self.log_dir, file_name)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Dictionary data saved to '{file_path}'")

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Logs metrics to TensorBoard.

        Args:
            metrics (Dict[str, float]): A dictionary of metric names and their values.
            step (int): The current step (epoch or iteration) of training.
        """
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        print(f"Metrics logged to TensorBoard at step {step}: {metrics}")

    def log_params(self, params: Dict) -> None:
        """
        Logs hyperparameters or other parameter settings.

        Args:
            params (Dict): The parameters to log.

        """
        for key, value in params.items():
            self.writer.add_text(f"param_{key}", str(value))
        print(f"Parameters logged: {params}")

    def log_artifact(
        self, artifact_path: str, artifact_name: Optional[str] = None
    ) -> None:
        """
        Logs a file or directory as an artifact.

        Args:
            artifact_path (str): The path to the artifact to log.
            artifact_name (Optional[str]): The name under which to log the artifact.

        """
        if not artifact_name:
            artifact_name = os.path.basename(artifact_path)
        destination_path = os.path.join(self.log_dir, artifact_name)
        if os.path.isdir(artifact_path):
            shutil.copytree(artifact_path, destination_path)
        else:
            shutil.copy2(artifact_path, destination_path)
        print(f"Artifact '{artifact_path}' logged to '{destination_path}'")

    def close(self) -> None:
        """
        Closes the TensorBoard writer.

        """
        self.writer.close()
        print("Closed TensorBoard logger.")
