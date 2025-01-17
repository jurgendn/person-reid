from abc import abstractmethod
from typing import Dict

import numpy as np
import torch
from PIL import Image


class Logger:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def log_text(self, tag: str, text: str, step: int):
        """
        Logs a text message.

        Args:
            tag (str): Tag associated with the text message.
            text (str): The text to log.
            step (int): The training step at which this text is logged.

        """

    @abstractmethod
    def log_image(
        self, tag: str, image: str | Image.Image | np.ndarray | torch.Tensor, step: int
    ):
        """
        Logs an image.

        Args:
            tag (str): Tag associated with the image.
            image (str | Image.Image | np.ndarray | torch.Tensor): The image to log.
            step (int): The training step at which this image is logged.

        """

    @abstractmethod
    def log_model(self, model: torch.nn.Module, model_name: str):
        """
        Logs a model using MLFlow's PyTorch API.

        Args:
            model (torch.nn.Module): The model to log.
            model_name (str): The name under which the model will be logged.

        """

    @abstractmethod
    def log_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        checkpoint_name: str = "checkpoint.pth",
    ):
        """
        Logs a checkpoint, including model and optimizer states.

        Args:
            model (torch.nn.Module): The model whose state is to be logged.
            optimizer (torch.optim.Optimizer): The optimizer whose state is to be logged.
            epoch (int): The epoch number for the checkpoint.
            checkpoint_name (str): The name under which the checkpoint will be saved.

        """

    @abstractmethod
    def log_dict(self, data: Dict, file_name: str):
        """
        Logs arbitrary dictionary data to a JSON file.

        Args:
            data (Dict): The dictionary to log.
            file_name (str): The file name to save the dictionary data.

        """

    @abstractmethod
    def log_params(self, params: Dict):
        """
        Logs hyperparameters or other parameter settings.

        Args:
            params (Dict): The parameters to log.

        """

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Logs metrics to TensorBoard.

        Args:
            metrics (Dict[str, float]): A dictionary of metric names and their values.
            step (int): The current step (epoch or iteration) of training.
        """

    @abstractmethod
    def log_artifact(self, artifact_path: str):
        """
        Logs a file or directory as an artifact.

        Args:
            artifact_path (str): The path to the artifact to log.

        """
