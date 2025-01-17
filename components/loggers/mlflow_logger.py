"""
Module for MLFlow logging.

This module defines the MLFlowLogger class, which facilitates logging of text, images, models, checkpoints,
dictionaries, parameters, and artifacts to MLFlow. It manages MLFlow experiment tracking and saves data
for reproducibility and analysis.

Classes:
    MLFlowLogger: Manages logging to MLFlow.
"""

import json
from typing import Dict, Optional

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from PIL import Image

from .base import Logger


class MLFlowLogger(Logger):
    """
    A logger for MLFlow.

    This class manages logging of text, images, models, checkpoints, dictionaries, parameters, and artifacts
    to MLFlow. It uses MLFlow's tracking and logging features to save and manage experiment data.

    Attributes:
        experiment_name (str): Name of the MLFlow experiment.
        run (mlflow.entities.Run): The current MLFlow run object.
    """

    def __init__(
        self, experiment_name: str = "default", connection_url: Optional[str] = None
    ) -> None:
        """
        Initializes the MLFlowLogger.

        Args:
            experiment_name (str): Name of the experiment to log.
            connection_url (Optional[str]): MLFlow tracking URI.

        """
        if connection_url:
            mlflow.set_tracking_uri(connection_url)
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run()
        print(
            f"MLFlow logging started for experiment '{experiment_name}' with tracking URI '{connection_url}'"
        )

    def log_text(self, tag: str, text: str, step: int) -> None:
        """
        Logs a text message.

        Args:
            tag (str): Tag associated with the text message.
            text (str): The text to log.
            step (int): The training step at which this text is logged.

        """
        mlflow.log_text(text, f"{tag}_{step}.txt")
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
            mlflow.log_artifact(image, artifact_path=f"images/{tag}")
            print(f"Logged image from '{image}' under tag '{tag}' at step {step}")
        else:
            image_path = f"temp_image_{tag}_{step}.png"
            if isinstance(image, Image.Image):
                image.save(image_path)
            elif isinstance(image, torch.Tensor):
                if image.ndim == 4 and image.shape[0] == 1:  # single batch
                    image = image.squeeze(0)
                np_image = image.permute(1, 2, 0).numpy()
                Image.fromarray((np_image * 255).astype(np.uint8)).save(image_path)
            elif isinstance(image, np.ndarray):
                Image.fromarray(image.astype(np.uint8)).save(image_path)
            else:
                print("Unsupported image type.")
                return
            mlflow.log_artifact(image_path, artifact_path=f"images/{tag}")
            print(f"Logged image from '{image_path}' under tag '{tag}' at step {step}")

    def log_model(self, model: torch.nn.Module, model_name: str = "model") -> None:
        """
        Logs a model using MLFlow's PyTorch API.

        Args:
            model (torch.nn.Module): The model to log.
            model_name (str): The name under which the model will be logged.

        """
        mlflow.pytorch.log_model(model, model_name)
        print(f"Model saved as '{model_name}' in MLFlow")

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
        torch.save(checkpoint, checkpoint_name)
        mlflow.log_artifact(checkpoint_name)
        print(f"Checkpoint saved as '{checkpoint_name}' at epoch {epoch} in MLFlow")

    def log_dict(self, data: Dict, file_name: str = "data.json") -> None:
        """
        Logs arbitrary dictionary data to a JSON file.

        Args:
            data (Dict): The dictionary to log.
            file_name (str): The file name to save the dictionary data.

        """
        with open(file_name, "w") as f:
            json.dump(data, f, indent=4)
        mlflow.log_artifact(file_name)
        print(f"Dictionary data saved to '{file_name}' in MLFlow")

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Logs metrics to MLFlow.

        Args:
            metrics (Dict[str, float]): A dictionary of metric names and their values.
            step (int): The current step (epoch or iteration) of training.
        """
        mlflow.log_metrics(metrics, step=step)
        print(f"Metrics logged to MLFlow at step {step}: {metrics}")

    def log_params(self, params: Dict) -> None:
        """
        Logs hyperparameters or other parameter settings.

        Args:
            params (Dict): The parameters to log.

        """
        mlflow.log_params(params)
        print(f"Parameters logged: {params}")

    def log_artifact(self, artifact_path: str) -> None:
        """
        Logs a file or directory as an artifact.

        Args:
            artifact_path (str): The path to the artifact to log.

        """
        mlflow.log_artifact(artifact_path)
        print(f"Artifact '{artifact_path}' logged to MLFlow")

    def close(self) -> None:
        """
        Ends the MLFlow run.

        """
        mlflow.end_run()
        print("Closed MLFlow logger.")
