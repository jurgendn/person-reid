"""
Module for WandB (Weights and Biases) logging.

This module includes the WandBLogger class, which supports logging of text, images, models, checkpoints,
dictionaries, parameters, and artifacts to WandB. It integrates with WandB's API for experiment tracking
and visualization.

Classes:
    WandBLogger: Handles logging to WandB.
"""


import json
import os
from typing import Dict

import numpy as np
import torch
from PIL import Image

import components.loggers.wandb_logger as wandb_logger

from .wandb_logger import Logger


class WandBLogger(Logger):
    """
    A logger for WandB (Weights and Biases).

    This class provides logging capabilities for text, images, models, checkpoints, dictionaries, parameters,
    and artifacts using WandB. It integrates with WandB's API to manage experiment tracking and visualization.

    Attributes:
        project_name (str): Name of the WandB project.
    """

    def __init__(self, project_name: str = "default_project") -> None:
        """
        Initializes the WandBLogger.

        Args:
            project_name (str): Name of the WandB project to log.
            connection_url (Optional[str]): Base URL for WandB logging.

        """
        wandb_logger.init(project=project_name)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """
        Logs a text message.

        Args:
            tag (str): Tag associated with the text message.
            text (str): The text to log.
            step (int): The training step at which this text is logged.

        """
        wandb_logger.log({tag: text}, step=step)
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
                image = Image.open(image)
                wandb_logger.log({tag: wandb_logger.Image(image)}, step=step)
                print(f"Logged image from '{image}' under tag '{tag}' at step {step}")
            else:
                print(f"Image path '{image}' does not exist.")
        elif isinstance(image, Image.Image):
            wandb_logger.log({tag: wandb_logger.Image(image)}, step=step)
            print(f"Logged PIL image under tag '{tag}' at step {step}")
        elif isinstance(image, torch.Tensor):
            if image.ndim == 4 and image.shape[0] == 1:  # single batch
                image = image.squeeze(0)
            image = image.permute(1, 2, 0).numpy()
            wandb_logger.log({tag: wandb_logger.Image(image)}, step=step)
            print(f"Logged torch.Tensor image under tag '{tag}' at step {step}")
        elif isinstance(image, np.ndarray):
            wandb_logger.log({tag: wandb_logger.Image(image)}, step=step)
            print(f"Logged numpy.ndarray image under tag '{tag}' at step {step}")
        else:
            print("Unsupported image type.")

    def log_model(self, model: torch.nn.Module, model_name: str = "model") -> None:
        """
        Logs a model by saving its state_dict.

        Args:
            model (torch.nn.Module): The model to log.
            model_name (str): The name under which the model will be saved.

        """
        model_path = f"{model_name}.pth"
        torch.save(model.state_dict(), model_path)
        wandb_logger.save(model_path)
        print(f"Model saved as '{model_path}' in WandB")

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
        wandb_logger.save(checkpoint_name)
        print(f"Checkpoint saved as '{checkpoint_name}' at epoch {epoch} in WandB")

    def log_dict(self, data: Dict, file_name: str = "data.json") -> None:
        """
        Logs arbitrary dictionary data to a JSON file.

        Args:
            data (Dict): The dictionary to log.
            file_name (str): The file name to save the dictionary data.

        """
        with open(file_name, "w") as f:
            json.dump(data, f, indent=4)
        wandb_logger.save(file_name)
        print(f"Dictionary data saved to '{file_name}' in WandB")

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Logs metrics to WandB.

        Args:
            metrics (Dict[str, float]): A dictionary of metric names and their values.
            step (int): The current step (epoch or iteration) of training.
        """
        wandb_logger.log(metrics, step=step)
        print(f"Metrics logged to WandB at step {step}: {metrics}")

    def log_params(self, params: Dict) -> None:
        """
        Logs hyperparameters or other parameter settings.

        Args:
            params (Dict): The parameters to log.

        """
        wandb_logger.config.update(params)
        print(f"Parameters logged: {params}")

    def log_artifact(self, artifact_path: str) -> None:
        """
        Logs a file or directory as an artifact.

        Args:
            artifact_path (str): The path to the artifact to log.

        """
        wandb_logger.save(artifact_path)
        print(f"Artifact '{artifact_path}' logged to WandB")

    def close(self) -> None:
        """
        Closes the WandB run.

        """
        wandb_logger.finish()
        print("Closed WandB logger.")