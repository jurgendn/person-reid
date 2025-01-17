from torch import nn

from config_refactored import LossConfig
from src.losses.arcface_loss import ArcFaceLoss
from src.losses.circle_loss import CircleLoss, PairwiseCircleLoss
from src.losses.clothes_based_adversarial_loss import ClothesBasedAdversarialLoss
from src.losses.contrastive_loss import ContrastiveLoss
from src.losses.cosface_loss import CosFaceLoss, PairwiseCosFaceLoss
from src.losses.cross_entropy_loss_with_label_smooth import CrossEntropyWithLabelSmooth
from src.losses.triplet_loss import TripletLoss


def build_losses(config: LossConfig) -> dict:
    """Builds the loss functions based on the provided configuration.

    Args:
        config: Configuration object containing loss settings.

    Returns:
        dict: A dictionary of loss functions.
    """
    losses = {}

    # Build identity classification loss
    losses["cla"] = _build_classification_loss(config)

    if config.use_pairwise_loss:
        losses["pair"] = _build_pairwise_loss(config)

    if config.use_triplet_loss:
        losses["triplet"] = _build_triplet_loss(config)

    if config.use_clothes_loss:
        losses["clothes"], losses["cal"] = _build_clothes_losses(config)

    return losses


def _build_classification_loss(config: LossConfig) -> nn.Module:
    loss = {
        "crossentropy": nn.CrossEntropyLoss(),
        "crossentropylabelsmooth": CrossEntropyWithLabelSmooth(),
        "arcface": ArcFaceLoss(scale=config.cla_scale, margin=config.cla_margin),
        "cosface": CosFaceLoss(scale=config.cla_scale, margin=config.cla_margin),
        "circle": CircleLoss(scale=config.cla_scale, margin=config.cla_margin),
    }
    """Builds the classification loss based on the configuration."""
    f = loss.get(config.clothes_cla_loss)
    if f is None:
        raise KeyError(f"Invalid classification loss: '{config.clothes_cla_loss}'")
    return f


def _build_pairwise_loss(config: LossConfig) -> nn.Module:
    """Builds the pairwise loss based on the configuration."""
    loss = {
        "triplet": TripletLoss(margin=config.pair_m),
        "contrastive": ContrastiveLoss(scale=config.pair_s),
        "cosface": PairwiseCosFaceLoss(scale=config.pair_s, margin=config.pair_m),
        "circle": PairwiseCircleLoss(scale=config.pair_s, margin=config.pair_m),
    }
    f = loss.get(config.pair_loss, None)
    if f is None:
        raise KeyError(f"Invalid pairwise loss: '{config.pair_loss}'")
    return f


def _build_triplet_loss(config: LossConfig) -> nn.Module:
    """Builds the triplet loss based on the configuration."""
    loss = {
        "triplet": nn.TripletMarginLoss(margin=config.triplet_m),
    }
    f = loss.get(config.triplet_loss, None)
    if f is None:
        raise KeyError(f"Invalid pairwise loss: '{config.triplet_loss}'")
    return f


def _build_clothes_losses(config: LossConfig) -> tuple:
    """Builds the clothes classification and adversarial loss based on the configuration."""
    loss = {
        "crossentropy": nn.CrossEntropyLoss(),
        "cosface": CosFaceLoss(scale=config.cla_scale),
    }
    criterion_clothes = loss.get(config.clothes_cla_loss, None)
    if criterion_clothes is None:
        raise KeyError(
            f"Invalid clothes classification loss: '{config.clothes_cla_loss}'"
        )
    # Build clothes-based adversarial loss
    if config.cal == "cal":
        criterion_cal = ClothesBasedAdversarialLoss(
            scale=config.cla_scale, epsilon=config.epsilon
        )
    else:
        raise KeyError(f"Invalid clothing classification loss: '{config.cal}'")

    return criterion_clothes, criterion_cal
