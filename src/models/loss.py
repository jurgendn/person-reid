import math

import torch
import torch.nn.functional as F
from torch import distributed as dist
from torch import nn

from src.models.utils import GatherLayer


def l2_norm(v):
    fnorm = torch.norm(v, p=2, dim=1, keepdim=True) + 1e-6
    v = v.div(fnorm.expand_as(v))
    return v


class ArcFaceLoss(nn.Module):
    """ArcFace loss.

    Reference:
        Deng et al. ArcFace: Additive Angular Margin Loss for Deep Face Recognition. In CVPR, 2019.

    Args:
        scale (float): scaling factor.
        margin (float): pre-defined margin.
    """

    def __init__(self, scale=16, margin=0.1):
        super().__init__()
        self.s = scale
        self.m = margin

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        # get a one-hot index
        index = inputs.data * 0.0
        index.scatter_(1, targets.data.view(-1, 1), 1)
        index = index.bool()

        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        cos_t = inputs[index]
        sin_t = torch.sqrt(1.0 - cos_t * cos_t)
        cos_t_add_m = cos_t * cos_m - sin_t * sin_m

        cond_v = cos_t - math.cos(math.pi - self.m)
        cond = F.relu(cond_v)
        keep = cos_t - math.sin(math.pi - self.m) * self.m

        cos_t_add_m = torch.where(cond.bool(), cos_t_add_m, keep)

        output = inputs * 1.0
        output[index] = cos_t_add_m
        output = self.s * output

        return F.cross_entropy(output, targets)


class CircleLoss(nn.Module):
    def __init__(self, scale=96, margin=0.3, **kwargs):
        super(CircleLoss, self).__init__()
        self.s = scale
        self.m = margin

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        mask = torch.zeros_like(inputs).to(input.device)
        mask.scatter_(1, targets.view(-1, 1), 1.0)

        pos_scale = self.s * F.relu(1 + self.m - inputs.detach())
        neg_scale = self.s * F.relu(inputs.detach() + self.m)
        scale_matrix = pos_scale * mask + neg_scale * (1 - mask)

        scores = (inputs - (1 - self.m) * mask - self.m * (1 - mask)) * scale_matrix

        loss = F.cross_entropy(scores, targets)

        return loss


class PairwiseCircleLoss(nn.Module):
    def __init__(self, scale=48, margin=0.35, **kwargs):
        super(PairwiseCircleLoss, self).__init__()
        self.s = scale
        self.m = margin

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs = F.normalize(inputs, p=2, dim=1)
        similarities = torch.matmul(inputs, inputs.t())

        targets = targets.view(-1, 1)
        mask = torch.eq(targets, targets.T).float().to(input.device)
        mask_self = torch.eye(targets.size(0)).float().to(input.device)
        mask_pos = mask - mask_self
        mask_neg = 1 - mask

        pos_scale = self.s * F.relu(1 + self.m - similarities.detach())
        neg_scale = self.s * F.relu(similarities.detach() + self.m)
        scale_matrix = pos_scale * mask_pos + neg_scale * mask_neg

        scores = (similarities - self.m) * mask_neg + (
            1 - self.m - similarities
        ) * mask_pos
        scores = scores * scale_matrix

        neg_scores_lse = torch.logsumexp(
            scores * mask_neg - 99999999 * (1 - mask_neg), dim=1
        )
        pos_scores_lse = torch.logsumexp(
            scores * mask_pos - 99999999 * (1 - mask_pos), dim=1
        )

        loss = F.softplus(neg_scores_lse + pos_scores_lse).mean()

        return loss


class ClothesBasedAdversarialLoss(nn.Module):
    """Clothes-based Adversarial Loss.

    Reference:
        Gu et al. Clothes-Changing Person Re-identification with RGB Modality Only. In CVPR, 2022.

    Args:
        scale (float): scaling factor.
        epsilon (float): a trade-off hyper-parameter.
    """

    def __init__(self, scale=16, epsilon=0.1):
        super().__init__()
        self.scale = scale
        self.epsilon = epsilon

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor, positive_mask: torch.Tensor
    ):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
            positive_mask: positive mask matrix with shape (batch_size, num_classes). The clothes classes with
                the same identity as the anchor sample are defined as positive clothes classes and their mask
                values are 1. The clothes classes with different identities from the anchor sample are defined
                as negative clothes classes and their mask values in positive_mask are 0.
        """
        inputs = self.scale * inputs
        negtive_mask = 1 - positive_mask
        identity_mask = (
            torch.zeros(inputs.size())
            .scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
            .to(self.parameters.device)
        )

        exp_logits = torch.exp(inputs)
        log_sum_exp_pos_and_all_neg = torch.log(
            (exp_logits * negtive_mask).sum(1, keepdim=True) + exp_logits
        )
        log_prob = inputs - log_sum_exp_pos_and_all_neg

        mask = (1 - self.epsilon) * identity_mask + self.epsilon / positive_mask.sum(
            1, keepdim=True
        ) * positive_mask
        loss = (-mask * log_prob).sum(1).mean()

        return loss


class ClothesBasedAdversarialLossWithMemoryBank(nn.Module):
    """Clothes-based Adversarial Loss between mini batch and the samples in memory.

    Reference:
        Gu et al. Clothes-Changing Person Re-identification with RGB Modality Only. In CVPR, 2022.

    Args:
        num_clothes (int): the number of clothes classes.
        feat_dim (int): the dimensions of feature.
        momentum (float): momentum to update memory.
        scale (float): scaling factor.
        epsilon (float): a trade-off hyper-parameter.
    """

    def __init__(self, num_clothes, feat_dim, momentum=0.0, scale=16, epsilon=0.1):
        super().__init__()
        self.num_clothes = num_clothes
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.epsilon = epsilon
        self.scale = scale

        self.register_buffer("feature_memory", torch.zeros((num_clothes, feat_dim)))
        self.register_buffer(
            "label_memory", torch.zeros(num_clothes, dtype=torch.int64) - 1
        )
        self.has_been_filled = False

    def forward(self, inputs, targets, positive_mask):
        """
        Args:
            inputs: sample features (before classifier) with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
            positive_mask: positive mask matrix with shape (batch_size, num_classes).
        """
        # gather all samples from different GPUs to update memory.
        gathered_inputs = torch.cat(GatherLayer.apply(inputs), dim=0)
        gathered_targets = torch.cat(GatherLayer.apply(targets), dim=0)
        self._update_memory(gathered_inputs.detach(), gathered_targets)

        inputs_norm = F.normalize(inputs, p=2, dim=1)
        memory_norm = F.normalize(self.feature_memory.detach(), p=2, dim=1)
        similarities = torch.matmul(inputs_norm, memory_norm.t()) * self.scale

        negtive_mask = 1 - positive_mask
        mask_identity = (
            torch.zeros(positive_mask.size())
            .scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
            .cuda()
        )

        if not self.has_been_filled:
            invalid_index = self.label_memory == -1
            positive_mask[:, invalid_index] = 0
            negtive_mask[:, invalid_index] = 0
            if sum(invalid_index.type(torch.int)) == 0:
                self.has_been_filled = True
                print("Memory bank is full")

        # compute log_prob
        exp_logits = torch.exp(similarities)
        log_sum_exp_pos_and_all_neg = torch.log(
            (exp_logits * negtive_mask).sum(1, keepdim=True) + exp_logits
        )
        log_prob = similarities - log_sum_exp_pos_and_all_neg

        # compute mean of log-likelihood over positive
        mask = (1 - self.epsilon) * mask_identity + self.epsilon / positive_mask.sum(
            1, keepdim=True
        ) * positive_mask
        loss = (-mask * log_prob).sum(1).mean()

        return loss

    def _update_memory(self, features, labels):
        label_to_feat = {}
        for x, y in zip(features, labels):
            if y not in label_to_feat:
                label_to_feat[y] = [x.unsqueeze(0)]
            else:
                label_to_feat[y].append(x.unsqueeze(0))
        if not self.has_been_filled:
            for y in label_to_feat:
                feat = torch.mean(torch.cat(label_to_feat[y], dim=0), dim=0)
                self.feature_memory[y] = feat
                self.label_memory[y] = y
        else:
            for y in label_to_feat:
                feat = torch.mean(torch.cat(label_to_feat[y], dim=0), dim=0)
                self.feature_memory[y] = (
                    self.momentum * self.feature_memory[y]
                    + (1.0 - self.momentum) * feat
                )
                # self.embedding_memory[y] /= self.embedding_memory[y].norm()


class ContrastiveLoss(nn.Module):
    """Supervised Contrastive Learning Loss among sample pairs.

    Args:
        scale (float): scaling factor.
    """

    def __init__(self, scale=16, **kwargs):
        super().__init__()
        self.s = scale

    def forward(self, inputs, targets):
        """
        Args:
            inputs: sample features (before classifier) with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """
        # l2-normalize
        inputs = F.normalize(inputs, p=2, dim=1)

        # gather all samples from different GPUs as gallery to compute pairwise loss.
        gallery_inputs = torch.cat(GatherLayer.apply(inputs), dim=0)
        gallery_targets = torch.cat(GatherLayer.apply(targets), dim=0)
        m, n = targets.size(0), gallery_targets.size(0)

        # compute cosine similarity
        similarities = torch.matmul(inputs, gallery_inputs.t()) * self.s

        # get mask for pos/neg pairs
        targets, gallery_targets = targets.view(-1, 1), gallery_targets.view(-1, 1)
        mask = torch.eq(targets, gallery_targets.T).float().cuda()
        mask_self = torch.zeros_like(mask)
        rank = dist.get_rank()
        mask_self[:, rank * m : (rank + 1) * m] += torch.eye(m).float().cuda()
        mask_pos = mask - mask_self
        mask_neg = 1 - mask

        # compute log_prob
        exp_logits = torch.exp(similarities) * (1 - mask_self)
        # log_prob = similarities - torch.log(exp_logits.sum(1, keepdim=True))
        log_sum_exp_pos_and_all_neg = torch.log(
            (exp_logits * mask_neg).sum(1, keepdim=True) + exp_logits
        )
        log_prob = similarities - log_sum_exp_pos_and_all_neg

        # compute mean of log-likelihood over positive
        loss = (mask_pos * log_prob).sum(1) / mask_pos.sum(1)

        loss = -loss.mean()

        return loss


class CosFaceLoss(nn.Module):
    """CosFace Loss based on the predictions of classifier.

    Reference:
        Wang et al. CosFace: Large Margin Cosine Loss for Deep Face Recognition. In CVPR, 2018.

    Args:
        scale (float): scaling factor.
        margin (float): pre-defined margin.
    """

    def __init__(self, scale=16, margin=0.1, **kwargs):
        super().__init__()
        self.s = scale
        self.m = margin

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        one_hot = torch.zeros_like(inputs)
        one_hot.scatter_(1, targets.view(-1, 1), 1.0)

        output = self.s * (inputs - one_hot * self.m)

        return F.cross_entropy(output, targets)


class PairwiseCosFaceLoss(nn.Module):
    """CosFace Loss among sample pairs.

    Reference:
        Sun et al. Circle Loss: A Unified Perspective of Pair Similarity Optimization. In CVPR, 2020.

    Args:
        scale (float): scaling factor.
        margin (float): pre-defined margin.
    """

    def __init__(self, scale=16, margin=0):
        super().__init__()
        self.s = scale
        self.m = margin

    def forward(self, inputs, targets):
        """
        Args:
            inputs: sample features (before classifier) with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """
        # l2-normalize
        inputs = F.normalize(inputs, p=2, dim=1)

        # gather all samples from different GPUs as gallery to compute pairwise loss.
        gallery_inputs = torch.cat(GatherLayer.apply(inputs), dim=0)
        gallery_targets = torch.cat(GatherLayer.apply(targets), dim=0)
        m, n = targets.size(0), gallery_targets.size(0)

        # compute cosine similarity
        similarities = torch.matmul(inputs, gallery_inputs.t())

        # get mask for pos/neg pairs
        targets, gallery_targets = targets.view(-1, 1), gallery_targets.view(-1, 1)
        mask = torch.eq(targets, gallery_targets.T).float().cuda()
        mask_self = torch.zeros_like(mask)
        rank = dist.get_rank()
        mask_self[:, rank * m : (rank + 1) * m] += torch.eye(m).float().cuda()
        mask_pos = mask - mask_self
        mask_neg = 1 - mask

        scores = (similarities + self.m) * mask_neg - similarities * mask_pos
        scores = scores * self.s

        neg_scores_LSE = torch.logsumexp(
            scores * mask_neg - 99999999 * (1 - mask_neg), dim=1
        )
        pos_scores_LSE = torch.logsumexp(
            scores * mask_pos - 99999999 * (1 - mask_pos), dim=1
        )

        loss = F.softplus(neg_scores_LSE + pos_scores_LSE).mean()

        return loss


class CrossEntropyWithLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularization.

    Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. In CVPR, 2016.
    Equation:
        y = (1 - epsilon) * y + epsilon / K.

    Args:
        epsilon (float): a hyper-parameter in the above equation.
    """

    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        _, num_classes = inputs.size()
        log_probs = self.logsoftmax(inputs)
        targets = (
            torch.zeros(log_probs.size())
            .scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
            .cuda()
        )
        targets = (1 - self.epsilon) * targets + self.epsilon / num_classes
        loss = (-targets * log_probs).mean(0).sum()

        return loss


class InstanceLoss(nn.Module):
    def __init__(self, gamma=1) -> None:
        super(InstanceLoss, self).__init__()
        self.gamma = gamma

    def forward(self, feature, label=None) -> torch.Tensor:
        # Dual-Path Convolutional Image-Text Embeddings with Instance Loss, ACM TOMM 2020
        # https://zdzheng.xyz/files/TOMM20.pdf
        # using cross-entropy loss for every sample if label is not available. else use given label.
        normed_feature = l2_norm(feature)
        sim1 = torch.mm(normed_feature * self.gamma, torch.t(normed_feature))
        # sim2 = sim1.t()
        if label is None:
            sim_label = torch.arange(sim1.size(0)).cuda().detach()
        else:
            _, sim_label = torch.unique(label, return_inverse=True)
        loss = F.cross_entropy(sim1, sim_label)  # + F.cross_entropy(sim2, sim_label)
        return loss


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, distance="euclidean"):
        super(TripletLoss, self).__init__()
        if distance not in ["euclidean", "cosine"]:
            raise KeyError("Unsupported distance: {}".format(distance))
        self.distance = distance
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        if self.distance == "euclidean":
            dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
            dist = dist + dist.t()
            dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        elif self.distance == "cosine":
            inputs = F.normalize(inputs, p=2, dim=1)
            dist = -torch.mm(inputs, inputs.t())

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss
