import torch
from torch import nn
import torch.distributed as dist


def get_num_channels(backbone: nn.Module, in_channels: int, output_name: str) -> int:
    dummies_input = torch.randn(1, in_channels, 224, 224)
    with torch.no_grad():
        features = backbone(dummies_input)[output_name]
    return features.shape[1]


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)

        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)

        # dist.reduce_scatter(grad_out, list(grads))
        # grad_out.div_(dist.get_world_size())

        grad_out[:] = grads[dist.get_rank()]

        return grad_out
