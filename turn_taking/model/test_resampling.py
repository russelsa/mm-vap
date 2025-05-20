import einops
import numpy as np
import torch
import torch.nn as nn


def resample(input_tensor):
        # Example usage:
    B, N, D, C = 32, 600, 58, 2
    input_tensor = torch.randn(B, N, D, C)

    scale_factor = 1000 / 600  # To increase from 600 to 1000


    # Set the slice at index 25 in the 3rd dimension to be 25
    for i in range(600):
        input_tensor[:, i, ...] = i

    print(input_tensor[:, 22, ...])


    # Apply the network to the input tensor
    input_tensor = torch.Tensor(input_tensor)

    rs = nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=True)

    input_tensor_reshaped = einops.rearrange(input_tensor, "b n d c -> (b d) c n")
    output_tensor = rs(input_tensor_reshaped)
    output_tensor = einops.rearrange(output_tensor, "(b d) c n -> b n d c", d=D, b=B)

    print(output_tensor[:, 37, ...])
    print(output_tensor.shape)
