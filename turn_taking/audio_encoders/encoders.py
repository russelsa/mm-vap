import sys
sys.path.append("/mnt/storage/github/turn-taking")
import torch
from torch import nn
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers import AutoProcessor, HubertModel
from torch import Tensor
import torch
import torch.nn as nn
import einops
from turn_taking.audio_encoders.encoder_cpc_components import load_CPC, get_cnn_layer


class EncoderCPC(nn.Module):
    """
    Encoder: waveform -> h
    pretrained: default='cpc'

    A simpler version of the Encoder
    check paper (branch) version to see other encoders...
    """

    def __init__(self, load_pretrained=True, freeze=True):
        super().__init__()
        self.sample_rate = 16000
        self.encoder = load_CPC(load_pretrained)
        self.output_dim = self.encoder.gEncoder.conv4.out_channels
        self.dim = self.output_dim

        self.downsample_ratio = 160
        self.downsample = get_cnn_layer(
            dim=self.output_dim,
            kernel=[5],
            stride=[2],
            dilation=[1],
            activation="GELU",
        )
        self.downsample_ratio = 320

        if freeze:
            self.freeze()

    def get_default_conf(self):
        return {""}

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        print(f"Froze {self.__class__.__name__}!")

    def unfreeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(True)
        print(f"Trainable {self.__class__.__name__}!")

    def forward(self, waveform):
        if waveform.ndim < 3:
            waveform = waveform.unsqueeze(1)  # channel dim

        # Backwards using only the encoder encounters:
        # ---------------------------------------------------
        # RuntimeError: one of the variables needed for gradient computation
        # has been modified by an inplace operation:
        # [torch.FloatTensor [4, 256, 1000]], which is output 0 of ReluBackward0, is at version 1;
        # expected version 0 instead. Hint: enable anomaly detection to find
        # the operation that failed to compute its gradient, with
        # torch.autograd.set_detect_anomaly(True).
        # HOWEVER, if we feed through encoder.gAR we do not encounter that problem...
        z = self.encoder.gEncoder(waveform)
        z = einops.rearrange(z, "b c n -> b n c")
        z = self.encoder.gAR(z)
        z = self.downsample(z)
        return z


class StereoEncoder(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = EncoderCPC()

    def forward(self, x):
        xl = self.encoder(x[:,:,0])
        xr = self.encoder(x[:,:,1])
        return torch.stack((xl,xr),dim=-1)
        



class Wav2VecEncoder(nn.Module):
    pass


class HubertEncoder(torch.nn.Module):
    pass
    

if __name__=="__main__":
    encoder = EncoderCPC()
    # N, B, D
    x = torch.rand([80_000,10,1])
    z = encoder(x)
    x=1