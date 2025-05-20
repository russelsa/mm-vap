from typing import Optional
import torch
from torch import Tensor
from torch import nn
from turn_taking.audio_encoders.encoders import StereoEncoder
from turn_taking.model.modules import *
import einops
import torch.nn.functional as F


class GPT(nn.Module):
    """
    GPT like transformer Decoder-only class.

    * Uses AliBi attention (no positional embeddings or max-sequence-length)
    """

    def __init__(
        self,
        dim: int,
        dff_k: int = 3,
        num_layers: int = 4,
        num_heads: int = 4,
        activation: str = "GELU",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.dff = int(dim * dff_k)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = dropout

        self._build_layers()
        self.apply(self._init_weights)

    def _build_layers(self):
        layers = []
        for _ in range(self.num_layers):
            layers.append(
                TransformerLayer(
                    dim=self.dim,
                    ffn_dim=self.dff,
                    num_heads=self.num_heads,
                    ffn_activation=self.activation,
                    dropout=self.dropout,
                )
            )
        self.layers = nn.ModuleList(layers)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        all_attention = []

        for layer in self.layers:
            x, self_attn_weights, _ = layer(x)
            if return_attention:
                all_attention.append(self_attn_weights)

        ret = {"x": x}

        if return_attention:
            self_attn_weights = torch.stack(all_attention, dim=1)
            ret["attn"] = self_attn_weights

        return ret


class GPTStereo(GPT):
    def _build_layers(self):
        layers = []
        for _ in range(self.num_layers):
            layers.append(
                TransformerStereoLayer(
                    dim=self.dim,
                    ffn_dim=self.dff,
                    num_heads=self.num_heads,
                    ffn_activation=self.activation,
                    dropout=self.dropout,
                    cross_attention=True,
                )
            )
        self.layers = nn.ModuleList(layers)

        # Combine output from both 'towers'
        self.combinator = Combinator(dim=self.dim, activation="GELU")

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:

        self_attn_a = []
        self_attn_b = []
        cross_attn_a = []
        cross_attn_b = []
        for layer in self.layers:
            x1, x2, attn_list = layer(x1=x1, x2=x2)
            if return_attention:
                # [sa1w, ca1w, sa2w, ca2w] = attn_list
                self_attn_a.append(attn_list[0])
                cross_attn_a.append(attn_list[1])
                self_attn_b.append(attn_list[2])
                cross_attn_b.append(attn_list[3])

        x = self.combinator(x1, x2)
        ret = {"x": x, "x1": x1, "x2": x2}

        if return_attention:
            # B, num_layers, num_heads, N, N
            self_attn_a = torch.stack(self_attn_a, dim=1)  # stack on layer dim
            self_attn_b = torch.stack(self_attn_b, dim=1)  # stack on layer dim
            cross_attn_a = torch.stack(cross_attn_a, dim=1)  # stack on layer dim
            cross_attn_b = torch.stack(cross_attn_b, dim=1)  # stack on layer dim
            ret["self_attn"] = torch.stack([self_attn_a, self_attn_b], dim=1)
            ret["cross_attn"] = torch.stack([cross_attn_a, cross_attn_b], dim=1)
        return ret


class Combinator(nn.Module):
    """
    Combines the "ego-centric" representations from identical 'towers'
    processing channel 0 and 1. The towers are identical (shared weights)
    and therefore channel agnostic, e.g. they don't know if they process information
    from the view of speaker A or B.

    Here we have specific layers associated with each channel to join the representations
    into a single coherent space with channel information included.
    """

    def __init__(self, dim: int, activation: str = "GELU"):
        super().__init__()
        self.dim = dim

        # Channel information
        self.h0_a = nn.Linear(dim, dim, bias=False)  # Channel 0
        self.h0_b = nn.Linear(dim, dim, bias=False)  # Channel 1
        self.ln = nn.LayerNorm(self.dim)

        # Activation
        self.activation = getattr(nn, activation)()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Combines the hidden states from identical 'towers' which have processed
        each channel from an 'ego-centric' view. However, the towers are channel agnostic
        by default (shared weights) so in this step we process the information from channel 0, 1
        separately into a joint representation.

        The final representation will (see GPTStereo -> ProjectionModel) go into a final linear
        layer to produce logits.
        """

        # Channel specific information
        ha = self.activation(self.ln(self.h0_a(x1)))
        hb = self.activation(self.ln(self.h0_b(x2)))
        h = ha + hb  # combine estimations from both parties
        return h


class StereoTransformerModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        
        self.cfg=cfg

        # encoder
        self.audio_encoder = StereoEncoder().to("cuda")
        
        # for the individual channels
        self.self_attention_transformer = GPT(
            dim=cfg['model_cfg']['d_model'],
            dff_k=cfg['model_cfg']['dff_k'],
            num_layers=cfg['model_cfg']['n_layers_self'],
            num_heads=cfg['model_cfg']['n_heads'],
            activation=cfg['model_cfg']['ffn_activation'],
            dropout=cfg['model_cfg']['dropout_prob']
        )

        # for channel information
        self.cross_attention_transformer = GPTStereo(
            dim=cfg['model_cfg']['d_model'],
            dff_k=cfg['model_cfg']['dff_k'],
            num_layers=cfg['model_cfg']['n_layers_cross'],
            num_heads=cfg['model_cfg']['n_heads'],
            activation=cfg['model_cfg']['ffn_activation'],
            dropout=cfg['model_cfg']['dropout_prob']
        )

        self.vad_classifier = nn.Linear(in_features=cfg['model_cfg']['d_model'], out_features=1)
        self.vap_classifier = nn.Linear(in_features=cfg['model_cfg']['d_model'], out_features=cfg['model_cfg']['d_out'])


    def forward(self, batch):
        
        with torch.no_grad():
            audio = self.audio_encoder(batch['audio_chunk'].to("cuda"))
            x_l, x_r = audio[:,:,:,0].to("cuda"), audio[:,:,:,1].to("cuda")
        
        # pass thru layers
        x_l = self.self_attention_transformer(x_l, return_attention=False)
        x_r = self.self_attention_transformer(x_r, return_attention=False)
        out = self.cross_attention_transformer(x_l["x"], x_r["x"], return_attention=False)

        # output
        v1 = self.vad_classifier(out["x1"])
        v2 = self.vad_classifier(out["x2"])

        vad = torch.cat((v1, v2), dim=-1)
        vap = self.vap_classifier(out['x'])

        return vad, vap


class StereoTransformerModelVideoOnly(StereoTransformerModel):

    def __init__(self, cfg):
        super().__init__(cfg)
        scale_factor = cfg['multimodal_model_cfg']['audio']['sequence_len'] / cfg['multimodal_model_cfg']['video']['sequence_len']
        scale_factor = scale_factor
        self.upsampler = nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=True)

        # project video to audio dim
        self.video_proj_ln = nn.LayerNorm(cfg['multimodal_model_cfg']['audio']['d_model'])
        self.video_proj = nn.Linear(in_features=cfg['multimodal_model_cfg']['video']['d_model'], out_features=cfg['multimodal_model_cfg']['audio']['d_model'], bias=False)

        # Activation
        self.activation = getattr(nn, cfg['multimodal_model_cfg']['ffn_activation'])()

    def upsample(self, x):
        B, N, D, C = x.shape
        x=einops.rearrange(x,"b n d c -> (b c) d n")
        x=self.upsampler(x)
        x=einops.rearrange(x,"(b c) d n -> b n d c", b=B, c=C)
        return x

    def forward(self, batch):
        
        batch['frames']=torch.nan_to_num(batch['frames'])
        video = self.upsample(batch['frames']).to(device='cuda')
        x_l, x_r = video[:,:,:,0].to("cuda"), video[:,:,:,1].to("cuda")
        
        # pass thru layers
        x_l = self.self_attention_transformer(x_l, return_attention=False)
        x_r = self.self_attention_transformer(x_r, return_attention=False)
        out = self.cross_attention_transformer(x_l["x"], x_r["x"], return_attention=False)

        # output
        v1 = self.vad_classifier(out["x1"])
        v2 = self.vad_classifier(out["x2"])

        vad = torch.cat((v1, v2), dim=-1)
        vap = self.vap_classifier(out['x'])

        return vad, vap

def vap_loss_fn(y_true: dict, y_pred: dict, mode: str):
    """loss function for the stereo transformer model with 256 prob output

    Args:
        batch (dict): _description_
        y_pred (dict): _description_

    Return:
        loss (float): loss
    """

    assert mode in ['VAP', 'ind-4', 'ind-40']

    y_vap_true = y_true['vap']
    y_vap_pred = y_pred['vap']

    if mode in ['ind-4', 'ind-40']:
        vap_loss = F.binary_cross_entropy_with_logits(y_vap_pred, y_vap_true.to(y_vap_pred.device))
    
    elif mode == 'VAP':
        y_vap_pred = einops.rearrange(y_vap_pred, "b n d -> (b n) d")
        y_vap_true = einops.rearrange(y_vap_true, "b n -> (b n)")
        vap_loss = F.cross_entropy(y_vap_pred, y_vap_true.to(y_vap_pred.device), reduction="mean")
    
    return vap_loss


def vad_loss_fn(y_true: dict, y_pred: dict):
    
    y_vad_true = y_true['vad']
    y_vad_pred = y_pred['vad']

    y_vad_true = y_vad_true.to(y_vad_pred.device)

    vad_loss = F.binary_cross_entropy_with_logits(y_vad_pred, y_vad_true)
    
    return vad_loss
