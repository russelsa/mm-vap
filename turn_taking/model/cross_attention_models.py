"""https://raw.githubusercontent.com/ErikEkstedt/VoiceActivityProjection/main/vap/modules.py"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from typing import Dict, Optional, Tuple
import numpy as np
import math
import yaml
from turn_taking.audio_encoders.encoders import StereoEncoder


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout_prob: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)

        self.register_buffer('positional_encodings', get_positional_encoding(d_model, max_len), False)

    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:x.shape[0]].detach().requires_grad_(False)
        x = x + pe
        x = self.dropout(x)
        return x


def get_positional_encoding(d_model: int, max_len: int = 5000):
    # Empty encodings vectors
    encodings = torch.zeros(max_len, d_model)
    # Position indexes
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    # $2 * i$
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)
    # $10000^{\frac{2i}{d_{model}}}$
    div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
    # $PE_{p,2i} = sin\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)$
    encodings[:, 0::2] = torch.sin(position * div_term)
    # $PE_{p,2i + 1} = cos\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)$
    encodings[:, 1::2] = torch.cos(position * div_term)

    # Add batch dimension
    encodings = encodings.unsqueeze(1).requires_grad_(False)

    return encodings


def ffn_block(
    din: int,
    dff: int,
    activation: str = "GELU",
    dropout: float = 0.0,
    bias: bool = False,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(din, dff, bias=bias),
        getattr(nn, activation)(),
        nn.Dropout(p=dropout),
        nn.Linear(dff, din, bias=bias),
    )


class MultiHeadAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, dim: int, d_src: int, num_heads: int, dropout: float, bias: bool = False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.dim = dim

        self.internal_dim = self.dim
        print("self.internal_dim ", self.internal_dim)

        # key, query, value projections for all heads
        self.query = nn.Linear(dim, self.internal_dim, bias=bias)
        self.key = nn.Linear(d_src, self.internal_dim, bias=bias)
        self.value = nn.Linear(d_src, self.internal_dim, bias=bias)

        # head re-shapers
        self.unstack_heads = Rearrange("b t (h d) -> b h t d", h=self.num_heads)
        self.stack_heads = Rearrange("b h t d -> b t (h d)")

        # regularization
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # output projection
        self.proj = nn.Linear(self.internal_dim, dim, bias=bias)
        self.scale = 1.0 / math.sqrt(self.internal_dim)

    def get_scores(self, q: torch.Tensor, k: torch.Tensor):
        """
        Arguments:
            q: (B, heads, T, D)
            k: (B, heads, T, D)

        Return:
            QK:     (B, heads, T, T)
        """
        return torch.einsum("bhid,bhjd->bhij", q, k)

    @staticmethod
    def prepare_causal_mask(T, device="cpu", dtype=torch.float32):
        mask = torch.tril(torch.ones((T, T), device=device, dtype=dtype)).view(
            1, 1, T, T
        )
        mask.requires_grad_(False)
        return mask

    def mask_scores(self, qk: torch.Tensor, mask=None):
        T = qk.size(-1)
        if mask is None:
            mask = MultiHeadAttention.prepare_causal_mask(
                T, device=qk.device, dtype=qk.dtype
            )
        qk = qk.masked_fill(mask == 0, float("-inf"))
        return qk

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, D = Q.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.unstack_heads(self.key(K))  # (B, heads, T, D_head)
        q = self.unstack_heads(self.query(Q))  # (B, heads, T, D_head)
        v = self.unstack_heads(self.value(V))  # (B, heads, T, D_head)

        # QK
        att = self.get_scores(q, k) * self.scale  #  (B, nh, T, T)
        att = self.mask_scores(att, mask)
        att = F.softmax(att, dim=-1)

        # Softmax, dropout, values
        y = self.attn_drop(att) @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # re-assemble all head outputs side by side
        y = self.stack_heads(y)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class MultiCrossAttention(MultiHeadAttention):
    
    @staticmethod
    def asymmetric_mask(i, j):
        s=1.67
        delay=3
        mask = np.zeros(i.shape)
        ii,jj = np.where(np.round(i/s) >= j)
        mask[ii,jj]=1
        return mask
    

    def mask_scores(self, qk: torch.Tensor, mask=None, device="cuda", dtype=torch.float32):

        B, H, I, J = qk.shape

        if mask is None:
            mask = np.fromfunction(self.asymmetric_mask, (I, J))
            mask = torch.Tensor(mask).view(1, 1, I, J)
        mask = mask.to(qk.device)
        mask.requires_grad_(False)

        qk = qk.masked_fill(mask == 0, float("-inf"))
        return qk


class TransformerLayer(nn.Module):
    """
    Transformer Layer

    Using pre-layer-normalization: https://arxiv.org/pdf/2002.04745.pdf
    Inspiration: https://nn.labml.ai/transformers/models.html
    AliBI Attention: https://ofir.io/train_short_test_long.pdf
    """

    def __init__(
        self,
        dim: int = 256,
        d_src: Optional[int] = None,
        ffn_dim: int = 768,
        num_heads: int = 4,
        ffn_activation: str = "GELU",
        dropout: float = 0.1,
        cross_attention: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.d_src = self.dim
        if d_src is not None:
            self.d_src = d_src
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.cross_attention = cross_attention

        self.dropout = nn.Dropout(p=dropout)
        self.ln_self_attn = nn.LayerNorm(dim)
        self.ln_ffnetwork = nn.LayerNorm(dim)

        self.mha = MultiHeadAttention(
            dim=dim, d_src=dim, num_heads=num_heads, dropout=dropout
        )
        self.ffnetwork = ffn_block(
            dim, ffn_dim, activation=ffn_activation, dropout=dropout
        )

        if cross_attention:
            self.ln_src_attn = nn.LayerNorm(dim)
            self.mha_cross = MultiCrossAttention(
                dim=dim, d_src=self.d_src, num_heads=num_heads, dropout=dropout
            )

    def forward(
        self,
        x: torch.Tensor,
        src: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Using pre-layer-normalization: https://arxiv.org/pdf/2002.04745.pdf
        """

        # Self-attention
        z = self.ln_self_attn(x)
        self_attn, self_attn_weights = self.mha(Q=z, K=z, V=z, mask=mask)

        # Residual
        x = x + self.dropout(self_attn)

        # Cross-attention
        cross_attn_weights = None
        if self.cross_attention and src is not None:
            z = self.ln_src_attn(x)
            # https://nn.labml.ai/transformers/models.html#section-16
            # Don't normalize src... why?
            cross_attn, cross_attn_weights = self.mha_cross(
                Q=z, K=src, V=src, mask=mask
            )
            x = x + self.dropout(cross_attn)

        x = x + self.dropout(self.ffnetwork(self.ln_ffnetwork(x)))
        return x, self_attn_weights, cross_attn_weights


class TransformerStereoLayer(TransformerLayer):
    """
        same x and src dimension
    """
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        # sa1w: self-attention-weights 1
        # ca1w: cross-attention-weights 1
        z1, sa1w, ca1w = super().forward(x=x1, src=x2, mask=mask)
        z2, sa2w, ca2w = super().forward(x=x2, src=x1, mask=mask)
        return z1, z2, [sa1w, ca1w, sa2w, ca2w]


class TransformerStereoLayerAsymmetric(nn.Module):
    """
        different x and src dimensions
    """
    def __init__(self, dim, d_src, **kwargs):

        super().__init__()
        
        self.left = TransformerLayer(dim=dim, d_src=d_src, **kwargs)
        self.right = TransformerLayer(dim=d_src, d_src=dim, **kwargs)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        # sa1w: self-attention-weights 1
        # ca1w: cross-attention-weights 1
        z1, sa1w, ca1w = self.left(x=x1, src=x2, mask=mask)
        z2, sa2w, ca2w = self.right(x=x2, src=x1, mask=mask)
        return z1, z2, [sa1w, ca1w, sa2w, ca2w]


class GPT(nn.Module):
    """
    GPT like transformer Decoder-only class.

    """

    def __init__(
        self,
        dim: int,
        dim_src: int,
        dff_k: int = 3,
        num_layers: int = 4,
        num_heads: int = 4,
        activation: str = "GELU",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dim = dim
        if dim_src is None:
            dim_src = self.dim
        self.dim_src = dim_src
        self.dff = int(dim * dff_k)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = dropout

        self.positional_embedding = PositionalEncoding(self.dim, dropout_prob=0.1)

        self._build_layers()
        self.apply(self._init_weights)

    def _build_layers(self):
        layers = []
        for _ in range(self.num_layers):
            layers.append(
                TransformerLayer(
                    dim=self.dim,
                    d_src=self.dim_src,
                    ffn_dim=self.dff,
                    num_heads=self.num_heads,
                    ffn_activation=self.activation,
                    dropout=self.dropout,
                    cross_attention=False                
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

        x = self.positional_embedding(x)

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

    def __init__(self, combine, *args, **kwargs):
        self.combine = combine
        super().__init__(*args, **kwargs)

    def _build_layers(self):
        layers = []

        if self.dim == self.dim_src:
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
        else:
            for _ in range(self.num_layers):
                layers.append(
                    TransformerStereoLayerAsymmetric(
                        dim=self.dim,
                        d_src=self.dim_src,
                        ffn_dim=self.dff,
                        num_heads=self.num_heads,
                        ffn_activation=self.activation,
                        dropout=self.dropout,
                        cross_attention=True,
                    )
                )
        
        self.layers = nn.ModuleList(layers)

        # Combine output from both 'towers'
        if self.combine=='add':
            self.combinator = Combinator(dim=self.dim, activation="GELU")
        elif self.combine=='cat':
            self.combinator = CombinatorCat(dim=self.dim, dim_src=self.dim_src, activation="GELU")

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

        x=None
        if self.combine:
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

class SoftDocBlock(GPT):

    """ block inspired by softblock paper"""

    def __init__(self, combine, *args, **kwargs):
        self.combine = combine
        super().__init__(*args, **kwargs)

    def _build_layers(self):
        layers = []

        if self.dim == self.dim_src:
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
            self.self_attn = TransformerLayer(
                dim=self.dim,
                d_src=self.dim_src,
                ffn_dim=self.dff,
                num_heads=self.num_heads,
                ffn_activation=self.activation,
                dropout=self.dropout,
                cross_attention=False
            )
            self.asymmetric=False
        else:
            for _ in range(self.num_layers):
                layers.append(
                    TransformerStereoLayerAsymmetric(
                        dim=self.dim,
                        d_src=self.dim_src,
                        ffn_dim=self.dff,
                        num_heads=self.num_heads,
                        ffn_activation=self.activation,
                        dropout=self.dropout,
                        cross_attention=True,
                    )
                )
            self.self_attn_l = TransformerLayer(
            dim=self.dim,
            d_src=self.dim_src,
            ffn_dim=self.dff,
            num_heads=self.num_heads,
            ffn_activation=self.activation,
            dropout=self.dropout,
            cross_attention=False
            )
            self.self_attn_r = TransformerLayer(
            dim=self.dim_src,
            d_src=self.dim,
            ffn_dim=self.dff,
            num_heads=self.num_heads,
            ffn_activation=self.activation,
            dropout=self.dropout,
            cross_attention=False
            )
            self.asymmetric=True

        # Combine output from both 'towers'
        if self.combine=='add':
            self.combinator = Combinator(dim=self.dim, activation="GELU")
        elif self.combine=='cat':
            self.combinator = CombinatorCat(dim=self.dim, dim_src=self.dim_src, activation="GELU")
        
        self.layers = nn.ModuleList(layers)

        # Combine output from both 'towers'
        if self.combine=='add':
            self.combinator = Combinator(dim=self.dim, activation="GELU")
        elif self.combine=='cat':
            self.combinator = CombinatorCat(dim=self.dim, dim_src=self.dim_src, activation="GELU")

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

        x=None

        if self.asymmetric:
            # not sharing weights
            x1, _, _ = self.self_attn_l(x1)
            x2, _, _ = self.self_attn_r(x2)
        else:
            x1, _, _ = self.self_attn(x1)
            x2, _, _ = self.self_attn(x2)

        if self.combine:
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

class CombinatorCat(Combinator):
    def __init__(self, dim_src, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.h0_b = nn.Linear(dim_src, dim_src, bias=False)
        self.ln2 = nn.LayerNorm(dim_src)

    def forward(self, x1, x2):
        # Channel specific information
        ha = self.activation(self.ln(self.h0_a(x1)))
        hb = self.activation(self.ln2(self.h0_b(x2)))
        h = torch.cat((ha,hb),dim=-1)  # combine estimations from both parties
        return h

class CrossAttnLateVAFusion(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()

        """
            cross channel information is learnt for audio video separately before fusion to output 
        """

        self.cfg = cfg

        self.video_cross_attention = GPTStereo(            
            dim=cfg['multimodal_model_cfg']['video']['d_model'],
            dim_src=cfg['multimodal_model_cfg']['video']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['fuse_speakers']['n_layers_cross'],
            num_heads=cfg['multimodal_model_cfg']['video']['n_heads'],
            activation=cfg['model_cfg']['ffn_activation'],
            dropout=cfg['model_cfg']['dropout_prob'],
            combine='add'
        )

        self.audio_individual_speaker_cross_attention = GPTStereo(
            dim=cfg['multimodal_model_cfg']['audio']['d_model'],
            dim_src=cfg['multimodal_model_cfg']['audio']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['fuse_speakers']['n_layers_cross'],
            num_heads=cfg['model_cfg']['n_heads'],
            activation=cfg['model_cfg']['ffn_activation'],
            dropout=cfg['model_cfg']['dropout_prob'],
            combine='add'
        )

        self.video_self_attention = GPT(            
            dim=cfg['multimodal_model_cfg']['video']['d_model'],
            dim_src=None,
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['video']['n_layers_self'],
            num_heads=cfg['multimodal_model_cfg']['video']['n_heads'],
            activation=cfg['model_cfg']['ffn_activation'],
            dropout=cfg['model_cfg']['dropout_prob']
        )

        self.audio_self_attention = GPT(
            dim=cfg['multimodal_model_cfg']['audio']['d_model'],
            dim_src=None,
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['audio']['n_layers_self'],
            num_heads=cfg['model_cfg']['n_heads'],
            activation=cfg['model_cfg']['ffn_activation'],
            dropout=cfg['model_cfg']['dropout_prob']
        )

        # audio --> video (per speaker)
        self.audio_video_cross_attention = GPTStereo(
            dim=cfg['multimodal_model_cfg']['audio']['d_model'],
            dim_src=cfg['multimodal_model_cfg']['video']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['fuse_speakers']['n_layers_cross'],
            num_heads=cfg['multimodal_model_cfg']['video']['n_heads'],
            activation=cfg['model_cfg']['ffn_activation'],
            dropout=cfg['model_cfg']['dropout_prob'],
            combine=False
            )
        
        self.audio_fused_self_attention = GPT(
            dim=cfg['multimodal_model_cfg']['audio']['d_model'],
            dim_src=None,
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['audio']['n_layers_self'],
            num_heads=cfg['model_cfg']['n_heads'],
            activation=cfg['model_cfg']['ffn_activation'],
            dropout=cfg['model_cfg']['dropout_prob']
            )
        
        self.audio_fused_cross_attention = GPTStereo(
            dim=cfg['multimodal_model_cfg']['audio']['d_model'],
            dim_src=cfg['multimodal_model_cfg']['audio']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['fuse_speakers']['n_layers_cross'],
            num_layers=cfg['model_cfg']['n_layers_self'],
            num_heads=cfg['model_cfg']['n_heads'],
            activation=cfg['model_cfg']['ffn_activation'],
            dropout=cfg['model_cfg']['dropout_prob'],
            combine='add'
            )

        self.audio_encoder = StereoEncoder().to("cuda")
        self.vad_classifier = nn.Linear(in_features=cfg['model_cfg']['d_model'], out_features=1)
        self.vap_classifier = nn.Linear(in_features=cfg['multimodal_model_cfg']['audio']['d_model'], out_features=cfg['model_cfg']['d_out'])

    def forward(self, batch):
        
        with torch.no_grad():
            # get the L and R channels of the audio
            audio = self.audio_encoder(batch['audio_chunk'].to("cuda"))
            x_l, x_r = audio[:,:,:,0].to("cuda"), audio[:,:,:,1].to("cuda")

            # upsample the video
            batch['frames']=torch.nan_to_num(batch['frames'])
            v_l, v_r = batch['frames'][:,:,:,0].to("cuda"), batch['frames'][:,:,:,1].to("cuda")

        x_l = self.audio_self_attention(x_l)
        x_r = self.audio_self_attention(x_r)

        out = self.audio_individual_speaker_cross_attention(x_l["x"], x_r["x"])
        x_l, x_r = out["x1"], out["x2"]

        v_l = self.video_self_attention(v_l)
        v_r = self.video_self_attention(v_r)

        out = self.video_cross_attention(v_l["x"], v_r["x"])
        v_l, v_r = out["x1"], out["x2"]

        xl_vl = self.audio_video_cross_attention(x_l, v_l)
        xl_vl = self.audio_fused_self_attention(xl_vl["x1"])
        xl_vl = xl_vl["x"]

        xr_vr = self.audio_video_cross_attention(x_r, v_r)
        xr_vr = self.audio_fused_self_attention(xr_vr["x1"])
        xr_vr = xr_vr["x"]

        out = self.audio_fused_cross_attention(xl_vl, xr_vr)

        vap = self.vap_classifier(out["x"])

        v1 = self.vad_classifier(out["x1"])
        v2 = self.vad_classifier(out["x2"])

        vad = torch.cat((v1, v2), dim=-1)

        return vad, vap

class CrossAttnEarlyVAFusion(nn.Module):

    def __init__(self, cfg):

        super().__init__()

        self.cfg = cfg

        self.video_self_attention = GPT(            
            dim=cfg['multimodal_model_cfg']['video']['d_model'],
            dim_src=None,
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['video']['n_layers_self'],
            num_heads=cfg['model_cfg']['n_heads'],
            activation=cfg['model_cfg']['ffn_activation'],
            dropout=cfg['model_cfg']['dropout_prob']
        )

        self.audio_self_attention = GPT(
            dim=cfg['multimodal_model_cfg']['audio']['d_model'],
            dim_src=None,
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['audio']['n_layers_self'],
            num_heads=cfg['model_cfg']['n_heads'],
            activation=cfg['model_cfg']['ffn_activation'],
            dropout=cfg['model_cfg']['dropout_prob']
        )

        self.audio_video_self_attention = GPT(
            dim=cfg['multimodal_model_cfg']['audio']['d_model'],
            dim_src=None,
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['audio']['n_layers_self'],
            num_heads=cfg['model_cfg']['n_heads'],
            activation=cfg['model_cfg']['ffn_activation'],
            dropout=cfg['model_cfg']['dropout_prob']
        )

        self.audio_video_cross_attention = GPTStereo(
            dim=cfg['multimodal_model_cfg']['audio']['d_model'],
            dim_src=cfg['multimodal_model_cfg']['video']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['fuse_av']['n_layers_cross'],
            num_heads=cfg['model_cfg']['n_heads'],
            activation=cfg['model_cfg']['ffn_activation'],
            dropout=cfg['model_cfg']['dropout_prob'],
            combine=False
            )
        
        self.audio_cross_attention = GPTStereo(
            dim=cfg['multimodal_model_cfg']['audio']['d_model'],
            dim_src=cfg['multimodal_model_cfg']['audio']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['fuse_speakers']['n_layers_cross'],
            num_heads=cfg['model_cfg']['n_heads'],
            activation=cfg['model_cfg']['ffn_activation'],
            dropout=cfg['model_cfg']['dropout_prob'],
            combine='add'
        )
     
        self.audio_encoder = StereoEncoder().to("cuda")
        self.vad_classifier = nn.Linear(in_features=cfg['model_cfg']['d_model'], out_features=1)
        self.vap_classifier = nn.Linear(in_features=cfg['multimodal_model_cfg']['audio']['d_model'], out_features=cfg['model_cfg']['d_out'])

    def forward(self, batch):

        with torch.no_grad():
            audio = self.audio_encoder(batch['audio_chunk'].to("cuda"))
            x_l, x_r = audio[:,:,:,0].to("cuda"), audio[:,:,:,1].to("cuda")

            batch['frames']=torch.nan_to_num(batch['frames'])
            v_l, v_r = batch['frames'][:,:,:,0].to("cuda"), batch['frames'][:,:,:,1].to("cuda")

        x_l = self.audio_self_attention(x_l)
        x_r = self.audio_self_attention(x_r)

        v_l = self.video_self_attention(v_l)
        v_r = self.video_self_attention(v_r)

        x = self.audio_video_cross_attention(x_l["x"], v_l["x"])
        x_l = x["x1"]

        x = self.audio_video_cross_attention(x_r["x"], v_r["x"])
        x_r = x["x1"]
        
        # both are now 256
        x_l = self.audio_video_self_attention(x_l)
        x_r = self.audio_video_self_attention(x_r)

        x_r = x_r["x"]
        x_l = x_l["x"]
        
        out = self.audio_cross_attention(x_r, x_l)

        v1 = self.vad_classifier(out["x1"])
        v2 = self.vad_classifier(out["x2"])

        vad = torch.cat((v1, v2), dim=-1)
        
        vap = self.vap_classifier(out["x"])

        return vad, vap

class VideoEncoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.stack = nn.Sequential(
                      nn.Linear(in_features=cfg['multimodal_model_cfg']['video']['d_model'],
                      out_features=cfg['multimodal_model_cfg']['video']['d_model']
                      ),
                      nn.LayerNorm(
                          normalized_shape=cfg['multimodal_model_cfg']['video']['d_model']
                      ),
                      getattr(nn, cfg['model_cfg']['ffn_activation'])()
        )

    def forward(self, x):

        return self.stack(x)


class SoftDocEarly(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.block_1 = SoftDocBlock(
            dim=cfg['multimodal_model_cfg']['audio']['d_model'],
            dim_src=cfg['multimodal_model_cfg']['video']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['video']['n_layers_self'],
            num_heads=cfg['model_cfg']['n_heads'],
            activation=cfg['model_cfg']['ffn_activation'],
            dropout=cfg['model_cfg']['dropout_prob'],
            combine=False
        )

        self.block_2 = SoftDocBlock(
            dim=cfg['multimodal_model_cfg']['audio']['d_model'],
            dim_src=cfg['multimodal_model_cfg']['audio']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['video']['n_layers_self'],
            num_heads=cfg['model_cfg']['n_heads'],
            activation=cfg['model_cfg']['ffn_activation'],
            dropout=cfg['model_cfg']['dropout_prob'],
            combine='add'
        )

        self.audio_encoder = StereoEncoder().to("cuda")
        self.vad_classifier = nn.Linear(in_features=cfg['multimodal_model_cfg']['audio']['d_model'], out_features=1)
        self.vap_classifier = nn.Linear(in_features=cfg['multimodal_model_cfg']['audio']['d_model'], out_features=cfg['model_cfg']['d_out'])
        self.video_encoder = VideoEncoder(cfg).to("cuda")

    def forward(self, batch):

        with torch.no_grad():
            audio = self.audio_encoder(batch['audio_chunk'].to("cuda"))
            x_l, x_r = audio[:,:,:,0].to("cuda"), audio[:,:,:,1].to("cuda")

            batch['frames']=torch.nan_to_num(batch['frames'])
            v_l, v_r = batch['frames'][:,:,:,0].to("cuda"), batch['frames'][:,:,:,1].to("cuda")

        v_l, v_r = self.video_encoder(v_l), self.video_encoder(v_r)

        x = self.block_1(x_l, v_l)
        x_l = x["x1"]

        x = self.block_1(x_r, v_r)
        x_r = x["x1"]

        out = self.block_2(x_l, x_r)

        v1 = self.vad_classifier(out["x1"])
        v2 = self.vad_classifier(out["x2"])

        vad = torch.cat((v1, v2), dim=-1)
        vap = self.vap_classifier(out["x"])

        return vad, vap

class SoftDocLate(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.block_1 = SoftDocBlock(
            dim=cfg['multimodal_model_cfg']['audio']['d_model'],
            dim_src=cfg['multimodal_model_cfg']['audio']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['video']['n_layers_self'],
            num_heads=cfg['model_cfg']['n_heads'],
            activation=cfg['model_cfg']['ffn_activation'],
            dropout=cfg['model_cfg']['dropout_prob'],
            combine='add'
        )

        self.block_2 = SoftDocBlock(
            dim=cfg['multimodal_model_cfg']['video']['d_model'],
            dim_src=cfg['multimodal_model_cfg']['video']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['video']['n_layers_self'],
            num_heads=cfg['model_cfg']['n_heads'],
            activation=cfg['model_cfg']['ffn_activation'],
            dropout=cfg['model_cfg']['dropout_prob'],
            combine='add'
        )

        self.block_3 = SoftDocBlock(
            dim=cfg['multimodal_model_cfg']['audio']['d_model'],
            dim_src=cfg['multimodal_model_cfg']['video']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['video']['n_layers_self'],
            num_heads=cfg['model_cfg']['n_heads'],
            activation=cfg['model_cfg']['ffn_activation'],
            dropout=cfg['model_cfg']['dropout_prob'],
            combine=None
        )

        self.audio_encoder = StereoEncoder().to("cuda")
        self.video_encoder = VideoEncoder(cfg).to("cuda")

        self.vad_classifier = nn.Linear(in_features=cfg['multimodal_model_cfg']['audio']['d_model'], out_features=2)
        self.vap_classifier = nn.Linear(in_features=cfg['multimodal_model_cfg']['audio']['d_model'], out_features=cfg['model_cfg']['d_out'])

    def forward(self, batch):

        with torch.no_grad():
            audio = self.audio_encoder(batch['audio_chunk'].to("cuda"))
            x_l, x_r = audio[:,:,:,0].to("cuda"), audio[:,:,:,1].to("cuda")

            batch['frames']=torch.nan_to_num(batch['frames'])
            v_l, v_r = batch['frames'][:,:,:,0].to("cuda"), batch['frames'][:,:,:,1].to("cuda")
        
        v_l, v_r = self.video_encoder(v_l), self.video_encoder(v_r)

        x = self.block_1(x_l, x_r)
        x_a = x["x"]

        x = self.block_2(v_l, v_r)
        x_v = x["x"]

        out = self.block_3(x_a, x_v)

        vad = self.vad_classifier(out["x1"])
        # v2 = self.vad_classifier(out["x2"])

        # vad = torch.cat((v1, v2), dim=-1)
        vap = self.vap_classifier(out["x1"])

        return vad, vap

        
     

def test_gpt_stereo():
    tl = GPTStereo(
        dim=256,
        dim_src=20
    )

    x = torch.rand((5,100,256))
    src = torch.rand((5,60,20))

    tl.forward(x, src)
    
    x=1


def get_dataloader_sample():
    from torch.utils.data import DataLoader
    from dataset_management.dataset_manager.dataloader.dataloader import AudioVisualDataset

    pickle_file = "/home/russelsa@ad.mee.tcd.ie/github/dataset_management/dataset_manager/assets/candor_filtered_folds/candor/fold_0/train.pkl"
    # pickle_file = "/home/russelsa@ad.mee.tcd.ie/github/dataset_management/dataset_manager/assets/candor_filtered_folds/candor/fold_0/val.pkl"
    wavdir = "turn-taking-projects/corpora/candor/candor_wav"
    video_directory = "/data/ssd3/russelsa/candor_openface_pkl_brief"
    channelmaps = "turn-taking-projects/corpora/candor/channelmaps"

    av_ds = AudioVisualDataset(pickle_file=pickle_file, video_directory=video_directory, wavdir=wavdir, channelmaps=channelmaps, video_format='.pkl', mode='VAP')
    dl = DataLoader(av_ds, batch_size=2, shuffle=True, pin_memory=False)

    from tqdm import tqdm
    pbar = tqdm(total=len(dl))

    item = next(iter(dl))
    
    return item

if __name__ == "__main__":

    batch = get_dataloader_sample()

    config_file = "turn_taking/assets/config.yaml"
    cfg = yaml.safe_load(open(config_file))

    # model = LateVAFusion(cfg=cfg)
    # model = CrossAttnLateVAFusion(cfg=cfg)
    model = SoftDocLate(cfg=cfg)
    model.to(cfg['device'])
    ret = model.forward(batch)
    x=1