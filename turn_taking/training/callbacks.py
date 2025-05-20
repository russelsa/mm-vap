import numpy as np
import torch
import einops
import torchaudio.functional as AF


@torch.no_grad()
def mask_around_vad(
    waveform: torch.Tensor,
    vad: torch.Tensor,
    vad_hz: int,
    sample_rate,
    scale: float = 0.1,
) -> torch.Tensor:
    # assert (
    #     vad.shape[-1] == 2
    # ), f"Expects vad of shape (B, N_FRAMES, 2) but got {vad.shape}"

    if vad.shape[-1]!=2:
        vad = einops.rearrange(vad, "b c n -> b n c")
    non_vad_mask = vad.permute(0, 2, 1).logical_not().float()  # -> B, 2, N_frames
    waveform = einops.rearrange(waveform, "b s c -> b c s")
    B, C, _ = waveform.shape
    if C > 1:
        w_tmp = einops.rearrange(waveform, "b c s -> (b c) s")
        non_vad_mask = einops.rearrange(non_vad_mask, "b c f -> (b c) f")
        if vad_hz != sample_rate:
            non_vad_mask = AF.resample(
                non_vad_mask, orig_freq=50, new_freq=16_000
            )
            non_vad_mask = non_vad_mask > 0.5
        non_vad_mask = non_vad_mask[..., : waveform.shape[-1]]
        # z_mask *= scale
        w_tmp[non_vad_mask] *= scale
        # w_tmp = w_tmp * v_mask[:, : w_tmp.shape[-1]]
        waveform = einops.rearrange(w_tmp, "(b c) s -> b c s", b=B, c=C)
    else:
        if vad_hz != sample_rate:
            non_vad_mask = AF.resample(
                non_vad_mask, orig_freq=vad_hz, new_freq=sample_rate
            )
        if C == 1:
            non_vad_mask = non_vad_mask.sum(-2).unsqueeze(1)

        non_vad_mask = non_vad_mask > 0.5
        non_vad_mask = non_vad_mask[..., : waveform.shape[-1]]
        # z_mask *= scale
        waveform[non_vad_mask] *= scale
        # waveform = waveform * non_vad_mask[:, :, : waveform.shape[-1]]
    waveform = einops.rearrange(waveform, "b c s -> b s c")
    return waveform



class FlipChannel():
    """
    Randomly "flips" the speakers such that we get a fair evaluation not dependent on the
    biased speaker-order / speaker-activity
    """

    def __init__(
        self,
        probability: float = 0.5
    ):
        self.probability = probability

    @torch.no_grad()
    def get_flipped_batch(self, batch):
        for k, v in batch.items():
            
            if k == "vad":
                flipped = torch.stack((v[..., 1], v[..., 0]), dim=-1)
            elif k == "audio_chunk":
                flipped = torch.stack((v[..., 1], v[..., 0]), dim=-1)
            elif k == "frames":
                flipped = torch.stack((v[..., 1], v[..., 0]), dim=-1)
            elif k== "vap":
                flipped = batch['inverse_vap']
            else:
                flipped = v

            batch[k] = flipped
        return batch

    @torch.no_grad()
    def __call__(self, x):
        r = np.random.uniform()

        if r > self.probability:
            return x

        return self.get_flipped_batch(x)
    
class MaskVad():
    def __init__(
        self,
        probability: float,
        feature_hz: int,
        audio_hz: int,
        scale: float = 0.1
    ):
        self.probability = probability
        self.feature_hz = feature_hz
        self.audio_hz = audio_hz
        self.scale = scale

    @torch.no_grad()
    def mask_vad(self, batch):
        batch['audio_chunk'] = mask_around_vad(batch['audio_chunk'], batch['vad'], self.audio_hz, self.feature_hz, self.scale)
        return batch

    @torch.no_grad()
    def __call__(self, x):
        r = np.random.uniform()

        if r > self.probability:
            return x

        return self.mask_vad(x)
    

@torch.no_grad()
def video_feature_subset(batch, feature_subset):

    if 'all' in batch:
        return batch

    video_features = batch['frames']
    output_features = []

    if 'gaze' in feature_subset:
        output_features.append(video_features[:, :, 0:6, :])
    if 'pose' in feature_subset:
        output_features.append(video_features[:, :, 6:12, :])
    if 'lmks' in feature_subset:
        output_features.append(video_features[:, :, 12:42, :])
    if 'faus' in feature_subset:
        output_features.append(video_features[:, :, 42:, :])
    
    output_features = torch.concat(output_features, dim=-2)

    batch['frames'] = output_features
    return batch



    # # range selection
    # gaze = cols[0:6]
    # pose = cols[6:12]
    # lmks = cols[12:42]
    # faus = cols[42:59]

    # # always include the confidence? what is this 
    # # confidence how confident is the tracker in current landmark detection estimage
    # confidence = cols[60]
