import torch
from torch import Tensor
from torch import nn 
from turn_taking.model.model import StereoTransformerModel
import yaml
import einops
from dataset_management.dataset_manager.dataloader.dataloader import ValidationAudioDataset, AudioDataset
from turn_taking.analysis.validation.probabilities import VAPDecoder
from turn_taking.training.callbacks import MaskVad
from turn_taking.audio_encoders.encoders import StereoEncoder
from dataset_management.dataset_manager.src.audio_manager import AudioManager
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def merge(start, list_of_tensors):

    list_of_tensors = torch.cat(list_of_tensors, dim=0)
    list_of_tensors = einops.rearrange(list_of_tensors, "b n ... -> (b n) ...")
    ret = torch.cat((start, list_of_tensors))
    
    return ret


@torch.no_grad
def run_model(model: nn.Module, dataloader: Dataset, mask_vad: bool, feature_extraction_hz: int = 50, window_size=10, step_size=9, mode='VAP'):
    """ run the model on an entire audio file"""

    if mask_vad:
        mask_vad_batch = MaskVad(probability=1, feature_hz=50, audio_hz=16_000, scale=0.0)

    # discard the overlapping segment
    overlap = int(feature_extraction_hz * (window_size-step_size))

    for i, batch in enumerate(dataloader):


        if mask_vad:
            batch = mask_vad_batch(batch)

        with torch.no_grad():
            vad_pred, vap_pred = model(batch)
        ret = {"vad": vad_pred, "vap": vap_pred}

        if mode == 'VAP':
            ret['vap'] = torch.softmax(ret['vap'], dim=-1).to(device='cpu')
        else:
            ret['vap'] = torch.sigmoid(ret['vap'])

        ret['vad'] = torch.sigmoid(ret['vad'].to(device='cpu'))
        
        vad_pred = ret['vad'][:, overlap:, :]
        vap_pred = ret['vap'][:, overlap:, ...]

        # do not cut the start of the file
        if i == 0:
            vads = [ret['vad'][0, :overlap, :]]
            vaps = [ret['vap'][0, :overlap, ...]]

        vads.append(vad_pred)
        vaps.append(vap_pred)

    # merge back together
    vads = merge(vads[0], vads[1:])
    vaps = merge(vaps[0], vaps[1:])
    
    return vaps, vads

@torch.no_grad
def run_model_unbatched(model: nn.Module, dataset: Dataset, mask_vad: bool, feature_extraction_hz: int = 50, window_size=10, step_size=9, mode='VAP'):
    """ run the model on an entire audio file"""

    if mask_vad:
        mask_vad_batch = MaskVad(probability=1, feature_hz=50, audio_hz=16_000, scale=0.0)

    # discard the overlapping segment
    overlap = int(feature_extraction_hz * (window_size-step_size))

    for idx in range(len(dataset)):

        batch = dataset.__getitem__(idx)
    
        if mask_vad:
            batch = mask_vad_batch(batch)

        with torch.no_grad():
            batch['audio_chunk'] = batch['audio_chunk'].unsqueeze(0)
            batch['frames'] = batch['frames'].unsqueeze(0)
            vad_pred, vap_pred = model(batch)
        ret = {"vad": vad_pred, "vap": vap_pred}

        if mode == 'VAP':
            ret['vap'] = torch.softmax(ret['vap'], dim=-1).to(device='cpu')
        else:
            ret['vap'] = torch.sigmoid(ret['vap'])

        ret['vad'] = torch.sigmoid(ret['vad'].to(device='cpu'))
        
        vad_pred = ret['vad'][:, overlap:, :]
        vap_pred = ret['vap'][:, overlap:, ...]

        # do not cut the start of the file
        if idx == 0:
            vads = [ret['vad'][0, :overlap, :]]
            vaps = [ret['vap'][0, :overlap, ...]]

        vads.append(vad_pred)
        vaps.append(vap_pred)

    # merge back together
    vads = merge(vads[0], vads[1:])
    vaps = merge(vaps[0], vaps[1:])
    
    return vaps, vads



def plot_model(vap_pred, gt_vad, gt_vap, vad_pred, audio, start_time, stop_time, vertical_lines, output_name, sr, sr_features):

    start_y, stop_y = start_time*sr_features, stop_time*sr_features 
    start_a, stop_a = start_time*sr, stop_time*sr
    
    fig, axs = plt.subplots(6, 1, sharex=True) 
    plt.suptitle(output_name)

    colours = ['r', 'g']
    for channel in [0,1]:
        
        ax = axs[channel].twinx()
        ax.set_ylim([-1,+1])
        x = np.linspace(start_time, stop_time, stop_a-start_a)
        ax.plot(x, audio[start_a:stop_a, channel], colours[channel])

        ax = axs[channel]
        x = np.linspace(start_time, stop_time, stop_y-start_y)
        ax.plot(x, vad_pred[start_y:stop_y, channel], "k--")
        ax.plot(x, gt_vad[start_y:stop_y, channel]*0.8, "k--")
    
    for bin in [0,1,2,3]: #,
        ax = axs[bin + 2]
        ax.set_ylim([-.1,1.1])
        if vertical_lines:
            for vertical_line in vertical_lines:
                ax.vlines(x=vertical_line, ymin=-.1, ymax=1.1)
        x = np.linspace(start_time, stop_time, stop_y - start_y)
        
        ax.plot(x, vap_pred[start_y:stop_y, 0, bin], "r--")
        # ax.plot(x, 1-vap_pred[start_y:stop_y, 0, bin], "r-")

        ax.plot(x, vap_pred[start_y:stop_y, 1, bin], "g--")
        # ax.plot(x, 1-vap_pred[start_y:stop_y, 1, bin], "g-")

        ax.plot(x, [0.5]*len(x), '--')
        # ax.plot(x, gt_vap[start_y:stop_y, 0, bin], "k")

    plt.savefig("test.png")


def run_file():

    window_size = 20
    step_size = 19
    
    state_dict = "runs_VAP_switchboard_lr_sweep_phonwords_with_crosstalk/20240712_164319/0.0001_32_epoch_2"
    checkpoint = torch.load(state_dict, map_location ='cpu')

    cfg = yaml.safe_load(open("turn_taking/assets/config.yaml", "r"))
    model = StereoTransformerModel(cfg=cfg)
    model.load_state_dict(state_dict=checkpoint)
    model = model.to("cuda")
    model.eval()

    audio_file="turn-taking-projects/corpora/switchboard/switchboard_wav/sw3222.wav"
    transcript_file="turn-taking-projects/corpora/switchboard/textgrids_phonwords/sw3222.TextGrid"

    audio_dataset = ValidationAudioDataset(audio_file=audio_file, transcript_file=transcript_file, sr=16_000, feature_sr=50, window_size=window_size, step_size=step_size, mode='VAP')
    audio_dl = DataLoader(audio_dataset, batch_size=10, shuffle=False)

    vaps, vads = run_model(model, audio_dl, mask_vad=False, feature_extraction_hz=50, window_size=window_size, step_size=step_size, mode='VAP')

    gt_vap = audio_dataset.vap_unbatched
    gt_vad = audio_dataset.vad_unbatched
    audio = audio_dataset.audio_unbathed

    decoder = VAPDecoder(bin_times=[0.2, 0.4, 0.6, 0.8])
    
    vaps = vaps.unsqueeze(dim=0)
    vaps = decoder.p_all(vaps)
    vaps = vaps.squeeze()

    plot_model(
        vap_pred=vaps,
        vad_pred=vads,
        gt_vad=gt_vad,
        gt_vap=gt_vap,
        audio=audio,
        start_time=40,
        stop_time=50,
        vertical_lines=[45.4],
        # vertical_lines=None,
        output_name="test.png",
        sr=16_000,
        sr_features=50
        )


def run_batch():

    mask_vad = MaskVad(probability=1, feature_hz=50, audio_hz=16_000, scale=0.01)

    prefix="20240709_151434_fold_0_epoch_1"
    state_dict =f"/home/russelsa@ad.mee.tcd.ie/github/runs_VAP_switchboard/20240709_151434_fold_0_epoch_1"
    checkpoint = torch.load(state_dict)

    cfg = yaml.safe_load(open("turn_taking/assets/config.yaml", "r"))
    torch.cuda.set_device("cuda")

    model = StereoTransformerModel(cfg=cfg)
    model.load_state_dict(state_dict=checkpoint)
    model = model.to("cuda")
    model.eval()

    wavdir="turn-taking-projects/corpora/switchboard/switchboard_wav"
    val_pickle_file="dataset_management/dataset_manager/assets/folds/switchboard/fold_0/val.pkl"
    
    val_audio_dataset = AudioDataset(val_pickle_file, mode='ind-4', wavdir=wavdir)
    val_dataloader = DataLoader(val_audio_dataset, batch_size=1, shuffle=True, pin_memory=True)

    for i, batch in enumerate(val_dataloader):

        batch = mask_vad(batch)
        
        with torch.no_grad():
            vad_pred, vap_pred = model(batch)
            ret = {
                "vad": vad_pred,
                "vap": vap_pred
            }
            # ret = model(batch)

        batch['vap'] = einops.rearrange(batch['vap'], "b n (c d) -> b n d c", d=4, c=2)
        ret['vad'] = torch.sigmoid(ret['vad'].to(device='cpu'))

        if cfg['model_cfg']['mode']=='VAP':
            ret['vap'] = torch.softmax(ret['vap'], dim=-1).to(device='cpu')
            ret['vap'] = decode_probabilities(ret['vap'])

        else:
            ret['vap'] = torch.sigmoid(ret['vap']).to(device='cpu')

        batch_no=0
        fig, axs = plt.subplots(6, 1, sharex=True)
        plt.suptitle(batch['id'])

        speaker_colours=['r','g']
        for channel in [0,1]:
            
            ax = axs[channel]
            ax.set_ylim((0,1))
            x = np.linspace(batch['start_time'], batch['end_time'], num=batch['vad'].shape[1])
            ax.plot(x, batch['vad'][batch_no, :, channel], f"{speaker_colours[channel]}")
            ax.plot(x, ret['vad'][batch_no, :, channel], f"{speaker_colours[channel]}")

            ax = axs[channel].twinx()
            ax.set_ylim((-1,1))
            x = np.linspace(batch['start_time'], batch['end_time'], batch['audio_chunk'].shape[1])
            ax.plot(x, batch['audio_chunk'][batch_no, :, channel])
        
        for bin in [0,1,2,3]:
            ax = axs[bin + 2].twinx()
            ax.set_ylim([0,1])
            x = np.linspace(batch['start_time'], batch['end_time'], num=batch['vap'].shape[1])

            for speaker in [0]:
                ax.plot(x, ret['vap'][batch_no, :, speaker, bin], f"{speaker_colours[speaker]}")
                ax.plot(x, batch['vap'][batch_no, :, bin, speaker], "k")

            ax.plot(x, [0.5]*len(x), '--')

        plt.savefig(f"{prefix}_{i}.png")
        x=1


if __name__=="__main__":
    # torch.cuda.set_device(1)
    # run_batch()
    import os 
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    run_file()