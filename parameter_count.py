import argparse
import turn_taking
from turn_taking.model.multimodal_model import EarlyVAFusion, LateVAFusion
from turn_taking.model.model import StereoTransformerModel, StereoTransformerModelVideoOnly
from  dataset_management.dataset_manager.dataloader.dataloader import ValidationAudioVisualDataset
from turn_taking.analysis.validation.probabilities import VAPDecoder
from turn_taking.analysis.validation.run_model import run_model_unbatched
import os
import pickle
import yaml
import torch
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import numpy as np


def plot_model(pred, vad_pred, audio, start_time, stop_time, sr=16_000, sr_features=50, output_name="test.png"):

    start_y, stop_y = start_time*sr_features, stop_time*sr_features 
    start_a, stop_a = start_time*sr, stop_time*sr
    
    fig, axs = plt.subplots(4, 1, sharex=True) 
    plt.suptitle(os.path.basename(output_name).split('.')[0])

    colours = ['r', 'g']
    for channel in [0,1]:
        
        ax = axs[channel].twinx()
        ax.set_ylim([-1,+1])
        x = np.linspace(start_time, stop_time, stop_a-start_a)
        a = audio[start_a:stop_a, channel]
        ax.plot(x, a, colours[channel])

        ax = axs[channel]
        ax.set_ylim([-1,+1])
        x = np.linspace(start_time, stop_time, stop_y-start_y)
        ax.plot(x, vad_pred[start_y:stop_y, channel], "k--")
        ax.set_ylabel(f'VAD {channel}')
    
    for bin in [2,3]: #,
        ax = axs[bin]
        ax.set_ylim([-.1,1.1])

        channel=bin-2

        x = np.linspace(start_time, stop_time, stop_y - start_y)
        
        ax.plot(x, pred[start_y:stop_y, channel], f"{colours[channel]}--")
        ax.plot(x, [0.5]*len(x), '--')

        ax.set_ylabel(f'VAP {channel}')

    plt.xlabel("Time [seconds]")
    plt.savefig(output_name)


def param_count(cfg, model, model_weights, audio, video, transcript):


    model.load_state_dict(state_dict=torch.load(model_weights))
    model = model.to("cuda")
    model.eval()

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, help="the turn-taking model, e.g. early_fusion_candor")
    # parser.add_argument("--start_time", type=str, help="start of probability trace in seconds")
    # parser.add_argument("--end_time", type=str, help="end of probability trace in seconds")

    args = parser.parse_args()

    models = ['early_fusion_candor', 'candor_video_only', 'VAP_candor', 'VAP_switchboard_ASR', 'VAP_switchboard_ground_truth']
    assert args.model in models, f"use a valid model from {models}"
    
    if args.model == "early_fusion_candor":
        cfg = "sample_trained_models/early_fusion_candor/config.yaml"
        model_weights = "sample_trained_models/early_fusion_candor/weights"
        cfg = yaml.safe_load(open(cfg, "r"))
        model = EarlyVAFusion(cfg=cfg)

    elif args.model == "VAP_candor":
        cfg = "sample_trained_models/VAP_candor/20240822_105427_params.yaml"
        model_weights = "sample_trained_models/VAP_candor/20240822_105427_fold_0_epoch_10"
        cfg = yaml.safe_load(open(cfg, "r"))
        model = StereoTransformerModel(cfg=cfg)

    elif args.model == "VAP_switchboard_ASR":
        cfg = "sample_trained_models/VAP_switchboard_ASR/20240716_144644_params.yaml"
        model_weights = "sample_trained_models/VAP_switchboard_ASR/20240716_144644_fold_0_epoch_10"
        cfg = yaml.safe_load(open(cfg, "r"))
        model = StereoTransformerModel(cfg=cfg)

    elif args.model == "VAP_switchboard_ground_truth":
        cfg = "sample_trained_models/VAP_switchboard_ground_truth/20240716_144749_params.yaml"
        model_weights = "sample_trained_models/VAP_switchboard_ground_truth/20240716_144749_fold_0_epoch_10"
        cfg = yaml.safe_load(open(cfg, "r"))
        model = StereoTransformerModel(cfg=cfg)

    elif args.model == "candor_video_only":
        cfg = "sample_trained_models/video_only_candor/20241023_164016_params.yaml"
        model_weights = "sample_trained_models/video_only_candor/20241023_164016_fold_0_epoch_10"
        cfg = yaml.safe_load(open(cfg, "r"))
        model = StereoTransformerModelVideoOnly(cfg=cfg)

    audio = "sample_data/1f7e582c-c6bc-46b6-b5a4-e5d78e8a46ac.wav"
    video = "sample_data"
    transcript = "sample_data/1f7e582c-c6bc-46b6-b5a4-e5d78e8a46ac.TextGrid"

    param_count(cfg, model, model_weights, audio, video, transcript)
