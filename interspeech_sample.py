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


def run_sample(cfg, model, model_weights, audio, video, transcript, start_time, end_time, output_name):


    model.load_state_dict(state_dict=torch.load(model_weights))
    model = model.to("cuda")
    model.eval()

    channelmap = "turn-taking-projects/corpora/candor/channelmaps.pkl"
    window_size=20
    step_size=19
    mode="VAP"

    if video is not None:
        dataset = ValidationAudioVisualDataset(video_pkl_dir=video, channelmap=channelmap, audio_file=audio, transcript_file=transcript, sr=16_000, feature_sr=50, window_size=window_size, step_size=step_size, mode=mode)

    vaps, vads = run_model_unbatched(model=model, dataset=dataset, mask_vad=False, feature_extraction_hz=50, window_size=window_size, step_size=step_size, mode=mode)

    pickle.dump(vads, open("sample_data/vads.pkl", "wb"))
    pickle.dump(vaps, open("sample_data/vaps.pkl", "wb"))

    vads = pickle.load(open("sample_data/vads.pkl", "rb"))
    vaps = pickle.load(open("sample_data/vaps.pkl", "rb"))

    decoder = VAPDecoder(bin_times=[0.2, 0.4, 0.6, 0.8])
    p_future = decoder.p_future(vaps)

    sr, audio_file = wavfile.read(audio)
    audio_file = audio_file.astype('float')
    audio_file = audio_file/audio_file.max()

    plot_model(pred=p_future, vad_pred=vads, audio=audio_file, start_time=start_time, stop_time=end_time, output_name=output_name)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, help="the turn-taking model, e.g. clean_audio_clean_alignment/audio_only")
    parser.add_argument("--audio", type=str, help="the audio e.g. babble")
    parser.add_argument("--start_time", type=str, help="start of probability trace in seconds")
    parser.add_argument("--end_time", type=str, help="end of probability trace in seconds")

    args = parser.parse_args()

    models = ['clean_audio_clean_alignment/audio_only',
              'clean_audio_clean_alignment/audio_and_video',
              'augmented_audio_clean_alignment/audio_only',
              'augmented_audio_clean_alignment/audio_and_video',
              'augmented_audio_augmented_alignment/audio_only',
              'augmented_audio_augmented_alignment/audio_and_video',
              'video_only'
              ]
    
    assert args.model in models, f"use a valid model from {models}"

    audios = [
        "clean",
        "babble",
        "music",
        "speech"
    ]

    assert args.audio in audios, f"use a valid audio from {audios}"
    
    if args.model == "clean_audio_clean_alignment/audio_only":
        cfg = "sample_trained_models/clean_audio_clean_alignment/audio/20240822_105427_params.yaml"
        model_weights = "sample_trained_models/clean_audio_clean_alignment/audio/20240822_105427_fold_0_epoch_10"
        cfg = yaml.safe_load(open(cfg, "r"))
        model = StereoTransformerModel(cfg=cfg)

    elif args.model == "clean_audio_clean_alignment/audio_and_video":
        cfg = "sample_trained_models/clean_audio_clean_alignment/audio_and_video/config.yaml"
        model_weights = "sample_trained_models/clean_audio_clean_alignment/audio_and_video/weights"
        cfg = yaml.safe_load(open(cfg, "r"))
        model = EarlyVAFusion(cfg=cfg)

    if args.model == "augmented_audio_clean_alignment/audio_only":
        cfg = "sample_trained_models/augmented_audio_clean_alignment/audio/20250113_124943_params.yaml"
        model_weights = "sample_trained_models/augmented_audio_clean_alignment/audio/20250113_124943_fold_0_epoch_10"
        cfg = yaml.safe_load(open(cfg, "r"))
        model = StereoTransformerModel(cfg=cfg)

    elif args.model == "augmented_audio_clean_alignment/audio_and_video":
        cfg = "sample_trained_models/augmented_audio_clean_alignment/audio_and_video/20250113_125258_params.yaml"
        model_weights = "sample_trained_models/augmented_audio_clean_alignment/audio_and_video/20250113_125258_fold_0_epoch_10"
        cfg = yaml.safe_load(open(cfg, "r"))
        model = LateVAFusion(cfg=cfg)

    if args.model == "augmented_audio_augmented_alignment/audio_only":
        cfg = "sample_trained_models/augmented_audio_augmented_alignment/audio/20250120_231454_params.yaml"
        model_weights = "sample_trained_models/augmented_audio_augmented_alignment/audio/20250120_231454_fold_0_epoch_10"
        cfg = yaml.safe_load(open(cfg, "r"))
        model = StereoTransformerModel(cfg=cfg)

    elif args.model == "augmented_audio_augmented_alignment/audio_and_video":
        cfg = "sample_trained_models/augmented_audio_augmented_alignment/audio_and_video/20250120_231330_params.yaml"
        model_weights = "sample_trained_models/augmented_audio_augmented_alignment/audio_and_video/20250120_231330_fold_0_epoch_10"
        cfg = yaml.safe_load(open(cfg, "r"))
        model = LateVAFusion(cfg=cfg)

    elif args.model == "video_only":
        cfg = "sample_trained_models/video_only/20241023_164016_params.yaml"
        model_weights = "/home/russelsa@ad.mee.tcd.ie/github/Interspeech25_code/sample_trained_models/video_only/20241023_164016_fold_0_epoch_10"
        cfg = yaml.safe_load(open(cfg, "r"))
        model = StereoTransformerModelVideoOnly(cfg=cfg)

    audio = f"sample_data/{args.audio}/1f7e582c-c6bc-46b6-b5a4-e5d78e8a46ac.wav"
    video = "sample_data"
    transcript = "sample_data/1f7e582c-c6bc-46b6-b5a4-e5d78e8a46ac.TextGrid"

    output_name = 'images/' + args.model.replace('/', '_') + f'_{args.audio}_{args.start_time}_{args.end_time}' + '.png'

    run_sample(cfg, model, model_weights, audio, video, transcript, int(args.start_time), int(args.end_time), output_name)
