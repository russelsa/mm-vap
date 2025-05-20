import argparse
import turn_taking
from turn_taking.model.multimodal_model import EarlyVAFusion, LateVAFusion
from turn_taking.model.model import StereoTransformerModel, StereoTransformerModelVideoOnly
from  dataset_management.dataset_manager.dataloader.dataloader import ValidationAudioVisualDataset
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from turn_taking.analysis.validation.probabilities import VAPDecoder
from turn_taking.analysis.validation.run_model import run_model_unbatched
import os
import pickle
import yaml
import torch
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import numpy as np


# def plot_model(pred, vad_pred, audio, start_time, stop_time, sr=16_000, sr_features=50, output_name="test.png"):

if __name__ == "__main__":

    audio = "/home/russelsa@ad.mee.tcd.ie/github/Multimodal_Turn_Taking/sample_data/1f7e582c-c6bc-46b6-b5a4-e5d78e8a46ac.wav"
    start_time=959
    stop_time=968 

    start_pause=963.2
    stop_pause=963.83

    decoder = VAPDecoder(bin_times=[0.2, 0.4, 0.6, 0.8])

    sr_features=50

    sr, audio_file = wavfile.read(audio)
    audio_file = audio_file.astype('float')
    audio_file = audio_file/audio_file.max()

    start_y, stop_y = start_time*sr_features, stop_time*sr_features 
    start_a, stop_a = start_time*sr, stop_time*sr
    
    fig, axs = plt.subplots(4, 1, sharex=True) 


    colours = ['red', 'gray']
    for channel in [0,1]:
        
        ax = axs[0]
        ax.set_ylim([-1,+1])
        x = np.linspace(start_time, stop_time, stop_a-start_a)
        a = audio_file[start_a:stop_a, channel]
        ax.plot(x, a, colours[channel], alpha=0.5)
        ax.get_yaxis().set_ticks([])

        ax.plot([start_pause]*5, np.linspace(-1,1,5), 'k')
        ax.plot([stop_pause]*5, np.linspace(-1,1,5), 'k')

    # ----------------- clean-trained vap deployed on clean speech ----------------------------------
    bin=1
    ax = axs[bin]
    ax.set_ylim([-.1,1.1])

    channel=bin-2

    x = np.linspace(start_time, stop_time, stop_y - start_y)
    ax.plot(x, [0.5]*len(x), 'k', alpha=0.2)
    
    # vap 
    pred = pickle.load(open("outputs/clean/clean_VAP/20240717_142924_fold_0_epoch_1_vap.pkl", "rb"))
    pred = decoder.p_future(pred)
    p = pred[start_y:stop_y, 0]
    ax.plot(x, p, color="#43a047", linestyle="dotted")
    ax.fill_between(x, 0.5, p, where=p>0.5, interpolate=True, facecolor=colours[0], alpha=0.2)    
    # ax.fill_between(x, 0.5, p, where=p<0.5, interpolate=True, facecolor=colours[1], alpha=0.2)    

    # mmvap
    pred = pickle.load(open("outputs/clean/clean_MMVAP/20240821_160422_fold_0_epoch_10_vap.pkl", "rb"))
    pred = decoder.p_future(pred)
    p = pred[start_y:stop_y, 0]
    ax.plot(x, p, color="#43a047", linestyle="-")
    ax.fill_between(x, 0.5, p, where=p>0.5, interpolate=True, facecolor=colours[0], alpha=0.2)    
    # ax.fill_between(x, 0.5, p, where=p<0.5, interpolate=True, facecolor=colours[1], alpha=0.2)    

    # ax.plot([start_pause]*5, np.linspace(0,1,5), 'k')
    # ax.plot([stop_pause]*5, np.linspace(0,1,5), 'k')

    ax.get_yaxis().set_ticks([0,0.5,1])


    # ----------------- clean-trained vap deployed on 10dB speech interference ---------------------
    bin=2
    ax = axs[bin]
    ax.set_ylim([-.1,1.1])

    channel=bin-2

    x = np.linspace(start_time, stop_time, stop_y - start_y)
    ax.plot(x, [0.5]*len(x), 'k', alpha=0.2)
    
    # vap 
    pred = pickle.load(open("outputs/babble/clean_VAP_babble/20240717_142924_fold_0_epoch_7_vap.pkl", "rb"))
    pred = decoder.p_future(pred)
    p = pred[start_y:stop_y, 0]
    ax.plot(x, p, color="#43a047", linestyle="dotted")
    ax.fill_between(x, 0.5, p, where=p>0.5, interpolate=True, facecolor=colours[0], alpha=0.2)    
    # ax.fill_between(x, 0.5, p, where=p<0.5, interpolate=True, facecolor=colours[1], alpha=0.2)    

    # mmvap
    pred = pickle.load(open("outputs/babble/clean_MMVAP_babble/20240821_160422_fold_0_epoch_10_vap.pkl", "rb"))
    pred = decoder.p_future(pred)
    p = pred[start_y:stop_y, 0]
    ax.plot(x, p, color="#43a047", linestyle="-")
    ax.fill_between(x, 0.5, p, where=p>0.5, interpolate=True, facecolor=colours[0], alpha=0.2)    
    # ax.fill_between(x, 0.5, p, where=p<0.5, interpolate=True, facecolor=colours[1], alpha=0.2)    

    ax.get_yaxis().set_ticks([0,0.5,1])
    
    # ax.plot([start_pause]*5, np.linspace(0,1,5), 'k')
    # ax.plot([stop_pause]*5, np.linspace(0,1,5), 'k')

    # ----------------- augmented-trained vap deployed on 10dB speech interference ---------------------
    bin=3
    ax = axs[bin]
    ax.set_ylim([-.1,1.1])

    channel=bin-2

    x = np.linspace(start_time, stop_time, stop_y - start_y)
    ax.plot(x, [0.5]*len(x), 'k', alpha=0.2)
    
    # vap 
    pred = pickle.load(open("outputs/babble/augmented_VAP_babble/20250113_124943_fold_0_epoch_10_vap.pkl", "rb"))
    pred = decoder.p_future(pred)
    p = pred[start_y:stop_y, 0]
    ax.plot(x, p, color="#1e88e5", linestyle="dotted")
    ax.fill_between(x, 0.5, p, where=p>0.5, interpolate=True, facecolor=colours[0], alpha=0.2)    
    # ax.fill_between(x, 0.5, p, where=p<0.5, interpolate=True, facecolor=colours[1], alpha=0.2)    

    # mmvap
    pred = pickle.load(open("outputs/babble/augmentred_MMVAP_babbble/20250113_125258_fold_0_epoch_10_vap.pkl", "rb"))
    pred = decoder.p_future(pred)
    p = pred[start_y:stop_y, 0]
    ax.plot(x, p, color="#1e88e5", linestyle="-")
    ax.fill_between(x, 0.5, p, where=p>0.5, interpolate=True, facecolor=colours[0], alpha=0.2)    
    # ax.fill_between(x, 0.5, p, where=p<0.5, interpolate=True, facecolor=colours[1], alpha=0.2)    

    ax.get_yaxis().set_ticks([0,0.5,1])

    for label, ax in zip(["A", "B", "C", "D"], axs):
        ax.annotate(
            label,
            xy=(0, 1), xycoords='axes fraction',
            xytext=(+0.5, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top',
            bbox=dict(facecolor='0.7', edgecolor='k', pad=3.0))

    methods = [
        mpatches.Patch(facecolor='#ffffff', label="$\mathbf{speaker}$"),
        mpatches.Patch(facecolor=colours[1], label='speaker 0'),
        mpatches.Patch(facecolor=colours[0], label='speaber 1'),
        mpatches.Patch(facecolor='#ffffff', label="$\mathbf{audio + align.}$"),
        Line2D([0], [0], color="#43a047", label='clean + clean'),
        Line2D([0], [0], color="#1e88e5", label='aug. + clean'),
        mpatches.Patch(facecolor='#ffffff', label=r"$\mathbf{model}$"+r" $\mathbf{(modality)}$"),
        Line2D([0], [0], color='k', label='VAP (A)', linestyle='dotted'),
        Line2D([0], [0], color='k', label='MM-VAP (A + V)', linestyle='-'),
    ]
    
    ax.legend(handles=methods, ncol=3, prop={'size': 10},loc='upper center', bbox_to_anchor=(0.5, -0.6)) # loc='upper center', bbox_to_anchor=(0.5, -0.15),

    
    plt.suptitle("Shift Prediction in VAP and MM-VAP", fontsize=16, y=0.92)
    ax=axs[0]
    # ax.yaxis.set_label_position("right")
    # ax.set_ylabel(, fontsize=12)

    ax=axs[1]
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("clean", fontsize=14)

    # ax=axs[2]
    # ax.yaxis.set_label_position("right")
    # ax.set_ylabel("0 dB babble", fontsize=14)

    # ax=axs[3]
    # ax.yaxis.set_label_position("right")
    # ax.set_ylabel("babble", fontsize=14)

    fig.text(0.04, 0.4, 'Speaker 1 Probability', fontsize=14, va='center', rotation='vertical')

    fig.text(0.910, 0.282, '0dB babble', fontsize=14, va='center', rotation='vertical')

    
    plt.xlabel("Time [seconds]", fontsize=14)
    output_name=f"shift_vap_mmvap.pdf"
    plt.savefig(output_name, bbox_inches='tight')

