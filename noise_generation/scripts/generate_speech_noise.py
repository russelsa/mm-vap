import textgrid
import pandas as pd
from copy import copy
import numpy as np
import glob
import random
import os 
import json
import torch
import torchaudio
import torchaudio.functional as F
random.seed(0)
from dataset_management.dataset_manager.src.audio_manager import AudioManager


def candor_split_utterances():
    """
        split the utterances of the candor corpus into individual noise files 
    """

    ipu_files = "/home/russelsa@ad.mee.tcd.ie/github/turn-taking-projects/corpora/candor/candor_ipu"
    channelmaps = "/home/russelsa@ad.mee.tcd.ie/github/turn-taking-projects/corpora/candor/channelmaps.pkl"
    wavfiles = "/home/russelsa@ad.mee.tcd.ie/github/turn-taking-projects/corpora/candor/candor_wav"

    output_directory = "/data/ssd4/russelsa/candor_utts/fold_0/train"
    ids = "/home/russelsa@ad.mee.tcd.ie/github/dataset_management/dataset_manager/assets/new_folds/candor/fold_0/train.csv"

    ids = pd.read_csv(ids)
    ids.columns = [c.strip() for c in ids.columns]
    ids = ids['id'].to_list()

    channelmaps = pd.read_pickle(channelmaps)

    for id in ids:

        textgrid_file = textgrid.TextGrid.fromFile(os.path.join(ipu_files, id+'.TextGrid'))
        wav, audio_sr  = AudioManager.load_waveform(os.path.join(wavfiles, id+'.wav'), sample_rate=16_000, mono=False, normalize=True, channel_first=True)

        for i, intervaltier in enumerate(textgrid_file):

            for j, interval in enumerate(intervaltier):
                if interval.mark != '':
                    start, end = interval.minTime, interval.maxTime
                    if end-start < 10:
                        continue
                    start, end = int(audio_sr*start), int(audio_sr*end)
                    utterance = wav[i, start:end].unsqueeze(dim=0)

                    output_file = os.path.join(output_directory, intervaltier.name + '--' + str(j) + '.wav')
                    torchaudio.save(output_file, utterance, audio_sr)


def candor_remove_silences():
    """
        remove the silences from the candor corpus, and create one file per speaker per session
        containing all utterances with no silences between utterances
    """

    ipu_files = "/home/russelsa@ad.mee.tcd.ie/github/turn-taking-projects/corpora/candor/candor_ipu"
    channelmaps = "/home/russelsa@ad.mee.tcd.ie/github/turn-taking-projects/corpora/candor/channelmaps.pkl"
    wavfiles = "/home/russelsa@ad.mee.tcd.ie/github/turn-taking-projects/corpora/candor/candor_wav"

    output_directory = "/data/ssd4/russelsa/candor_utts/silences_removed"
    ids = [o.split('.')[0] for o in os.listdir(wavfiles)]

    channelmaps = pd.read_pickle(channelmaps)

    for id in ids:

        textgrid_file = textgrid.TextGrid.fromFile(os.path.join(ipu_files, id+'.TextGrid'))
        wav, audio_sr  = AudioManager.load_waveform(os.path.join(wavfiles, id+'.wav'), sample_rate=16_000, mono=False, normalize=True, channel_first=True)

        # ensure both have already started speaking
        starts = []
        for i, intervaltier in enumerate(textgrid_file):
            start = intervaltier[0]
            starts.append(start.minTime)
        MIN_START = max(starts)

        for i, intervaltier in enumerate(textgrid_file):

            new_wav = torch.tensor([])
            output_file = os.path.join(output_directory, intervaltier.name + '.wav')

            for j, interval in enumerate(intervaltier):
                if interval.mark != '':

                    if start<MIN_START:
                        continue

                    start, end = interval.minTime, interval.maxTime
                    start, end = int(audio_sr*start), int(audio_sr*end)
                    utterance = wav[i, start:end].unsqueeze(dim=0)        

                    new_wav = torch.concatenate((new_wav, utterance), dim=-1)
            
            torchaudio.save(output_file, new_wav, audio_sr)

if __name__ == "__main__":
    candor_remove_silences()