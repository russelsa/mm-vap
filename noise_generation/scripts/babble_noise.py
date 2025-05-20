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
from utils import get_noise, apply_noise_to_channel


def apply_babble_noise(clean_audio_files, no_silences_directory, noise_directory, ids, output_top_dir):

    # get the relevant subset of noise files
    pattern = os.path.join(noise_directory, "*.wav")
    noise_files = glob.glob(pattern)

    random.shuffle(noise_files)

    snrs = list(np.arange(-10, 12.5, 2.5))
    for snr in snrs:
        
        print(snr)
        outdir_snr = os.path.join(output_top_dir, f'{snr}')

        if not os.path.exists(outdir_snr):
            os.mkdir(outdir_snr)

        for clean_audio_file in clean_audio_files:

            output_file = os.path.join(outdir_snr, os.path.basename(clean_audio_file))
            add_babble_noise_to_file_stereo(input_file=clean_audio_file, 
                                     no_silences_directory=no_silences_directory,
                                     output_file=output_file, 
                                     noise_files=noise_files, 
                                     snr_dbs=snr)
            # exit()

    return


def apply_babble_noise_to_corpus_testset():

    rootdir = "/home/russelsa@ad.mee.tcd.ie/github/turn-taking-projects/corpora/candor/candor_wav_test_SNR_sweep/"
    clean_audio_files = "/data/ssd2/russelsa/candor_wav"

    output_top_dir = os.path.join(rootdir, "babble")

    noise_directory = "/data/ssd4/russelsa/lrs3_babble/val"

    no_silences_directory = "/data/ssd4/russelsa/candor_utts/silences_removed"

    # test set ids
    test_ids = pd.read_csv("/home/russelsa@ad.mee.tcd.ie/github/dataset_management/dataset_manager/assets/new_folds/candor/test.csv")
    test_ids = test_ids['id'].tolist()

    clean_audio_files = [os.path.join(clean_audio_files, id + '.wav') for id in test_ids]

    apply_babble_noise(clean_audio_files, no_silences_directory, noise_directory, test_ids, output_top_dir)


def add_babble_noise_to_file_stereo(input_file, no_silences_directory, output_file, noise_files, snr_dbs, repeat=True):

    # load dialogue
    speech, sr_speech = AudioManager.load_waveform(input_file, normalize=True, sample_rate=16000, channel_first=True)

    # original speech
    speech_0 = speech[0, :].unsqueeze(dim=0)
    speech_0_trimmed = os.path.join(no_silences_directory, os.path.basename(input_file).split('.')[0]+'_0.wav')

    speech_1 = speech[1, :].unsqueeze(dim=0)
    speech_1_trimmed = os.path.join(no_silences_directory, os.path.basename(input_file).split('.')[0]+'_1.wav')

    # read speech without silences -- don't normalise it already has been? 
    try:
        speech_0_trimmed, sr = AudioManager.load_waveform(speech_0_trimmed, mono=True, normalize=True, sample_rate=16000, channel_first=True)
    except RuntimeError:
        return
    speech_1_trimmed, sr = AudioManager.load_waveform(speech_1_trimmed, mono=True, normalize=True, sample_rate=16000, channel_first=True)

    # noise chosen at random from a different session
    noise = get_noise(speech_0, noise_files, repeat=repeat, normalize=True, volume_norm=True)
    n1 = apply_noise_to_channel(speech_0, speech_0_trimmed, noise, snr_dbs, mode='standard')

    noise = get_noise(speech_1, noise_files, repeat=repeat, normalize=True, volume_norm=True)
    n2 = apply_noise_to_channel(speech_1, speech_1_trimmed, noise, snr_dbs, mode='standard')

    noisy_speech = torch.concatenate((n1, n2), axis=0)

    torchaudio.save(output_file, noisy_speech, sr)


if __name__ == "__main__":
    # apply_babble_noise_to_corpus_testset()
    noise_files = glob.glob("/data/ssd4/russelsa/lrs3_babble/val/*.wav")
    add_babble_noise_to_file_stereo(
        input_file="/data/ssd2/russelsa/candor_wav/1f7e582c-c6bc-46b6-b5a4-e5d78e8a46ac.wav",
        no_silences_directory="/data/ssd4/russelsa/candor_utts/silences_removed",
        output_file="/home/russelsa@ad.mee.tcd.ie/github/Multimodal_Turn_Taking/sample_data/0db_babble.wav",
        noise_files=noise_files,
        snr_dbs=0
    )
