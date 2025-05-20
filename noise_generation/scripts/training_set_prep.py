import json
import glob
import shutil
import torch
import torchaudio
import pandas as pd
import tqdm
import numpy as np
from noise_generation.scripts.babble_noise import add_babble_noise_to_file_stereo
from noise_generation.scripts.speech_noise import add_speech_noise_to_file_stereo
from noise_generation.scripts.music_noise import add_music_noise_to_file_stereo
import os 
from dataset_management.dataset_manager.src.audio_manager import AudioManager
from utils import get_noise, apply_noise_to_channel
import shutil
np.random.seed(0)


def generate_augmentations():
    augmentations = []
    for candor_file in candor_files:
        s = np.random.uniform(0,1)
        if s < 0.25:
            aug = np.random.choice(possible_augmentations)
        else:
            aug = -1
        augmentations.append((candor_file, aug))
    
    augmentations = pd.DataFrame(augmentations, columns=['file', 'augmentation'])
    augmentations.to_csv(augmentation_csv)


def add_noise_to_file_stereo(input_file, no_silences_directory, output_file, noise_files, snr_dbs, repeat=False):

    # load dialogue
    speech, sr_speech = AudioManager.load_waveform(input_file, normalize=True, sample_rate=16000, channel_first=True)

    # original speech
    speech_0 = speech[0, :].unsqueeze(dim=0)
    speech_0_trimmed = os.path.join(no_silences_directory, os.path.basename(input_file).split('.')[0]+'_0.wav')

    speech_1 = speech[1, :].unsqueeze(dim=0)
    speech_1_trimmed = os.path.join(no_silences_directory, os.path.basename(input_file).split('.')[0]+'_1.wav')

    # read speech without silences -- don't normalise it already has been? 
    speech_0_trimmed, sr = AudioManager.load_waveform(speech_0_trimmed, mono=True, normalize=False, sample_rate=16000, channel_first=True)
    speech_1_trimmed, sr = AudioManager.load_waveform(speech_1_trimmed, mono=True, normalize=False, sample_rate=16000, channel_first=True)

    # noise chosen at random from a different session
    noise = get_noise(speech_0, noise_files, repeat=repeat, normalize=False)
    n1 = apply_noise_to_channel(speech_0, speech_0_trimmed, noise, snr_dbs)

    noise = get_noise(speech_1, noise_files, repeat=repeat, normalize=False)
    n2 = apply_noise_to_channel(speech_1, speech_1_trimmed, noise, snr_dbs)

    noisy_speech = torch.concatenate((n1, n2), axis=0)

    torchaudio.save(output_file, noisy_speech, sr)


def training_set():

    clean_files_dir = "/data/ssd2/russelsa/candor_wav"
    output_files_dir = "turn-taking-projects/corpora/candor/candor_wav_test_SNR_sweep"
    no_silences_directory = "/data/ssd4/russelsa/candor_utts/silences_removed"

    possible_augmentations = ['music', 'speech', 'babble']

    # list all input files 
    candor_files = [os.path.join(clean_files_dir, o) for o in os.listdir(clean_files_dir)]

    # what to do to what file
    augmentation_csv = "noise_generation/training_set_augmentations.csv"

    # generate_augmentations()

    snr = 0

    augmentations = pd.read_csv(augmentation_csv)

    # training noise files

    # music 
    music_dirs = ['noise_generation/data/short-musan/music/fma-merged',
                  'noise_generation/data/short-musan/music/fma-western-art-merged',
                  'noise_generation/data/short-musan/music/hd-classical-merged',
                  'noise_generation/data/short-musan/music/rfm-merged']
    music_files = []
    for music_dir in music_dirs:
        music_files += glob.glob(music_dir+'/*.wav')

    ## LRS3
    # speech_split = json.load(open("noise_generation/data/lrs3_split.json", "rb"))
    # speech_files = glob.glob("/data/ssd4/pretrain/**/*.wav")
    # speech_files = [s for s in speech_files if s.split('/')[-2] in speech_split['train_ids']]
    candor_root = "/home/russelsa@ad.mee.tcd.ie/github/dataset_management/dataset_manager/assets/new_folds/candor/"
    candor_utts = "/data/ssd4/russelsa/candor_utts/silences_removed/"
    candor_utts = glob.glob(candor_utts + "*.wav")
    test_ids = pd.read_csv(os.path.join(candor_root, "test.csv"))
    test_ids = test_ids['id'].tolist()

    # babble 
    babble_files = glob.glob("/data/ssd4/russelsa/lrs3_babble/train/*.wav")

    for candor_file in candor_files:

        output_file = os.path.join(output_files_dir, os.path.basename(candor_file))
        aug = augmentations.loc[augmentations['file']==candor_file, "augmentation"].item()

        id = os.path.basename(candor_file).split('.')[0]

        # dont need these now
        if id in test_ids:
            continue

        if aug == '-1':
            shutil.copy(candor_file, output_file)
        
        if aug == 'babble':
            # add_noise_to_file(candor_file, output_file, babble_files, snr, repeat=True)
            print("babble", output_file)
            add_babble_noise_to_file_stereo(candor_file, no_silences_directory, output_file, babble_files, snr, repeat=True)
        
        elif aug == 'speech':
            
            # add_noise_to_file(candor_file, output_file, speech_files, snr, repeat=False)
            for fold in range(0,5):

                output_file_fold = os.path.join(output_files_dir, os.path.basename(output_file).split('.')[0] + f'_fold_{fold}' + '.wav')
                print("speech", output_file_fold)
                
                # train ids / val ids for this fold
                val_ids = pd.read_csv(os.path.join(candor_root, f"fold_{fold}", "val.csv"))
                train_ids = pd.read_csv(os.path.join(candor_root, f"fold_{fold}", "train.csv"))

                val_ids.columns = [c.strip() for c in val_ids.columns]
                train_ids.columns = [c.strip() for c in train_ids.columns]
                
                val_ids = val_ids['id'].tolist()
                train_ids = train_ids['id'].tolist()

                # if this file is used for training then sample the noise added to it from the training set!
                if id in train_ids:
                    print(id, f"fold_{fold}", "train")
                    speech_files = [f for f in candor_utts if os.path.basename(f).split('_')[0] in train_ids] 
                # ditto validation
                elif id in val_ids:
                    print(id, f"fold_{fold}", "val")
                    speech_files = [f for f in candor_utts if os.path.basename(f).split('_')[0] in val_ids] 

                add_speech_noise_to_file_stereo(candor_file, no_silences_directory, output_file_fold, speech_files, snr, repeat=True)            

        elif aug == 'music':
            print("music", output_file)
            add_music_noise_to_file_stereo(candor_file, no_silences_directory, output_file, music_files, snr, repeat=False)


def babble_0db():
    test_set = "dataset_management/dataset_manager/assets/new_folds/candor/test.csv"
    test_set = pd.read_csv(test_set)['id'].to_list()
    output_test = "/data/ssd3/russelsa/candor_0dB_babble/test"
    output_train = "/data/ssd3/russelsa/candor_0dB_babble/train"
    babble_train = glob.glob("/data/ssd4/russelsa/lrs3_babble/train/*.wav")
    babble_test =  glob.glob("/data/ssd4/russelsa/lrs3_babble/val/*.wav")
    no_silences_directory="/data/ssd4/russelsa/candor_utts/silences_removed"
    candor_wav = "/data/ssd2/russelsa/candor_wav/*.wav"
    for file in tqdm.tqdm(glob.glob(candor_wav)):
        id = os.path.basename(file).split('.')[0]
        if id in test_set:
            output_file=os.path.join(output_test, f"{id}.wav")
            if not os.path.exists(output_file):
                add_babble_noise_to_file_stereo(file, no_silences_directory, output_file, babble_test, 0, repeat=True)
        else:
            output_file=os.path.join(output_train, f"{id}.wav")
            if not os.path.exists(output_file):
                add_babble_noise_to_file_stereo(file, no_silences_directory, output_file, babble_train, 0, repeat=True)

if __name__ == "__main__":
    # training_set()
    babble_0db()
