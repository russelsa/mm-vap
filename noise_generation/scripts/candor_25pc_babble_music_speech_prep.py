import pandas as pd
import os 
import shutil
import glob
from noise_generation.scripts.babble_noise import add_babble_noise_to_file_stereo
from noise_generation.scripts.speech_noise import add_speech_noise_to_file_stereo
from noise_generation.scripts.music_noise import add_music_noise_to_file_stereo
import pandas as pd


def babble_prep():
    augmentations = "noise_generation/training_set_augmentations.csv"
    candor_babble = "/data/ssd3/russelsa/candor_0dB_babble/train/"
    candor_original = "turn-taking-projects/candor_wav/"

    aug = pd.read_csv(augmentations)

    for _, row in aug.iterrows():
        clean_file = row['file']
        augmentation = row['augmentation']

        if augmentation != '-1':
            print(clean_file)
            os.remove(clean_file)
            babble_file = os.path.join(candor_babble, os.path.basename(clean_file))
            shutil.copy(babble_file, clean_file)

def music_prep():

    augmentations = "noise_generation/training_set_augmentations.csv"
    candor_original = "/data/ssd2/russelsa/candor_wav"
    candor_output = "turn-taking-projects/corpora/candor/candor_0db_25pc_music"

    music_dirs = ['noise_generation/data/short-musan/music/fma-merged',
                'noise_generation/data/short-musan/music/fma-western-art-merged',
                'noise_generation/data/short-musan/music/hd-classical-merged',
                'noise_generation/data/short-musan/music/rfm-merged']
    music_files = []
    for music_dir in music_dirs:
        music_files += glob.glob(music_dir+'/*.wav')

    no_silences_directory = "/data/ssd4/russelsa/candor_utts/silences_removed"
    snr=0

    aug = pd.read_csv(augmentations)

    for _, row in aug.iterrows():
        clean_file = row['file']
        augmentation = row['augmentation']

        if augmentation != '-1':
            print(clean_file)
            candor_clean = os.path.join(candor_original, os.path.basename(clean_file))
            candor_music = os.path.join(candor_output, os.path.basename(clean_file))
            add_music_noise_to_file_stereo(candor_clean, no_silences_directory, candor_music, music_files, snr, repeat=False)


def speech_prep():

    augmentations = "noise_generation/training_set_augmentations.csv"
    candor_original = "/data/ssd2/russelsa/candor_wav"
    candor_output = "turn-taking-projects/corpora/candor/candor_0db_25pc_speech"

    no_silences_directory = "/data/ssd4/russelsa/candor_utts/silences_removed"
    
    speech_files = glob.glob(no_silences_directory+'/*.wav')
    test_ids = pd.read_csv("/home/russelsa@ad.mee.tcd.ie/github/dataset_management/dataset_manager/assets/new_folds/candor/test.csv")['id'].to_list()
    val_ids = pd.read_csv("/home/russelsa@ad.mee.tcd.ie/github/dataset_management/dataset_manager/assets/new_folds/candor/fold_0/val.csv")['id'].to_list()
    exclude = test_ids + val_ids

    speech_files = [s for s in speech_files if os.path.basename(s).split('.')[0] not in exclude]

    snr=0
    aug = pd.read_csv(augmentations)

    for _, row in aug.iterrows():
        clean_file = row['file']
        augmentation = row['augmentation']

        if augmentation != '-1':
            print(clean_file)
            candor_clean = os.path.join(candor_original, os.path.basename(clean_file))
            candor_music = os.path.join(candor_output, os.path.basename(clean_file))
            add_speech_noise_to_file_stereo(candor_clean, no_silences_directory, candor_music, speech_files, snr, repeat=False)


if __name__=="__main__":

    # babble_prep()
    # music_prep()
    speech_prep()
