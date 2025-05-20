import numpy as np
import cv2
import glob
import pandas as pd
import os
from dataset_management.dataset_manager.dataloader.dataloader import AudioVisualDataset
from dataset_manager.src.openface_plotting import draw_landmarks
from torch.utils.data import DataLoader
from scipy.io import wavfile


def write_audio(audio, output_file, mono=False):

    if mono:
        wavfile.write(output_file, 16_000, np.array(audio))


def analyse_batch(item, batch_idx, video_dir, csv_dir):

    id = item['id'][batch_idx]
    
    audio = item['audio_chunk'][batch_idx, :, :]

    left_video_id = item['channel_ids'][0]['L'][batch_idx]
    right_video_id = item['channel_ids'][0]['R'][batch_idx]

    videos = [os.path.join(video_dir, id+'--'+left_video_id+'.mp4'), os.path.join(video_dir, id+'--'+right_video_id+'.mp4')]

    landmarks = item['frames'][batch_idx, ...]

    xx, yy = landmarks[:, 5:21, 0], landmarks[:, 22:40, 0]
    start_frame = int(item['start_time'][batch_idx].item()*30)
    draw_landmarks(videos[0], xx, yy, start_frame=start_frame)
    write_audio(audio[:, 0], output_file='out.wav', mono=True)

    return

def test_batch():

    pickle_file = "/home/russelsa@ad.mee.tcd.ie/github/dataset_management/dataset_manager/assets/new_folds/candor/fold_0/train.pkl"
    wavdir = "turn-taking-projects/corpora/candor/candor_wav"
    video_features = "/data/ssd3/russelsa/candor_openface_pkl"
    mp4_directory = "/data/ssd2/russelsa/candor_video/"
    channelmaps = "turn-taking-projects/corpora/candor/channelmaps"
    csv_directory = "/data/ssd2/russelsa/candor_openface_features/"

    sample_file = os.path.join(video_features, "aced00ba-3582-4d63-9f08-f9c444986729--5edc3f9696a72790ae256a27.pkl")
    sample_file = pd.read_pickle(sample_file)
    openface_features = [d.strip() for d in sample_file.columns]

    av_ds = AudioVisualDataset(pickle_file=pickle_file, video_directory=video_features, wavdir=wavdir, channelmaps=channelmaps, video_format='.pkl', mode='VAP')
    dl = DataLoader(av_ds, batch_size=2, shuffle=False, pin_memory=True)

    from tqdm import tqdm
    # pbar = tqdm(total=len(dl))
    while True:
        item = next(iter(dl))
        analyse_batch(item, 0, mp4_directory, csv_directory)
        exit()
