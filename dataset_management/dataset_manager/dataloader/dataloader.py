import pandas as pd
import csv
import json
import torch
import time 
import os 
import pickle
from torch.utils.data import Dataset, DataLoader
from dataset_management.dataset_manager.src.audio_manager import AudioManager, GeMAPSManager
from dataset_management.dataset_manager.src.video_manager_cpu import VideoManager
from dataset_management.dataset_manager.config.config import config
from dataset_management.dataset_manager.scripts.transcript_processing import vads_from_transcript, vad_list_to_one_hot, training_labels_projection
from dataset_management.dataset_manager.dataloader.codebook import Codebook, bin_times_to_frames
import einops
from copy import copy
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Manager
import tensordict as td


def read_csv_lines(file, start_line, end_line):
    """ 
        looking at the fastest way to read specific lines from a csv file...
    """

    with open(file, "r") as f:
        # lines = f.readlines()
        # target = lines[start_line:end_line]
        lines = list(csv.reader(f))[start_line:end_line]
        
    return lines



class ChunkedDataset(Dataset):


    def __init__(self, pickle_file: str, mode: str) -> None:
        super().__init__()

        self.pickle_file = pickle_file
        self.dataset = None
        self.manager = Manager()

        self.load_data()

        self.sr = config['audio_feature_extraction_Hz']
        self.window_size = [2, self.sr*config['audio_window_length']]
        self.future_window_size = [2, self.sr*config['future_context']]

        self.mode = mode

        assert mode in ['VAP', 'ind-4','ind-40', None]
        if mode is None:
            mode='VAP'
        
        if self.mode == 'VAP':
            bin_times = [0.2, 0.4, 0.6, 0.8]
            bin_frames = bin_times_to_frames(bin_times, frame_hz=self.sr)
            self.codebook = Codebook(bin_frames=bin_frames)
        

    def load_data(self):

        # self.dataset = self.manager.list(pd.read_pickle(self.pickle_file))
        self.ids = np.load(self.pickle_file+'_ids.npy')
        # self.ids = self.ids.tolist()
        self.start_times = np.load(self.pickle_file+'_starts.npy')
        self.end_times = np.load(self.pickle_file+'_ends.npy')
        self.vads = torch.load(self.pickle_file+'_vads.pt')
        self.vaps = torch.load(self.pickle_file+'_vaps.pt')


    def  __len__(self):
        return len(self.ids)
    

    def __getitem__(self, index):

        id =  self.ids[index].decode('utf8')
        start_time = self.start_times[index]
        end_time = self.end_times[index]
        vad = self.vads[index, ...]
        vap = self.vaps[index, ...]

        # pad the current window with the next 2 seconds of future context to enable projection
        window_with_future = torch.concat((vad, vap), dim=0)

        if self.mode == 'VAP':
            vap = training_labels_projection(window_with_future, mode='VAP')

            inverse_vap = torch.stack((vap[:, 1, :], vap[:, 0, :]), dim=1)
            inverse_vap = self.codebook.encode(inverse_vap)
            vap = self.codebook.encode(vap)

            return id, start_time, end_time, vad, vap, inverse_vap

        elif self.mode == 'ind-4':
            vap = training_labels_projection(window_with_future, mode='ind-4')

        elif self.mode == 'ind-40':
            vap = training_labels_projection(window_with_future, mode='ind-40')
        
        vap = einops.rearrange(vap, "n c d -> n (c d)")
        inverse_vap = torch.stack((vap[:, 1, :], vap[:, 0, :]), dim=1)
        inverse_vap = einops.rearrange(inverse_vap, "n c d -> n (c d)")
        
        return id, start_time, end_time, vad, vap, inverse_vap


class AudioDataset(ChunkedDataset):

    
    def __init__(self, pickle_file: str, mode: str, wavdir: str) -> None:
        self.wavdir = wavdir
        self.normalize = True
        super().__init__(pickle_file, mode)

    
    def get_audio_chunk(self, id, start, end, sample_rate):

        # # id of the file 
        # id = item[0]

        # # start and end of the chunk
        # start, end = item[1], item[2]

        # locate the audio file 
        wavfile = os.path.join(self.wavdir, f"{id}.wav")
        
        # audio segment 
        audio = AudioManager(wavfile, mono=False)
        audio_chunk = audio.get_segment(start, end, sample_rate, normalize=self.normalize)

        return audio_chunk[0]


    def __getitem__(self, index):

        id, start_time, end_time, vad, vap, inverse_vap = super().__getitem__(index)
        audio_chunk = self.get_audio_chunk(id, start_time, end_time, sample_rate=config['sample_rate'])
        # return_item = {"id": id, "start_time": start_time, "end_time": end_time, "audio_chunk": audio_chunk, "vad": vad, "vap": vap, "inverse_vap": inverse_vap}
        return id, start_time, end_time, vad, vap, inverse_vap, audio_chunk


class GeMAPSDataset(ChunkedDataset):
    
    def __init__(self, pickle_file: str, gemaps_dir: str, normalize_gemaps: bool, mode=None) -> None:
        self.gemaps_dir = gemaps_dir
        self.normalize_gemaps = normalize_gemaps
        super().__init__(pickle_file, mode=mode)


    def get_gemaps_chunk(self, item):
        
        # id of the file 
        id = item[0]

        # start and end of the chunk
        start, end = item[1], item[2]

        # locate gemaps file 
        gemaps_chunk = {}
        for i in [0,1]:
            gemaps_file = os.path.join(self.gemaps_dir, f"{id}_{i}.pkl")
            gemaps_mgr = GeMAPSManager(gemaps_file, normalize=self.normalize_gemaps, gemaps_sample_rate=config['audio_feature_extraction_Hz'])
            gemaps_chunk[i] = gemaps_mgr.get_segment(start, end)
        
        return gemaps_chunk


    def __getitem__(self, index):
        item = super().__getitem__(index)
        id =  item[0]
        start_time = item[1]
        end_time = item[2]

        gemaps_chunk = self.get_gemaps_chunk(item)
        return_item = {"id": id, "start_time": start_time, "end_time": end_time, "gemaps_chunk": gemaps_chunk}

        return return_item


class ValidationAudioDataset(Dataset):
    """ returns everything in order with overlapping windows to enable validation of a file """

    def __init__(self, audio_file: str, transcript_file: str, sr: int, feature_sr: int, window_size: int, step_size: int, mode: str):

        super().__init__()
        
        self.audio_file = audio_file
        self.transcript_file = transcript_file

        self.id = os.path.basename(audio_file).split('.')[0]

        self.sr = sr
        self.feature_sr = feature_sr
        self.window_size = window_size
        self.step_size = step_size

        self.mode = mode
        self.audio_normalize = True

        self.load_audio()
        self.load_transcript()

    def load_audio(self):
        audio, _= AudioManager.load_waveform(self.audio_file, self.sr, normalize=self.audio_normalize)
        self.track_size = audio.shape
        size, step = int(self.sr*self.window_size), int(self.sr*self.step_size)

        self.audio_unbathed = audio

        audio = audio.unfold(0, size, step)
        audio = einops.rearrange(audio, "b c n -> b n c")

        self.audio = audio      
    
    def load_transcript(self):
        
        vad_list = vads_from_transcript(self.transcript_file, samples=False)
        vad = vad_list_to_one_hot(vad_list, [2, int(self.feature_sr*self.track_size[0]/self.sr)], samples=False, sr=self.feature_sr)
        vap = training_labels_projection(vad, mode=self.mode)
        
        size, step = int(self.feature_sr*self.window_size), int(self.feature_sr*self.step_size)

        self.vad_unbatched, self.vap_unbatched = vad, vap

        vad, vap = vad.unfold(0, size, step), vap.unfold(0, size, step)

        self.vad, self.vap = vad, vap


    def __getitem__(self, index):
        start_time, end_time = index * self.step_size, index * self.step_size + self.window_size
        return {"id": [self.id], "audio_chunk": self.audio[index, :], "start_time": start_time, "end_time": end_time, "vad": self.vad[index, :], "vap": self.vap[index, :]}
    
    def __len__(self):
        return self.audio.shape[0]-1


class AudioVisualDataset(AudioDataset):


    def __init__(self, video_directory, video_format, channelmaps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_directory = video_directory
        self.video_format = video_format
        self.channel_hash = pd.read_pickle(channelmaps)
        self.fps=30


    def __getitem__(self, idx):

        id, start, end, vad, vap, inverse_vap, audio_chunk = super().__getitem__(idx)

        # load the corresponding video file
        frames_lr = None
        ch = self.channel_hash[self.channel_hash['id']==id]
    
        for lr in ("L", "R"):

            video_id = ch[lr].item()
            
            video_file = os.path.join(self.video_directory, id + '--' + video_id + self.video_format)
            # frames = read_csv_lines(video_file, int(start*self.fps), int(end*self.fps))
            df = pd.read_pickle(video_file)
            frames = df.iloc[int(start*self.fps):int(end*self.fps), :]
            frames = torch.Tensor(frames.values)
            if frames.shape[0]==0:
                print(id, frames.shape)

            if frames_lr is None:
                frames_lr = frames
            else:
                frames_lr = torch.stack((frames_lr, frames), dim=-1)

            if lr == "L":
                channel_id_l = video_id

            if lr == "R":
                channel_id_r = video_id

        # if 'openface_features' not in ret:
        #     ret['openface_features'] = [d.strip() for d in  df.columns]

        return id, start, end, vad, vap, inverse_vap, audio_chunk, channel_id_l, channel_id_r, frames_lr


class ValidationAudioVisualDataset(ValidationAudioDataset):

    
    def __init__(self, video_pkl_dir, channelmap, **kwargs):
        super().__init__(**kwargs)
        
        self.channelmap = pd.read_pickle(channelmap)
        self.video_pkl_dir = video_pkl_dir
        self.id = os.path.basename(self.audio_file).split('.')[0]
        ch = self.channelmap[self.channelmap.id==self.id]
        self.channelmap = {"L": ch["L"].item(), "R": ch["R"].item()}
        self.fps=30


    def __getitem__(self, index):
        # return {"id": self.audio, "audio_chunk": self.audio[index, :], "start_time": start_time, "end_time": end_time, "vad": self.vad[index, :], "vap": self.vap[index, :]}
        ret = super().__getitem__(index)

        start = ret['start_time']
        end = ret['end_time']

        frames_lr = None
        for video_id in [self.channelmap["L"], self.channelmap["R"]]:
            
            video_file = os.path.join(self.video_pkl_dir, ret['id'][0] + '--' + video_id + '.pkl')

            df = pd.read_pickle(video_file)
            frames = df.iloc[int(start*self.fps):int(end*self.fps), :]
            frames = torch.Tensor(frames.values)
            if frames.shape[0]==0:
                print(ret['id'], frames.shape)
            if frames_lr is None:
                frames_lr = frames
            else:
                frames_lr = torch.stack((frames_lr, frames), dim=-1)
            
        # frames = torch.stack(frames_lr, dim=-1)

        # time last
        ret['channel_ids'] = np.array([self.channelmap["L"], self.channelmap["R"]])
        ret['frames'] = frames_lr

        return ret
    
    def __len__(self):
        return super().__len__()


def test_dataloader():

    pickle_file = "/home/russelsa@ad.mee.tcd.ie/github/dataset_management/dataset_manager/assets/candor_filtered_folds/candor/fold_0/train.pkl"
    # pickle_file = "/home/russelsa@ad.mee.tcd.ie/github/dataset_management/dataset_manager/assets/candor_filtered_folds/candor/fold_0/val.pkl"
    wavdir = "turn-taking-projects/corpora/candor/candor_wav"
    video_directory = "turn-taking-projects/corpora/candor/candor_openface_pkl"
    channelmaps = "turn-taking-projects/corpora/candor/channelmaps"

    av_ds = AudioVisualDataset(pickle_file=pickle_file, video_directory=video_directory, wavdir=wavdir, channelmaps=channelmaps, video_format='.pkl', mode='VAP')
    dl = DataLoader(av_ds, batch_size=32, shuffle=True, pin_memory=False)

    from tqdm import tqdm
    pbar = tqdm(total=len(dl))

    print(pickle_file)
    while True:
        # start = time.time()
        try:
            item = next(iter(dl))
        except FileNotFoundError as e:
            print(e)
            with open("dataloader_missing_file_train_fold0.txt", "a") as f:
                f.write(str(e)+"\n")
                f.flush()
        pbar.update(1)


if __name__=="__main__":

    dl = ValidationAudioVisualDataset(
        video_pkl_dir="turn-taking-projects/corpora/candor/candor_openface_pkl/",
        channelmap="/home/russelsa@ad.mee.tcd.ie/github/turn-taking-projects/corpora/candor/channelmaps/0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a.channelmap.json",
        audio_file="/home/russelsa@ad.mee.tcd.ie/github/turn-taking-projects/corpora/candor/candor_wav/0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a.wav",
        transcript_file="/home/russelsa@ad.mee.tcd.ie/github/turn-taking-projects/corpora/candor/candor_speechmatics/0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a.TextGrid", 
        sr=16_000, 
        feature_sr=50,
        window_size=20,
        step_size=19,
        mode='VAP'
    )

    item = dl.__getitem__(0)

    x=1