from tqdm import tqdm
import pickle
import os 
import pandas as pd
from dataset_management.dataset_manager.config.config import config
from dataset_management.dataset_manager.scripts import transcript_processing
from dataset_management.dataset_manager.scripts.transcript_processing import vads_from_transcript, vad_list_to_one_hot, training_labels_projection
from pydub import AudioSegment
import copy
import torch
import numpy as np


def get_chunked_dataset(csv_path, audio_dir, transcripts_dir):
    """generate a sliding window dataset, with possibly overlapping windows (see the config file audio_window_length, audio_window_stride)

    Args:
        csv_path (str): path to the train / val csv 
        audio_dir (str): path to the directory containing the .wav files
        transcripts_dir (str): path to the directory with the .TextGrid files

    Returns:
        list[list]: a list of format [
                                      id (session_id), 
                                      start, end (start and end times of the windows),
                                      vads (the VAD times in the window)
                                      future_vads (the VAD times in the next N seconds after the window, see future_context config file)
                                      ]
    """

    df = pd.read_csv(csv_path,skipinitialspace=True)
    df.columns = [d.strip() for d in df.columns]

    # window size and overlap
    window_size = config['audio_window_length']
    stride_size = config['audio_window_stride']

    ids = [] 
    starts = [] 
    ends = [] 
    vads = None
    vaps = None

    kk=0

    pbar = tqdm(total=df.shape[0])
    for i, row in df.iterrows():
        
        id = row['id']

        # get the audio file and the duration of this dyad
        audio_file = os.path.join(audio_dir, f"{id}.wav")
        audio = AudioSegment.from_file(audio_file)

        duration_ms = len(audio)
        
        # convert and remove seconds from the end
        duration_seconds = duration_ms / 1000.0 - config['future_context']

        start_times_all = list(range(0, int(duration_seconds - stride_size), stride_size))
        end_times_all = [f + window_size for f in start_times_all]

        start_times = []
        end_times = []

        # ensure that duration is not exceeded
        # remove the last incomplete windows for training
        for start, end in zip(start_times_all, end_times_all):
            if end < duration_seconds:
                start_times.append(start)
                end_times.append(end)

        # transcript file for this 
        transcript_file = os.path.join(transcripts_dir, f"{id}.TextGrid")

        try:
            vad_list_master = transcript_processing.vads_from_transcript(transcript_file, samples=False)
        except Exception as e:
            print(f"missing transcript for {id}")
            continue

        for start, end in zip(start_times, end_times):

            vad_list = copy.deepcopy(vad_list_master)

            # adding some seconds for the vap
            vads_in_frame_left = [v for v in vad_list[0] if v[1] > start and v[0] < end]
            vads_in_frame_right = [v for v in vad_list[1] if v[1] > start and v[0] < end]

            # now trim the start and the end 
            if vads_in_frame_left != []:
                vads_in_frame_left[0][0] = max(start, vads_in_frame_left[0][0])
                vads_in_frame_left[-1][1] = min(end, vads_in_frame_left[-1][1])

            if vads_in_frame_right != []:
                vads_in_frame_right[0][0] = max(start, vads_in_frame_right[0][0])
                vads_in_frame_right[-1][1] = min(end, vads_in_frame_right[-1][1])
            
            vads_in_frame_left = [[v[0] - start, v[1] - start] for v in vads_in_frame_left]
            vads_in_frame_right = [[v[0] - start, v[1] - start] for v in vads_in_frame_right]

            # future vads
            future_vad_left = [v for v in vad_list[0] if v[0] > end and v[0] < end + config['future_context']]
            future_vad_right = [v for v in vad_list[1] if v[0] > end and v[0] < end + config['future_context']]

            # now trim the start and the end 
            if future_vad_left != []:
                future_vad_left[-1][1] = min(end + config['future_context'], future_vad_left[-1][1])
                future_vad_left = [[v[0] - start, v[1] - start] for v in future_vad_left]

            if future_vad_right != []:
                future_vad_right[-1][1] = min(end + config['future_context'], future_vad_right[-1][1])
                future_vad_right = [[v[0] - start, v[1] - start] for v in future_vad_right]

            vad = [vads_in_frame_left, vads_in_frame_right]
            future_vad = [future_vad_left, future_vad_right]

            vad = vad_list_to_one_hot(vad, track_size_window, samples=False, sr=sr, start=0)
            vap = vad_list_to_one_hot(future_vad, track_size_future_window, samples=False, sr=sr, start=0)

            ids.append(id)
            starts.append(start)
            ends.append(end)

            if vads is None:
                vads = vad.unsqueeze(dim=0)
            else:
                vads = torch.concat((vads, vad.unsqueeze(dim=0)), dim=0)
            if vaps is None:
                vaps = vap.unsqueeze(dim=0)
            else:
                vaps = torch.concat((vaps, vap.unsqueeze(dim=0)), dim=0)

            # kk+=1
            # if kk >= 2000:


        pbar.update(1)
            # 
    ids = np.array(ids, dtype=np.string_)
    starts = np.array(starts, dtype=np.int32)
    ends = np.array(ends, dtype=np.int32)
    return ids, starts, ends, vads, vaps
    
            # pbar.update(1)


def generate_chunks_for_one_fold(audio_dir, transcripts_dir, csv_path, pkl_path):

    # if os.path.exists(pkl_path):
    #     return

    ids, starts, ends, vads, vaps = get_chunked_dataset(csv_path, audio_dir, transcripts_dir)
    
    np.save(pkl_path+'_ids', ids)
    np.save(pkl_path+'_starts', starts)
    np.save(pkl_path+'_ends', ends)
    torch.save(vads, pkl_path+'_vads.pt')
    torch.save(vaps, pkl_path+'_vaps.pt')

if __name__=="__main__":

    # candor wav
    audio_dir = "turn-taking-projects/corpora/candor/candor_wav"
    transcripts_dir = "turn-taking-projects/corpora/candor/candor_speechmatics"

    sr = config['audio_feature_extraction_Hz']
    track_size_window = [2, sr*config['audio_window_length']]
    track_size_future_window = [2, sr*config['future_context']]

    for i in range(0,5):
        pkl_path = f"dataset_management/dataset_manager/assets/new_folds/candor/fold_{i}/train"
        csv_path = f"dataset_management/dataset_manager/assets/new_folds/candor/fold_{i}/train.csv"
        generate_chunks_for_one_fold(audio_dir, transcripts_dir, csv_path, pkl_path)

        for i in [2,3,4]:
            print(i)
            pkl_path = f"dataset_management/dataset_manager/assets/new_folds/candor/fold_{i}/val"
            csv_path = f"dataset_management/dataset_manager/assets/new_folds/candor/fold_{i}/val.csv"
            generate_chunks_for_one_fold(audio_dir, transcripts_dir, csv_path, pkl_path)

    pkl_path = f"/mnt/storage/dataset_management/dataset_manager/assets/new_folds/candor/test"
    csv_path = f"/mnt/storage/dataset_management/dataset_manager/assets/new_folds/candor/test.csv"
    generate_chunks_for_one_fold(audio_dir, transcripts_dir, csv_path, pkl_path)
