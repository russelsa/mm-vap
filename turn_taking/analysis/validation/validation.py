import glob
import os 
import pickle
import numpy as np
import json
import torch
torch.cuda.set_device(1)
import yaml
from turn_taking.analysis.validation import run_model
from turn_taking.model.model import StereoTransformerModel
from turn_taking.model.multimodal_model import EarlyVAFusion, LateVAFusion
from torch.utils.data import DataLoader
from dataset_management.dataset_manager.dataloader.dataloader import ValidationAudioDataset, ValidationAudioVisualDataset
from turn_taking.analysis.validation import probabilities
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import torchmetrics
from torchmetrics.functional import f1_score
from collections import defaultdict
import copy


def run_all(model, ids, wav_dir, transcript_dir, output_dir, window_size=20, step_size=19, mode='VAP'):

    if ids==[]:
        wavs = [os.path.join(wav_dir, o) for o in os.listdir(wav_dir) if '.wav' in o]
    else:
        wavs = [os.path.join(wav_dir, f"{o}.wav") for o in ids]

    progress_bar = tqdm(total=len(wavs))
    for audio_file in wavs:
        
        transcript_file = os.path.join(transcript_dir, os.path.basename(audio_file).split('.')[0]+'.TextGrid')
        output_file = os.path.join(output_dir, os.path.basename(audio_file).split('.')[0]+'.pkl')

        if os.path.exists(output_file):
            continue
        
        try:
            audio_dataset = ValidationAudioDataset(audio_file=audio_file, transcript_file=transcript_file, sr=16_000, feature_sr=50, window_size=window_size, step_size=step_size, mode=mode)
        except AttributeError:
            # raised when the transcription is blank
            continue
        audio_dl = DataLoader(audio_dataset, batch_size=10, shuffle=False)

        vaps, vads = run_model.run_model(model, audio_dl, mask_vad=False, feature_extraction_hz=50, window_size=window_size, step_size=step_size, mode='VAP')

        with open(output_file, "wb") as f:

            pickle.dump([vaps, vads], f)

        progress_bar.update(1)

    progress_bar.close()





def samples(time, sr=50):
    return int(time*sr)


def get_shifts_holds(events, p, window=0.1):

    # shift = 0
    # hold = 1
    if window<0:
        sign=-1
    else:
        sign=1
    window = samples(np.abs(window))*sign

    max_time = p.shape[0]/50

    y_true = []
    y_pred = []
    score = []

    shift_events = events['shifts']
    for speaker, shifts in shift_events.items():

        for shift in shifts:

            if shift>max_time:
                continue

            speaker = int(speaker)

            end_original, start_original = samples(shift), samples(shift) + window
            start = min(end_original, start_original)
            end = max(end_original, start_original)

            # '0' means a shift '0' --> '1'
            prev_speaker = p[start:end, speaker].sum()
            next_speaker = p[start:end, 1-speaker].sum()
            
            y_true.append(1)
            score.append(float(next_speaker))

    hold_events = events['holds']
    for speaker, holds in hold_events.items():

        for hold in holds:

            speaker = int(speaker)

            if hold>max_time:
                continue

            end_original, start_original = samples(hold), samples(hold) + window
            start = min(end_original, start_original)
            end = max(end_original, start_original)

            # '0' means a shift '0' --> '1'
            prev_speaker = p[start:end, speaker].sum()
            next_speaker = p[start:end, 1-speaker].sum()

            score.append(float(next_speaker))
            y_true.append(0)

    return y_true, score


def get_spred(events, p, window_size=0.5):

    # shift = 0
    # hold = 1
    y_true, y_pred = [], []
    score = []

    max_time = p.shape[0]/50
    
    for speaker, shifts in events['shifts'].items():
        for shift in shifts:

            if shift > max_time:
                continue

            speaker = int(speaker)

            # '0' means a shift '0' --> '1'
            
            # previous speaker --> previous
            prev_speaker = p[samples(shift-window_size):samples(shift), speaker].sum()
            
            # previous speaker --> other speaker
            next_speaker = p[samples(shift-window_size):samples(shift), 1-speaker].sum()

            y_true.append(1)
            score.append(float(next_speaker))

            if next_speaker > prev_speaker:
                # shift_shift += 1
                y_pred.append(1)
            else:
                # shift_hold += 1
                y_pred.append(0)

    for speaker, negative_samples in events['s_pred_neg'].items():

        for negative_sample in negative_samples:

            speaker = int(speaker)

            if negative_sample > max_time:
                continue

            # '0' means a shift '0' --> '1'
            
            # previous speaker --> previous
            prev_speaker = p[samples(negative_sample-window_size):samples(negative_sample), speaker].sum()
            
            # previous speaker --> other speaker
            next_speaker = p[samples(negative_sample-window_size):samples(negative_sample), 1-speaker].sum()

            y_true.append(0)
            score.append(float(next_speaker))

            if next_speaker > prev_speaker:
                # shift_shift += 1
                y_pred.append(1)
            else:
                # shift_hold += 1
                y_pred.append(0)

    return y_true, score


def get_overlap(events, p, window):

    if window<0:
        sign=-1
    else:
        sign=1
    window = samples(np.abs(window))*sign

    y_true = []
    score = []
    for speaker, times in events['overlaps_shift'].items():
        
        other_speaker = int(1-int(speaker))
        for time in times:
            end_original, start_original = samples(time), samples(time) + window
            start = min(end_original, start_original)
            end = max(end_original, start_original)
            score.append(p[start:end, other_speaker].sum().item())
            y_true.append(1)

    for speaker, times in events['overlaps_hold'].items():
        speaker = int(speaker)
        other_speaker = int(1-int(speaker))
        for time in times:
            end_original, start_original = samples(time), samples(time) + window
            start = min(end_original, start_original)
            end = max(end_original, start_original)
            score.append(p[start:end, other_speaker].sum().item())
            y_true.append(0)
    
    return y_true, score

def get_overlap_spred(events, p, window, spred):

    if window<0:
        sign=-1
    else:
        sign=1
    window = samples(np.abs(window))*sign

    y_true = []
    score = []
    for speaker, times in events['overlaps_shift'].items():
        
        other_speaker = int(1-int(speaker))
        for time in times:
            end_original, start_original = samples(time), samples(time) + window
            start = min(end_original, start_original)
            end = max(end_original, start_original)
            score.append(p[start:end, other_speaker].sum().item())
            y_true.append(1)

    for speaker, times in spred.items():
        speaker = int(speaker)
        other_speaker = int(1-int(speaker))
        for time in times:
            end_original, start_original = samples(time), samples(time) + window
            start = min(end_original, start_original)
            end = max(end_original, start_original)
            score.append(p[start:end, other_speaker].sum().item())
            y_true.append(0)
    
    return y_true, score


def get_bc(events, p_bc, window):

    y_true, scores = [], []
    for speaker, backchannels in events['backchannels'].items():

        for backchannel in backchannels:

            other_speaker = 1-int(speaker)
            
            # half a second before a bc 
            y_true.append(1)
            scores.append(p_bc[samples(backchannel-window):samples(backchannel), int(speaker)].sum().item())
    
    for speaker, backchannel_negs in events['bc_pred_neg'].items():

        for backchannel_neg in backchannel_negs:

            # half a second before 
            y_true.append(0)
            scores.append(p_bc[samples(backchannel_neg-window):samples(backchannel_neg), int(speaker)].sum().item())
    
    return y_true, scores


def get_sl(events, p_sl, window_size):

    y_true, scores = [], []
    for speaker, short_longs in events['short'].items():

        other_speaker = 1-int(speaker)

        for short_long in short_longs:
            
            # half a second before a bc 
            y_true.append(1)
            scores.append(p_sl[samples(short_long-window_size):samples(short_long), int(speaker)].sum().item())
    
    for speaker, short_longs in events['long'].items():

        other_speaker = 1-int(speaker)

        for short_long in short_longs:
            
            # half a second before 
            y_true.append(0)
            scores.append(p_sl[samples(short_long-window_size):samples(short_long), int(speaker)].sum().item())

    for speaker, short_longs in events['short_overlap'].items():

        other_speaker = 1-int(speaker)

        for short_long in short_longs:
            
            # half a second before a bc 
            y_true.append(1)
            scores.append(p_sl[samples(short_long-window_size):samples(short_long), int(speaker)].sum().item())
    
    for speaker, short_longs in events['long_overlap'].items():

        other_speaker = 1-int(speaker)

        for short_long in short_longs:
            
            # half a second before 
            y_true.append(0)
            scores.append(p_sl[samples(short_long-window_size):samples(short_long), int(speaker)].sum().item())
    
    return y_true, scores


def get_superset(events, p, window=0.1):

    # shift = 0
    # hold = 1
    if window<0:
        sign=-1
    else:
        sign=1
    window = samples(np.abs(window))*sign

    max_time = p.shape[0]/50

    y_true = []
    y_pred = []
    score = []

    shift_events = [events['shifts'], events['overlaps_shift'],  events['long']]
    events_shift_combined = {"1": [], "0": []}
    for item in shift_events:
        for speaker, times in item.items():
            events_shift_combined[speaker] += times
    events_shift_combined = {k: list(set(v)) for k, v in events_shift_combined.items()}
    for speaker, shifts in events_shift_combined.items():

        for shift in shifts:

            if shift>max_time:
                continue

            speaker = int(speaker)

            end_original, start_original = samples(shift), samples(shift) + window
            start = min(end_original, start_original)
            end = max(end_original, start_original)

            # '0' means a shift '0' --> '1'
            prev_speaker = p[start:end, speaker].sum()
            next_speaker = p[start:end, 1-speaker].sum()
            
            y_true.append(1)
            score.append(float(next_speaker))

    hold_events = [events['holds'], events['s_pred_neg'],  events['overlaps_hold']]
    events_hold_combined = {"1": [], "0": []}
    for item in hold_events:
        for speaker, times in item.items():
            events_hold_combined[speaker] += times
    events_hold_combined = {k: list(set(v)) for k, v in events_hold_combined.items()}
    for speaker, holds in events_hold_combined.items():

        for hold in holds:

            speaker = int(speaker)

            if hold>max_time:
                continue

            end_original, start_original = samples(hold), samples(hold) + window
            start = min(end_original, start_original)
            end = max(end_original, start_original)

            # '0' means a shift '0' --> '1'
            prev_speaker = p[start:end, speaker].sum()
            next_speaker = p[start:end, 1-speaker].sum()

            score.append(float(next_speaker))
            y_true.append(0)

    return y_true, score


def plot_all_rocs(y_trues, scores, corpus):

    for k, v in y_trues.items():
        for kk, vv in y_trues[k].items():
            y, s = y_trues[k][kk], scores[k][kk]
            if y[0]==[]:
                continue
            plot_roc(y,s,corpus+f'{k}_{kk}.png')


def plot_roc(y_trues, scores, path):

    # plt.figure()
    fig, ax = plt.subplots(1,2)
    i=0
    
    for y_true, score in zip(y_trues, scores):

        fpr, tpr, _ = metrics.roc_curve(y_true, score, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        ax[0].plot(fpr, tpr, lw=2, label=f'ROC curve (area = {round(roc_auc, 2)}) fold {i}')

        prescision, recall, _ = metrics.precision_recall_curve(y_true, score, pos_label=1)
        ax[1].plot(recall, prescision, lw=2, label=f'ROC curve (area = {round(roc_auc, 2)}) fold {i}')
        i+=1
    
    ax[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_ylim([0.0, 1.05])

    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])

    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')

    ax[1].set_xlabel('prescision')
    ax[1].set_ylabel('recall')
    
    ax[0].set_title(f'Receiver Operating Characteristic')
    ax[1].set_title(f'PR Curve')

    plt.legend(loc="lower right")
    plt.savefig(path)

    return tpr, fpr


def turn_taking_stats():
    """ 
        output all gap, overlap, etc durations for the corpus
    """
    pass

@torch.no_grad
def run_trained_model(state_dict, validation_csv, wav_dir, transcript_dir, out_dir):

    # run the model on all files
    validation_csv = pd.read_csv(validation_csv)
    ids = validation_csv.id.to_list()
    
    checkpoint = torch.load(state_dict, map_location ='cpu')

    cfg = yaml.safe_load(open("turn_taking/assets/config.yaml", "r"))
    model = StereoTransformerModel(cfg=cfg)
    model.load_state_dict(state_dict=checkpoint)
    model = model.to("cuda")
    model.eval()

    run_all(
        model=model,
        ids=ids,
        wav_dir=wav_dir,
        transcript_dir=transcript_dir,
        output_dir=out_dir
    )


def score_trained_model(model_name, model_probabilities_dir, json_event_dir, validation_csv):

    validation_csv = pd.read_csv(validation_csv)
    ids = validation_csv.id.to_list()

    y_true, score = get_preds(ids, model_probabilities_dir, json_event_dir)

    return y_true, score

def optimal_threshold(y_true, score, func_to_optimise, min_thresh=0.01):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, score, pos_label=1)
    ts = thresholds[::10]
    over = torch.tensor( min_thresh <= ts)
    under = torch.tensor(ts <= (1 - min_thresh))
    w = torch.where(torch.logical_and(over, under))
    # find the one that maximises the score 
    values = []
    for threshold in ts:
        y_pred = np.zeros_like(score)
        y_pred[score>threshold]=1
        values.append(func_to_optimise(y_true, y_pred))
    values = torch.tensor(values)
    _, best_idx = values.max(0)
    return ts[best_idx]


def optimal_thresholds(y_true, score, func_to_optimise):
    
    thresholds = defaultdict(float)
    for k, v in y_true.items():
        if k not in thresholds:
            thresholds[k]=defaultdict(float)
        for kk, vv in y_true[k].items():
            y,s=y_true[k][kk],score[k][kk]
            if len(y)<1:
                continue
            threshold = optimal_threshold(y,s,func_to_optimise)
            thresholds[k][kk]=threshold
    return thresholds

def get_preds(ids, model_probabilities_dir, json_event_dir):
    
    y, score = [], []

    score_parser = probabilities.VAPDecoder(bin_times=[0.2, 0.4, 0.6, 0.8])

    # output scores and predictions
    y_true = defaultdict(list)
    score = defaultdict(list)

    for id in tqdm(ids):
        
        pkl_file = os.path.join(model_probabilities_dir, f"{id}.pkl")
        event_json = os.path.join(json_event_dir, f"{id}.json")

        if not os.path.exists(event_json):
            continue

        events = json.load(open(event_json))
        try:
            vap, vad = pickle.load(open(pkl_file, "rb"))
        except Exception as e:
            continue

        # convert the 256 output into probabilities for evaluation
        p_now = score_parser.p_now(vap)
        p_future = score_parser.p_future(vap)
        p_bc = score_parser.p_bc(vap)
        p_all = score_parser.decode_probabilities(vap)

        spred = events['ekstedt_events']['s_pred_neg']
       
        for event_class, event_times in events.items():

            # skip turn stats
            if 'turn' in event_class:
                continue
            
            if event_class not in y_true:
                y_true[event_class] = defaultdict(list)
            if event_class not in score:
                score[event_class] = defaultdict(list)

            # original events ekstedt
            if event_class=='ekstedt_events':

                # p future
                y, s = get_shifts_holds(events=event_times, p=p_future, window=0.2)
                y_true[event_class]['shift_hold_p_future'] += y
                score[event_class]['shift_hold_p_future'] += s

                y, s = get_spred(events=event_times, p=p_future, window_size=0.2)
                y_true[event_class]['s_pred_p_future'] += y
                score[event_class]['s_pred_p_future'] += s

                # p now 
                y, s = get_shifts_holds(events=event_times, p=p_now, window=0.2)
                y_true[event_class]['shift_hold_p_now'] += y
                score[event_class]['shift_hold_p_now'] += s

                y, s = get_spred(events=event_times, p=p_now, window_size=0.2)
                y_true[event_class]['s_pred_p_now'] += y
                score[event_class]['s_pred_p_now'] += s

                # backchannel
                y, s = get_bc(events=event_times, p_bc=p_bc, window=0.2)
                y_true[event_class]['backchannel'] += y
                score[event_class]['backchannel'] += s

                # short long 
                y, s = get_sl(events=event_times, p_sl=p_bc, window_size=0.2)
                y_true[event_class]['short_long'] += y
                score[event_class]['short_long'] += s
            
            elif event_class=='roddy_events':

                # overlaps_shift
                # overlaps_hold

                # before
                y, s = get_overlap(event_times, p_now, window = -0.2)
                y_true[event_class]['overlaps_before_p_now'] += y
                score[event_class]['overlaps_before_p_now'] += s

                y, s = get_overlap(event_times, p_future, window= -0.2)
                y_true[event_class]['overlaps_before_p_future'] += y
                score[event_class]['overlaps_before_p_future'] += s
                
                # after
                y, s = get_overlap(event_times, p_now, window = 0.2)
                y_true[event_class]['overlaps_after_p_now'] += y
                score[event_class]['overlaps_after_p_now'] += s

                y, s = get_overlap(event_times, p_future, window = 0.2)
                y_true[event_class]['overlaps_after_p_future'] += y
                score[event_class]['overlaps_after_p_future'] += s

                # before
                y, s = get_overlap_spred(event_times, p_future, window = -0.2, spred=spred)
                y_true[event_class]['overlap_spred_before_p_future'] += y
                score[event_class]['overlap_spred_before_p_future'] += s

            elif event_class=='gap_0':

                y, s = get_shifts_holds(event_times, p_future, window=-0.2)
                y_true[event_class]['gap_0_p_future'] += y
                score[event_class]['gap_0_p_future'] += s

                y, s = get_shifts_holds(event_times, p_now, window=-0.2)
                y_true[event_class]['gap_0_p_now'] += y
                score[event_class]['gap_0_p_now'] += s

                y, s = get_spred(events=event_times, p=p_now, window_size=0.2)
                y_true[event_class]['s_pred_p_now'] += y
                score[event_class]['s_pred_p_now'] += s

                y, s = get_spred(events=event_times, p=p_future, window_size=0.2)
                y_true[event_class]['s_pred_p_future'] += y
                score[event_class]['s_pred_p_future'] += s

                y, s = get_superset(events=event_times, p=p_future, window=-0.2)
                y_true[event_class]['superset_p_future'] += y
                score[event_class]['superset_p_future'] += s

    return y_true, score


def run_trained_model_n_folds(mode, best_epoch=True):

    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    assert mode in ['SWB_GT', 'SWB_ASR', 'CND', 'CND-->SWB_ASR', 'SWB_ASR-->SWB_GT', 'SWB_GT-->SWB_ASR', 'SWB_ASR-->CND', 'combined-->SWB_ASR', 'combined-->CND']

    print("mode:: ", mode)

    if best_epoch:
        best_epoch_per_fold = pd.read_csv("/home/russelsa@ad.mee.tcd.ie/github/turn_taking/results/best_epoch_fold.csv")
    
    top_outdir=f"/data/ssd3/russelsa/model_runs/{mode}"

    if mode=="SWB_GT":
        # switchboard phonwords 
        wav_dir="turn-taking-projects/corpora/switchboard/switchboard_wav"
        transcript_dir="turn-taking-projects/corpora/switchboard/textgrids_phonwords"
        validation = "dataset_management/dataset_manager/assets/new_folds/switchboard_phonwords"
        test = "dataset_management/dataset_manager/assets/new_folds/switchboard_phonwords/test.csv"
        dir="/data/ssd1/russelsa/checkpoints/runs_VAP_switchboard_phonwords"

    if mode=="SWB_ASR":
        # switchboard asr
        wav_dir="turn-taking-projects/corpora/switchboard/switchboard_wav"
        transcript_dir="turn-taking-projects/corpora/switchboard/switchboard_speechmatics"
        validation = "dataset_management/dataset_manager/assets/new_folds/switchboard_asr"
        test = "dataset_management/dataset_manager/assets/new_folds/switchboard_asr/test.csv"
        dir="/data/ssd1/russelsa/checkpoints/runs_VAP_switchboard_asr"

    if mode=="CND":
        # candor
        wav_dir="turn-taking-projects/corpora/candor/candor_wav"
        transcript_dir="turn-taking-projects/corpora/candor/candor_speechmatics"
        validation = "/home/russelsa@ad.mee.tcd.ie/github/dataset_management/dataset_manager/assets/candor_filtered_folds/candor"
        test = "/home/russelsa@ad.mee.tcd.ie/github/dataset_management/dataset_manager/assets/candor_filtered_folds/candor/test.csv"
        dir = "/data/ssd1/russelsa/checkpoints/runs_VAP_candor"

    if mode == "CND-->SWB_ASR":
        
        # trained on...
        dir="/data/ssd1/russelsa/checkpoints/runs_VAP_candor"

        # deployed on...
        wav_dir="turn-taking-projects/corpora/switchboard/switchboard_wav"
        transcript_dir="turn-taking-projects/corpora/switchboard/switchboard_speechmatics"
        validation = "dataset_management/dataset_manager/assets/new_folds/switchboard_asr"
        test = "dataset_management/dataset_manager/assets/new_folds/switchboard_asr/test.csv"

        # mode
        mode = mode.split('-->')[1]
        
    if mode == "SWB_ASR-->SWB_GT":

        # trained on...
        dir="/data/ssd1/russelsa/checkpoints/runs_VAP_switchboard_asr"

        # deployed on...
        wav_dir="turn-taking-projects/corpora/switchboard/switchboard_wav"
        transcript_dir="turn-taking-projects/corpora/switchboard/textgrids_phonwords"
        validation = "dataset_management/dataset_manager/assets/new_folds/switchboard_phonwords"
        test = "dataset_management/dataset_manager/assets/new_folds/switchboard_phonwords/test.csv"

        # mode
        mode = mode.split('-->')[1]

    if mode == "SWB_GT-->SWB_ASR":

        # trained on...
        dir="/data/ssd1/russelsa/checkpoints/runs_VAP_switchboard_phonwords"

        # deployed on...
        wav_dir="turn-taking-projects/corpora/switchboard/switchboard_wav"
        transcript_dir="turn-taking-projects/corpora/switchboard/switchboard_speechmatics"
        validation = "dataset_management/dataset_manager/assets/new_folds/switchboard_asr"
        test = "dataset_management/dataset_manager/assets/new_folds/switchboard_asr/test.csv"

        # mode
        mode = mode.split('-->')[1]

    if mode == "SWB_ASR-->CND":

        # trained on...
        dir="/data/ssd1/russelsa/checkpoints/runs_VAP_switchboard_asr"

        # deployed on...
        wav_dir="turn-taking-projects/corpora/candor/candor_wav"
        transcript_dir="turn-taking-projects/corpora/candor/candor_speechmatics"
        # validation = "dataset_management/dataset_manager/assets/new_folds/candor"
        # test = "dataset_management/dataset_manager/assets/new_folds/candor/test.csv"
        validation = "dataset_management/dataset_manager/assets/candor_filtered_folds/candor"
        test = "dataset_management/dataset_manager/assets/candor_filtered_folds/candor/test.csv"

        # mode
        mode = mode.split('-->')[1]
        combined=True

    if mode == "combined-->SWB_ASR":

        # trained on...
        dir="/data/ssd1/russelsa/checkpoints/runs_VAP_combined_run"

        # deployed on...
        wav_dir="turn-taking-projects/corpora/switchboard/switchboard_wav"
        transcript_dir="turn-taking-projects/corpora/switchboard/switchboard_speechmatics"
        validation = "dataset_management/dataset_manager/assets/new_folds/switchboard_asr"
        test = "dataset_management/dataset_manager/assets/new_folds/switchboard_asr/test.csv"

        # mode
        mode = mode.split('-->')[1]
        combined=True

    if mode == "combined-->CND":

        # trained on...
        dir="/data/ssd1/russelsa/checkpoints/runs_VAP_combined_run"

        # deployed on...
        wav_dir="turn-taking-projects/corpora/candor/candor_wav"
        transcript_dir="turn-taking-projects/corpora/candor/candor_speechmatics"
        validation = "dataset_management/dataset_manager/assets/candor_filtered_folds/candor"
        test = "dataset_management/dataset_manager/assets/candor_filtered_folds/candor/test.csv"

        # mode
        mode = mode.split('-->')[1]
        combined=True


    # lowest loss model for this fold
    if best_epoch:
        models = []
        for fold in range(5):
            epoch = best_epoch_per_fold[best_epoch_per_fold['fold']==fold]
            if not combined:
                epoch = epoch[epoch['run']==mode]
            else:
                epoch = epoch[epoch['run']=='combined']
            epoch = epoch['min_step'].item()
            model = [m for m in os.listdir(dir) if f'fold_{fold}' in m and f'epoch_{epoch}' in m][0]
            models.append(model)

    else:
        models = []
        for fold in range(5):
            model_fold = [m for m in os.listdir(dir) if f'fold_{fold}']
            model_fold = sorted(model_fold, key=lambda x: x[-1])
            models.append(model_fold)[-1]

    for file in models:

        file = os.path.join(dir, file)

        # store this run
        out_dir_epoch=os.path.join(top_outdir, os.path.basename(file))
        
        if not os.path.exists(out_dir_epoch):
            os.mkdir(out_dir_epoch)

        # validation csv fold
        fold=os.path.basename(file).split('_')[-3]
        fold=f"{validation}/fold_{fold}/val.csv"

        # run the trained model
        run_trained_model(
            state_dict=file,
            validation_csv=fold,
            wav_dir=wav_dir,
            transcript_dir=transcript_dir,
            out_dir=out_dir_epoch
        )

        # on the test set 
        testdir = out_dir_epoch + '_test'

        # store this run
        if not os.path.exists(testdir):
            os.mkdir(testdir)

        run_trained_model(
            state_dict=file,
            validation_csv=test,
            wav_dir=wav_dir,
            transcript_dir=transcript_dir,
            out_dir=testdir
        )


def f1_score_func(y_true, y_pred):
    x = f1_score(preds=torch.tensor(y_pred), target=torch.tensor(y_true), task="multiclass", num_classes=2, average='weighted').item()
    return x

def f1_score_func_detailed(y_true, y_pred):
    weighed = f1_score(preds=torch.tensor(y_pred), target=torch.tensor(y_true), task="multiclass", num_classes=2, average='weighted').item()
    sh = f1_score(preds=torch.tensor(y_pred), target=torch.tensor(y_true), task="multiclass", num_classes=2, average=None).tolist()
    return weighed, sh[0], sh[1]



def append_defaultdicts(d,d1):
    for k,v in d1.items():
        if k not in d:
            d[k]=defaultdict(list)
        for kk,vv in v.items():
            d[k][kk].append(vv)
    return d


def merge_defaultdicts(d,d1):
    for k,v in d1.items():
        if k not in d:
            d[k]=defaultdict(list)
        for kk,vv in v.items():
            d[k][kk] += vv
    return d


def apply_thresholds(score, threshold):

    for k, v in score.items():
        for kk, vv in score[k].items():
            if len(score[k][kk])>0:
                p = score[k][kk]
                s = np.zeros_like(p)
                s[p>=threshold[k][kk]]=1
                s[p<threshold[k][kk]]=0
                score[k][kk]=s
    return score


def apply_score(y_true, y_pred, func):

    score = defaultdict(list)

    for k, v in y_true.items():
        if k not in score:
            score[k] = defaultdict(float)
        for kk, vv in y_true[k].items():
            y,p=y_true[k][kk], y_pred[k][kk]
            if len(y)<1:
                continue
            score[k][kk] = [func(y, p)]

    return score


def evaluate_model(mode):

    assert mode in ['SWB_GT', 'SWB_ASR', 'CND', 'CND-->SWB_ASR', 'SWB_ASR-->SWB_GT', 'SWB_GT-->SWB_ASR', 'SWB_ASR-->CND', 'combined-->CND', 'combined-->SWB_ASR']
    
    if mode=="SWB_GT":
        which="phonwords"
        code="20240716_144749"
        corpus="switchboard"
        turns=f"switchboard_turns_{which}"
        val=f"switchboard_{which}"

    if mode=="SWB_ASR":
        which="asr"
        code="20240716_144644"
        corpus="switchboard"
        turns=f"switchboard_turns_{which}"
        val=f"switchboard_{which}"

    if mode=="CND":
        code="20240822_105427"
        corpus="candor"
        turns="candor_turns"
        val=corpus

    if mode=="CND-->SWB_ASR":
        code="20240822_105427"
        corpus="switchboard"
        turns="switchboard_turns_asr"
        val="switchboard_asr"

    if mode=="SWB_ASR-->CND":
        code="20240716_144644"
        corpus="candor"
        turns="candor_turns"
        val=corpus

    if mode=="combined-->CND":
        code="20240912_104736"
        corpus="candor"
        turns="candor_turns"
        val=corpus

    if mode=="combined-->SWB_ASR":
        code="20240912_104736"
        corpus="switchboard"
        turns="switchboard_turns_asr"
        val="switchboard_asr"
        
    f1s=defaultdict(list)
    bal_Accs=defaultdict(list)
    
    y_trues = defaultdict(list)
    scores = defaultdict(list)
    for fold in range(5):

        model_probabilities_dir=glob.glob(f"/data/ssd3/russelsa/model_runs/{mode}/{code}_fold_{fold}_epoch*")
        model_probabilities_dir=[g for g in model_probabilities_dir if 'test' not in g][0]

        # run it on the validation fold 
        y_true, score = score_trained_model(
            model_name="asr",
            model_probabilities_dir=model_probabilities_dir,
            json_event_dir=f"turn-taking-projects/corpora/{corpus}/{turns}/",
            validation_csv=f"dataset_management/dataset_manager/assets/new_folds/{val}/fold_{fold}/val.csv"
        )

        y_trues = append_defaultdicts(y_trues, y_true)
        scores = append_defaultdicts(scores, score) 

        # score the test data
        test_csv=f"dataset_management/dataset_manager/assets/new_folds/{val}/test.csv"
        if mode=="SWB_ASR-->CND" or mode=="CND":
            test_csv=f"/home/russelsa@ad.mee.tcd.ie/github/dataset_management/dataset_manager/assets/candor_filtered_folds/candor/test.csv"
        
        y_true, y_p = score_trained_model(
            model_name="asr",
            model_probabilities_dir=model_probabilities_dir+'_test',
            json_event_dir=f"turn-taking-projects/corpora/{corpus}/{turns}/",
            validation_csv=test_csv
        )

        y_score = copy.deepcopy(y_p)

        # find the optimal threshold for each task! 
        threshold = optimal_thresholds(y_true, y_score, func_to_optimise=f1_score_func)
        y_pred = apply_thresholds(y_score, threshold)
        f1 = apply_score(y_true, y_pred, f1_score_func_detailed)

        y_score = copy.deepcopy(y_p)

        threshold = optimal_thresholds(y_true, y_score, func_to_optimise=metrics.balanced_accuracy_score)
        y_pred = apply_thresholds(y_score, threshold)
        bal_Acc = apply_score(y_true, y_pred, metrics.balanced_accuracy_score)

        bal_Accs = merge_defaultdicts(bal_Accs, bal_Acc)
        f1s = merge_defaultdicts(f1s, f1)

    plot_all_rocs(y_trues, scores, f"turn_taking/results/figures/{mode}_")
    
    with open(f"turn_taking/results/{mode}.json", "w") as f:

        results = {"f1_score": f1s, "bal_Accs": bal_Accs}
        json.dump(results, f)

    return



if __name__ == "__main__":
    
    mode = "combined-->SWB_ASR"
    # run_trained_model_n_folds(mode=mode)

    # mode = "combined-->CND"
    # run_trained_model_n_folds(mode=mode)
    evaluate_model(mode=mode)
