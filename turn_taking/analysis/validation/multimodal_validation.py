import glob
import os 
import pandas as pd
import pickle
import numpy as np
import json
import torch
from sklearn import metrics
torch.cuda.set_device(1)
from torch.utils.data import DataLoader
from dataset_management.dataset_manager.dataloader.dataloader import ValidationAudioDataset, ValidationAudioVisualDataset
from tqdm import tqdm
import copy
from turn_taking.analysis.validation import run_model
from turn_taking.analysis.validation.validation import score_trained_model, optimal_thresholds, f1_score_func, f1_score_func_detailed, apply_thresholds, apply_score, plot_all_rocs
import yaml 
from turn_taking.model.model import StereoTransformerModel, StereoTransformerModelVideoOnly
from turn_taking.model.multimodal_model import EarlyVAFusion, LateVAFusion
from turn_taking.model.cross_attention_models import SoftDocEarly, SoftDocLate
from collections import defaultdict
from turn_taking.analysis.validation.validation import merge_defaultdicts, append_defaultdicts
import re


def run_all_multimodal(model, ids, wav_dir, transcript_dir, output_dir, channelmap, video_pkl, window_size=20, step_size=19, mode='VAP'):

    if ids==[]:
        wavs = [os.path.join(wav_dir, o) for o in os.listdir(wav_dir) if '.wav' in o]
    else:
        wavs = [os.path.join(wav_dir, f"{o}.wav") for o in ids]

    progress_bar = tqdm(total=len(wavs))
    for audio_file in wavs:

        id = os.path.basename(audio_file).split('.')[0]
        
        transcript_file = os.path.join(transcript_dir, os.path.basename(audio_file).split('.')[0]+'.TextGrid')
        output_file = os.path.join(output_dir, os.path.basename(audio_file).split('.')[0]+'.pkl')

        if os.path.exists(output_file):
            continue
        
        try:
            audio_dataset = ValidationAudioVisualDataset(video_pkl_dir=video_pkl, channelmap=channelmap, audio_file=audio_file, transcript_file=transcript_file, sr=16_000, feature_sr=50, window_size=window_size, step_size=step_size, mode=mode)
        except AttributeError:
            # raised when the transcription is blank
            continue
        audio_dl = DataLoader(audio_dataset, batch_size=10, shuffle=False)

        vaps, vads = run_model.run_model(model, audio_dl, mask_vad=False, feature_extraction_hz=50, window_size=window_size, step_size=step_size, mode='VAP')

        with open(output_file, "wb") as f:

            pickle.dump([vaps, vads], f)

        progress_bar.update(1)

    progress_bar.close()


def run_fold(model, checkpoint, outdir, test_csv, val_csv):

    checkpoint = torch.load(checkpoint, map_location ='cpu')

    model.load_state_dict(state_dict=checkpoint)
    model = model.to("cuda")
    model.eval()

    val_ids = pd.read_csv(val_csv)['id'].to_list()
    test_ids = pd.read_csv(test_csv)['id'].to_list()

    # run everything... 
    run_all_multimodal(model=model, 
                       ids=val_ids, 
                       wav_dir='turn-taking-projects/corpora/candor/candor_wav', 
                       transcript_dir='turn-taking-projects/corpora/candor/candor_speechmatics', 
                       output_dir=outdir, 
                       channelmap='/mnt/storage/turn-taking-projects/corpora/candor/channelmaps.pkl', 
                       video_pkl=video_pkl, 
                       window_size=20, step_size=19, mode='VAP')
    
    run_all_multimodal(model=model, 
                    ids=test_ids, 
                    wav_dir='turn-taking-projects/corpora/candor/candor_wav', 
                    transcript_dir='turn-taking-projects/corpora/candor/candor_speechmatics', 
                    output_dir=outdir+'_test', 
                    channelmap='/mnt/storage/turn-taking-projects/corpora/candor/channelmaps.pkl', 
                    video_pkl=video_pkl, 
                    window_size=20, step_size=19, mode='VAP')


if __name__=="__main__":

    cfg = yaml.safe_load(open("/mnt/Bandon/dodder_ssd1/checkpoints/runs_VAP_candor/20240822_105427_params.yaml", "r"))

    video_pkl = '/mnt/Bandon/dodder_ssd3/candor_openface_pkl'
    # video_pkl = '/data/ssd3/russelsa/candor_openface_pkl_brief'

    run = 'CND'
    # run="CND_multimodal_late_fusion"
    # run = "CND_multimodal_early_fusion"
    # run = "CND_softdoc_early"
    # run = "CND_softdoc_late"
    # run = "CND_video_only"

    outdir = "/mnt/Bandon/dodder_ssd1/model_runs/CND"
    # outdir = '/mnt/Bandon/dodder_ssd1/model_runs/CND_early_fusion'
    # outdir = '/mnt/Bandon/dodder_ssd1/model_runs/CND_late_fusion'
    # outdir = '/data/ssd3/russelsa/model_runs/multimodal_test_early_fusion'
    # outdir = '/data/ssd3/russelsa/model_runs/softdoc_early'
    # outdir = '/data/ssd3/russelsa/model_runs/softdoc_late'
    # outdir = '/mnt/Bandon/dodder_ssd1/model_runs/CND_video_only'

    checkpoint = '/mnt/Bandon/dodder_ssd1/checkpoints/runs_VAP_candor/20240717_142924'
    # checkpoint = '/mnt/Bandon/dodder_ssd1/checkpoints/runs_VAP_candor_early_fusion/20240826_084027'
    # checkpoint = '/data/ssd1/russelsa/checkpoints/runs_VAP_candor_early_fusion/20240826_084027'
    # checkpoint = '/mnt/Bandon/dodder_ssd1/checkpoints/runs_VAP_candor_late_fusion/20240821_160422'
    # checkpoint = '/data/ssd1/russelsa/multimodal_candor_run/runs_VAP_candor_SoftDocEarly/20240903_120545'
    # checkpoint = '/data/ssd1/russelsa/multimodal_candor_run/runs_VAP_candor_SoftDocLate/20240903_115618'
    # checkpoint = '/mnt/Bandon/dodder_ssd1/checkpoints/runs_VAP_candor_video_only/20241023_164016'
    
    model = StereoTransformerModel(cfg=cfg)
    # model = EarlyVAFusion(cfg=cfg)  
    # model = LateVAFusion(cfg=cfg)
    # model = SoftDocEarly(cfg=cfg)
    # model = SoftDocLate(cfg=cfg)
    model = StereoTransformerModelVideoOnly(cfg=cfg)
    
    test_csv = "dataset_management/dataset_manager/assets/new_folds/candor/test"
    json_event_dir = "/mnt/Bandon/dodder_home_dir/turn-taking-projects/corpora/candor/candor_turns"

    best_epoch = pd.read_csv("turn_taking/results/best_epoch_fold.csv")
    best_epoch = best_epoch[best_epoch['run']==run]

    checkpoints = []
    for fold in range(5):
        
        best = best_epoch[best_epoch.fold==fold]['Step'].item()
        checkpoints.append(checkpoint+f'_fold_{fold}_epoch_{best}')

    for checkpoint in checkpoints:
        
        fold = re.findall(r'fold_[0-9]', checkpoint)[0].split('_')[-1]
        output_dir = os.path.join(outdir, os.path.basename(checkpoint))

        if not os.path.exists(output_dir):
            # print(output_dir)
            os.mkdir(output_dir)
        if not os.path.exists(output_dir + "_test"):
            # print(output_dir+'_test')
            os.mkdir(output_dir + "_test")
        
        # validation_csv=f"dataset_management/dataset_manager/assets/new_folds/candor/fold_{fold}/val"
        # run_fold(model, checkpoint, output_dir, test_csv+'.csv', validation_csv+'.csv')
    # exit()
    f1s=defaultdict(list)
    bal_Accs=defaultdict(list)
    
    y_trues = defaultdict(list)
    scores = defaultdict(list)
    for fold in range(5):

        model_probabilities_dir=glob.glob(os.path.join(outdir, f"*_fold_{fold}_epoch*"))
        model_probabilities_dir=[g for g in model_probabilities_dir if '_test' not in g.split('/')[-1]][0]

        # run it on the validation fold 
        y_true, score = score_trained_model(
            model_name="asr",
            model_probabilities_dir=model_probabilities_dir,
            json_event_dir=json_event_dir,
            validation_csv=f"dataset_management/dataset_manager/assets/new_folds/candor/fold_{fold}/val.csv"
        )

        y_trues = append_defaultdicts(y_trues, y_true)
        scores = append_defaultdicts(scores, score) 

        # find the optimal threshold for each task! 
        threshold = optimal_thresholds(y_true, score, func_to_optimise=f1_score_func)

        # score the test data
        y_true, y_p = score_trained_model(
            model_name="asr",
            model_probabilities_dir=model_probabilities_dir+'_test',
            json_event_dir=json_event_dir,
            validation_csv=test_csv+'.csv'
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

    plot_all_rocs(y_trues, scores, f"turn_taking/results/figures/{run}_")
    
    with open(f"turn_taking/results/{run}.json", "w") as f:

        results = {"f1_score": f1s, "bal_Accs": bal_Accs}
        json.dump(results, f)
