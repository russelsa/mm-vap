import pickle
from dataset_management.dataset_manager.dataloader.dataloader import GeMAPSDataset, ChunkedDataset
import os
import json
from collections import defaultdict
from typing import List
from tqdm import tqdm
import numpy as np
import pandas as pd


class GeMAPSAnalysis():
    """Analyse features before and after turn-taking events
    """

    def __init__(self, json_dir: str, dataset: ChunkedDataset, feature_names: List[str], output_file: str):

        MAX=-1
        self.max_event_times = 20
        self.json_files = [os.path.join(json_dir, o) for o in os.listdir(json_dir) if '.json' in o][0:MAX]
        self.dataset = dataset
        self.before_window = 0.5
        self.after_window = 0.5
        self.output_file = output_file
        self.feature_names = feature_names
        self.events_record = self.analyse_events()
        self.events_record.to_csv(output_file)


    def analyse_events(self):
        
        pbar = tqdm(total=len(self.json_files))
        for i, json_file in enumerate(self.json_files):
            
            events = json.load(open(json_file, 'r'))
            id = os.path.basename(json_file).split('.')[0]

            events_record = []

            for speaker in ['1', '0']:

                for turn_type in events.keys():
                    
                    try:
                        event_times = events[turn_type][speaker]
                    except KeyError:
                        continue
                
                    for event_time in event_times[0:min(len(event_times),self.max_event_times)]:
                        
                        # probably have to ensure doesnt exceed duration

                        # before and after
                        try:
                            gemaps_before = self.dataset.get_gemaps_chunk((id, event_time, event_time+self.before_window))
                            gemaps_after = self.dataset.get_gemaps_chunk((id, event_time, event_time+self.after_window))
                        except FileNotFoundError:
                            continue

                        # before speaker
                        before_speaker = gemaps_before[int(speaker)]

                        # before listener (silent?)
                        before_listener = gemaps_before[int(not int(speaker))]

                        # after speaker (silent?)
                        after_speaker = gemaps_after[int(speaker)]

                        # after listener 
                        after_listener = gemaps_after[int(not int(speaker))]

                        for gemap, value in zip(self.feature_names, range(len(self.feature_names))):

                            num_pts = before_speaker[:, value].shape[0]

                            # before_speaker
                            events_record.append(
                                {
                                    "turn_type": turn_type,
                                    "window": "before",
                                    "who": "speaker",
                                    "gemap": gemap,
                                    "mean_value": before_speaker[:, value].mean(),
                                    "max_value": before_speaker[:, value].max(),
                                    "min_value": before_speaker[:, value].min(),
                                    "slope": np.polyfit(np.linspace(0,num_pts,num_pts), before_speaker[:, value], 1)[0]
                                }
                            )

                            # after speaker 
                            events_record.append(
                                {
                                    "turn_type": turn_type,
                                    "window": "after",
                                    "who": "speaker",
                                    "gemap": gemap,
                                    "mean_value": after_speaker[:, value].mean(),
                                    "max_value": after_speaker[:, value].max(),
                                    "min_value": after_speaker[:, value].min(),
                                    "slope": np.polyfit(np.linspace(0,num_pts,num_pts), after_speaker[:, value], 1)[0]
                                }
                            )

                            # before listener
                            events_record.append(
                                {
                                    "turn_type": turn_type,
                                    "window": "before",
                                    "who": "listener",
                                    "gemap": gemap,
                                    "mean_value": before_listener[:, value].mean(),
                                    "max_value": before_listener[:, value].max(),
                                    "min_value": before_listener[:, value].min(),
                                    "slope": np.polyfit(np.linspace(0,num_pts,num_pts), before_listener[:, value], 1)[0]
                                }
                            )

                            # after listener
                            events_record.append(
                                {
                                    "turn_type": turn_type,
                                    "window": "after",
                                    "who": "listener",
                                    "gemap": gemap,
                                    "mean_value": after_listener[:, value].mean(),
                                    "max_value": after_listener[:, value].max(),
                                    "min_value": after_listener[:, value].min(),
                                    "slope": np.polyfit(np.linspace(0,num_pts,num_pts), after_listener[:, value], 1)[0]
                                }
                            )

            pbar.update(1)

            if i%20==0:
                events_record = pd.DataFrame(events_record)
                events_record.to_csv(self.output_file)

        events_record = pd.DataFrame(events_record)
        return events_record



def switchboard():

    pickle_file = "dataset_management/dataset_manager/assets/folds/switchboard/fold_0/train.pkl"
    gemaps_dir = "/mnt/storage/turn-taking-projects/corpora/switchboard/switchboard_gemaps"
    json_dir = "turn-taking-projects/corpora/switchboard/switchboard_turns"

    output_file = "/mnt/storage/turn-taking-projects/corpora/switchboard/switchboard_turn_analysis/gemaps.csv"
    
    gemaps_file = "/mnt/storage/turn-taking-projects/corpora/switchboard/switchboard_gemaps/sw2001_1.pkl"
    feature_names, _ = pickle.load(open(gemaps_file, 'rb'))
    
    MAX=-1
    dataset = GeMAPSDataset(pickle_file=pickle_file, gemaps_dir=gemaps_dir, normalize_gemaps=True)
    analysis = GeMAPSAnalysis(json_dir=json_dir, dataset=dataset, feature_names=feature_names, output_file=output_file)


def candor():
    pickle_file = "dataset_management/dataset_manager/assets/folds/candor/fold_0/train.pkl"
    gemaps_dir = "/mnt/storage/turn-taking-projects/corpora/candor/candor_gemaps"
    json_dir = "turn-taking-projects/corpora/candor/candor_turns"

    output_file = "/mnt/storage/turn-taking-projects/corpora/candor/candor_turn_analysis/gemaps.csv"
    
    gemaps_file = "/mnt/storage/turn-taking-projects/corpora/switchboard/switchboard_gemaps/sw2001_1.pkl"
    feature_names, _ = pickle.load(open(gemaps_file, 'rb'))
    
    MAX=-1
    dataset = GeMAPSDataset(pickle_file=pickle_file, gemaps_dir=gemaps_dir, normalize_gemaps=True)
    analysis = GeMAPSAnalysis(json_dir=json_dir, dataset=dataset, feature_names=feature_names, output_file=output_file)


if __name__=="__main__":
    candor()
    # switchboard()
