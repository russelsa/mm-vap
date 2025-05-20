import numpy as np
import os
from collections import defaultdict
import json


def events_in_dialogue(json_file):

    all_events = json.load(open(json_file))

    counter = defaultdict(list)

    for category, events in all_events.items():

        if category=='turns' or category=='turn-taking-events':
            continue

        for event, speaker_events in events.items():
        
            for speaker, times in speaker_events.items():
                if category not in counter:
                    counter[category] = defaultdict(list)

                if event in counter[category]:
                    counter[category][event] = [counter[category][event][0] + len(times)]
                else:
                    counter[category][event] = [len(times)]

    return counter

def events_duration_in_dialogue(json_file):

    all_events = json.load(open(json_file))

    durations = defaultdict(list)

    events =  all_events['turn-taking-events'].keys()

    for event in events:
        durations[event] = all_events['turn-taking-events'][event]

    durations['turns'] = all_events['turns']['0'] + all_events['turns']['1']

    for k, v in durations.items():
        durations[k] = [vv['end']-vv['start'] for vv in v]

    return durations


def append_default_dicts(a,b):
    res = defaultdict(list)
    for k, v in a.items():
        res[k] = v
    for k,v in b.items():
        res[k] = res[k] + v
    return res

def append_default_dicts_2(a,b):
    res = defaultdict()
    for k, v in a.items():
        if k not in res:
            res[k] = defaultdict()
        for kk, vv in v.items():
            if kk in res[k]:
                res[k][kk] = vv
            else:
                res[k][kk] = vv
    for k,v in b.items():
        for kk, vv in v.items():
            if kk in res[k]:
                res[k][kk] = res[k][kk] + vv
            else:
                res[k][kk] = vv
    return res

def count_all_events(directory):

    event_stats = defaultdict(list)
    jsonfiles = [os.path.join(directory, j) for j in os.listdir(directory) if '.json' in j]

    durations = None
    counts = None

    for jsonfile in jsonfiles:
        
        # number of events in this dialogue
        if counts is None:
            counts = events_in_dialogue(jsonfile)
        else:
            counts = append_default_dicts_2(counts, events_in_dialogue(jsonfile))

        if durations is None:
            durations = events_duration_in_dialogue(jsonfile)
        else:
            durations = append_default_dicts(durations, events_duration_in_dialogue(jsonfile))

    for k,v in counts.items():
        for kk,vv in counts[k].items():
            vv=np.array(vv)
            print(k, kk, np.sum(vv), np.mean(vv), np.std(vv))

    for k, v in durations.items():
        v = np.array(v)
        print(k, np.sum(v), np.mean(v), np.std(v), np.median(v), np.quantile(v, 0.25), np.quantile(v, 0.75))

    return


if __name__=="__main__":

    # counts = count_all_events("turn-taking-projects/corpora/switchboard/switchboard_turns_phonwords_with_timings")
    counts = count_all_events("turn-taking-projects/corpora/switchboard/switchboard_turns_asr_with_timings")
    # counts = count_all_events("turn-taking-projects/corpora/candor/candor_turns_with_timings")

    x=1