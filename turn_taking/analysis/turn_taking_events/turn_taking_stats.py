import pickle
import os
import json
import numpy as np


def median_iqr(x):
    median = np.median(x)
    q75, q25 = np.percentile(x, [75 ,25])
    print(len(x), median, q25, q75)


def analyse_turn_taking_stats(directory):

    jsons = os.listdir(directory)
    print(f"num sessions: {len(jsons)}")

    # number of turns per speaker
    turns_per_speaker = []
    gap_duration = []
    overlap_duration = []
    ftos = []
    turn_duration = []

    for jsonfile in jsons:
        jsonfile = json.load(open(os.path.join(directory, jsonfile), "r"))

        s0_num_turns = len(jsonfile['turns']['0'])
        turns_per_speaker.append(s0_num_turns)

        s1_num_turns = len(jsonfile['turns']['1'])
        turns_per_speaker.append(s1_num_turns)

        # gaps 
        if 'gaps' in jsonfile['turn-taking-events']:
            gaps = jsonfile['turn-taking-events']['gaps']
            gap_lengths = [g['end']-g['start'] for g in gaps]
            gap_duration += gap_lengths

        # overlaps 
        if 'overlaps' in jsonfile['turn-taking-events']:
            overlaps = jsonfile['turn-taking-events']['overlaps']
            overlap_lengths = [g['end']-g['start'] for g in overlaps]
            overlap_duration += overlap_lengths

        # turn duration
        turns_0 = jsonfile['turns']['0']
        turns_0_lengths = [g['end']-g['start'] for g in turns_0]
        turn_duration += turns_0_lengths

        turns_1 = jsonfile['turns']['1']
        turns_1_lengths = [g['end']-g['start'] for g in turns_1]
        turn_duration += turns_1_lengths

        # fto 
        ftos += gap_lengths
        ftos += [o*-1 for o in overlap_lengths]
    
    print(np.sum(turns_per_speaker))
    median_iqr(gap_duration)
    median_iqr(overlap_duration)
    median_iqr(ftos)
    median_iqr(turn_duration)

    pickle.dump({
        "ftos": ftos
    }, open(os.path.basename(directory)+'.pkl', "wb"))


if __name__ == "__main__":
    # directory = "turn-taking-projects/corpora/candor/candor_turns"
    directory = "turn-taking-projects/corpora/switchboard/switchboard_turns_phonwords"
    # directory = "turn-taking-projects/corpora/switchboard/switchboard_turns_asr"

    analyse_turn_taking_stats(directory)
