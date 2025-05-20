from dataset_management.dataset_manager.scripts.transcript_processing import vad_list_to_one_hot, vads_from_transcript, vad_list_to_one_hot, vad_list_to_turns, interpausal_units
import numpy as np
import textgrid
from collections import defaultdict
import os 
import json
import tqdm
from dataset_management.dataset_manager.src.turn_event_manager import TurnEvents, TurnEventsModifier
np.random.seed(0)

def get_gaps_and_overlaps2(vad_list, maxlen=-1):
    """
    https://github.com/dopefishh/pympi/blob/master/pympi/Elan.py
    ad = list((start, stop)) vad segments
    Faster variant of :func:`get_gaps_and_overlaps`. Faster in this case
    means almost 100 times faster...

    :param str tier1: Name of the first tier.
    :param str tier2: Name of the second tier.
    :param int maxlen: Maximum length of gaps (skip longer ones), if ``-1``
                       no maximum will be used.
    :yields: Tuples of the form ``[(start, end, type)]``.
    :raises KeyError: If a tier is non existent.
        +-----+---------------------------------------------+
        | id  | Description                                 |
        +=====+=============================================+
        | O12 | Overlap from tier1 to tier2                 |
        +-----+---------------------------------------------+
        | O21 | Overlap from tier2 to tier1                 |
        +-----+---------------------------------------------+
        | G12 | Between speaker gap from tier1 to tier2     |
        +-----+---------------------------------------------+
        | G21 | Between speaker gap from tier2 to tier1     |
        +-----+---------------------------------------------+
        | W12 | Within speaker overlap from tier2 in tier1  |
        +-----+---------------------------------------------+
        | W21 | Within speaker overlap from tier1 in tier2  |
        +-----+---------------------------------------------+
        | P1  | Pause for tier1                             |
        +-----+---------------------------------------------+
        | P2  | Pause for tier2                             |
        +-----+---------------------------------------------+
    """
    
    ad = sorted(((a, i + 1) for i, t in enumerate(vad_list) for a in t), reverse=True)

    if ad:
        last = (lambda x: (x[0][0], x[0][1], x[1]))(ad.pop())

        def thr(x, y):
            return maxlen == -1 or abs(x-y) < maxlen
        while ad:
            (begin, end), current = ad.pop()
            if last[2] == current and thr(begin, last[1]) and last[1] != begin:
                yield (last[1], begin, 'P{}'.format(current))
            elif last[0] < begin and last[1] > end:
                yield (begin, end, 'W{}{}'.format(last[2], current))
                continue
            elif last[1] > begin:
                yield (begin, last[1], 'O{}{}'.format(last[2], current))
            elif last[1] < begin and thr(begin, last[1]):
                yield (last[1], begin, 'G{}{}'.format(last[2], current))
            last = (begin, end, current)


def get_turn_taking_events(vad_list, return_seconds=False):
    """
        get the pauses gaps and overlaps in a dialogue from a list of VAD events
        vad: [[[start,end]....[start,end]],  [[start,end]....[start,end]]]
    """
    events = get_gaps_and_overlaps2(vad_list)
    events = [e for e in events]
    
    if return_seconds:
        def seconds(x):
            return round(x/16_000, 2)
        events = [(seconds(x[0]), seconds(x[1]), x[2]) for x in events]

    return events



def get_track_shape_from_tg(textgrid_file, mono, sr=16_000):
    tg = textgrid.TextGrid.fromFile(textgrid_file)
    time = tg.maxTime
    samples = int(time*sr)
    if mono:
        return [1, samples]
    return [2, samples]


def get_events_from_tg(transcript):

    shape = get_track_shape_from_tg(transcript, mono=False, sr=16_000)
    vad_list = vads_from_transcript(transcript, samples=True)

    vad_list_merge = [[],[]]

    vad_list_merge[0] = interpausal_units(vad_list[0], max_silence_between=0.2)
    vad_list_merge[1] = interpausal_units(vad_list[1], max_silence_between=0.2)

    events_list = get_turn_taking_events(vad_list_merge, return_seconds=False)
    vad_one_hot = vad_list_to_one_hot(vad_list_merge, track_size=shape, samples=True)
    turns = vad_list_to_turns(vad_list_merge, vad_one_hot, samples=True, sr=16_000) 

    events = {}

    # ------------------------------------ events from prior work -----------------------------------------------------  #

    # original events from the vap paper 
    turn_manager = TurnEvents()
    turn_manager.overlap = None
    turn_manager.overlap = None
    turn_manager.short_long = None
    events['ekstedt_events'] = turn_manager.get_events(events_list, vad_list_merge, vad_one_hot, return_seconds=True)

    # matt roddy overlap 
    turn_manager = TurnEvents()
    turn_manager.shift_hold_config = None
    turn_manager.s_pred_config = None
    turn_manager.backchannel_pred = None
    turn_manager.short_long = None
    events['roddy_events'] = turn_manager.get_events(events_list, vad_list_merge, vad_one_hot, return_seconds=True)

    # ------------------------------------ // events from prior work ---------------------------------------------------  #


    # ------------------------------------ newly introduced events -----------------------------------------------------  #
    
    # with no minimum gap duration for a shift/hold event
    turn_manager_gap = TurnEventsModifier(gap_duration=0)
    turn_manager_gap.backchannel_pred = None
    turn_manager_gap.short_long = None
    events['gap_0'] = turn_manager_gap.get_events(events_list, vad_list_merge, vad_one_hot, return_seconds=True)

    # with no minimum gap duration for a shift/hold event
    turn_manager_gap = TurnEventsModifier(gap_duration=0.500)
    turn_manager_gap.backchannel_pred = None
    turn_manager_gap.short_long = None
    events['gap_500'] = turn_manager_gap.get_events(events_list, vad_list_merge, vad_one_hot, return_seconds=True)

    # with no minimum gap duration for a shift/hold event
    turn_manager_gap = TurnEventsModifier(gap_duration=0.750)
    turn_manager_gap.backchannel_pred = None
    turn_manager_gap.short_long = None
    events['gap_750'] = turn_manager_gap.get_events(events_list, vad_list_merge, vad_one_hot, return_seconds=True)

    # with no minimum gap duration for a shift/hold event
    turn_manager_gap = TurnEventsModifier(gap_duration=1.000)
    turn_manager_gap.backchannel_pred = None
    turn_manager_gap.short_long = None
    events['gap_1000'] = turn_manager_gap.get_events(events_list, vad_list_merge, vad_one_hot, return_seconds=True)

    # with no minimum gap duration for a shift/hold event
    turn_manager_gap = TurnEventsModifier(gap_duration=1.250)
    turn_manager_gap.backchannel_pred = None
    turn_manager_gap.short_long = None
    events['gap_1250'] = turn_manager_gap.get_events(events_list, vad_list_merge, vad_one_hot, return_seconds=True)

    # with no minimum gap duration for a shift/hold event
    turn_manager_gap = TurnEventsModifier(gap_duration=1.500)
    turn_manager_gap.backchannel_pred = None
    turn_manager_gap.short_long = None
    events['gap_1500'] = turn_manager_gap.get_events(events_list, vad_list_merge, vad_one_hot, return_seconds=True)

    # with no minimum gap duration for a shift/hold event
    turn_manager_gap = TurnEventsModifier(gap_duration=2.000)
    turn_manager_gap.backchannel_pred = None
    turn_manager_gap.short_long = None
    events['gap_2000'] = turn_manager_gap.get_events(events_list, vad_list_merge, vad_one_hot, return_seconds=True)

    # with no minimum gap duration for a shift/hold event
    turn_manager_gap = TurnEventsModifier(gap_duration=2.500)
    turn_manager_gap.backchannel_pred = None
    turn_manager_gap.short_long = None
    events['gap_2500'] = turn_manager_gap.get_events(events_list, vad_list_merge, vad_one_hot, return_seconds=True)

    # with no minimum gap duration for a shift/hold event
    turn_manager_gap = TurnEventsModifier(gap_duration=5.000)
    turn_manager_gap.backchannel_pred = None
    turn_manager_gap.short_long = None
    events['gap_5000'] = turn_manager_gap.get_events(events_list, vad_list_merge, vad_one_hot, return_seconds=True)

    # faster == there should be one person only in the near future (600ms)
    # faster turn shifts
    turn_manager_fast_shift = TurnEventsModifier(gap_duration=0, sh_pre_onset=0.5, sh_post_offset=0.6, overlap_pre_onset=0, overlap_post_offset=0.6)
    turn_manager_fast_shift.shift_hold_config = None
    turn_manager_fast_shift.s_pred_config = None
    turn_manager_fast_shift.backchannel_pred = None
    turn_manager_fast_shift.short_long = None
    events['fast_shift'] = turn_manager_fast_shift.get_events(events_list, vad_list_merge, vad_one_hot, return_seconds=True)

    # slow == there should be one person only in the near future (600ms)
    # slow turn shifts
    turn_manager_slow_shift = TurnEventsModifier(gap_duration=0, sh_pre_onset=0.5, sh_post_offset=2, overlap_pre_onset=0, overlap_post_offset=2)
    turn_manager_slow_shift.shift_hold_config = None
    turn_manager_slow_shift.s_pred_config = None
    turn_manager_slow_shift.backchannel_pred = None
    turn_manager_slow_shift.short_long = None
    events['slow_shift'] = turn_manager_gap.get_events(events_list, vad_list_merge, vad_one_hot, return_seconds=True)

    # short and long to include overlapping segments
    turn_manager_sl = TurnEventsModifier(gap_duration=0, overlap_pre_onset=0, overlap_post_offset=0.6, overlap_max_duration=1)
    turn_manager_sl.shift_hold_config = None
    turn_manager_sl.s_pred_config = None
    turn_manager_sl.backchannel_pred = None
    events['short_long'] = turn_manager_sl.get_events(events_list, vad_list_merge, vad_one_hot, return_seconds=True)
    events['short_long']['overlaps_shift'] = {}
    events['short_long']['overlaps_hold'] = {}
    events['short_long']['short'] = events['ekstedt_events']['short']
    events['short_long']['long'] = events['ekstedt_events']['long']

    # turn taking statistics, gap overlap etc duration
    events = parse_gaps_overlaps(events, events_list, sr=16_000)
    events['turns'] = turns

    return events


def parse_gaps_overlaps(events, events_raw, sr):
    events_parsed = defaultdict(list)
    for event in events_raw:
        start, end = round(event[0]/sr, 2), round(event[1]/sr, 2)
        if event[2][0]=='G':
            events_parsed['gaps'].append({"start": start, "end": end})
        elif event[2][0]=='O':
            events_parsed['overlaps'].append({"start": start, "end": end})
        elif event[2][0]=='P':
            events_parsed['pauses'].append({"start": start, "end": end})
    events['turn-taking-events'] = events_parsed
    return events
    

def candor():

    output_dir = "turn-taking-projects/corpora/candor/candor_turns_with_timings/"
    candor_dir = "turn-taking-projects/corpora/candor/candor_speechmatics"
    textgrid_files = [os.path.join(candor_dir, t) for t in os.listdir(candor_dir) if '.TextGrid' in t]
    for textgrid_file in tqdm.tqdm(textgrid_files):

        output_file = os.path.join(output_dir, os.path.basename(textgrid_file).split('.')[0]+'.json')
        if os.path.exists(output_file):
            continue

        id = os.path.basename(textgrid_file).split('.')[0]

        try:
            events = get_events_from_tg(textgrid_file)
        except Exception as e:
            print(f"error with {textgrid_file}")
            continue
        json.dump(events, open(output_file, 'w'), indent=5)


def switchboard():

    # output_dir = "turn-taking-projects/corpora/switchboard/switchboard_turns_phonwords_with_timings/"
    # input_dir = "turn-taking-projects/corpora/switchboard/textgrids_phonwords"
    # textgrid_files = [os.path.join(input_dir, t) for t in os.listdir(input_dir) if '.TextGrid' in t]
    # for textgrid_file in tqdm.tqdm(textgrid_files):
    #     output_file = os.path.join(output_dir, os.path.basename(textgrid_file).split('.')[0]+'.json')
    #     if os.path.exists(output_file):
    #         continue
    #     events = get_events_from_tg(textgrid_file)
    #     json.dump(events, open(output_file, 'w'), indent=5)

    output_dir = "turn-taking-projects/corpora/switchboard/switchboard_turns_asr_with_timings/"
    input_dir = "turn-taking-projects/corpora/switchboard/switchboard_speechmatics"
    textgrid_files = [os.path.join(input_dir, t) for t in os.listdir(input_dir) if '.TextGrid' in t]
    for textgrid_file in tqdm.tqdm(textgrid_files):
        events = get_events_from_tg(textgrid_file)
        output_file = os.path.join(output_dir, os.path.basename(textgrid_file).split('.')[0]+'.json')
        json.dump(events, open(output_file, 'w'), indent=5)


def single():
    output_dir = "turn-taking-projects/corpora/switchboard/switchboard_turns_phonwords/"
    textgrid_file = "turn-taking-projects/corpora/switchboard/textgrids_phonwords/sw2001.TextGrid"
    events = get_events_from_tg(textgrid_file)
    output_file = os.path.join(output_dir, os.path.basename(textgrid_file).split('.')[0]+'.json')
    json.dump(events, open(output_file, 'w'), indent=5)

    output_dir = "turn-taking-projects/corpora/switchboard/switchboard_turns_asr/"
    textgrid_file = "turn-taking-projects/corpora/switchboard/switchboard_speechmatics/sw2001.TextGrid"
    events = get_events_from_tg(textgrid_file)
    output_file = os.path.join(output_dir, os.path.basename(textgrid_file).split('.')[0]+'.json')
    json.dump(events, open(output_file, 'w'), indent=5)

    output_dir = "turn-taking-projects/corpora/candor/candor_turns/"
    textgrid_file = "turn-taking-projects/corpora/candor/candor_speechmatics/0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a.TextGrid"
    events = get_events_from_tg(textgrid_file)
    output_file = os.path.join(output_dir, os.path.basename(textgrid_file).split('.')[0]+'.json')
    json.dump(events, open(output_file, 'w'), indent=5)


if __name__ == "__main__":
    # single()
    switchboard()
    # candor()
    