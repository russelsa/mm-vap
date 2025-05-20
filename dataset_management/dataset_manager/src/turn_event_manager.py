import numpy as np
import textgrid
from collections import defaultdict
import os 
import json
import tqdm



def is_pause(e):
    if e[2][0] == 'P':
        return True
    return False


def is_gap(e):
    if e[2][0] == 'G':
        return True
    return False


def is_overlap(e):
    if e[2][0] == 'O':
        return True
    return False

def is_overlap_within(e):
    if e[2][0] == 'W':
        return True
    return False


def seconds(x):
    return round(x/16_000, 2)


def samples(x):
    return int(x*16_000)


class TurnEvents():
    
    def __init__(self):

        # criteria in the VAP interspeech paper
        self.shift_hold_config = {
            'min_length': 0.25,
            'max_length': 2,
            'pre_onset': 1,
            'post_offset': 1
        }

        self.s_pred_config = {
            "range_before_onset": 0.5,
            "neg_future_range": 2
        }

        self.backchannel_pred = {
            "max_bc_length": 1,
            "pre_silence": 1,
            "post_silence": 2,
            "pre_other_speaker": 1,
            "range_before_onset": 0.5,
            "neg_future_range": 2
        }

        # matt roddy work
        self.overlap = {
            "min_overlap_duration": 0.2,
            "pre_onset": 1.5,
            "post_offset": 1,
            "overlap_max_duration": 1
        }

        # short long with complete turns
        self.short_long = {}

    def get_events(self, events, vad_list, vad_one_hot, return_seconds=False):
        """
            the original VAP paper 
        """

        if vad_one_hot.shape[0] != 2:
            vad_one_hot = np.stack((vad_one_hot[:,0], vad_one_hot[:,1]),axis=0)


        shifts = defaultdict(list)
        holds = defaultdict(list)
        overlaps = defaultdict(list)
        overlaps_hold = defaultdict(list)
        overlaps_endtime = defaultdict(list)
        overlaps_hold_endtime = defaultdict(list)
        shorts = defaultdict(list)
        longs = defaultdict(list)

        for event in events:
            
            # shift / hold (roddy and ekstedt)
            # pauses and gaps within the dialogue of a certain length 
            # post-onset and pre-onset periods, minimum times befpre and after 
            # where only a single speaker can be active in the dialogue 
            # roddy: pause-50 and pause-500 lengths 
            # evaluation frames start 50ms within the silence and cover 100ms total (ekstedt)
            if (is_pause(event) or is_gap(event)) and self.shift_hold_config is not None:
                
                # check the pause meets the minimum / maximum duration threshold
                duration = event[1]-event[0]
                dur = seconds(duration)
                if dur < self.shift_hold_config['min_length'] or dur > self.shift_hold_config['max_length']:
                    continue
                
                start = event[0]
                stop = event[1]

                # check that there's no change in speakers in this region
                onset_start = max(0, start - samples(self.shift_hold_config['pre_onset']))
                offset_stop = min(stop + samples(self.shift_hold_config['post_offset']), vad_one_hot.shape[1])

                # who is speaking
                s1 = vad_one_hot[:, onset_start : start].sum(axis=1)
                s2 = vad_one_hot[:, stop : offset_stop].sum(axis=1)

                # before check
                before_check = s1[0] > 1 and s1[1] > 1
                after_check =  s2[0] > 1 and s2[1] > 1

                # exclude if there's more than 1 person speaking before or after 
                if before_check or after_check:
                    continue
                    
                else:
                
                    if is_gap(event):
                        shifts[int(event[2][1])-1].append(start)
                    
                    elif is_pause(event):
                        holds[int(event[2][1])-1].append(start)

            if is_overlap(event) and (self.overlap is not None or self.short_long is not None): 

                start = event[0]
                end = event[1]

                start_s, end_s = seconds(start), seconds(end)

                # needs to be at least 100 ms overlap
                duration = seconds(end-start)
                if duration < self.overlap['min_overlap_duration']:
                    continue

                prev_speaker = int(event[-1][1])-1
                next_speaker = int(event[-1][2])-1

                # check that there's been a minimum amount of speech by the previous speaker
                onset_start = max(0, start - samples(self.overlap['pre_onset']))
                offset_stop = min(end + samples(self.overlap['post_offset']), vad_one_hot.shape[1])

                # previous speaker should have been speaking for a minimum period of time 
                psc=vad_one_hot[prev_speaker, onset_start : start]
                prev_speaker_check = (psc==0).any()
                
                # only one person can speak in the next time
                s1 = vad_one_hot[:, onset_start : start].sum(axis=1)
                s2 = vad_one_hot[:, end : offset_stop].sum(axis=1)

                # before check
                before_check = s1[0] > 1 and s1[1] > 1
                after_check =  s2[0] > 1 and s2[1] > 1

                if prev_speaker_check or after_check:
                    continue
                
                # overlap is either a hold or a shift?
                if self.overlap is not None:
                    overlaps[prev_speaker].append(start)
                    overlaps_endtime[prev_speaker].append(end)
                if self.short_long is not None:
                    shorts[prev_speaker].append(start)

            elif is_overlap_within(event) and (self.overlap is not None or self.short_long is not None):

                start = event[0]
                end = event[1]

                start_s, end_s = seconds(start), seconds(end)

                # needs to be at least 100 ms overlap
                duration = seconds(end-start)
                if duration > self.overlap['overlap_max_duration']:
                    continue

                prev_speaker = int(event[-1][1])-1
                next_speaker = int(event[-1][2])-1

                # check that there's been a minimum amount of speech by the previous speaker
                onset_start = max(0, start - samples(self.overlap['pre_onset']))
                offset_stop = min(end + samples(self.overlap['post_offset']), vad_one_hot.shape[1])

                # previous speaker should have been speaking for a minimum period of time 
                psc=vad_one_hot[prev_speaker, onset_start : start]
                prev_speaker_check = (psc==0).any()
                
                # only one person can speak in the next time
                s1 = vad_one_hot[:, onset_start : start].sum(axis=1)
                s2 = vad_one_hot[:, end : offset_stop].sum(axis=1)

                # before check
                before_check = s1[0] > 1 and s1[1] > 1
                after_check =  s2[0] > 1 and s2[1] > 1

                if prev_speaker_check or after_check:
                    continue
                
                if self.overlap is not None:
                    overlaps_hold[prev_speaker].append(start)
                    overlaps_hold_endtime[prev_speaker].append(end)
                if self.short_long is not None:
                    longs[prev_speaker].append(start)

        # shift prediction (ekstedt) and negative samples
        # roddy calls it prediction at overlap
        # how well the model can continuously predict upcoming shifts
        # while a current speaker is still active
        # postitive: consider a range 500ms towards the end of a VA segment, before a shift occurs. one person only should be speaking at all times
        # negative samples: regions where a single speaker is active but far from the future activity of a single speaker (2s)

        # sample an equal number of negative samples
                
        # all regions with a single active speaker (from both!)
        single_active_speaker = np.where(vad_one_hot.sum(axis=0) == 1)[0]

        num_0 = 0
        num_1 = 0

        s_pred_neg = defaultdict(list)
        while (num_0<=len(shifts[0])+1) and (num_1<=len(shifts[1])+1) and self.s_pred_config is not None:
            
            # random start point
            start = np.random.choice(single_active_speaker)

            # check one speaker active before
            start_window = start - samples(self.s_pred_config['range_before_onset'])
            two_or_no_speaker = (vad_one_hot[:, start_window : start].sum(axis=0) != 1).any()

            if two_or_no_speaker:
                continue

            # check only this speaker active after
            this_speaker = vad_one_hot[:, start].argmax()
            other_speaker = vad_one_hot[:, start].argmin()
            other_speaker = (vad_one_hot[other_speaker, start : start + samples(self.s_pred_config['neg_future_range'])] == 1).any()

            if other_speaker: 
                continue
            
            s_pred_neg[this_speaker.item()].append(start.item())
            if this_speaker==1:
                num_1 += 1
            elif this_speaker==0:
                num_0 += 1

        # random periods of silence 
        random_no_speech = defaultdict(list)

        # minimum start time must be AFTER both have spoken
        try:
            a=np.where(vad_one_hot[1, :]==1)[0][0]
        except IndexError:
            a=0
        try:
            b=np.where(vad_one_hot[0, :]==1)[0][0]
        except IndexError:
            b=0

        x = min(a,b)
        
        for i in [0,1]:
            
            num=0
            while num < len(s_pred_neg[i]) and self.s_pred_config is not None:
                
                # random start point
                start = np.random.choice(vad_one_hot.shape[1])
                if start < x:
                    continue

                # check who is speaking
                if (vad_one_hot[i, start - samples(self.s_pred_config['range_before_onset']): 
                            start + samples(self.s_pred_config['neg_future_range'])] == 1).any():
                    continue
                    
                random_no_speech[i].append(start)
                num+=1
                continue

        # backchannel prediction 
        # segment must be less than 1s
        # segment must be surrounded by pre and post silence 
        # consider the regions 500ms before bc and a positive sample
        # consdier the regions shift, and during silences (backchannels can be predicted during silences as well)
        backchannels = defaultdict(list)

        for segment in vad_list[0] + vad_list[1]:

            if self.backchannel_pred is None:
                continue
            
            segment_length = segment[1] - segment[0]
            
            # if exceeds length
            if seconds(segment_length) > self.backchannel_pred['max_bc_length']:
                continue

            # start and end
            start = segment[0]
            stop = segment[1]

            # who could be talking
            try:
                speakers = np.where(vad_one_hot[:, start + 160] == 1)
            except IndexError:
                continue

            try:    
                np.nditer(speakers)
            except ValueError:
                continue

            # identify the person who uttered the backchannel from the people talking
            for this_speaker in np.nditer(speakers):

                other_speaker = 1-this_speaker

                # before and
                before_check = (vad_one_hot[this_speaker, start - samples(self.backchannel_pred['pre_silence']) : start - 160] == 1).any()

                # after 
                after_check = (vad_one_hot[this_speaker, stop + 160 : stop + samples(self.backchannel_pred['post_silence'])] == 1).any()

                # the other speaker 
                # final condition: must be preceeded by voice activity from the pther speaker 
                other_speaker_check = (vad_one_hot[other_speaker, start - samples(self.backchannel_pred['pre_other_speaker']) : start - 160] == 1).any()

                # person cant have been talking or about to take a full turn
                if before_check or after_check:
                    continue
                
                # the other person must have been talking 
                if other_speaker_check:
                
                    # include who spoke the backchannel
                    backchannels[this_speaker.item()].append(start)

        # bc pred neg
        single_active_speaker = np.where(vad_one_hot.sum(axis=0) == 1)[0]
        silence = np.where(vad_one_hot.sum(axis=0) == 0)[0]
        single_speaker_or_silence = np.concatenate((single_active_speaker, silence))

        num_0 = 0
        num_1 = 0

        bc_pred_neg = defaultdict(list)
        while (num_0<len(backchannels[0])) and (num_1<len(backchannels[1])) and self.backchannel_pred is not None:
            
            # random start point
            start = np.random.choice(single_speaker_or_silence)

            # randomly choose silence
            if np.random.choice([0,1])==1:
                
                # silent portion
                start = np.random.choice(silence)

                # randomly choose who to assign it to 
                speaker = np.random.choice((this_speaker, other_speaker))
                bc_pred_neg[speaker.item()].append(start.item())

            # check one speaker active before
            start_window = start - samples(self.backchannel_pred['range_before_onset'])
            two_or_no_speaker = (vad_one_hot[:, start_window : start].sum(axis=0) != 1).any()

            if two_or_no_speaker:
                continue

            # check only this speaker active after
            this_speaker = vad_one_hot[:, start].argmax()
            other_speaker = vad_one_hot[:, start].argmin()
            other_speaker = (vad_one_hot[other_speaker, start : start + samples(self.backchannel_pred['neg_future_range'])] == 1).any()

            if other_speaker: 
                continue
            
            bc_pred_neg[this_speaker.item()].append(start.item())
            if this_speaker==1:
                num_1 += 1
            elif this_speaker==0:
                num_0 += 1

        # short / long prediction 
        # useful to know if the upcoming "turn" is a backchannel or a long turn, should the system continue speaking or yield? 
        # postitive: onset of a backchannel
        # negatives: onset of a shift 
        # region covers 200ms of the insent 
        short = backchannels
        long = shifts

        if return_seconds:
            shifts = {k: [seconds(xx) for xx in x] for k, x in shifts.items()}
            holds = {k: [seconds(xx) for xx in x] for k, x in holds.items()}
            s_pred_neg = {k: [seconds(xx) for xx in x] for k, x in s_pred_neg.items()}
            random_no_speech = {k: [seconds(xx) for xx in x] for k, x in random_no_speech.items()}
            backchannels = {k: [seconds(xx) for xx in x] for k, x in backchannels.items()}
            bc_pred_neg = {k: [seconds(xx) for xx in x] for k, x in bc_pred_neg.items()}
            overlaps = {k: [seconds(xx) for xx in x] for k, x in overlaps.items()}
            overlaps_hold = {k: [seconds(xx) for xx in x] for k, x in overlaps_hold.items()}
            overlaps_endtime = {k: [seconds(xx) for xx in x] for k, x in overlaps_endtime.items()}
            overlaps_hold_endtime = {k: [seconds(xx) for xx in x] for k, x in overlaps_hold_endtime.items()}
            short = backchannels
            long = shifts
            shorts = {k: [seconds(xx) for xx in x] for k, x in shorts.items()}
            longs = {k: [seconds(xx) for xx in x] for k, x in longs.items()}


        return {
            "shifts": shifts,
            "holds": holds,
            "s_pred_neg": s_pred_neg,
            "random_no_speech": random_no_speech,
            "backchannels": backchannels,
            "bc_pred_neg": bc_pred_neg,
            "short": short, 
            "long": long,
            "overlaps_shift": overlaps, # start of the overlap
            "overlaps_hold": overlaps_hold, # start of the overlap
            "overlaps_shift_endtime": overlaps_endtime,
            "overlaps_hold_endtime": overlaps_hold_endtime,
            "short_overlap": shorts,
            "long_overlap": longs
        }


class TurnEventsModifier(TurnEvents):  
    """
        modify the hyperparameters for turn-taking event ID
    """

    def __init__(self, gap_duration=None, sh_pre_onset=None, sh_post_offset=None, overlap_duration=None, overlap_pre_onset=None, overlap_post_offset=None, overlap_max_duration=None):
        super().__init__()

        # modifying the gap duration between the turns
        if gap_duration is not None:
            self.shift_hold_config['min_length'] = gap_duration

        if sh_pre_onset is not None:
            self.shift_hold_config['pre_onset'] = sh_pre_onset

        if sh_post_offset is not None:
            self.shift_hold_config['post_offset'] = sh_post_offset

        # overlaps 
        if overlap_duration is not None:
            self.overlap["min_overlap_duration"] = overlap_duration

        if overlap_pre_onset is not None:
            self.overlap["pre_onset"] = overlap_pre_onset

        if overlap_post_offset is not None:
            self.overlap["post_offset"] = overlap_post_offset

        if overlap_max_duration is not None:
            self.overlap["overlap_max_duration"] = overlap_max_duration  
