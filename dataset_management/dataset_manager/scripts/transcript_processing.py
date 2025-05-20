from collections import defaultdict
from einops import rearrange
import textgrid
import collections
from dataset_management.dataset_manager.config.config import config
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os 
from torch.nn import functional as F


def bin_times_to_frames(bin_times, Hz):
    return (torch.tensor(bin_times) * Hz).long().tolist()


def vads_from_transcript(transcript_file, samples):
    
    tg = textgrid.TextGrid.fromFile(transcript_file)

    # one vad list for each tier
    vad_list = collections.defaultdict(list)

    # filtering minimum pauses
    minimum_pause_length = config['minimum_pause_length']
    minimum_pause_length = int(minimum_pause_length * config['sample_rate'])

    # for each speaker
    for i, tier in enumerate(tg.tiers):

        # for the intervals in the tier (i.e. the words spoken)
        intervals = tier.intervals
        
        for interval in intervals:
            
            # if this is a silence tier
            if interval.mark == '':
                continue

            # start and end of the voice activity segment
            start, end = interval.minTime, interval.maxTime
            start, end = start, end 

            if samples:
                start, end = int(start*config['sample_rate']), int(end*config['sample_rate'])

            # record it 
            vad_list[i].append([start, end])
    
    vad_list = [vad_list[0], vad_list[1]]
    return vad_list


def vad_list_reduce(vad_list):
    """ concatenate adjacent vad lists """

    return vad_list



def vad_list_windowing(vad_list):
    """
        vad_list: tensor (N_channels, N_windows, W_window_width)
        return:
            tensor (N_channels, N_windows)
    """

    # threshold for determining presence or absence of speech
    threshold = config['vad_threshold_window']

    # window it at the audio sample rate with non overlapping windows
    width = int(config['sample_rate']/config['audio_feature_extraction_Hz'])
    blocks = vad_list.unfold(1, size=int(width), step=int(width))

    # determine the ratio of VADs in the window
    sb = blocks.sum(dim=-1)
    sum_d =  sb / width

    # set if greater than threshold
    vad_threshold = torch.zeros_like(sb)
    mask = sum_d >= threshold
    vad_threshold[mask] = 1.
    
    return vad_threshold


def vad_list_to_one_hot(vad_list, track_size, samples, sr=None, start=0):
    """
        convert a list of start and end times of vocalisations 
        to a one hot encoded representation of voice activity 
    """
    output = torch.zeros(track_size)

    if len(vad_list) > 1:
        for i, channel_vad in enumerate(vad_list):
            for v in channel_vad:
                if samples:
                    output[i, v[0]-start:v[1]-start] = 1.0
                else:
                    output[i, int((v[0]-start)*sr):int((v[1]-start)*sr)] = 1.0

    output = rearrange(output, "c n -> n c")
    return output


def interpausal_units(vad_list, samples=False, sr=None, max_silence_between=0.2):
    """
    forms IPUs from textgrid files
    concatenates adjacent units if the silence between them is less than 200ms or 0.2 seconds
    uses the definition of an interpausal unit from
    "Neural Generation of Dialogue Response Timings" Roddy and Harte, 2020s
    """

    prev_word = None
    ipu_list = []
    current_ipu = []
    
    for curr_word in vad_list:

        if prev_word is None:
            prev_word = curr_word
            current_ipu = [prev_word]
            continue

        time = (curr_word[0] - prev_word[1])
        if samples:
            time = time/sr

        # start a new ipu
        if time > max_silence_between:

            ipu = (current_ipu[0][0], current_ipu[-1][1])
            ipu_list.append(ipu)

            current_ipu = [curr_word]

        else:

            current_ipu.append(curr_word)

        prev_word = curr_word

    return ipu_list

def vad_list_to_turns(vad_list, vad_one_hot, samples, sr):

    ipus_0 = interpausal_units(vad_list[0], samples, sr)
    ipus_1 = interpausal_units(vad_list[1], samples, sr)

    speaker_turns = defaultdict(list)

    for i, speaker in enumerate([ipus_0, ipus_1]):

        turn = None
        for curr_ipu, next_ipu in zip(speaker, speaker[1:]):

            # check this interval for silence
            a = curr_ipu[1]
            b = next_ipu[0]

            # turn has ended
            x = vad_one_hot[a:b, 1-i]
            if x.any() == 1:

                if not turn:
                    
                    start, end = curr_ipu[0], curr_ipu[1]
                    if samples:
                        start, end = round(start/sr, 2), round(end/sr, 2)
                        
                    turn = [ start, end ]
                
                else:
                    start, end = turn[0], curr_ipu[1]
                    if samples:
                        start, end = start, round(end/sr, 2)
                    turn = [start, end]

                speaker_turns[i].append({"start": turn[0], "end": turn[1]})
                turn=None
                
            else:

                if not turn:
                    start,end = curr_ipu[0], next_ipu[1]
                    if samples:
                        start, end = round(start/sr, 2), round(end/sr, 2)
                    turn = [start,end]

                else:
                    start, end = turn[0], next_ipu[1]
                    if samples:
                        start, end = start, round(end/sr, 2)
                    turn = [start, end]

        if turn:
            speaker_turns[i].append({"start": turn[0], "end": turn[1]})

    return {"0": speaker_turns[0], "1": speaker_turns[1]}


def get_activity_history(vad_frames, bin_end_frames, channel_last=False):
    """

    Uses convolutions to sum the activity over each segment of interest.

    The kernel size is set to be the number of frames of any particular segment i.e.

    ---------------------------------------------------


    ```
    ... h0       | h1 | h2 | h3 | h4 +
    distant past |    |    |    |    +
    -inf -> -t0  |    |    |    |    +

    ```

    ---------------------------------------------------

    Arguments:
        vad_frames:         torch.tensor: (Channels, N_Frames) or (N_Frames, Channels)
        bin_end_frames:     list: boundaries for the activity history windows i.e. [6000, 3000, 1000, 500]
        channel_last:       bool: if true we expect `vad_frames` to be (N_Frames, Channels)

    Returns:
        ratios:             torch.tensor: (Channels, N_frames, bins) or (N_frames, bins, Channels) (dependent on `channel_last`)
        history_bins:       torch.tesnor: same size as ratio but contains the number of active frames, over each segment, for both speakers.
    """

    N = vad_frames.shape[0]

    # container for the activity of the defined bins
    hist_bins = []

    # Distance past activity history/ratio
    # The segment from negative infinity to the first bin_end_frames
    if vad_frames.shape[1] > bin_end_frames[0]:
        h0 = vad_frames[:, : -bin_end_frames[0]].cumsum(dim=-1)
        diff_pad = torch.ones(2, bin_end_frames[0]) * -1
        h0 = torch.cat((diff_pad, h0), dim=-1)
    else:
        # there is not enough duration to get any long time information
        # -> set to prior of equal speech
        # negative values for debugging to see where we provide prior
        # (not seen outside of this after r0/r1 further down)
        h0 = torch.ones(2, N) * -1
    hist_bins.append(h0)

    # Activity of segments defined by the the `bin_end_frames`

    # If 0 is not included in the window (i.e. the current frame)
    # we append it for consistency in loop below
    if bin_end_frames[-1] != 0:
        bin_end_frames = bin_end_frames + [0]

    # Loop over each segment window, construct conv1d (summation: all weights are 1.)
    # Omit end-frames which are not used for the current bin
    # concatenate activity sum with pad (= -1) at the start where the bin values are
    # not defined.
    for start, end in zip(bin_end_frames[:-1], bin_end_frames[1:]):
        ks = start - end
        if end > 0:
            vf = vad_frames[:, :-end]
        else:
            vf = vad_frames
        if vf.shape[1] > 0:
            filters = torch.ones((1, 1, ks), dtype=torch.float)
            vf = F.pad(vf, [ks - 1, 0]).unsqueeze(1)  # add channel dim
            o = F.conv1d(vf, weight=filters).squeeze(1)  # remove channel dim
            if end > 0:
                # print('diffpad: ', end)
                diff_pad = torch.ones(2, end) * -1
                o = torch.cat((diff_pad, o), dim=-1)
        else:
            # there is not enough duration to get any long time information
            # -> set to prior of equal speech
            # negative values for debugging to see where we provide prior
            # (not seen outside of this after r0/r1 further down)
            o = torch.ones(2, N) * -1
        hist_bins.append(o)

    # stack together -> (2, N, len(bin_end_frames) + 1) default: (2, N, 5)
    hist_bins = torch.stack(hist_bins, dim=-1)

    # find the ratios for each speaker
    r0 = hist_bins[0] / hist_bins.sum(dim=0)
    r1 = hist_bins[1] / hist_bins.sum(dim=0)

    # segments where both speakers are silent (i.e. [0, 0] activation)
    # are not defined (i.e. hist_bins / hist_bins.sum = 0 / 0 ).
    # Where both speakers are silent they have equal amount of
    nan_inds = torch.where(r0.isnan())
    r0[nan_inds] = 0.5
    r1[nan_inds] = 0.5

    # Consistent input/output with `channel_last` VAD
    if channel_last:
        ratio = torch.stack((r0, r1), dim=-1)
    else:
        ratio = torch.stack((r0, r1))
    return ratio, hist_bins



def training_labels_projection(vad_one_hot, mode):
    """
        operates at the audio encoder sampler rate, pass it the original vad list at 8kHz (Switchboard)

        input:
            vad_one_hot: tensor (N_frames, N_channels)
        return:
            projection_vad: tensor (N_channels, N_projection_bins, P_projection_bin_width)
    """

    if config['shift_one_frame']:
        vad_one_hot = torch.concat((vad_one_hot[1:, :], vad_one_hot[-1, :].unsqueeze(0)), dim=0)

    modes = ['VAP', 'ind-40', 'vap', 'ind-4']
    assert mode in modes, f"mode must be in {modes}"

    Hz = config['audio_feature_extraction_Hz']
    threshold_ratio = config['projection_threshold_ratio']

    # bin times are different for different objectives
    if mode.lower() == 'vap':
        bin_times = config['bin_times']

    if mode == 'ind-40':
        bin_times = [50/1000]*40
    
    elif mode == 'ind-4':
        bin_times = config['bin_times']
    
    # convert to frames
    bin_frames = bin_times_to_frames(bin_times, Hz)

    # width of the projection bins in samples
    bin_sample_width = np.sum(bin_frames)
    
    # get the horizon windows: bin-width into the *future*, spaced apart at the audio sampler rate
    horizon_windows = vad_one_hot.unfold(0, size=int(bin_sample_width), step=1)

    bins = []
    # if config['shift_one_frame']:
    #     start=1 # shift one frame
    # else:
    start=0
    for b in bin_frames:
        
        start = int(start) 
        end = start + b

        m = horizon_windows[..., start:end]
        m = m.sum(dim=-1) / b
        m = (m >= threshold_ratio).float()

        start = end
        
        bins.append(m)
    
    projection_bins = torch.stack(bins, dim=1)
    projection_bins = projection_bins.permute(0,2,1)
    projection_bins = projection_bins[:-1,...]

    return projection_bins


# if __name__=="__main__":

#     transcript_file='/mnt/storage/turn-taking-projects/corpora/switchboard/switchboard_speechmatics/sw2699.TextGrid'
#     vads = vads_from_transcript(transcript_file, samples=False)
#     x=1
    
    # # testing the VAP
    # transcript_file = "/mnt/storage/Switchboard/processed/textgrids_phonwords/sw2045.TextGrid"
    # audio = "/mnt/storage/Switchboard/processed/wavs_8k_crosstalk_removed/sw2045.wav"

    # audio, fs = librosa.load(audio, sr=config['sample_rate'], mono=False)

    # vads_from_transcript = vads_from_transcript(transcript_file, samples=True)
    # vad_oh = vad_list_to_one_hot(vads_from_transcript, audio.shape, samples=True)

    # vap = training_labels_projection(vad_oh, mode='VAP')
    # ind40 = training_labels_projection(vad_oh, mode='Independent-40')

    # max_time = 30
    # max_time_audio = int(max_time*config['sample_rate'])
    # max_time_label = int(max_time*config['audio_feature_extraction_Hz'])
    
    # channel = 1
    
    # x = np.linspace(0, max_time_audio, max_time_label)

    # audio = audio[channel, :max_time_audio].squeeze()
    # vad_oh = vad_oh[channel, :max_time_label].squeeze()
    
    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax1.plot(audio)
    # ax2.plot(vad_oh, "r--")
    # ax2.plot(x, ind40[channel, 0:max_time_label, 5], "g--")
    # plt.savefig("tmp.png")

    # x=1