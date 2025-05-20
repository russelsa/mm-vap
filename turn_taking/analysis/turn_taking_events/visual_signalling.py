import pandas as pd
import os 
import json

features = ['gaze_0_x',
 'gaze_0_y',
 'gaze_0_z',
 'gaze_1_x',
 'gaze_1_y',
 'gaze_1_z',
 'pose_Tx',
 'pose_Ty',
 'pose_Tz',
 'pose_Rx',
 'pose_Ry',
 'pose_Rz',
 'x_3',
 'x_13',
 'x_6',
 'x_10',
 'x_8',
 'y_3',
 'y_13',
 'y_6',
 'y_10',
 'y_8',
 'x_19',
 'x_24',
 'y_19',
 'y_24',
 'x_27',
 'x_28',
 'x_29',
 'x_30',
 'x_31',
 'x_32',
 'x_33',
 'x_34',
 'y_27',
 'y_28',
 'y_29',
 'y_30',
 'y_31',
 'y_32',
 'y_33',
 'y_34',
 'AU01_r',
 'AU02_r',
 'AU04_r',
 'AU05_r',
 'AU06_r',
 'AU07_r',
 'AU09_r',
 'AU10_r',
 'AU12_r',
 'AU14_r',
 'AU15_r',
 'AU17_r',
 'AU20_r',
 'AU23_r',
 'AU25_r',
 'AU26_r',
 'AU45_r',
 'confidence']


def load_channelmaps(channelmaps_dir):

    # create a dictionary linking...
    # id --> [left_channel_id, right_channel_id]
    # to link with the videos
    channelmaps = os.listdir(channelmaps_dir)
    channel_hash = {}

    for channelmap in channelmaps:
        
        id = os.path.basename(channelmap.split('.')[0])
        channelmap_file = os.path.join(channelmaps_dir, channelmap)
        channel_id = json.load(open(channelmap_file))

        channel_hash[id] = {"L": channel_id["L"], "R": channel_id["R"]}

    return channel_hash


def convert(x):

    # speaker (just spoke) and listener (about to speak)
    x = int(x)
    y = 1-x

    if x==0:
        x='L'
    else:
        x='R'

    if y==0:
        y='L'
    else:
        y='R'

    return x,y


def get_vel_acc(visual_features, pose_R):

    for k in visual_features.keys():

        head_pose_radians = visual_features[k][pose_R]
        velocity = 30*(head_pose_radians - head_pose_radians.shift(-1))
        acceleration = 30*(velocity - velocity.shift(-1))

        velocity.columns = [c + '_acc' for c in velocity.columns]
        acceleration.columns = [c + '_vel' for c in acceleration.columns]

        concat = pd.concat([visual_features[k], velocity, acceleration], axis=1)

        visual_features[k] = concat

    return visual_features, list(velocity.columns) + list(acceleration.columns)


def get_max(features, event_name, event_times, listener_id, speaker_id):
    
    # speaker + listener
    speaker_feats = visual_features[speaker_id][features]
    listener_feats = visual_features[listener_id][features]

    ret = []

    # for each time 
    for time in event_times:

        # window of analysis 
        time_frames = int(time*fps)
        start = time_frames - analysis_window
        end = time_frames + analysis_window

        # get the peak intensity 
        speaker_after = speaker_feats.iloc[time_frames:end, :]
        speaker_after = speaker_after.max()
        speaker_after['id'] = speaker_id
        speaker_after['role'] = 'speaker'
        speaker_after['time'] = 'after'

        # for random silence and speech we're only interested in what that person is doing 
        if (event_name != 'random_no_speech') and (event_name != 's_pred_neg'):

            speaker_before = speaker_feats.iloc[start:time_frames, :]
            speaker_before = speaker_before.max()
            speaker_before['id'] = speaker_id
            speaker_before['role'] = 'speaker'
            speaker_before['time'] = 'before'

            listener_before = listener_feats.iloc[start:time_frames, :].max()
            listener_before['id'] = listener_id
            listener_before['role'] = 'listener'
            listener_before['time'] = 'before'

            listener_after = listener_feats.iloc[time_frames:end, :].max()
            listener_after['id'] = listener_id
            listener_after['role'] = 'listener'
            listener_after['time'] = 'after'

            ret += [speaker_before, speaker_after, listener_before, listener_after]

        else:
            ret += [speaker_after]
    
    return ret


def get_analysis_windows(event_name, event_class='ekstedt_events'):

    # SHIFTS
    # turn below into a general function
    
    events = turns[event_class][event_name]

    faus = []
    pose_angle = []
    pose_position = []
    gaze_angle = []

    for speaker, event_times in events.items():

        # convert 
        speaker, listener = convert(speaker)

        # get the subset 
        speaker_id = channelhash[session_id][speaker]
        listener_id = channelhash[session_id][listener]

        faus += get_max(fau_intensity, event_name, event_times, listener_id, speaker_id)
        pose_angle += get_max(pose_R, event_name, event_times, listener_id, speaker_id)
        pose_position += get_max(pose_T, event_name, event_times, listener_id, speaker_id)
        gaze_angle += get_max(gaze, event_name, event_times, listener_id, speaker_id)

    faus = pd.concat(faus, axis=1).T
    pose_angle = pd.concat(pose_angle, axis=1).T
    pose_position = pd.concat(pose_position, axis=1).T
    gaze_angle = pd.concat(gaze_angle, axis=1).T

    windows_of_analysis = pd.concat([faus, pose_angle, pose_position, gaze_angle], axis=1)
    windows_of_analysis = windows_of_analysis.loc[:,~windows_of_analysis.columns.duplicated()].copy()
    windows_of_analysis['event'] = event_class +'-'+ event_name
    windows_of_analysis['session_id'] = session_id

    return windows_of_analysis


if __name__ == "__main__":

    # session_id 
    session_ids = [o.split('.')[0].split('--')[0] for o in os.listdir("/data/ssd3/russelsa/candor_openface_pkl")]
    session_ids = list(set(session_ids))

    # outputdir 
    output_dir = "/data/ssd3/russelsa/analysis_windows_200ms_final"

    for session_id in session_ids:
        try:

            # channelhash
            channelhash = load_channelmaps("/home/russelsa@ad.mee.tcd.ie/github/turn-taking-projects/corpora/candor/channelmaps")

            # for a single session 
            visual_features_dir = "/data/ssd3/russelsa/candor_openface_pkl/"
            visual_features = [o for o in os.listdir(visual_features_dir) if session_id in o]

            # load in the info
            visual_features = {visual_feature.split('.')[0].split('--')[1]: pd.read_pickle(os.path.join(visual_features_dir, visual_feature)).fillna(0) for visual_feature in visual_features}

            # turns for this session
            turns_dir = "turn-taking-projects/corpora/candor/candor_turns/"
            turns = os.path.join(turns_dir, f"{session_id}.json")

            if not os.path.exists(turns):
                # continue
                exit()

            turns = json.load(open(turns, "rb"))

            # WINDOW OF ANALYSIS (milliseconds)
            fps = 30
            analysis_window_dur = 275 
            analysis_window = int((analysis_window_dur/1000)*fps)

            # TARGETS
            fau_intensity = [f for f in features if '_r' in f]
            fau_presence = [f for f in features if '_c' in f]
            gaze = [f for f in features if 'gaze' in f]
            pose_R = [f for f in features if 'pose_R' in f]
            pose_T = [f for f in features if 'pose_T' in f]
            landmarks = [f for f in features if 'x_' in f or 'y_' in f]

            # compute the angular info
            visual_features, cols = get_vel_acc(visual_features, pose_R)
            pose_R = cols

            visual_features, cols = get_vel_acc(visual_features, gaze)
            gaze = cols

            visual_features, cols = get_vel_acc(visual_features, pose_T)
            pose_T = cols

            shifts = get_analysis_windows('shifts')
            holds = get_analysis_windows('holds')
            backchannels = get_analysis_windows('backchannels')
            short = get_analysis_windows('short')
            long = get_analysis_windows('long')
            random_speech = get_analysis_windows('s_pred_neg')
            random_silence = get_analysis_windows('random_no_speech')
            overlaps_shift = get_analysis_windows('overlaps_shift', event_class='roddy_events')
            overlaps_hold = get_analysis_windows('overlaps_hold', event_class='roddy_events')
            gap_0_shift = get_analysis_windows('shifts', event_class='gap_0')
            gap_0_hold = get_analysis_windows('holds', event_class='gap_0')

            features_to_save = []
            features_to_save += [shifts, holds, backchannels, short, long, random_speech, random_silence, overlaps_shift, overlaps_hold, gap_0_shift, gap_0_hold]

            # concat everything 
            windows = pd.concat(features_to_save, axis=0)

            # output 
            output_file = os.path.join(output_dir, f"{session_id}.pkl")
            windows.to_pickle(output_file)

        except ValueError as e:
            print("error")
            # continue
