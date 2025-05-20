"""

    extract the gemaps feature representation for a wav file
    https://github.com/audeering/opensmile-python/issues/76
    to change step size and overlap
    find the location of the opensmile-python package
    go to the "opensmile-python/opensmile/core/config/compare" folder
    open the config file ComParE_2016_core.lld.conf.inc
    change the frame size and frame step at lines 18-19
    make the same change to the frame size and frame step at lines 51-52

"""
import os
import librosa
import opensmile
import numpy as np
import pickle

def extract_gemaps(input_file, output_path, target_sr, stereo):

    if stereo:
        channels = [0,1]
        y, fs = librosa.load(input_file, sr=target_sr, mono=False)

    else:
        channels = [0]
        y, fs = librosa.load(input_file, sr=target_sr, mono=True)

    smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv01a,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    channels=[0]
    )
    features = smile.feature_names
    
    for channel_no in channels:
        output_filepath = f"{output_path}_{channel_no}.pkl"

        if stereo:
            y_c = y[channel_no, :]
        else:
            y_c = y
        if not os.path.exists(output_filepath):
            x = smile.process_signal(y_c, sampling_rate=target_sr).to_numpy()
            pickle.dump((features, x), open(output_filepath, "wb"))

    return


if __name__=="__main__":

    import argparse

    arg = argparse.ArgumentParser()
    arg.add_argument('--target_sr')
    arg.add_argument('--input_file')
    arg.add_argument('--stereo')

    args = arg.parse_args()

    target_sr = args.target_sr
    input_file = args.input_file
    stereo = args.stereo
    
    output_path = input_file.split('.')[0]
    stereo = int(stereo)
    target_sr = int(target_sr)

    extract_gemaps(input_file, output_path, target_sr, stereo)
