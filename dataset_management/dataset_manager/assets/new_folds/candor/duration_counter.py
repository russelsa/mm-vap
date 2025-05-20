import os 
import pandas as pd 
import wave
import contextlib



wavdir = "/home/russelsa@ad.mee.tcd.ie/github/turn-taking-projects/corpora/switchboard/switchboard_wav"
for fold in [0,1,2,3,4]:
    dur = 0
    with open(f"/home/russelsa@ad.mee.tcd.ie/github/dataset_management/dataset_manager/assets/new_folds/switchboard_asr/test.csv", "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            fname = os.path.join(wavdir, line).strip()
            with contextlib.closing(wave.open(fname+'.wav','r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
                dur += duration 
    print(dur)