import torch
import torchaudio 
import os 


def concatenate_tracks():

    topdir = "/home/russelsa@ad.mee.tcd.ie/github/noise_generation/data/short-musan/music/fma-western-art"
    output_dir = "/home/russelsa@ad.mee.tcd.ie/github/noise_generation/data/short-musan/music/fma-western-art-merged"

    # format
    # music-jamendo-0000-0

    # get set of recordings
    recordings = os.listdir(topdir)
    prefixes = list(set(['-'.join(oo for oo in o.split('-')[:-1]) for o in recordings]))

    for prefix in prefixes:
        
        music_files = [f for f in recordings if prefix in f]
        music_files = sorted(music_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))

        res = []
        for music_file in music_files:
            music_clip, sr = torchaudio.load(os.path.join(topdir, music_file))

            res += [music_clip]

        # merge and save
        res = torch.concat(res, dim=-1)
        torchaudio.save(os.path.join(output_dir, prefix + '.wav'), res, sr)



