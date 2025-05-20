import os
import torch
import numpy as np
import random
import json
import glob
import torchaudio
from dataset_management.dataset_manager.src.audio_manager import AudioManager


def generate_babble(speech_files, output_dir, num_per_shard, num_samples):

    
    for i in range(num_samples):

        # randomly select num_per_shard files
        noise_sample = np.random.choice(speech_files, size=num_per_shard)

        speech_samples = []
        for n in noise_sample:
            file, sr = AudioManager.load_waveform(n, sample_rate=16_000, mono=True, normalize=True, channel_first=True)
            speech_samples.append(file)
        min_shape = min([f.shape[-1] for f in speech_samples])
        speech_samples = [f[..., :min_shape] for f in speech_samples]
        speech_samples = torch.stack(speech_samples, dim=-1)

        babble = speech_samples.mean(dim=-1)

        output_file = os.path.join(output_dir, f"{i}.wav")
        torchaudio.save(output_file, babble, sr)

    return


if __name__ == "__main__":

    split = json.load(open("noise_generation/data/lrs3_split.json", "r"))
    lrs3_directory = "/data/ssd4/russelsa/pretrain"
    outdir_val = "/data/ssd4/russelsa/lrs3_babble/train"

    ids = split['train_ids']

    pattern = lrs3_directory+"/**/*.wav"
    noise_files = glob.glob(pattern)
    noise_files = [n for n in noise_files if n.split('/')[-2] in ids]

    random.shuffle(noise_files)

    generate_babble(noise_files, outdir_val, num_per_shard=30, num_samples=50_000)
