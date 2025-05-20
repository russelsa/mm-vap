import os
import torchaudio 
import torch
from copy import copy 
import numpy as np
import torchaudio.functional as F
from dataset_management.dataset_manager.src.audio_manager import AudioManager


def add_noise_to_file(input_file, output_file, noise_files, snr_dbs, repeat=False):

    speech, sr_speech = AudioManager.load_waveform(input_file, normalize=True, sample_rate=16000, channel_first=True)
    noise, sr_noise = AudioManager.load_waveform(np.random.choice(noise_files), normalize=True, mono=True, sample_rate=16000, channel_first=True)

    start_sample = 0

    noisy_speech = copy(speech)

    i=0
    while noise.shape[-1] < speech.shape[-1]:
        i+=1

        if i%10==0 or not repeat:
            n2, sr_noise = AudioManager.load_waveform(np.random.choice(noise_files), normalize=True, mono=True, sample_rate=16000, channel_first=True)
            # n2 = n2/n2.max()
        else:
            n2=noise
        
        noise = torch.cat((noise, n2), dim=-1)

    noise = noise[:, :speech.shape[-1]]
    noise_stereo = torch.concatenate((noise, noise), dim=0)

    noisy_speech = F.add_noise(speech, noise_stereo, torch.tensor([snr_dbs]))

    torchaudio.save(output_file, noisy_speech, sr_speech)


def get_noise(speech, noise_files, repeat, normalize, volume_norm=False):
    """
        repeatedly concatenated noise files together with or without repetiion until it is the same length as the speech
    """

    new_segment = np.random.choice(noise_files)
    new_segment, _ = AudioManager.load_waveform(new_segment, mono=True, normalize=normalize, sample_rate=16000, channel_first=True)

    noise = copy(new_segment)

    i=0
    while noise.shape[-1] < speech.shape[-1]:

        if repeat and i%10 == 0:
            new_segment = noise

        else:
            new_segment = np.random.choice(noise_files)
            new_segment, _ = AudioManager.load_waveform(new_segment, mono=True, normalize=normalize, sample_rate=16000, channel_first=True) 
            
            if volume_norm:
                new_segment = new_segment / new_segment.max()

        
        noise = torch.concat((noise, new_segment), dim=-1)
        i+=1
    noise = noise[:, :speech.shape[-1]]

    return noise


def apply_noise_to_channel(speech, speech_no_silences, noise, snr, mode):
    """
        apply noise to a mono audio channel
        speech: the speech 
        speech_no_silences: identical to speech but silences removed
        noise: a noise file the same length as speech
    """

    assert mode in ['standard', 'RMS-cutoff']

    # Parameters
    rms_thresh = 0.05  # use top fraction of RMS values measured in 1 sec chunks
    fs = 16_000

    # Compute RMS values
    rms_speech = torch.sqrt(torch.mean(speech_no_silences**2))
    rms_nse = torch.sqrt(torch.mean(noise**2))

    # Calculate steady-state alpha
    alpha_steady = rms_speech / (rms_nse * (10**(snr/10)))
    x_noisy_naomi = speech + (alpha_steady * noise)
    
    # normalise
    x_noisy_naomi = x_noisy_naomi / torch.max(torch.abs(x_noisy_naomi))

    if mode=='standard':
        return x_noisy_naomi

    # Compute RMS in 1-second chunks
    num_secs = speech_no_silences.shape[-1] // fs
    track_rms = []

    for i in range(num_secs - 1):
        data = speech_no_silences[:, i * fs: int((i+0.2)*fs)]
        data_rms = torch.sqrt(torch.mean(data**2))
        track_rms.append(data_rms.item())

    # Order RMS values starting with the highest
    track_rms = sorted(track_rms, reverse=True)
    cutoff = int(num_secs * rms_thresh)
    # cutoff=1
    use_rms = track_rms[:cutoff]

    # Compute target alpha for higher power elements
    alpha = np.mean(use_rms) / (rms_nse * (10**(snr/10)))
    x_noisy_naomi = speech + (alpha * noise)

    # Normalize the noisy signal
    x_noisy_naomi_norm = x_noisy_naomi / torch.max(torch.abs(x_noisy_naomi))

    return x_noisy_naomi_norm

   