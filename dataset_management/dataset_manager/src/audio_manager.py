from os.path import basename, dirname
from os import remove
import einops
import torchaudio
import torchaudio.functional as AF
from torchaudio.backend.sox_io_backend import info as info_sox
import pickle


def samples_to_frames(s, hop_len):
    return int(s / hop_len)


def sample_to_time(n_samples, sample_rate):
    return n_samples / sample_rate


def frames_to_time(f, hop_time):
    return f * hop_time


def time_to_frames(t, hop_time):
    return int(t / hop_time)


def time_to_frames_samples(t, sample_rate, hop_length):
    return int(t * sample_rate / hop_length)


def time_to_samples(t, sample_rate):
    return int(t * sample_rate)


def get_audio_info(audio_path):
    info = info_sox(audio_path)
    return {
        "name": basename(audio_path),
        "duration": sample_to_time(info.num_frames, info.sample_rate),
        "sample_rate": info.sample_rate,
        "num_frames": info.num_frames,
        "bits_per_sample": info.bits_per_sample,
        "num_channels": info.bits_per_sample,
    }


class AudioManager():

    """ extracting segments of an audio file

    """
    

    def __init__(self, audio_path: str, mono: bool):
        self.audio_path = audio_path

    def get_segment(self, start_time, stop_time, sample_rate, normalize):
        return self.load_waveform(self.audio_path, start_time=start_time, end_time=stop_time, sample_rate=sample_rate, normalize=normalize)

    @staticmethod
    def load_waveform(
        path,
        sample_rate=None,
        start_time=None,
        end_time=None,
        normalize=False,
        mono=False,
        audio_normalize_threshold=0.05,
    ):
        
        info = get_audio_info(path)
        assert sample_rate == info['sample_rate'], f"resample from {info['sample_rate']} to {sample_rate}"

        if start_time is not None:

            frame_offset = time_to_samples(start_time, info["sample_rate"])
            num_frames = info["num_frames"]
            if end_time is not None:
                num_frames = time_to_samples(end_time, info["sample_rate"]) - frame_offset
            else:
                num_frames = num_frames - frame_offset
            x, sr = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames, backend='sox')
        else:
            x, sr = torchaudio.load(path, backend='sox')

        if normalize:
            if x.shape[0] > 1:
                if x[0].abs().max() > audio_normalize_threshold:
                    x[0] /= x[0].abs().max()
                if x[1].abs().max() > audio_normalize_threshold:
                    x[1] /= x[1].abs().max()
            else:
                if x.abs().max() > audio_normalize_threshold:
                    x /= x.abs().max()

        if mono and x.shape[0] > 1:
            x = x.mean(dim=0).unsqueeze(0)
            if normalize:
                if x.abs().max() > audio_normalize_threshold:
                    x /= x.abs().max()

        # if sample_rate:
        #     resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        #     if sr != sample_rate:
        #         x = resampler(x)
        #         sr = sample_rate
        
        x = einops.rearrange(x, "c n -> n c")
        return x, sr


class GeMAPSManager():
    """extracting segments of a GeMAPS pkl file
    """
    
    def __init__(self, gemaps_path: str, gemaps_sample_rate: float, normalize = False) -> None:
        self.gemaps_path = gemaps_path
        self.Hz = gemaps_sample_rate
        self.normalize = normalize

    def get_segment(self, start_time, end_time):

        pkl = self.load_gemaps(self.gemaps_path)

        start_sample = int(start_time*self.Hz)
        end_sample = int(end_time*self.Hz)

        segment_gemaps = pkl[start_sample:end_sample, :]

        return segment_gemaps

    def load_gemaps(self, gemaps_path):

        features, pkl = pickle.load(open(gemaps_path, "rb"))

        if self.normalize:
            pkl = (pkl - pkl.mean(axis=0)) / pkl.std(axis=0)

        # debug=True
        # if debug:
        #     import seaborn as sns
        #     import matplotlib.pyplot as plt
        #     sns.boxplot(pkl, showfliers=False)
        #     plt.savefig("/mnt/storage/gemaps_text.png")

        return pkl



def test_loadtime():
    import time 
    import pickle 


    start = time.time()
    # pkl = pickle.load(open("/mnt/storage/turn-taking-projects/corpora/candor/candor_gemaps/0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a_0.pkl", "rb"))
    audio = "/mnt/storage/turn-taking-projects/corpora/candor/candor_wav/0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a.wav"
    audio = AudioManager.load_waveform(audio, start_time=50, end_time=60)
    stop = time.time()

    print(stop-start)


if __name__=="__main__":

    from dataset_manager.config.config import config

    # gemaps = GeMAPSManager()
    gemaps = GeMAPSManager("/mnt/storage/turn-taking-projects/corpora/switchboard/switchboard_wav/sw2344_0.pkl", config['audio_feature_extraction_Hz'])
    segment = gemaps.get_segment(0, 10)
    x=1
