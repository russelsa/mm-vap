

config = {
    # audio file parameters
    "sample_rate": 16_000,
    # feature extraction
    # make sure that the gemaps feature extraction settings match this
    "audio_feature_extraction_Hz": 50,
    # overlapping window start and end times
    "audio_window_length": 20,
    "audio_window_stride": 10,
    # for the VAD and the future windows
    "future_context": 2,
    "bin_times": [0.2, 0.4, 0.6, 0.8],
    "minimum_pause_length": 0.0,
    "projection_threshold_ratio": 0.5,
    "vad_threshold_window": 0.5,
    # shift one frame?
    "shift_one_frame": True
}