import random
import os 
import textgrid
import torchaudio
import shutil
random.seed(0)


def chunk_all(audio, transcripts, output_dir):

    textgrid_files = [os.path.join(transcripts, o) for o in os.listdir(transcripts)]

    for textgrid_file_name in textgrid_files:

        textgrid_file = textgrid.TextGrid().fromFile(textgrid_file_name)

        for interval_tier in textgrid_file:

            speaker = interval_tier.name

            wavfile_id = os.path.basename(textgrid_file_name).split('.')[0] + '_' + speaker

            try:
                audio_file = [a for a in os.listdir(audio) if wavfile_id in a][0]
            except Exception as e:
                print(f"error with {wavfile_id} skipping")
                continue            

            audio_file = os.path.join(audio, audio_file)

            waveform, sr = torchaudio.load(audio_file)

            for i, utterance in enumerate(interval_tier.intervals):

                if utterance.mark != '':
                    start, end = int(utterance.minTime*sr), int(utterance.maxTime*sr)
                    utterance = waveform[..., start:end]

                    speaker_id = wavfile_id.split('_')[1] + '_' + wavfile_id.split('_')[0]
                    output_file = os.path.join(output_dir, speaker_id + '_' + str(i) + '.wav')
                    torchaudio.save(output_file, utterance, sr)


def train_test_split(audio_files, p):
    speaker_ids = list(set([a.split('.')[0].split('_')[1] for a in os.listdir(audio_files)]))
    speaker_ids = [s for s in speaker_ids if 'T' not in s]
    random.shuffle(speaker_ids)

    L = len(speaker_ids)
    val_part = int(L*p)

    val_ids = speaker_ids[:val_part]
    train_ids = speaker_ids[val_part:]

    val_ids += ['T001']
    train_ids += ['T002']

    return val_ids, train_ids


def move_utterances(utterances_dir, test_speaker_ids, test_dir):

    utterances = [os.path.join(utterances_dir, u) for u in os.listdir(utterances_dir)]

    for speaker in test_speaker_ids:
        for utterance in utterances:
            if speaker in utterance:
                shutil.move(utterance, os.path.join(test_dir, os.path.basename(utterance)))

    return

if __name__=="__main__":

    roomreader_audio_dir = "data/RoomReader_Audio_16k"
    roomreader_textgrid_dir = "data/roomreader_ipus"
    utterances_dir = "data/roomreader_utterances"

    # divide the original recordings into individual utterances
    chunk_all(roomreader_audio_dir, roomreader_textgrid_dir, utterances_dir)

    # train test split the speakers
    test_speakers, train_speakers = train_test_split(roomreader_audio_dir, p=0.10)

    train_dir = "data/roomreader_utterances_train"
    move_utterances(utterances_dir, train_speakers, train_dir)
    