import os
import random
random.seed(0)


def train_val_split(X_dir, output_dir, split=0.8):
    """ generate a train test split 

    Args:
        X_dir str: path to a directory of .wav files
        output_dir str: output directory path for train/val.csv
        split (float, optional): percentage of items to be in the training set Defaults to 0.8.
    """

    X = [o for o in os.listdir(X_dir) if '.wav' in o]

    ids = [os.path.basename(o).split('.')[0] for o in X]
    random.shuffle(ids)
    L = len(ids)
    train_idx = int(L*split)

    X = ids
    
    X_train, X_val = X[:train_idx], X[train_idx:]
    
    
    val_output_csv = os.path.join(output_dir, 'val.csv')
    with open(val_output_csv, 'w+') as f:
        f.writelines("id\n")
        for xx in X_val:
            f.writelines(f'{xx}\n')

    val_output_csv = os.path.join(output_dir, 'train.csv')
    with open(val_output_csv, 'w') as f:
        f.writelines("id \n")
        for xx in X_train:
            f.writelines(f'{xx}\n')


if __name__ == "__main__":
    import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--switchboard_directory')

    # args = parser.parse_args()
    # topdir = args['switchboard_directory']

    # topdir = '/mnt/storage/Switchboard/Switchboard/'
    # audio_dir = '/mnt/storage/Switchboard/processed/wavs_16k_mono'
    # ground_truth_dir = '/mnt/storage/Switchboard/processed/textgrids_phonwords'
    # output_dir = 'turn_taking/assets/data'
    # gen_train_val_split(topdir, audio_dir, ground_truth_dir, output_dir)
    
    X = "/mnt/storage/turn-taking-projects/corpora/switchboard/switchboard_wav"
    output_dir = "/mnt/storage/dataset-management/dataset_manager/assets/switchboard_20_80_split"

    train_val_split(X, output_dir)
