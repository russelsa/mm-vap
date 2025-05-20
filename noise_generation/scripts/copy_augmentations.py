import pandas as pd
import shutil 
import os 


indir = "/home/russelsa@ad.mee.tcd.ie/github/turn-taking-projects/corpora/candor/candor_wav_noise_augment"
outdir = "/data/ssd2/russelsa/candor_wav_augmentations"

augmentations = pd.read_csv("/home/russelsa@ad.mee.tcd.ie/github/noise_generation/training_set_augmentations.csv")
augmentations = augmentations[augmentations['augmentation']!='-1']['file'].to_list()
augmentations = [os.path.basename(a).split('.')[0] for a in augmentations]

test_set_ids = "/home/russelsa@ad.mee.tcd.ie/github/dataset_management/dataset_manager/assets/new_folds/candor/test.csv"
test_set_ids = pd.read_csv(test_set_ids)
test_set_ids = test_set_ids['id'].to_list()

augmented_files = []
for file in augmentations:
    id = os.path.basename(file)
    if id in test_set_ids:
        continue
    test_file = os.path.join(indir, id+'.wav')
    if os.path.exists(test_file):
        augmented_files.append(test_file)
    else:
        for i in range(5):
            test_file_fold = os.path.join(indir, os.path.basename(file).split('.')[0]+f'_fold_{i}.wav')
            if os.path.exists(test_file_fold):
                augmented_files.append(test_file_fold)

for aug_file in augmented_files:
    shutil.copy(aug_file, os.path.join(outdir, os.path.basename(aug_file)))
