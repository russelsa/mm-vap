import random
import os
random.seed(0)


def makeifnotexists(path: str):

    try:
        os.makedirs(path)
    except FileExistsError:
        return


def generate_folds(k: int, X_dir: str, out_dir: str, filter: list[str]=None):

    X = [o for o in os.listdir(X_dir) if '.wav' in o]

    ids = [os.path.basename(o).split('.')[0] for o in X]
    random.shuffle(ids)

    # filter candor
    if filter is not None:
        ids = [i for i in ids if i.split('--')[0] not in filter]

    # 5% for testing...
    start = int(len(ids)*0.05)
    test_ids = ids[:start]
    test_csv = os.path.join(out_dir, "test.csv")
    with open(test_csv, 'w') as f:
        f.writelines("id\n")
        for xx in test_ids:
            f.writelines(f'{xx}\n')

    ids = ids[start:]
    L = len(ids)
    step = int(L/k)

    for i, start in enumerate(range(0, L-step, step)):
        
        print(start, start + step)
        
        fold_dir = os.path.join(out_dir, f"fold_{i}")
        makeifnotexists(fold_dir)

        val_csv = os.path.join(fold_dir, "val.csv")
        train_csv = os.path.join(fold_dir, "train.csv")

        train_ids = ids[:start] + ids[start + step:]
        val_ids = ids[start:start + step]

        with open(val_csv, 'w') as f:
            f.writelines("id\n")
            for xx in val_ids:
                f.writelines(f'{xx}\n')

        with open(train_csv, 'w') as f:
            f.writelines("id \n")
            for xx in train_ids:
                f.writelines(f'{xx}\n')

    return


if __name__=="__main__":

    # filter out the problem videos
    with open("dataset_management/dataset_manager/assets/candor_filtered_folds/candor_problematic_files.txt", "r") as f:
        lines = f.readlines()
        lines = [l.strip().split('--')[0] for l in lines]
    filter = list(set(lines))

    generate_folds(5, X_dir="/home/russelsa@ad.mee.tcd.ie/github/turn-taking-projects/corpora/candor/candor_wav", 
                   out_dir="/home/russelsa@ad.mee.tcd.ie/github/dataset_management/dataset_manager/assets/candor_filtered_folds/candor", 
                   filter=filter)
