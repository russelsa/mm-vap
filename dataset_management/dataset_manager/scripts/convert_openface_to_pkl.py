import numpy as np
import argparse
import os
import glob
import pandas as pd 
from multiprocessing import Process
from tqdm import tqdm
import subprocess
import soundfile as sf
import numpy as np



def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)


def converter(in_dir, out_dir, pkl_file, start, stop, fps=30):
    """
        generate chunked pickles for manageable parsing...
    """

    # read in the chunks
    chunks = pd.read_pickle(pkl_file)
    memory=None

    # columns
    columns = None

    for chunk in chunks[start:stop]:
        id = chunk[0]
        start = int(chunk[1]*fps)
        stop = int(chunk[2]*fps)

        # get the openface file 
        openface_files = glob.glob(os.path.join(in_dir, id+"*"))

        for openface_file in openface_files:

            output_filename = os.path.basename(openface_file).split('.')[0] + f"_{start}_{stop}.pkl"
            output_filename = os.path.join(out_dir, output_filename)

            if os.path.exists(output_filename):
                continue

            if memory is None or openface_file not in memory:
                memory={}
                df=pd.read_csv(openface_file)
                df.columns = [c.strip() for c in df.columns]
                memory[openface_file]=df
                pkl = memory[openface_file]
            elif openface_file in memory:
                pkl = memory[openface_file]
            
            # read the file 
            chunk_to_write = pkl.iloc[start:stop, :]

            if columns is None:

                columns_FAU_r = [c for c in chunk_to_write.columns if '_r' in c] # fau intensity
                columns_pose  = [c for c in chunk_to_write.columns if 'pose_R' in c] # pose angle
                columns_gaze =  [c for c in chunk_to_write.columns if 'gaze_angle' in c] # gaze angle

                # columns_face_x = [c for c in chunk_to_write.columns if ' x_' in c and 'eye' not in c] # should be 67 landmarks in pixel space
                # columns_face_y = [c for c in chunk_to_write.columns if ' y_' in c and 'eye' not in c] # 

                # mouth movement
                mouth = [f'x_{i}' for i in range(50,67)] + [f'y_{i}' for i in range(50,67)]

                # nose tip
                nose_tip = ['x_33', 'y_33']

                columns_normalize = columns_gaze + columns_pose + mouth + nose_tip  # columns to z normalize per file
                columns_other = columns_FAU_r # leave them unchanged

            # # get the subset
            # mean_x = chunk_to_write[columns_face_x].mean(axis=1)
            # mean_y = chunk_to_write[columns_face_y].mean(axis=1)

            # columns_normalize += ['head_x', 'head_y']

            # chunk_to_write['head_x'] = mean_x
            # chunk_to_write['head_y'] = mean_y

            # # normalize
            # for x in columns_normalize:
            #     chunk_to_write.loc[:, x] = centre(chunk_to_write[x])

            columns = columns_normalize + columns_other
            chunk_to_write = chunk_to_write[columns]
            chunk_to_write.to_pickle(output_filename)
            
            # print(openface_file)
            # print(output_filename)


def centre(df):
    df = df.astype(float)
    df = (df - df.mean()) 
    return df


def test_vis(csv):

    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv)
    columns_face_x = [c for c in df.columns if ' x_' in c and 'eye' not in c] # should be 67
    columns_face_y = [c for c in df.columns if ' y_' in c and 'eye' not in c]
    df['mean_x'] = df[columns_face_x].mean(axis=1)
    df['mean_y'] = df[columns_face_y].mean(axis=1)
    df.columns = [c.strip() for c in df.columns]

    # df['mean_y_norm'] = z_normalize(df['mean_y'])

    plt.figure()
    sns.histplot(df, x='x_62')
    plt.savefig(os.path.basename(csv).split('.')[0]+".png")


def stip_header_and_select_features(csv_file):

    try:
        df=pd.read_csv(csv_file)
        df = df.astype(float)
    except Exception as e:
        return None
    df.columns = [c.strip() for c in df.columns]

    # read the file 
    chunk_to_write = df

    # if columns is None:

    columns_FAU_r = [c for c in chunk_to_write.columns if '_r' in c] # fau intensity
    columns_FAU_c = [c for c in chunk_to_write.columns if '_c' in c] # fau presence
    columns_pose  = [c for c in chunk_to_write.columns if 'pose_' in c] # pose angle
    columns_gaze =  [c for c in chunk_to_write.columns if 'gaze' in c and '_0_' in c or '_1_' in c] # gaze 

    # nose 
    nose = [f'x_{i}' for i in range(27,35)] + [f'y_{i}' for i in range(27,35)]

    # jaw
    jaw = ['x_3', 'x_13', 'x_6', 'x_10', 'x_8', 'y_3', 'y_13', 'y_6', 'y_10', 'y_8']

    # brow 
    brow = ['x_19', 'x_24', 'y_19', 'y_24']

    columns_normalize = columns_gaze + columns_pose + jaw + brow + nose  # columns to z normalize per file
    columns_other = columns_FAU_r + columns_FAU_c + ['confidence'] # leave them unchanged
    columns_other = ['confidence']

    minimum = chunk_to_write[columns_normalize].min()
    chunk_to_write[columns_normalize] = (chunk_to_write[columns_normalize] - minimum)/(chunk_to_write[columns_normalize].max()-minimum)
    chunk_to_write[columns_normalize] = (chunk_to_write[columns_normalize] - chunk_to_write[columns_normalize].mean())

    columns = columns_normalize + columns_other
    chunk_to_write = chunk_to_write[columns]
    return chunk_to_write


def convert_to_pkl():

    """
        convert openface csvs to pkl 
        some videos have trailing silence crop them 
        could still be misaligned with audio ?
    """

    in_dir = "/data/ssd2/russelsa/candor_openface_features"
    output_dir = "/data/ssd3/russelsa/candor_openface_pkl_with_fau_c"
    video_dir = "/data/ssd2/russelsa/candor_video"

    csvs = os.listdir(in_dir)
    csvs = sorted(csvs)

    # csvs = csvs[start:stop]

    ids = [o.split('--')[0] for o in csvs]

    pbar = tqdm(total=len(csvs))
    
    for id in ids:
        
        csvs = glob.glob(os.path.join(in_dir, id+"*"))
        dfs=[]

        for csv in csvs:
            if os.path.exists(os.path.join(output_dir, os.path.basename(csv).split('.')[0]+'.pkl')):
                continue
            df = stip_header_and_select_features(csv)
            if df is None:
                with open(f"problematic_file_{start}_{stop}.txt", "a+") as f:
                    f.write(os.path.basename(csv).split('.')[0]+".mp4"+"\n")
                    f.flush()
                continue

            video_file = os.path.join(video_dir, os.path.basename(csv).split('.')[0]+'.mp4')
            vl = get_length(video_file)

            if not np.isclose(vl, df.shape[0]/30, atol=1):
                with open(f"problematic_file_{start}_{stop}.txt", "a+") as f:
                    f.write(os.path.basename(video_file)+"\n")
                    f.flush()
                continue
            
            # everything is fine ? 
            else:
                dfs.append(df)
                with open(f"grand_{start}_{stop}.txt", "a+") as f:
                    f.write(os.path.basename(video_file)+"\n")
                    f.flush()
        
        # if both are fine ! 
        if len(dfs)==2:
            max_len = min([d.shape[0] for d in dfs])
            for i, df in enumerate(dfs):
                df = df.iloc[:max_len, :]
                df.to_pickle(os.path.join(output_dir, os.path.basename(csvs[i]).split('.')[0]+'.pkl'))
            
        pbar.update(1)
    pbar.close()


if __name__=="__main__":
    start=0
    stop=-1
    convert_to_pkl()
    

    # indir = "/data/ssd3/russelsa/candor_openface_pkl"
    # outdir = "/data/ssd3/russelsa/candor_openface_pkl_brief"

    # for pkl in os.listdir(indir):
    #     if '.pkl' not in pkl:
    #         continue
    #     input_pkl = os.path.join(indir, pkl)
    #     output_pkl = os.path.join(outdir, pkl)
    #     df = pd.read_pickle(input_pkl)
    #     df = df[[d for d in df.columns if ('gaze' in d) or ('pose' in d)]]
    #     df.to_pickle(output_pkl)
    
