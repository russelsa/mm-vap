import pandas as pd
import os 
import seaborn as sns
import matplotlib.pyplot as plt
import re


def lr_sweep_tabulator():
    dir = "/data/ssd1/russelsa/tabulated/lr_sweep"
    output = os.path.join(dir, 'tabulated.csv')

    csvs = [os.path.join(dir, o) for o in os.listdir(dir) if 'tabulated' not in o and 'png' not in o]
    dfs = []
    for csv in csvs:
        x = os.path.basename(csv).split('_')
        lr = float(x[2])
        batch_size = int(x[3].split('-')[0])
        df = pd.read_csv(csv)
        df['lr'] = lr
        df['batch_size'] = batch_size
        dfs.append(df)
    df = pd.concat(dfs)

    df.to_csv(output)

    df = df[df['Step']<10]

    sns.catplot(df, x='Step', y='Value', row='lr', hue='batch_size', kind='point', sharey=False, log_scale=True, errorbar=None)
    plt.savefig("/data/ssd1/russelsa/tabulated/lr_sweep/output.png")

    df.to_csv(output)
    
    return


def tb_tabulator(directory):
    dfs = []
    corpora = [o for o in os.listdir(directory) if os.path.isdir(os.path.join(directory, o)) and 'lr_sweep' not in o]
    for corpus in corpora:
        for split in ['train', 'val']:
            for objective in ['vad', 'vap']:
                csv_topdir = os.path.join(directory, corpus, split, objective)
                csvs = [os.path.join(csv_topdir, o) for o in os.listdir(csv_topdir)]
                for csv in csvs:
                    _df = pd.read_csv(csv)
                    fold = int(re.findall(r'fold\_[0-9]',csv)[0][-1])
                    _df['run']=corpus
                    _df['fold']=fold
                    _df['split']=split
                    _df['objective']=objective
                    dfs.append(_df)
    df = pd.concat(dfs)
    df.to_csv("/data/ssd1/russelsa/tabulated/runs.csv")
    return


def best_epochs(csv):
    
    df = pd.read_csv(csv)
    df_val = df[df.split=='val']
    df_val = df_val[df_val['objective']=='vap']
    df_val['loss'] = df_val.groupby(['run', 'fold', 'Step'])['Value'].transform('sum')

    best = df_val.groupby(['run', 'fold'])['loss'].idxmin() 
    best= df_val.loc[best]

    best.to_csv("turn_taking/results/best_epoch_fold.csv")
    return
    

if __name__=="__main__":
    # lr_sweep_tabulator()
    # directory = "/data/ssd1/russelsa/tabulated"
    # tb_tabulator(directory)
    # csv = "/home/russelsa@ad.mee.tcd.ie/github/turn_taking/results/results - runs.csv"
    # best_epochs(csv)
