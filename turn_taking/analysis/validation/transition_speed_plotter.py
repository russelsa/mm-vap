import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


if __name__ == "__main__":

    df = pd.read_csv("/home/russelsa@ad.mee.tcd.ie/github/turn_taking/results/f1_bal_acc_timings.csv")
    print(df.columns)

    # have 5 folds here... draw a confidence interval? 
    # df = df[df.metric<=1250]
    df['bal_acc'] = df['bal_acc']*100
    df['dataset'] = df['corpus'].str.split('_').str[0]
    # for y in ['f1_0', 'f1_1', 'f1_weighted', 'bal_acc']:
    y='bal_acc'
    fig, ax = plt.subplots()
    plt.suptitle("Balanced Accuracy vs. Minimum Silence between Turns")
    sns.lineplot(df, x='kind', y=y, hue='corpus', markers=True, linestyle='--', errorbar="se", err_style="bars",
                 palette=['lightgray', 'k', 'lawngreen', 'skyblue', 'fuchsia', 'darkviolet'])
    # ax2 = plt.twinx()
    # sns.lineplot(df, x='metric', y='proportion shifts', hue='dataset', linestyle='--')
    plt.legend(prop={'size': 8}, loc='lower left', 
               labels=['SWB (ground-truth)', 'SWB (ASR)', 'CND (audio)', 'CND (video)', 'CND (a+v early fusion)', 'CND (a+v late fusion)'])
    plt.xlabel("Minimum silence between turns [ms]")
    plt.ylabel("Balanced accuracy (%)")
    # g = sns.catplot(x="metric", y=y,  data=df, kind="box", hue='corpus')
    # g.map(sns.swarmplot, 'metric', y, color='k', order=sorted(df.metric.unique()))
    x=[0,250,500,750,1000,1250,1500]
    ax.set(xticks=x)
    plt.savefig(f"lineplot_test_{y}.pdf")