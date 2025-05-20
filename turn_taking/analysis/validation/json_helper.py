import pandas as pd
import json
import collections
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


jsonpaths = ["turn_taking/results/SWB_GT.json",
             "turn_taking/results/SWB_ASR.json",
             "turn_taking/results/CND.json",
             "turn_taking/results/CND_video_only.json",
            #  "turn_taking/results/CND-->SWB_ASR.json",
            #  "turn_taking/results/SWB_ASR-->CND.json",
             "turn_taking/results/CND_multimodal_early_fusion.json",
             "turn_taking/results/CND_multimodal_late_fusion.json",
            #  "turn_taking/results/combined-->CND.json",
            #  "turn_taking/results/combined-->SWB_ASR.json"
]

# corpora = ['SWB_ASR', 'SWB_GT']
# corpora = ['SWB_ASR', 'CND']
# corpora = ['SWB_ASR', 'CND-->SWB_ASR', 'CND', 'SWB_ASR-->CND']
# corpora = ['CND', 'CND_multimodal_early_fusion', 'CND_multimodal_late_fusion']

dfs_all=[]
for jsonpath in jsonpaths:
    with open(jsonpath, "r") as f:
        jsonobj = json.load(f)
    
    f1s=[]
    score="f1_score"
    score = jsonobj[score]
    for k, v in score.items():
        for kk, vv in v.items():
            f1 = pd.DataFrame(vv)
            f1.columns=['f1_weighted','f1_0', 'f1_1']
            f1['kind'] = k
            f1['metric'] = kk
            f1s.append(f1)

    bas=[]
    score="bal_Accs"
    score = jsonobj[score]
    for k, v in score.items():
        for kk, vv in v.items():
            ba = pd.DataFrame(vv)
            ba.columns = ['bal_acc']
            ba['kind'] = k
            ba['metric'] = kk
            bas.append(ba)
        
    f1 = pd.concat(f1s)
    ba = pd.concat(bas)
    df = f1
    df['bal_acc'] = ba['bal_acc']
    df['corpus'] = os.path.basename(jsonpath).split('.')[0]
    dfs_all.append(df)

df = pd.concat(dfs_all)
# df = df[df.corpus.isin(corpora)]

# metrics = list(set(df.metric))

# for metric in metrics:
#     _df = df[df.metric==metric]
#     _df = _df.melt(id_vars=['metric', 'kind', 'corpus'], value_vars=['bal_acc', 'f1_weighted','f1_0', 'f1_1'], var_name='metric', value_name='value', col_level=None, ignore_index=True)
#     _df = _df.loc[:,~_df.columns.duplicated()].copy()
#     plt.figure()
#     sns.catplot(data=_df, hue='corpus', y='value', col='kind', kind='box', row='metric', sharey=False)
#     plt.savefig(f'turn_taking/results/figures_box/{metric}.png')

print(df.columns)
# df=df[~df['corpus'].str.contains('SWB')]
# df=df[~df['corpus'].str.contains('combined')]
# df = df.groupby(['kind','metric', 'corpus']).mean().reset_index(drop=False)
print(set(df.metric))
df = df[(df.metric.isin(['shift_hold_p_future_before', 'gap_0_p_future', 'gap_500_p_future', 'gap_750_p_future', 'gap_1000_p_future', 'gap_1250_p_future', 'gap_1500_p_future']))]
df['metric'] = df['metric'].replace({"shift_hold_p_future_before": "gap_250"})
df.to_csv("turn_taking/results/f1_bal_acc_timings.csv")

# # do ttests 
# df['event'] = df['kind']+'-'+df['metric']
# for event in set(df.event):

#     A, B = 'CND', 'CND_multimodal_late_fusion'
#     _df = df[df['event']==event]
#     _dfA, _dfB = _df[_df['corpus']==A], _df[_df['corpus']==B]

#     #f0 
#     t = ttest_ind(_dfA.f1_weighted, _dfB.f1_weighted)
#     print(event, A, B, "f0_weighted", _dfA.f1_weighted.mean(), _dfB.f1_weighted.mean(), t[0], t[1])

#     t = ttest_ind(_dfA.f1_0, _dfB.f1_0)
#     print(event, A, B, "f1_0", _dfA.f1_0.mean(), _dfB.f1_0.mean(), t[0], t[1])

#     t = ttest_ind(_dfA.f1_1, _dfB.f1_1)
#     print(event, A, B, "f1_1", _dfA.f1_1.mean(), _dfB.f1_1.mean(), t[0], t[1])

#     t = ttest_ind(_dfA.bal_acc, _dfB.bal_acc)
#     print(event, A, B, "bal_acc", _dfA.bal_acc.mean(), _dfB.bal_acc.mean(), t[0], t[1])

#     x=1
    