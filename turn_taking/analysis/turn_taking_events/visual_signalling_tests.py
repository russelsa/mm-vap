import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import collections


feature_names = {'AU01_r': 'inner brow raise', 
                 'AU02_r': 'outer brow raise', 
                 'AU04_r': 'brow lower',
                 'AU05_r': 'eyelid raise', 
                 'AU06_r': 'cheek raise', 
                 'AU07_r': 'eyelid tighten', 
                 'AU09_r': 'nose wrinkle', 
                 'AU10_r': 'upper lip raise', 
                 'AU12_r': 'lip corner pull', 
                 'AU14_r': 'mouth dimple', 
                 'AU15_r': 'lip corner depressor', 
                 'AU17_r': 'chin raise', 
                 'AU20_r': 'lip stretch', 
                 'AU23_r': 'lip tighten', 
                 'AU25_r': 'lips part', 
                 'AU26_r': 'jaw drop', 
                 'AU45_r': 'blink'}

renaming = {'ekstedt_events-s_pred_neg-speaker-after': "random-speech",
            'ekstedt_events-s_pred_neg-speaker-before': "random-speech",
             'ekstedt_events-random_no_speech-speaker-before': "random-silence",
             'ekstedt_events-random_no_speech-speaker-after': "random-silence"}


def heatmap(features, target_feature, output_name, column_orders, column_renaming):

    features.index = features['key']

    features = features[[f for f in features.columns if target_feature in f and 'acc' not in f and 'T' not in f]]

    features['index'] = features.index

    mean_features = features.groupby('index').median().reset_index()
    mean_features.index = mean_features['index']
    mean_features = mean_features[[f for f in mean_features.columns if 'index' not in f]].astype('float')

    mean_features = mean_features.rename(feature_names, axis=1).T
    mean_features = mean_features[column_orders]

    mean_features.columns = column_renaming

    # plt.figure()
    # ax = sns.heatmap(mean_features, cmap="crest", annot=True, fmt=".2f", annot_kws={'size': 5, 'rotation': 90}, vmin=0, vmax=2, cbar=True)
    # ax.vlines([3,5], *ax.get_ylim(), colors=['k', 'k'], linestyles='dotted')
    # ax.vlines([4], *ax.get_ylim(), colors=['k'], linestyles='solid')
    # # ax.set_yticklabels(''*len(ax.get_xticklabels()))
    # plt.savefig(f"{output_name}.pdf", bbox_inches='tight')

    return mean_features


def generate_fau_plot(feature, kill_list=[]):

    windows = pd.read_pickle(analysis_window)

    windows['key'] = windows['key'].replace(renaming)

    column_renaming = ['gap', 'no gap', 'overlap', 'gap', 'no gap', 'overlap']

    print(set(windows['key']))

    # heatmap

    p_values = pd.read_pickle(f"turn_taking/analysis/turn_taking_events/statsig/p_vals_{feature}.pkl")

    targets = [
            'ekstedt_events-holds-speaker-before',
            # 'ekstedt_events-holds-speaker-after',
            'gap_0-holds-speaker-before',
            'roddy_events-overlaps_hold-speaker-before', 

            'ekstedt_events-holds-listener-before',
            # 'ekstedt_events-holds-listener-after',
            'gap_0-holds-listener-before',
            'roddy_events-overlaps_hold-listener-before' 
            ]
    windows_subset = windows[(windows.key.str.contains('|'.join(targets)))]
    h1_holds = heatmap(windows_subset, feature, f"test_{feature}", column_orders=targets, column_renaming=column_renaming)

    ps = []
    for comparison in targets:
        p = p_values[p_values.event==comparison]
        p = p[['p']]
        ps.append(p.reset_index(drop=True))
    ps = pd.concat(ps,axis=1,ignore_index=True)

    h1_holds_statsig = ps
    h1_holds_statsig.columns = h1_holds.columns
    h1_holds_statsig.index = h1_holds.index

    targets = ['ekstedt_events-shifts-speaker-before',
            #    'ekstedt_events-shifts-speaker-after',
               'gap_0-shifts-speaker-before',
               'roddy_events-overlaps_shift-speaker-before', 

                'ekstedt_events-shifts-listener-before',
                # 'ekstedt_events-shifts-listener-after',
                'gap_0-shifts-listener-before',
                'roddy_events-overlaps_shift-listener-before' 
               ]

    windows_subset = windows[(windows.key.str.contains('|'.join(targets)))]
    h2_shifts = heatmap(windows_subset, feature, f"test_{feature}", column_orders=targets, column_renaming=column_renaming)

    ps = []
    for comparison in targets:
        p = p_values[p_values.event==comparison]
        p = p[['p']]
        ps.append(p.reset_index(drop=True))
    ps = pd.concat(ps,axis=1,ignore_index=True)

    h2_shifts_statsig = ps
    h2_shifts_statsig.columns = h2_shifts.columns
    h2_shifts_statsig.index = h2_shifts.index

    targets = ['random-speech', 
            'random-silence'
            ]

    windows_subset = windows[(windows.key.str.contains('|'.join(targets)))]
    h3_speech_no_speech = heatmap(windows_subset, feature, f"test_{feature}", column_orders=targets, column_renaming=['random speech', 'random silence'])

    fig, ax =plt.subplots(1, 3, figsize=(5.5, 6), gridspec_kw={'width_ratios': [1, 0.4, 1.2]})

    max_val = round(max(h1_holds.max().max(), h3_speech_no_speech.max().max(), h2_shifts.max().max()), 2)

    psig = 0.001

    # remove 
    if kill_list != []:
        h1_holds = h1_holds[~h1_holds.index.isin(kill_list)]
        h1_holds_statsig = h1_holds_statsig[~h1_holds_statsig.index.isin(kill_list)]

        h2_shifts = h2_shifts[~h2_shifts.index.isin(kill_list)]
        h2_shifts_statsig = h2_shifts_statsig[~h2_shifts_statsig.index.isin(kill_list)]

        h3_speech_no_speech = h3_speech_no_speech[~h3_speech_no_speech.index.isin(kill_list)]
        
    # HOLDS
    # 
    axx1 = sns.heatmap(h1_holds[h1_holds_statsig<=psig], ax=ax[0], cbar=False, vmin=0, vmax=max_val, cmap="gray_r", annot=False, fmt=".2f", annot_kws={'size': 5, 'rotation': 90})
    axx1.set_title('Before a Hold')
    sec = axx1.secondary_xaxis(location=-0.3)
    sec.set_xticks([1.5, 4.5], labels=['current speaker', 'listener'], rotation=90)
    for key, spine in sec.spines.items():
        spine.set_visible(False)
    # h1_holds[h1_holds_statsig>psig]==0
    # axx1 = sns.heatmap(h1_holds[h1_holds_statsig>psig], ax=ax[0], annot=False, cbar=False, cmap="gray_r")
    # for key, spine in sec.spines.items():
    # spine.set_visible(False)

    # Speech and silence
    axx = sns.heatmap(h3_speech_no_speech, ax=ax[1], cbar=False, vmin=0, vmax=max_val, cmap="gray_r", annot=False, fmt=".2f", annot_kws={'size': 5, 'rotation': 90})
    axx.set_yticklabels(''*len(axx.get_xticklabels()))
    axx.tick_params(left=False) ## other options are right and top

    # SHIFTS
    # 
    axx2 = sns.heatmap(h2_shifts[h2_shifts_statsig<=psig], ax=ax[2], cbar=True, vmin=0, vmax=max_val, cmap="gray_r", annot=False, fmt=".2f", annot_kws={'size': 5, 'rotation': 90}, cbar_kws={'label': 'median FAU intensity'})
    axx2.set_yticklabels(''*len(axx2.get_xticklabels()))
    axx2.tick_params(left=False) ## other options are right and top
    axx2.set_title('Before a Shift')
    sec = axx2.secondary_xaxis(location=-0.3)
    sec.set_xticks([1.5, 4.5], labels=['current speaker', 'future speaker'], rotation=90)
    for key, spine in sec.spines.items():
        spine.set_visible(False)
    # h2_shifts[h2_shifts_statsig>psig]=0
    # axx2 = sns.heatmap(h2_shifts[h2_shifts_statsig>psig], ax=ax[2], annot=False, cbar=False, cmap="gray_r")
    # axx2.set_yticklabels(''*len(axx2.get_xticklabels()))
    # axx2.tick_params(left=False) ## other options are right and top
    # for key, spine in sec.spines.items():
    #     spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{feature}.pdf", bbox_inches="tight")


def mwu_tests(feature = '_r'):

    analysis_window_df = pd.read_pickle(analysis_window)

    # heatmaps

    comparisons = collections.OrderedDict({            
            'ekstedt_events-holds-speaker-before': 'random-speech',
            # 'ekstedt_events-holds-speaker-after': 'random-silence',
            'gap_0-holds-speaker-before': 'random-speech',
            'roddy_events-overlaps_hold-speaker-before': 'random-speech',
            
            'ekstedt_events-holds-listener-before': 'random-silence',
            # 'ekstedt_events-holds-listener-after': 'random-silence',
            'gap_0-holds-listener-before': 'random-silence',
            'roddy_events-overlaps_hold-listener-before': 'random-silence',
            
            'ekstedt_events-shifts-speaker-before': 'random-speech',
            # 'ekstedt_events-shifts-speaker-after': 'random-silence',
            'gap_0-shifts-speaker-before': 'random-speech',
            'roddy_events-overlaps_shift-speaker-before': 'random-speech',
            
            'ekstedt_events-shifts-listener-before': 'random-silence',
            # 'ekstedt_events-shifts-listener-after': 'random-silence',
            'gap_0-shifts-listener-before': 'random-silence',
            'roddy_events-overlaps_shift-listener-before': 'random-silence'
    })

    results = []
    for k, v in comparisons.items():

        targets = [k, v]
    
        windows = analysis_window_df
        windows.index = windows['key']
        windows['index'] = windows.index

        A = windows[windows.index==k]
        B = windows[windows.index==v]

        features=A.columns
        features=windows[[f for f in windows.columns if feature in f and 'acc' not in f and 'T' not in f]]
        for column in features:

            a, b = A[column], B[column]
            a, b = a.dropna(), b.dropna()
            a_median, b_median = a.median(), b.median()
            a, b = a.to_list(), b.to_list()
            result = scipy.stats.mannwhitneyu(a, b)

            results.append({"event": k, "compared to": v, "feature": column, "p": result[1]})
    
    results = pd.DataFrame(results)
    results.to_pickle(f"turn_taking/analysis/turn_taking_events/statsig/p_vals_{feature}.pkl")
            


if __name__ == "__main__":

    # group all
    analysis_window = "analysis_windows_200ms_final.pkl"

    # windows_dir = f"/data/ssd3/russelsa/{analysis_window.split('.')[0]}"
    # windows = [o for o in os.listdir(windows_dir) if '.pkl' in o]
    # windows = [os.path.join(windows_dir, o) for o in windows]
    # windows = pd.concat([pd.read_pickle(o) for o in windows])
    # windows['key'] = windows['event']  + '-' +  windows['role'] + '-' + windows['time'] 
    # windows['key'] = windows['key'].replace(renaming)
    # windows.to_pickle(analysis_window)
    
    # mwu_tests(feature='_r')
    generate_fau_plot(feature='_r', kill_list=['outer brow raiser', 'upper lid raiser', 'nose wrinkler'])

    # mwu_tests(feature='pose')
    # generate_fau_plot(feature='pose')

    # mwu_tests(feature='gaze')
    # generate_fau_plot(feature='gaze')