import scipy
import pickle
import scipy.stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if __name__=="__main__":

    # gemaps_windows = "/mnt/storage/turn-taking-projects/corpora/switchboard/switchboard_turn_analysis/gemaps.csv"
    gemaps_windows = "/mnt/storage/turn-taking-projects/corpora/candor/candor_turn_analysis/gemaps.csv"

    df = pd.read_csv(gemaps_windows)

    gemaps_file = "/mnt/storage/turn-taking-projects/corpora/switchboard/switchboard_gemaps/sw2001_1.pkl"
    gemap_names, _ = pickle.load(open(gemaps_file, 'rb'))
    # turn_type,window,who,gemap,value
    print(list(set(df.turn_type)))

    for gemap_name in gemap_names:
        
        # before a shift versus random speech
        df11 = df[(df.turn_type=='s_pred_neg') & (df.window=='before') & (df.who=='speaker') & (df.gemap==gemap_name)].slope
        # df21 = df[(df.turn_type=='shifts') & (df.window=='before') & (df.who=='listener') & (df.gemap==gemap_name)].slope
        df21 = df[(df.turn_type=='holds') & (df.window=='before') & (df.who=='speaker') & (df.gemap==gemap_name)].slope

        df12 = df[(df.turn_type=='s_pred_neg') & (df.window=='before') & (df.who=='speaker') & (df.gemap==gemap_name)].mean_value
        # df22 = df[(df.turn_type=='shifts') & (df.window=='before') & (df.who=='listener') & (df.gemap==gemap_name)].mean_value
        df22 = df[(df.turn_type=='holds') & (df.window=='before') & (df.who=='speaker') & (df.gemap==gemap_name)].mean_value


        df13 = df[(df.turn_type=='s_pred_neg') & (df.window=='before') & (df.who=='speaker') & (df.gemap==gemap_name)].max_value
        # df23 = df[(df.turn_type=='shifts') & (df.window=='before') & (df.who=='listener') & (df.gemap==gemap_name)].max_value
        df23 = df[(df.turn_type=='holds') & (df.window=='before') & (df.who=='speaker') & (df.gemap==gemap_name)].max_value

        df14 = df[(df.turn_type=='s_pred_neg') & (df.window=='before') & (df.who=='speaker') & (df.gemap==gemap_name)].min_value
        # df24 = df[(df.turn_type=='shifts') & (df.window=='before') & (df.who=='listener') & (df.gemap==gemap_name)].min_value
        df24 = df[(df.turn_type=='holds') & (df.window=='before') & (df.who=='speaker') & (df.gemap==gemap_name)].min_value

        def effect_size(y , x):
            return (np.mean(x) - np.mean(y)) / np.sqrt((np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2.0)


        print(gemap_name, 
              scipy.stats.ttest_ind(df11, df21).pvalue, 
            #   effect_size(df11, df21),

              scipy.stats.ttest_ind(df12, df22).pvalue, 
            #   effect_size(df12, df22),
              
              scipy.stats.ttest_ind(df13, df23).pvalue, 
            #   effect_size(df13, df23),
              
              scipy.stats.ttest_ind(df14, df24).pvalue,
            #   effect_size(df14, df24)
              )

