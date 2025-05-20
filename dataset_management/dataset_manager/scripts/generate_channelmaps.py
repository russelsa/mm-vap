import pandas as pd 
import os 
import json



if __name__ == "__main__":
    channelmaps_dir = 'turn-taking-projects/corpora/candor/channelmaps'
    channelmaps = os.listdir(channelmaps_dir)

    channelmaps = os.listdir(channelmaps_dir)
    channel_hash = []

    for channelmap in channelmaps:
        
        id = os.path.basename(channelmap.split('.')[0])
        channelmap_file = os.path.join(channelmaps_dir, channelmap)
        channel_id = json.load(open(channelmap_file))

        channel_hash.append((id, channel_id["L"],channel_id["R"]))

    channel_hash = pd.DataFrame(channel_hash, columns=['id', 'L', 'R'])
    pd.to_pickle(channel_hash, 'turn-taking-projects/corpora/candor/channelmaps.pkl')
