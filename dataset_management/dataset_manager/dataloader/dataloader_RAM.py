from dataset_management.dataset_manager.dataloader.dataloader import ChunkedDataset, AudioVisualDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import psutil

# video_directory = 'turn-taking-projects/corpora/candor/candor_openface_pkl/'
# video_format = '.pkl'
# channelmaps = 'turn-taking-projects/corpora/candor/channelmaps.pkl'
# pickle_file = '/home/russelsa@ad.mee.tcd.ie/github/dataset_management/dataset_manager/assets/new_folds/candor/fold_0/train'
# wavdir = 'turn-taking-projects/corpora/candor/candor_wav'

# audio_dataset = AudioVisualDataset(video_directory, video_format, channelmaps, pickle_file=pickle_file, mode='VAP', wavdir=wavdir)
# dataloader = DataLoader(audio_dataset, batch_size=16, shuffle=True, pin_memory=False, num_workers=4, prefetch_factor=2)

# pbar = tqdm(total=len(dataloader))

# dl = iter(dataloader)
# # for i, batch in enumerate(dataloader):
# while True:

#     try:
#         item = next(dl)
#     except StopIteration:
#         dl = iter(dataloader)
#         item = next(dl)
#     memory = psutil.virtual_memory().used

#     # if i%100==0:
#     print(memory/1_000_000_000)
#     # pbar.update()

def z():
    x=2

def y():
    z()
    print(x)

if __name__ == "__main__":
    x = 1
    y()