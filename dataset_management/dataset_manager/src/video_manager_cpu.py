import time 
import pandas as pd
import torchvision
from torch.utils.data import DataLoader

class VideoManager():

    def __init__(self, video_path: str) -> None:
        self.video_path = video_path


    def get_frames(self, start_time, end_time):
        
        video = torchvision.io.read_video(self.video_path, start_time, end_time, pts_unit="sec")

        return video


class OpenFaceManager():

    def __init__(self, openface_csv_path: str, fps: float) -> None:
        self.openface_csv_path = openface_csv_path
        self.fps = fps


    def get_frames(self, start_time, end_time):
        
        with open(self.openface_csv_path, "r") as f:
            lines = f.readlines()
            frames = lines[1 + int(start_time*self.fps):int(end_time*self.fps)]

        frames = [[float(ff) for ff in f.split(',')] for f in frames]

        return frames
    

def csv_timer(file):
    """ 
        looking at the fastest way to read specific lines from a csv file...
    """

    start = time.time()
    # csv = pd.read_csv(file)
    with open(file, "rb") as f:
        lines = f.readlines()
        target = lines[30000:30500]
        x=1
    end = time.time()

    print(end-start)


def video_timer(file):

    start = time.time()
    video = torchvision.io.read_video(file,  60, 70)
    end = time.time()

    print(end-start)

# if __name__=="__main__":

#     video_path = "/home/russelsa@ad.mee.tcd.ie/github/turn-taking-projects/corpora/candor/candor_video/0a84a137-b947-441c-b94c-a03f4a5851ea--5eac6e70e481222bfb2a8872.mp4"
    
#     video_manager = VideoManager(video_path)
#     openface_manager = OpenFaceManager(csv_path, fps=30)

#     while True:

#         start = time.time()
#         video_frames = video_manager.get_frames(0, 20)
        

#         x=1
