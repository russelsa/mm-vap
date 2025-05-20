from feat.utils.io import get_test_data_path
import os
from feat import Detector
import time


def test_detector():
    detector = Detector(device='cpu', facepose_model="img2pose-c", n_jobs=40) # n_jobs=8

    test_video_path = "turn-taking-projects/corpora/candor/candor_video/0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a--5cf6231511ba980001e3aa3d.mp4"

    s = time.time()
    video_prediction = detector.detect_video(test_video_path, output_size=None, pin_memory=True, batch_size=64) #, batch_size=32
    print(time.time() - s)
    video_prediction.to_csv("test.csv")



if __name__ == "__main__":
    test_detector()
