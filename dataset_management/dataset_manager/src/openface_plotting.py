import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2


def draw_landmarks(video_path, xx, yy, start_frame, df_pose, df_gaze):

    # fps
    fps = 30

    # in
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(start_frame-1,0))
    # cap.set(2,max(start_frame-1,0))

    # out
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = None

    frame_no = 0
    ret, frame = cap.read()

    # camera intrinsics: estimated in the same way as openface
    image_width = max(frame.shape[0], frame.shape[1])
    image_height = min(frame.shape[0], frame.shape[1])
    cx, cy = image_width/2, image_height/2
    fx = 500*(image_width/640)
    fy = 500*(image_height/480)
    fx = (fx+fy)/2
    fy = fx


    cube_vertices = np.array([
        [0,0,0],  # Vertex 0
        [1,0,0], # 1
        [1,1,0], # 2
        [0,1,0], # 3
        [0,1,1], # 4
        [1,1,1],
        [1,0,1],
        [1,1,1], # 5
        [1,1,0],
        [0,1,0],
        [0,1,1],
        [0,0,1],
        [0,0,0],
        [0,1,0],
        [0,0,0],
        [0,0,1],
        [1,0,1],
        [1,0,0],
        [0,0,0],
        [0,0,1],
        [0,1,1]
        # [1,0,1], # 6
        # [0,1,1], # 4
        # [0,1,0],
        # [1,1,0],
    ])

    centre_of_head = np.array([0,0,0]).astype(float).reshape(3,1)

    cube_vertices = cube_vertices-0.5
    cube_vertices = cube_vertices*100


    P_c = np.array(cube_vertices).T.astype(float)
    cameraMatrix=np.array([[fx, 0, cx], [0, fy, cy], [0,0,1]])
            
    while ret and (frame_no<xx.shape[0]):

        # annotations idx
        annot_idx = frame_no

        # eye landmarks
        _x, _y = xx[annot_idx, :], yy[annot_idx, :]
        eye_lmks = list(zip(_x, _y))

        for lmk in eye_lmks:
            cv2.circle(frame, (lmk), 1, (0, 255, 0), -1)
        # cv2.circle(frame, (gx0p, gy0p), 1, (255, 255, 0), -1)

        if df_gaze is not None:
            gaze = df_gaze.iloc[annot_idx, :]
            tvec = np.array([gaze['gaze_0_x'].item(), gaze['gaze_0_y'].item(), gaze['gaze_0_z'].item()]).astype(float)
            point_of_gaze_left_eye = cv2.projectPoints(centre_of_head, None, tvec, cameraMatrix, distCoeffs=None)
            cv2.arrowedLine(frame, (centre_of_head), (point_of_gaze_left_eye), (255, 255, 0), 1)
        
        if df_pose is not None:
            pose = df_pose.iloc[annot_idx, :]
            rvec = np.array([pose['pose_Rx'].item(), pose['pose_Ry'].item(), pose['pose_Rz'].item()])
            tvec = np.array([pose['pose_Tx'].item(), pose['pose_Ty'].item(), pose['pose_Tz'].item()])
            x, _ =cv2.projectPoints(P_c, rvec, tvec, cameraMatrix, distCoeffs=None)
            cube_points = []
            for i in range(x.shape[0]):
                xp = int(x[i,0,0])
                yp = int(x[i,0,1])
                # cv2.circle(frame, (xp, yp), 1, (255, 255, 0), -1)
                cube_points.append((xp,yp))
            for prev, curr in zip(cube_points, cube_points[1:]):
                if prev != curr:
                    cv2.line(frame, prev, curr, (255, 0, 0), 1)
            x=1

        if out is None:
            out = cv2.VideoWriter("out.mp4", fourcc=fourcc, fps=fps, frameSize=(frame.shape[1], frame.shape[0]))

        # write
        out.write(frame)

        ret, frame = cap.read()
        frame_no += 1


def get_facial_landmarks(df):
    xx = df[[c for c in df.columns if 'x_' in c and 'gaze' not in c]]
    yy = df[[c for c in df.columns if 'y_' in c and 'gaze' not in c]]
    return xx, yy

def get_pose(df):
    xx = df[['pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz']]
    return xx

def get_gaze(df):
    return df[['gaze_0_x', 'gaze_0_y', 'gaze_0_z']]



if __name__ == "__main__":

    csv_file = "/data/ssd2/russelsa/candor_openface_features/67836c1d-1334-41a0-a33a-4f788e8b6fb3--5e0acbffdc79bd35336ed6e0.csv"
    video_file = "/data/ssd2/russelsa/candor_video/67836c1d-1334-41a0-a33a-4f788e8b6fb3--5e0acbffdc79bd35336ed6e0.mp4"

    df = pd.read_csv(csv_file)
    df.columns = [c.strip() for c in df.columns]

    xx, yy = get_facial_landmarks(df)
    xx, yy = xx.iloc[0:30*20,:].to_numpy().astype('int'), yy.iloc[0:30*20,:].to_numpy().astype('int')

    df_pose = get_pose(df)
    df_gaze = get_gaze(df)

    draw_landmarks(video_file,xx,yy, start_frame=0, df_pose=df_pose, df_gaze=None)

    csv_file = "/data/ssd2/russelsa/candor_openface_features/67836c1d-1334-41a0-a33a-4f788e8b6fb3--5cc604bd3bbb120018a3c0e2.csv"
    df1 = pd.read_csv(csv_file)
    df1.columns = [c.strip() for c in df1.columns]
     
    m=df.mean(axis=0)
    df = (df - df.min())/(df.max()-df.min())
    df1 = (df1 - df1.min())/(df1.max()-df1.min())

    df = (df - df.mean(axis=0))
    df1 = (df1 - df1.mean(axis=0))

    plt.figure()
    sns.histplot(df['pose_Ry'], label='A')
    sns.histplot(df1['pose_Ry'], label='B')
    plt.savefig("poseRy.png")

    plt.figure()
    sns.histplot(df['pose_Rx'])
    sns.histplot(df1['pose_Rx'])
    plt.savefig("poseRx.png")

    plt.figure()
    sns.histplot(df['pose_Rz'])
    sns.histplot(df1['pose_Rz'])
    plt.savefig("poseRz.png")

    plt.figure()
    sns.histplot(df['gaze_0_x'])
    sns.histplot(df1['gaze_0_x'])
    plt.savefig("gaze_0_x.png")

    plt.figure()
    sns.histplot(df['gaze_0_y'])
    sns.histplot(df1['gaze_0_y'])
    plt.savefig("gaze_0_y.png")

    plt.figure()
    sns.histplot(df['gaze_0_z'])
    sns.histplot(df1['gaze_0_z'])
    plt.savefig("gaze_0_z.png")

    plt.figure()
    sns.histplot(df['confidence'])
    sns.histplot(df1['confidence'])
    plt.savefig("confidence.png")

    plt.figure()
    sns.histplot(df['success'])
    sns.histplot(df1['success'])
    plt.savefig("success.png")

    plt.figure()
    sns.histplot(df['x_33'])
    sns.histplot(df1['x_33'])
    plt.savefig("x_33.png")

    plt.figure()
    sns.histplot(df['x_29'])
    sns.histplot(df1['x_29'])
    plt.savefig("x_29.png")

    plt.figure()
    sns.histplot(df['y_33'])
    sns.histplot(df1['y_33'])
    plt.savefig("x_33.png")

    plt.figure()
    sns.histplot(df['y_29'])
    sns.histplot(df1['y_29'])
    plt.savefig("y_29.png")
