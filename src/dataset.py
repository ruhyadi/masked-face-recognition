"""
Dataset preprocessing
"""

import os
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, random_split
from facenet_pytorch import MTCNN
import utils

class FaceDataset(Dataset):
    def __init__(self, dataset_dir, fps, render, rotate, device):
        # directory
        self.dataset_dir = dataset_dir
        self.fps = fps
        self.render = render
        self.rotate = rotate
        self.device = device

        # extract face
        self.FACES, self.LABELS = self._extract_face()

    def __len__(self):
        return len(self.LABELS)

    def __getitem__(self, index):
        # extract face
        face = self.FACES[index]
        label = self.LABELS[index]

        return face, label

    def _extract_face(self):
        # get face class
        CLASS = [class_ for class_ in os.listdir(self.dataset_dir)]
        # data and labels
        FACES = []
        LABELS = []

        for class_ in CLASS:
            # get videos filename
            class_dir = os.path.join(self.dataset_dir, class_)
            VIDEOS = [vid for vid in os.listdir(class_dir) if vid.endswith(('mp4', 'avi'))]
            # loop thru every videos
            for vid_fn in VIDEOS:
                frame_fn = vid_fn.split('.')[0] # for saving name
                vid_path = os.path.join(self.dataset_dir, class_, vid_fn)

                # read video
                cap = cv2.VideoCapture(vid_path)
                if cap.isOpened() == False:
                    print('[INFO] Cannot read video...')

                # get video intrinsic
                cap_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                cap_fps = cap.get(cv2.CAP_PROP_FPS)
                cap_length = int(cap_count / cap_fps) # second

                # get frame per minute from video
                num_frame = int(cap_length * self.fps)
                frame_ids = [int(i) for i in np.linspace(0, cap_count, num_frame)]
                face_id = 0 # id index for face
                i = 0 # index frame

                with tqdm(desc=f'Extracting Face {frame_fn}', total=cap_count) as pbar:
                    while(cap.isOpened()):
                        ret, frame = cap.read()
                        if ret==True:
                            if i in frame_ids:
                                # rotate face if needed
                                if self.rotate:
                                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                                if self.render:
                                    cv2.imshow('face', frame)
                                    cv2.waitKey(0)
                                # extract face
                                face = self._detect_face(frame)
                                if face is not None:
                                    FACES.append(face)
                                    LABELS.append(class_)
                                    face_id += 1

                                # if faces is not None:
                                #     for face in faces:
                                #         # append faces and labels
                                #         print(face.shape)
                                #         FACES.append(face)
                                #         LABELS.append(class_)
                                #         face_id += 1
                        else:
                            break
                        pbar.update(1)
                        i = i + 1

                print(f'[INFO] #Extracted Face from {class_} {frame_fn}: {face_id}')
            
            # # balancing dataset
            # self._balancing_dataset(class_dir)
            # print(f'[INFO] Balancing Dataset...')

        return FACES, LABELS

    def _detect_face(self, frame):
        # init detector
        detector = MTCNN(margin=10, min_face_size=50, device=self.device)

        # detect face from image
        # box format [x1, y1, x2, y2]
        # boxes, _ = detector.detect(frame)
        face = detector(frame)
        return face

        # slice face from image
        # img[y1:y2, x1:x2]
        # if boxes is not None:
        #     faces = [
        #         frame[int(box[1]): int(box[3]), 
        #             int(box[0]): int(box[2])] 
        #         for i, box in enumerate(boxes) \
        #         if box[0] > 0 and box[1] > 0  # non negative
        #         ]

        #     return faces
        # else:
        #     return None

    def _balancing_dataset(self, class_dir):
        IMAGES = [img for img in os.listdir(class_dir) if img.endswith(('png', 'jpg', 'jpeg'))]
        # len images
        key = ['Masked', 'Normal']
        len_masked = len([img for img in IMAGES if key[0] in img])
        len_normal = len([img for img in IMAGES if key[1] in img])
        
        # remove over data
        while(len_normal != len_masked):
            if len_masked > len_normal:
                fn = glob.glob(os.path.join(class_dir, 'Masked_*')) # random
                os.remove(fn[0])
                len_masked = len_masked - 1
            elif len_masked < len_normal:
                fn = glob.glob(os.path.join(class_dir, 'Normal_*')) # random
                os.remove(fn[0])
                len_normal = len_normal - 1

if __name__ == '__main__':
    dataset_dir = '/home/didi/Repository/masked-face-recognition/data'
    fps = 2
    dataset = FaceDataset(dataset_dir, fps, False, True, 'cpu')

    dataset_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in dataset_loader:
        x, y = batch
        print(x.shape, y)
