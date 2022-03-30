"""
Utils code for face recognition
"""

import os
from tqdm import tqdm
import glob

import numpy as np
from PIL import Image
import cv2
from facenet_pytorch import MTCNN

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

def extract_face(dataset_dir='data', fps=5, imshow=False, rotate=False):
    # get face class
    CLASS = [class_ for class_ in os.listdir(dataset_dir)]
    
    for class_ in CLASS:
        # get videos filename
        class_dir = os.path.join(dataset_dir, class_)
        VIDEOS = [vid for vid in os.listdir(class_dir) if vid.endswith(('mp4', 'avi'))]
        # loop thru every videos
        for vid_fn in VIDEOS:
            frame_fn = vid_fn.split('.')[0] # for saving name
            vid_path = os.path.join(dataset_dir, class_, vid_fn)

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
            num_frame = int(cap_length * fps)
            frame_ids = [int(i) for i in np.linspace(0, cap_count, num_frame)]
            i = 0 # frame index
            id = 0 # id index
            face_id = 0 # id index for face
            
            with tqdm(desc=f'Extracting Face {frame_fn}', total=cap_count) as pbar:
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if ret==True:
                        filename = os.path.join(dataset_dir, class_, f'{frame_fn}_{id:03d}_{face_id:02d}.png')
                        # save image if image in frame_ids
                        if i in frame_ids:
                            # rotate face if needed
                            if rotate:
                                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                            if imshow:
                                cv2.imshow('face', frame)
                                cv2.waitKey(0)
                            # extract face
                            faces = detect_face(frame)
                            id = id + 1 # add up id index
                            # save every detected face
                            if faces is not None:
                                for face in faces:
                                    cv2.imwrite(filename, face)
                                    face_id = face_id + 1 # add up face index
                    else:
                        break
                    pbar.update(1)
                    i = i + 1 # add up frame index
        
            print(f'[INFO] #Extracted Face from {class_} {frame_fn}: {face_id}')
        
        # balancing dataset
        balancing_dataset(class_dir)
        print(f'[INFO] Balancing Dataset...')

def detect_face(frame):
    # init detector
    detector = MTCNN(keep_all=True, device=device)

    # detect face from image
    # box format [x1, y1, x2, y2]
    boxes, _ = detector.detect(frame)

    # slice face from image
    # img[y1:y2, x1:x2]
    if boxes is not None:
        faces = [
            frame[int(box[1]): int(box[3]), 
                int(box[0]): int(box[2])] 
            for i, box in enumerate(boxes) \
            if box[0] > 0 and box[1] > 0  # non negative
            ]

        return faces
    else:
        return None

def balancing_dataset(class_dir, dataset_dir, valsize):
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

def train_val_split(dataset_dir, class_dir, valsize):
    try:
        os.makedirs(dataset_dir, 'Train')
        os.makedirs(dataset_dir, 'Val')
    except:
        pass
    IMAGES = [img for img in os.listdir(class_dir) if img.endswith(('png', 'jpg', 'jpeg'))]
    # len images
    key = ['Masked', 'Normal']
    len_masked = len([img for img in IMAGES if key[0] in img])
    len_normal = len([img for img in IMAGES if key[1] in img])

# testing
if __name__ == "__main__":
    dataset_dir = '/home/didi/Repository/masked-face-recognition/data'
    fps = 2
    extract_face(dataset_dir, fps, imshow=False, rotate=True)