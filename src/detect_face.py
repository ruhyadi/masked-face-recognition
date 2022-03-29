"""
Detecting face from image
"""

import os
import numpy as np
from PIL import Image
import cv2
from facenet_pytorch import MTCNN
import torch


from torchvision.transforms import transforms

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

def extract_face(filename, required_size=(160, 160)):
    """
    Return face image from whole image
    """
    # load image
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)

    # init detector
    detector = MTCNN(keep_all=True, device=device)

    # detect face from image
    # box format [x1, y1, x2, y2]
    boxes, _ = detector.detect(img)

    # slice face from image
    # img[y1:y2, x1:x2]
    faces = [
        img[int(box[1]): int(box[3]), 
            int(box[0]): int(box[2])] 
        for i, box in enumerate(boxes) \
        if box[0] > 0 and box[1] > 0  # non negative
        ]

    return faces

def load_faces(dir):
    faces = list()

    faces = [extract_face(path) for path in ]


    for filename in os.listdir(dir):
        
        path = os.path.join(dir, filename)


if __name__ == "__main__":
    img_path = '/home/didi/Repository/masked-face-recognition/src/face.png'

    faces = extract_face(img_path)
    for face in faces:
        cv2.imshow('img', face)
        cv2.waitKey(0)