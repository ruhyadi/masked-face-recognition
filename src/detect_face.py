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

if __name__ == "__main__":
    img_path = '/home/didi/Repository/masked-face-recognition/src/face.png'

    faces = extract_face(img_path)
    for face in faces:
        cv2.imshow('img', face)
        cv2.waitKey(0)

    # img = cv2.imread(img_path)
    # x1 = int(box[0][0])
    # # img = img[box.tolist()]
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    # faces = extract_face(img_path)
    # for face in faces:
    #     print(face.shape)
    #     cv2.imshow('img', face)
    #     cv2.waitKey(0)


    # faces = extract_face(img_path)
    # for face in faces:
    #     face = cv2.cvtColor(face.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
    #     cv2.imshow('face', face)
    #     cv2.waitKey(0)

    # transforms = transforms.ToPILImage()

    # res = extract_face(img_path)
    # print(res.shape)
    # res_arr = cv2.cvtColor(res.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
    # print(res_arr.shape)
    # cv2.imshow('image', res_arr)
    # cv2.waitKey(0)