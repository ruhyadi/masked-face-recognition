import stream
import embedd
import detect

import numpy as np
from keras.models import load_model

import os
import argparse
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# argument
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=ROOT / 'dataset', help='dataset for training')
parser.add_argument('--model', type=str, default=ROOT / 'model/facenet_keras.h5', help='dataset for training')
parser.add_argument('--faces_output', type=str, default=ROOT / 'dataset/faces.npz', help='faces output name')
parser.add_argument('--embedd_ouput', type=str, default=ROOT / 'dataset/embedd.npz', help='embedd output name')
opt = parser.parse_args()

# load train dataset
trainX, trainy = detect.load_dataset(os.path.join(opt.dataset, 'train'))
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = detect.load_dataset(os.path.join(opt.dataset, 'val'))
# save arrays to one file in compressed format
np.savez_compressed(opt.faces_output, trainX, trainy, testX, testy)

# load the face dataset
data = np.load(opt.output, allow_pickle=True)
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)

# load the facenet model
model = load_model(opt.model)
print('Loaded Model')

# convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX:
	embedding = embedd.get_embedding(model, face_pixels)
	newTrainX.append(embedding)
newTrainX = np.asarray(newTrainX)
print(newTrainX.shape)

# convert each face in the test set to an embedding
newTestX = list()
for face_pixels in testX:
	embedding = embedd.get_embedding(model, face_pixels)
	newTestX.append(embedding)
newTestX = np.asarray(newTestX)
print(newTestX.shape)

# save arrays to one file in compressed format
np.savez_compressed(opt.embedd_output, newTrainX, trainy, newTestX, testy)