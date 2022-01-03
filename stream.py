# develop a classifier for the 5 Celebrity Faces Dataset
from random import choice
import numpy as np
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
import cv2
from mtcnn.mtcnn import MTCNN
import embedd
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
parser.add_argument('--video', type=str, default=ROOT / 'videos/masked.mp4', help='dataset for training')
parser.add_argument('--model', type=str, default=ROOT / 'model/facenet_keras.h5', help='facenet model')
parser.add_argument('--embedding_data', type=str, default=ROOT / 'database/face_embedding.npz', help='faces output name')
parser.add_argument('--vis', action='store_true', default=False, help='Visualization')
parser.add_argument('--write', action='store_true', default=False, help='Write Video Output')
parser.add_argument('--output', type=str, default=ROOT / 'video/output.avi', help='output name')


opt = parser.parse_args()

# load face detector
detector = MTCNN()
# load the facenet model
embedd_model = load_model('/home/didi/Repository/Nodeflux/masked-face-recognition/model/facenet_keras.h5')
print('Loaded Model')

# load face embeddings
data = np.load(opt.embedding_data)
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)

# stream video
cap = cv2.VideoCapture(opt.video)

if (cap.isOpened() == False):
    print('Cannot Open Video')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

output = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*'MJPG'), 15, size)

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret==True:
        results = detector.detect_faces(frame)
        x1, y1, width, height = results[0]['box']

        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        # extract the face
        face = cv2.resize(frame[y1:y2, x1:x2], (160, 160))

        # embedd face
        face_embedd = embedd.get_embedding(embedd_model, face)

        # prediction for the face
        samples = expand_dims(face_embedd, axis=0)
        yhat_class = model.predict(samples)
        yhat_prob = model.predict_proba(samples)

        # get name
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)

        # draw bbox and name
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{predict_names} : {class_probability}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2, cv2.LINE_AA)

        if opt.vis:
            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        if opt.write:
            output.write(frame)
            
    else:
        break

# release video
cap.release()
# release output
output.release()
# destroy all windows
cv2.destroyAllWindows()