# https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
import cv2
import warnings
import joblib
import mediapipe as mp
import pandas as pd
import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import csv
import os
from os import path
import glob

from datetime import datetime
import time
import sys

import requests
import json



# crop  ##################################################################################################3
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
def crop(n, image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # print('grayshape', gray.shape)
   
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) != 0 :
        print('crop 영역 ', faces)
        (x,y,w,h)= faces[0]
            
        # img = image.copy()
        roi = image[y:y+h,x:x+w]
        resize = cv2.resize(roi,(96,96), interpolation=cv2.INTER_CUBIC)

        print('resize shape', resize.shape)

        # new_file_path = f'./mediapipe_crop/{cap_time_jpg}'
        # cv2.imwrite(new_file_path, resize)

        return resize
    else: 
        print('crop 안됨' )

##########################################################################################################
# csv 저장 (mediapipe_facemesh) #########################################################################

def landmark_csv(image, cap_time):
    
    result = holistic.process(image) 
    face = result.face_landmarks.landmark
    
    if face:
        face_list = []

        for temp in face:
            face_list.append([temp.x,temp.y,temp.z])
                
        face_row = list(np.array(face_list).flatten()) # [[x1,y1,z1],[x2,y2,z2]] -> [x1,y1,z1,x2,y2,
        face_row.insert(0,cap_time)

        if path.isfile('mediapipe_facemesh.csv')==False:# 1. csv파일이 없는 경우-> csv 파일 만들기 
                                                    
            landmarks = ['cap_time'] # 최종형태 : ['cap_time', 'x1', 'y1', 'z1', 'x2', 'x3',... ]
    
            for val in range(1,len(face)+1):
            
                landmarks += ['x{}'.format(val),'y{}'.format(val),'z{}'.format(val)]
    
            with open('mediapipe_facemesh.csv' , mode='w', newline='') as f:
    
                csv_writer = csv.writer(f,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(landmarks)

        else:
            with open('mediapipe_facemesh.csv',mode='w',newline='') as f: # mode='a'에서 'w'로 변경함
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(face_row)
            f.close()      
    else:
        pass
     
###################################################################

warnings.filterwarnings('ignore') #경고 뜨는거 무시 
model = joblib.load('ns.pkl') # 모델 부르는  
mp_holistic = mp.solutions.holistic


mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture(0)

video_length = 15
frames = []
start_time = cv2.getTickCount()  # 시작 시간


cycle = 30 # 5 프레임당 
n =0
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while True:        
        ret, frame = cap.read()
        cap_time = datetime.now().strftime('%Y%m%d_%H%M%S%f')
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    
        image.flags.writeable = False  # 영상에서 캡처한 이미지 변하지 않도록 w1. writable = False, w2. copy()

        if n == cycle:      
            try:
                resize=crop(n, image)
                landmark_csv(resize, cap_time)
            except:            
                pass                  
            n=0
        n += 1

        current_time = cv2.getTickCount()  # 현재 시간
        elapsed_time = (current_time - start_time) / cv2.getTickFrequency()  # 경과 시간 (초)

        if elapsed_time >= video_length:
            break
    
        cv2.imshow('face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  ## /////// flask 끝나면 q 눌러야 한다 ////// 
            break
    
    cap.release()
    cv2.destroyAllWindows()


################## predict #####################################################
# ###############################################################################
# yhat = 1 "neutral"  2 "sad"

df= pd.read_csv('mediapipe_facemesh.csv')
X = df.iloc[:,1:]

try:
    cname = ['1', '2']
    predict_li=model.predict(X)

    predict_li= list(predict_li)
    pred = ''
    if predict_li.count(1) > predict_li.count(0):
        
        pred = 'T'

    else:
        pred = 'F'

    

    # yhat = model.predict(X)[0]  #0,1
    # yhat = cname[yhat]
    # print(yhat)\

except:
    pass

url = 'http://192.168.35.145:5556/test/question_3' # 포트번호 : 5555, 5556으로 변경해보기
# headers = {'Content-Type': 'application/json'}
# data = {'key':pred}
# response = requests.post(url, data=json.dumps(data), headers=headers, timeout= 10)  # POST 요청 보내기

# response = requests.post(url, json= pred, headers=headers, timeout= 10)

response = requests.post(url, json=pred, timeout= 10)
# except:
#     time.sleep(10)
#     response = requests.post(url, pred)
sys.exit()


