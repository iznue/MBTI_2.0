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

import google.protobuf

import os

import pandas as pd
from sklearn.preprocessing import StandardScaler

# pkl 파일에 필요한 모듈 ############################################################

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression # linear 주로 회귀지만 분류모델도 있어 logistic regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle #모델 저장
from sklearn import preprocessing

##exe 파일에 필요한 데이터 넣기 ###########################################################################
# xml 파일+ n_s.pkl파일은 넣어놔야함 , csv 도 추가될 예정 


DATA_PATH= 'c:/Users/user/Desktop/MBTI_2.0/data/'

# crop  ##################################################################################################
face_cascade = cv2.CascadeClassifier(os.path.join(DATA_PATH,'haarcascade_frontalface_default.xml')) 


def crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # print('grayshape', gray.shape)
   
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) != 0 :
        print('crop 영역 ', faces)
        (x,y,w,h)= faces[0]
            
        # img = image.copy()
        roi = image[y:y+h,x:x+w]
        resize = cv2.resize(roi,(96,96), interpolation=cv2.INTER_CUBIC)

        # print('resize shape', resize.shape)

        # new_file_path = f'./mediapipe_crop/{cap_time_jpg}'
        # cv2.imwrite(new_file_path, resize)
        print('crop ')
        return resize
    else: 
        print('could not crop ' )

# ##########################################################################################################
# # csv 저장 (mediapipe_facemesh) #########################################################################

def landmark_csv(image, cap_time):
    
    result = holistic.process(image) 
    face = result.face_landmarks.landmark
    
    if face:
        face_list = []

        for temp in face:
            face_list.append([temp.x,temp.y,temp.z])
                
        face_row = list(np.array(face_list).flatten()) # [[x1,y1,z1],[x2,y2,z2]] -> [x1,y1,z1,x2,y2,
        face_row.insert(0,cap_time)


        if path.isfile(os.path.join(DATA_PATH,'mediapipe_facemesh.csv'))==False:# 1. csv파일이 없는 경우-> csv 파일 만들기 
                                                    
            landmarks = ['cap_time'] # 최종형태 : ['cap_time', 'x1', 'y1', 'z1', 'x2', 'x3',... ]
    
            for val in range(1,len(face)+1):
            
                landmarks += ['x{}'.format(val),'y{}'.format(val),'z{}'.format(val)]
    
            with open(os.path.join(DATA_PATH,'mediapipe_facemesh.csv') , mode='w', newline='') as f:
                
                print('write csv')
    
                csv_writer = csv.writer(f,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(landmarks)
        else:
            with open(os.path.join(DATA_PATH,'mediapipe_facemesh.csv'),mode='a',newline='') as f: # mode='a'에서 'w'로 변경함
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(face_row)
            f.close()      
    else:
        pass

     
# ###################################################################

warnings.filterwarnings('ignore') #경고 뜨는거 무시 
model = joblib.load(os.path.join(DATA_PATH,'ns.pkl')) # 모델 부르는  


mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture(0)

#300
video_length = 100 # 비디오 초 : 277.5454545454541
                   # 프레임수 : 6550
                   # 프레임 너비(픽셀) : 1920
                   # 프레임 너비(픽셀) : 1080  
                   # fps : 23.976023976023978

frames = []
start_time = cv2.getTickCount()  # 시작 시간


cycle = 10 # 10 프레임당 
n =0
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while True:
  
        ret, frame = cap.read()
        cap_time = datetime.now().strftime('%Y%m%d_%H%M%S%f')
        # print('captime', cap_time)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    
            
        image.flags.writeable = False  # 영상에서 캡처한 이미지 변하지 않도록 w1. writable = False, w2. copy()

        if n == cycle:      
            try:
                # print(n)
                resize=crop(image)
                landmark_csv(resize, cap_time)
            except:            
                pass                  
            n=0
        n += 1

        current_time = cv2.getTickCount()  # 현재 시간
        elapsed_time = (current_time - start_time) / cv2.getTickFrequency()  # 경과 시간(초)

        if elapsed_time >= video_length:
            break
    
        cv2.imshow('face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  ## /////// flask 끝나면 q 눌러야 한다 ////// 
            break
    
    cap.release()
    cv2.destroyAllWindows()


# ################## predict #####################################################
# # ###############################################################################
# yhat = 1 "neutral"  2 "sad"
percentage = 0.8


df= pd.read_csv(os.path.join(DATA_PATH,'mediapipe_facemesh.csv'))
X = df.iloc[:,1:]


try:
    cname = ['1', '2']
    predict_li=model.predict(X)
    predict_li= list(predict_li)

    pred = ''
    if predict_li.count(1) // len(predict_li) >= percentage:                   
        
        pred = 'F'

    else:
        pred = 'T'

    # yhat = model.predict(X)[0]  #0,1
    # yhat = cname[yhat]
    # print(yhat)\

except:
    pass

url = 'http://192.168.0.232:5556/test/get_device_PC2' # 포트번호 : 5555, 5556으로 변경해보기 
# headers = {'Content-Type': 'application/json'}
# data = {'key':pred}
# response = requests.post(url, data=json.dumps(data), headers=headers, timeout= 10)  # POST 요청 보내기

# response = requests.post(url, json= pred, headers=headers, timeout= 10)

data = {'predict':pred, 'user_id':3}
response = requests.post(url, json=data, timeout= 10)
# except:
#     time.sleep(10)
#     response = requests.post(url, pred)


# os.remove(os.path.join(DATA_PATH,'mediapipe_facemesh.csv'))

sys.exit()

############# cmd 창 안닫히게 ###################################################
input()
