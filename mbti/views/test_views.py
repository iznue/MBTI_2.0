from datetime import datetime
from flask import Blueprint, render_template, request, url_for, session, current_app, flash
from mbti.forms import Question_1_Form, Question_2_Form
from werkzeug.utils import redirect, secure_filename
from keras.preprocessing.sequence import pad_sequences
from mbti.models import E_I_answer, S_N_answer
from ultralytics import YOLO
from PIL import Image
import torch
from pytorch_transformers import BertTokenizer, BertForSequenceClassification, BertConfig 
import cv2
import re
import glob
import os
import numpy as np
import torch.nn.functional as F

# MBTI test 진행
bp = Blueprint('test', __name__, url_prefix='/test') 

# basic func
##########################################################################
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# 파일 업로드를 저장할 폴더를 설정하세요
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 허용할 파일을 확인하는 함수
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_save_path():
    return os.path.join(current_app.root_path, 'static', 'result_image')

def get_latest_image_path_for_template():
    save_path = get_save_path()
    existing_results = glob.glob(os.path.join(save_path, 'results*.jpg'))
    latest_image_path = max(existing_results, key=os.path.getmtime)

    # 상대 경로로 변환
    relative_path = os.path.relpath(latest_image_path, os.path.join(current_app.root_path, 'static'))

    # URL 변환
    url_path = relative_path.replace('\\', '/')

    return url_path

def pad_sequences_1(sequences, max_length, padding_value=0):
    padded_sequences = np.full((len(sequences), max_length), padding_value, dtype=int)
    for i, seq in enumerate(sequences):
        padded_sequences[i,:len(seq)] = seq
    return padded_sequences

def encode_texts_1(tokenizer, texts, max_length=None):
    encodings = []
    masks = []

    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=True)
        encodings.append(encoded)

    if max_length is None:
        max_length = max(len(encoding) for encoding in encodings)
    
    encodings = pad_sequences_1(encodings, max_length)
    masks = (encodings != 0) * 1

    return encodings, masks

# Models
##########################################################################

device = torch.device("cpu")

# E & I 추론을 위한 model
# config = BertConfig.from_pretrained('bert-base-multilingual-cased')
# model_1 = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', config=config)

# S & N 추론을 위한 model
tokenizer_2 = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)
model_2 = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
model_2.to(device)
model_2.load_state_dict(torch.load('best_model_0.7540_second_comment.pth'))
model_2.eval()

# E & I
##########################################################################
@bp.route('/question_1')
def E_I_question():
    session.clear() # 웹브라우저 방문(사용자마다) session 초기화
    return render_template('test.html')

# E & I 추론 작업
@bp.route('/question_2', methods=['GET','POST'])
def E_I_predict():
    data_1 = request.form['comment_1'] # test.html의 form key 값을 받아옴

    ########################## predict ##########################
    class_name_1=['E', 'I']
    
    encoding, attention_mask = encode_texts_1(tokenizer_2, [data_1], 128)
    input_ids = torch.tensor(encoding).to(device)
    attention_mask = torch.tensor(attention_mask).to(device)

    with torch.no_grad():
        outputs = model_2(input_ids, attention_mask=attention_mask)
        logits = outputs[0]        

        # 소프트맥스 함수를 적용하여 확률 값 계산
        # probabilities = F.softmax(logits, dim=1)
        predicted = torch.argmax(logits, 1)

    pred_1 = predicted.item()
    predict_1 = class_name_1[pred_1]

    session['E&I'] = predict_1
    print(session)
    return render_template('test_2.html') # test_2.html에 추론 결과 전달

# S & N
###########################################################################
# S & N 추론 작업
@bp.route('/question_3', methods=['GET','POST'])
def S_N_predict():
    data_2 = request.form['comment_2']
    
    ########################## predict ##########################
    class_name_2=['N', 'S']
    MAX_LEN = 87

    token_text = tokenizer_2.tokenize(data_2)
    input_ids = [[tokenizer_2.convert_tokens_to_ids(tokens) for tokens in token_text]]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    input_ids = torch.tensor(input_ids)
    
    input_ids = input_ids.to(device)

    with torch.no_grad():
        output = model_2(input_ids)

    output = output[0]
    pred_2 = torch.argmax(output, 1)
    
    predict_2 = class_name_2[pred_2]

    session['S&N'] = predict_2 # 추론 결과를 session에 value값으로 저장함
    # Mbti_pred['S&N'] = predict_2
    print(session)
    return render_template('test_3.html')

# T & F
###########################################################################
# T & F 추론 작업
@bp.route('/question_4', methods=['GET','POST'])
def T_F_predict():
    session['T&F'] = 'T'
    return render_template('test_4.html')


# P & J
###########################################################################
# P & J 추론 작업
@bp.route('/result', methods=['GET','POST'])
def P_J_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Load the trained model
            model = YOLO('best.pt')
            # Perform object detection on the image
            model_results = model.predict(source=file_path, show=False, save=True)

            for r in model_results:
                im_array = r.plot()
                im = Image.fromarray(im_array[..., ::-1])
                im.show()
                save_path = get_save_path()
                existing_results = glob.glob(os.path.join(save_path, 'results*.jpg'))
                results_count = len(existing_results)

                # Create a new file name with the next sequential number
                new_file_name = f'results{results_count + 1}.jpg'

                # Save the image with the new file name and the specified path
                full_save_path = os.path.join(save_path, new_file_name)
                im.save(full_save_path)

            icons_count = len(model_results[0].boxes)

            # Initialize object counter
            if icons_count < 9 or icons_count == 0:
                final_label = 'J'
            elif icons_count >= 9:
                final_label = 'P'

            # Save icon count and final label result in the results list
            results = [{'image_path': file_path, 'icons_count': icons_count, 'final_label': final_label}]

            session['P&J'] = final_label

    return render_template('result.html')
    # result.html에서 session 값을 통해 사용자의 mbti 검사 결과를 출력함