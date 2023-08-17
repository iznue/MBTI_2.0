from datetime import datetime
from flask import Blueprint, render_template, request, url_for
from mbti.forms import Question_1_Form, Question_2_Form
from werkzeug.utils import redirect

from mbti import db
from mbti.models import E_I_answer, S_N_answer

import torch
from pytorch_transformers import BertTokenizer, BertForSequenceClassification

# MBTI test 진행
bp = Blueprint('test', __name__, url_prefix='/test') 

Mbti_pred ={}

device = torch.device("cpu")
tokenizer_2 = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)
model_2 = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
model_2.to(device)
model_2.load_state_dict(torch.load())

# E & I
##########################################################################
@bp.route('/question_1')
def E_I_question():
    return render_template('test.html')

# E & I 추론 작업
@bp.route('/question_1', methods=['GET','POST'])
def E_I_predict():
    data_1 = request.form['comment_1'] # test.html의 form key 값을 받아옴
    Mbti_pred['E&I'] = data_1
    print(data_1)
    return render_template('test_2.html', data=data_1) # test_2.html에 추론 결과 전달

# S & N
###########################################################################
# S & N 추론 작업
@bp.route('/question_2', methods=['GET','POST'])
def S_N_predict():
    data_2 = request.form['comment_2']
    ## predict ##
    class_name=['N', 'S']
    print(data_2)
    print(Mbti_pred)
    Mbti_pred['S&N'] = data_2
    return render_template('test_3.html')