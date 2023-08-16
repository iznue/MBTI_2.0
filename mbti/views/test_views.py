from datetime import datetime
from flask import Blueprint, render_template, request, url_for
from mbti.forms import Question_1_Form, Question_2_Form
from werkzeug.utils import redirect

from mbti import db
from mbti.models import E_I_answer, S_N_answer

# MBTI test 진행
bp = Blueprint('test', __name__, url_prefix='/test') 

##########################################################################

@bp.route('/question_1')
def E_I_question():
    form = Question_1_Form()
    return render_template('test.html', form=form)

@bp.route('/question_1_create', methods=['GET','POST'])
def E_I_create():
    content = request.form['content']
    comment = E_I_answer(content=content, create_date=datetime.now())
    print('ok')
    return comment

###########################################################################
@bp.route('/question_2', methods=['GET','POST'])
def S_N_question():
    return render_template('test_2.html')