from flask import Blueprint, render_template, request, url_for
from mbti.forms import Question_1_Form, Question_2_Form
from werkzeug.utils import redirect

# MBTI test 진행
bp = Blueprint('test', __name__, url_prefix='/test') 

##########################################################################

@bp.route('/question_1', methods=['GET','POST'])
def E_I_question():
    form = Question_1_Form()

    print(request.method)
    print(form.validate_on_submit())

    if request.method == 'POST' and form.validate_on_submit():
        if request.form['comment'] == 'hi':
            print('ok')
        else:
            return redirect(url_for('question_2'))
    return render_template('test.html', form=form)

###########################################################################
@bp.route('/question_2', methods=['GET','POST'])
def S_N_question():
    return render_template('test_2.html')