from flask import Blueprint, render_template

# MBTI test 진행
bp = Blueprint('test', __name__, url_prefix='/test') 

@bp.route('/question_1')
def E_I_question():
    return render_template('test.html')