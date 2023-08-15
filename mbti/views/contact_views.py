from flask import Blueprint, render_template

# contact : 개발자 정보 및 기타내용 명시
bp = Blueprint('contact', __name__, url_prefix='/contact') 

@bp.route('/')
def information():
    return render_template('contact.html')