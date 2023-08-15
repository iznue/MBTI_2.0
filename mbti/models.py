from mbti import db

# E&I 질문에 대한 답변 - 고유 번호, 내용, 작성일시
class E_I_answer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text(), nullable=False) # 글자 제한 x
    create_date = db.Column(db.DateTime(), nullable=False) # 빈값 허용 x


# S&N 질문에 대한 답변 - 고유 번호, 내용, 작성일시
class S_N_answer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text(), nullable=False)
    create_date = db.Column(db.DateTime(), nullable=False)