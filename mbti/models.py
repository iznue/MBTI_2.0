from mbti import db

class MBTI_result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ei = db.Column(db.Text(), nullable=True)
    sn = db.Column(db.Text(), nullable=True)
    tf = db.Column(db.Text(), nullable=True)
    jp = db.Column(db.Text(), nullable=True)
    q2_answer = db.Column(db.Text(), nullable=True)
    create_date = db.Column(db.DateTime(), nullable=True)