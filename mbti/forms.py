from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField
from wtforms.validators import DataRequired

# E&I question answer
class Question_1_Form(FlaskForm):
    content = TextAreaField('content', validators=[DataRequired()])

# S&N question answer
class Question_2_Form(FlaskForm):
    content = TextAreaField('content', validators=[DataRequired()])