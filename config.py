import os

BASE_DIR = os.path.dirname(__file__)

SQLALCHEMY_DATABASE_URI = 'sqlite:///{}'.format(os.path.join(BASE_DIR, 'mbti_result.db'))

SQLALCHEMY_TRACK_MODIFICATIONS = False

SECRET_KEY = 'dev'
# secret_key가 'dev'라고 들어와야만 인증이됨