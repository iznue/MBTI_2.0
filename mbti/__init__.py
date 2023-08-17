from flask import Flask, session
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

import config

db = SQLAlchemy()
migrate = Migrate()

def create_app():
    app = Flask(__name__)
    app.config.from_object(config)
    app.secret_key = 'mbti_2.0' # session 사용을 위한 키 설정

    db.init_app(app)
    migrate.init_app(app, db)

    from . import models

    from .views import main_views, test_views, contact_views
    app.register_blueprint(main_views.bp)
    app.register_blueprint(test_views.bp)
    app.register_blueprint(contact_views.bp)

    return app