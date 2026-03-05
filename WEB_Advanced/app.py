from flask import Flask
from views import index, contato

def create_app():
    app = Flask(__name__)
    views.init_app(app)
    return app
