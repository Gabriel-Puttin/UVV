from app import app


def init_app(app):

@app.route("/")
    def index():
        return "Ola Mundo!"


    @app.route("/contato")
    def contato():
        return "<form><input type='text'></form>"
