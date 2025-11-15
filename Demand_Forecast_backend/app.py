from flask import Flask
from flask_cors import CORS
from inventory_api import inventory_api

app = Flask(__name__)
CORS(app)

app.register_blueprint(inventory_api)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
