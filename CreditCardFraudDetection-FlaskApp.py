from flask import Flask, render_template


cc_fraud_detection_app = Flask(__name__)


@cc_fraud_detection_app.route("/")
def welcome():
    return "welcome from flask app"


if __name__ == "__main__":
    cc_fraud_detection_app.run(debug=True, port=7200)
