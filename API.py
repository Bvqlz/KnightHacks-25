from flask import Flask, jsonify, render_template

app = Flask(__name__)

@app.route("/")
@app.route("/optimize", methods=['POST'])
def collect_data():
    return "<p>This is test</p>"