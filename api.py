from flask import Flask
from rag import run_engine

app = Flask(__name__)


@app.route("/<prompt>")
def hello_world(prompt):
    return run_engine(prompt)
