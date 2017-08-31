# -*- coding: utf-8 -*- 

"""
Простой REST API для нейросетевого лемматизатора.
https://flask-restful.readthedocs.io/en/latest/index.html
"""

from __future__ import print_function
from flask import Flask
from flask_restful import Resource, Api
from keras.models import Sequential
from keras.models import model_from_json
import numpy as np
import json
import codecs
import pickle
import os
import sys
from Lemmatizer import Lemmatizer


MODEL_FOLDER = '../../../../data'

lemmatizer = None

app = Flask(__name__)
api = Api(app)

# Лемматизатор загружается достаточно долго, поэтому делать это в контексте каждого запроса невозможно.
# https://stackoverflow.com/questions/24251307/flask-creating-objects-that-remain-over-multiple-requests
@app.before_first_request
def load_lemmatizer():
    global lemmatizer
    lemmatizer = Lemmatizer()
    lemmatizer.load(MODEL_FOLDER)


class LemmatizerResource(Resource):
    def get(self, word):
        lemma = lemmatizer.predict(word)
        return {'lemma': lemma}

api.add_resource(LemmatizerResource, '/<string:word>')

if __name__ == '__main__':
    app.run(debug=True)
  