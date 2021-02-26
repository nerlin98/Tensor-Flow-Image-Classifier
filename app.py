from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2

# Keras
import tensorflow
from tensorflow.keras.models import load_model

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
UPLOAD_FOLDER = r'C:\Users\mebin\OneDrive\Documents\CNN2\static'

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')
labels={0:'Narendra Modi',1:'Bill Gates',2:'Lionel Messi',3:'Elon Musk',4:'Mohanlal'}


def model_predict(img_path, model):
    x = []
    img = cv2.imread(str(img_path))
    img_arr_resized=cv2.resize(img,(224,224))
    img_arr_final=img_arr_resized/255

    # Preprocessing the image
    x.append(img_arr_final)
    X = np.array(x)

    preds = model.predict(X)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = os.path.join(UPLOAD_FOLDER,f.filename)
        f.save(file_path)

        # Make prediction
        pred = model_predict(file_path, model)

      # ImageNet Decode
        res = np.argmax(pred)
        result = labels[res]
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

