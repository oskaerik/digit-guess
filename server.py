import tensorflow as tf
import numpy as np
import cv2
import math
import base64
import imageio
import io
import random
from flask import Flask, render_template, request

CANVAS = 300
OUT = 28

app = Flask(__name__)
model = tf.keras.models.load_model('model.h5')

@app.route('/')
def server():
	return render_template('index.html', width=CANVAS, height=CANVAS, route='process')

@app.route('/process', methods=['POST'])
def process():
    # Read image in request
    base64_string = request.data
    image = imageio.imread(io.BytesIO(base64.b64decode(base64_string)))

    # Remove color channels and invert colors
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = 255-image

    # If the image if empty, do nothing
    if np.count_nonzero(image) == 0: return ''

    # Dilate image
    _, _, w, h = cv2.boundingRect(cv2.findNonZero(image))
    kernel = int(math.exp(max(w, h)/100))*2 + 1
    kernel = np.ones((kernel, kernel), dtype='uint8')
    image = cv2.dilate(image, kernel=kernel)

    # Crop image, keep non-zero pixels
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(image))
    image = image[y:y+h, x:x+w]

    # Resize image
    resize_to = OUT-8
    if image.shape[0] > image.shape[1]:
        ratio = resize_to / float(image.shape[0])
        dimensions = (int(image.shape[1] * ratio), resize_to)
    else:
        ratio = resize_to / float(image.shape[1])
        dimensions = (resize_to, int(image.shape[0] * ratio))
    image = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)

    # Pad image
    image = np.pad(image, ((math.ceil((OUT-image.shape[0])/2),),
                           (math.ceil((OUT-image.shape[1])/2),)),
                   'constant', constant_values=0)
    image = image[:OUT, :OUT]

    # Save image for debugging
    cv2.imwrite('processed.png', image)

    # Reshape and predict
    image = image.reshape((1, OUT, OUT, 1))
    prediction = model.predict_classes(image)[0]

    # Send response
    return str(prediction)

if __name__ == '__main__':
    app.run()
