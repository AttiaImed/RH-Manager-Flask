# face_recognition.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
import PIL
from flask import Flask, request, jsonify

app = Flask(__name__)

model_path = 'data/Files_for_Face_verification_and_Recognition/model'
model = tf.saved_model.load(model_path)

def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
    return loss

FRmodel = model

def img_to_encoding(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)  # Add a dimension of 1 as the first dimension
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)  # Ensure the tensor is of type float32
    # Use the model to make a prediction
    infer = model.signatures['serving_default']
    encoding = infer(input_1=x_train)['Bottleneck_BatchNorm'].numpy()
    return encoding / np.linalg.norm(encoding, ord=2)

database = {
    "kian": img_to_encoding("imed/imed1.jpg", FRmodel),
}

def verify(image_path, identity, database, model):
    encoding = img_to_encoding(image_path, model)
    dist = np.linalg.norm(tf.subtract(database[identity], encoding))
    if dist < 0.7:
        result = f"It's {identity}, welcome in!"
        door_open = True
    else:
        result = f"It's not {identity}, please go away"
        door_open = False
    return dist, door_open, result

def who_is_it(image_path, database, model):
    encoding = img_to_encoding(image_path, model)
    min_dist = 100
    identity = None

    for (name, db_enc) in database.items():
        dist = np.linalg.norm(tf.subtract(db_enc, encoding))
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
        result = "Not in the database."
    else:
        result = f"It's {identity}, the distance is {min_dist}"

    return min_dist, identity, result

@app.route('/verify', methods=['POST'])
def verify_endpoint():
    data = request.get_json()
    image_path = data['image_path']
    identity = data['identity']
    distance, door_open, result = verify(image_path, identity, database, FRmodel)
    return jsonify({
        'distance': distance,
        'door_open': door_open,
        'result': result
    })

@app.route('/whoisit', methods=['OPTIONS'])
def whoisit_endpoint():
    data = request.get_json()
    image_path = data['image_path']
    min_dist, identity, result = who_is_it(image_path, database, FRmodel)
    return jsonify({
        'min_dist': min_dist,
        'identity': identity,
        'result': result
    })

if __name__ == '__main__':
    app.run(debug=True)
