from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras import backend as K
from PIL import Image
import tensorflow as tf
import numpy as np
import flask
from flask_bootstrap import Bootstrap
from flask import Flask, render_template, request, url_for
import io
import os
import requests
import json

KERAS_REST_API_URL = "http://localhost:5000/predict"

app = flask.Flask(__name__, static_url_path='/static')
Bootstrap(app)
model = None
UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def get_model():
	global model
	model = ResNet50(weights="imagenet")
	global graph
	graph = tf.get_default_graph()


def prepare_image(image, target):
	if image.mode != "RGB":
		image = image.convert("RGB")

	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	return image


@app.route("/", methods=['GET', 'POST'])
def form():
	return render_template('test.html')

@app.route("/about", methods=['GET'])
def getAbout():
	return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_file():
	file = request.files['image']
	f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

	file.save(f)

	return render_template('indexs.html')


@app.route("/test", methods=["GET"])
def test():
	return render_template('test.html')


@app.route("/predict", methods=["POST"])
def predict():
	data = {"success": False}
	file = request.files['image']
	image = file

	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			image = prepare_image(image, target=(224, 224))

			with graph.as_default():
				preds = model.predict(image)
				results = imagenet_utils.decode_predictions(preds)
				data["predictions"] = []

			for (imagenetID, label, prob) in results[0]:
				r = {"imagenetID": imagenetID, "label": label, "probability": float(prob)}
				data["predictions"].append(r)

			data["success"] = True

	return render_template('result.html', data=data)



@app.route("/results", methods=['GET', 'POST'])
def results():
	file = request.files['image']
	image = file
	payload = {"image": image}
	r = requests.post(KERAS_REST_API_URL, files=payload)
	#data = r.json()
	#data = json.dumps(data, sort_keys = True, indent = 4, separators = (',', ': '))
	data = r
	return render_template('result.html', data=data)


if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
            "please wait until server has fully started"))
	get_model()
	app.run()
