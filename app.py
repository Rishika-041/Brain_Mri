#from flask_script  import Manager, Server
'''import numpy as np
import os
from keras.preprocessing import image 
import pandas as pd
import cv2
#from flask_ngrok import run_with_ngrok
from flask import Flask
import tensorflow as tf


# Flask utils
from flask import Flask,request, render_template
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model



# Define a flask app
app = Flask(__name__, static_url_path='', 
            static_folder='/Users/rishikapatwa/deploy/static',
            template_folder='/Users/rishikapatwa/deploy/templates')

# Load your trained model#model = load_model('/content/drive/MyDrive/clf-brain (1).hdf5')
model = tf.keras.models.load_model(
    '/Users/rishikapatwa/deploy/clf-brain (1).hdf5',
    custom_objects={'Functional':tf.keras.models.Model})

print('model-loaded')

@app.route('/', methods=['GET'])
def index():
    return render_template('brainintro.html')
#def index():
    # Main page
 #   return render_template('brainintro.html')

@app.route('/predict1', methods=['GET'])
def predict1():
    # Main page
    return render_template('brainpred2.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        #df= pd.read_csv('patient.csv')
        #f = request.files['image']
        input = request.files['image']
        input.save(os.path.join('/Users/rishikapatwa/deploy/uploads', secure_filename(input.filename)))
        #name=request.form['name']
        #age=request.form['age']
        
        # Save the file to ./uploads
        #basepath = os.path.dirname('/content')
        #file_path = os.path.join(
        #     '/Users/rishikapatwa/deploy/uploads', secure_filename(f.filename))
        #f.save(file_path)
        #test_datagen = image.ImageDataGenerator(rescale=1./255)
        #img_batch = test_datagen.flow_from_directory(file_path, target_size=(64, 64), shuffle=False)
        #predictions = model.predict_generator(img_batch, steps=1)
        img = image.load_img(input.filename, target_size=(256, 256))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        
        prediction = model.predict(x)[0][0]
        print(prediction)
        if prediction==0:
            text = "You are perfectly fine"
            inp = "No tumor"
        else:
            text = "You are infected! Please Consult Doctor"
            inp="Tumor detected"

        #df=df.append(pd.DataFrame({'name':[name],'age':[age],'status':[inp]}),ignore_index=True)
        #df.to_csv('patient.csv',index = False)
        return text
if __name__ == '__main__':
    app.run()
'''

from flask import Flask, render_template, request
#from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import image_utils
import tensorflow as tf
import numpy as np


app = Flask(__name__)

dic = {0.0 : 'You are perfectly fine', 1.0 : 'You are infected! Please Consult Doctor'}


model = tf.keras.models.load_model(
    '/Users/rishikapatwa/deploy/clf-brain (1).hdf5',
    custom_objects={'Functional':tf.keras.models.Model})

print('model-loaded')
#model._make_predict_function()

def predict_label(img_path):
    i = image_utils.load_img(img_path, target_size=(256,256))
    i = image_utils.img_to_array(i)
    i = np.expand_dims(i, axis=0)
    prediction = model.predict(i)[0][0]
    if prediction==0:
        text = "You are perfectly fine"
        inp = "No tumor"
    else:
        text = "You are infected! Please Consult Doctor"
        inp="Tumor detected"
    return text



# routes
@app.route("/", methods=['GET', 'POST'])
def kuch_bhi():
	return render_template("home.html")

@app.route("/about")
def about_page():
	return "About You..!!!"






@app.route("/submit", methods = ['GET', 'POST'])
def get_hours():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "/Users/rishikapatwa/deploy/static/" + img.filename	
		img.save(img_path)
        
		p = predict_label(img_path)



	return render_template("home.html", prediction = p, img_path = img_path)




	



if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)