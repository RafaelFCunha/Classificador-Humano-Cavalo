from flask import Flask, render_template, request
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.optimizers import RMSprop
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.applications.vgg16 import decode_predictions
import tensorflow 
from tensorflow import keras

app = Flask(__name__)
MODEL_PATH = 'model/classificador.h5'
model = keras.models.load_model(MODEL_PATH)
#model = load_model(MODEL_PATH)      

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    img = load_img(image_path, target_size=(300, 300))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images= np.vstack([x])
    classes = model.predict(images, batch_size=10)
    x = []
    print(classes[0])
    if classes[0]>0.5:
        print(" é um humano")
        x.append(" é um humano")
    else:
        print(" é um cavalo")
        x.append(" é um cavalo")

    #image = load_img(image_path, target_size=(300,300))
    #image = img_to_array(image)
    #image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #image = preprocess_input(image)
    #yhat =  model.predict(image, batch_size=10)
    #label = decode_predictions(yhat)
    #label = label[0]

    classification = lambda x: 'é um humano' if classes[0]>0.5  else 'é um cavalo'

    return render_template('index.html', prediction= x)
                                                    #classes
if __name__ == '__main__':
        app.run(port=3000, debug=True)