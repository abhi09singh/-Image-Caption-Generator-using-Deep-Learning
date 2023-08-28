from flask import Flask, render_template, request
import io
import os
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow import keras
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image

app = Flask(__name__, template_folder='templates')

WORKING_DIR = r"C:\Users\lenovo\Downloads"
MODEL_PATH = os.path.join(WORKING_DIR, 'best_model_collab.h5')
TOKENIZER_PATH = os.path.join(WORKING_DIR, 'tokenizer.pkl')
MAX_LENGTH = 74

def load_model_and_tokenizer():
    global model, tokenizer
    model = keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)

vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = idx_to_word(yhat_index, tokenizer)
        if word is None or word == 'endseq':
            break
        in_text += " " + word
    
     # Remove the "startseq" token from the generated caption
    if in_text.startswith("startseq"):
        in_text = in_text[len("startseq"):]

    return in_text 

@app.route('/', methods=['GET', 'POST'])
def index():
    caption = None
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No image part"
        
        image = request.files['image']
        
        if image.filename == '':
            return "No selected file"
        
        img = Image.open(io.BytesIO(image.read()))
        img = img.resize((224, 224))
        
        caption = generate_caption(img)
    
    return render_template('index.html', caption=caption)

vgg_model=VGG16()
vgg_model = Model(inputs=vgg_model.inputs,outputs=vgg_model.layers[-2].output)


def generate_caption(image):
  #convert image pixels to numpy array
  image = img_to_array(image)
  #reshape data for model
  image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
  # preprocess image for vgg
  image = preprocess_input(image)
  # extract features
  features = vgg_model.predict(image,verbose=1)
  # predict from the trained model
  caption = predict_caption(model,features,tokenizer,MAX_LENGTH)
  return caption


if __name__ == '__main__':
    load_model_and_tokenizer()
    app.run(debug=True)
