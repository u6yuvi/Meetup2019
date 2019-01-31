#! /usr/bin/env python
from flask import Flask, render_template, request, Response
import numpy as np
from binascii import a2b_base64
import imageio
from PIL import Image
import io
import tensorflow as tf
import time
import ast

global model_states, nb_epoch #to have access later
model_states = ['Not Trained']
nb_epoch=5


app = Flask(__name__)


#page to_train
@app.route('/')
def to_train():

    return render_template('to_train.html', nb_epoch=nb_epoch)

#train the model
@app.route("/train/", methods=['GET'])
def train():
    global model, graph, acc, time_delta #to have access later


    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    #Monitor the training
    callback = tf.keras.callbacks.RemoteMonitor(root='http://127.0.0.1:5000',
                                           path='/publish/epoch/end/',
                                            field='data',
                                            headers=None,
                                            send_as_json=False)

    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'),
      tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
      tf.keras.layers.Dropout(0.20),
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'),
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Dense(10, activation='softmax')
    #   tf.keras.layers.Conv2D(32, 3, 3, activation='relu', input_shape=(28,28,1),use_bias=False),
    #   tf.keras.layers.BatchNormalization(name='norm_1'),
    #   tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    #   tf.keras.layers.Conv2D(32, 3, 3, activation='relu',use_bias=False),
    #   tf.keras.layers.BatchNormalization(name='norm_2'),
    #   tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    #   tf.keras.layers.Conv2D(10, 5,use_bias=False),
    #   tf.keras.layers.Flatten(),
    #   tf.keras.layers.Activation('softmax')
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    t0 = time.time()
    x_train = x_train.reshape(x_train.shape[0], 28, 28,1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28,1)
    model.fit(x_train, y_train, epochs=nb_epoch, callbacks=[callback], verbose=1)
    time_delta = round((time.time() - t0),2) #time to train
    acc = round((model.evaluate(x_test, y_test)[1]),3) # test accuracy
    #model.save_weights('weights/nn_demo')
    graph = tf.get_default_graph()
    
    return "OK"

# send time and accuracy
@app.route("/time_accuracy/", methods=['GET'])
def time_accuracy():
    
    def generate():
        finish=False
        while not finish:  
            try:
                json_parse = '{ "time_delta":'+ str(time_delta)+',"acc":'+ str(acc) + '}'
                print(json_parse)
                yield "data:" + json_parse + "\n\n"
                finish=True
            except:
                time.sleep(1)


    return Response(generate(),mimetype='text/event-stream')

#receive state of training
@app.route('/publish/epoch/end/', methods=['POST'])
def publish():
    epoch="None"
    data = request.form.get('data')
    data = ast.literal_eval(data)
    model_states.append(data['epoch'])
  
    return "OK"

#yield state of training
@app.route('/state')
def state():
    
    def generate():

        epoch = model_states[len(model_states)-1]
        while epoch!=(nb_epoch-1):
            epoch = model_states[len(model_states)-1]
            if epoch!= "Not Trained":
                print(epoch)
                yield "data:" + str(epoch) + "\n\n"
            time.sleep(0.5)
        

    return Response(generate(),mimetype= 'text/event-stream')

#page where you draw the number
@app.route('/index/', methods=['GET','POST'])
def index():
    prediction='?'
    if request.method == 'POST':

        dataURL = request.get_data()
        drawURL_clean = dataURL[22:]
        binary_data=a2b_base64(drawURL_clean)
        img = Image.open(io.BytesIO(binary_data))
        img.thumbnail((28,28))
        img.save("data_img/draw.png")

    return render_template('index.html', prediction=prediction)

#display prediction
@app.route('/result/')
def result():
    time.sleep(0.2)
    img = Image.open("data_img/draw.png")
    draw_np = np.asarray(img)[:,:,3]
    with graph.as_default():
        prediction = str(model.predict(draw_np.reshape(1,28,28,1)).argmax())
        print(prediction)
    return render_template("index.html",prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)

