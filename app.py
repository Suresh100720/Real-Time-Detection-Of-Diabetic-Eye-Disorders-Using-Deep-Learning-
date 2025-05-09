import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import \
    tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for, request
import sqlite3
import cv2
import shutil




connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()

command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
 
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
        

        shutil.copy("test/"+fileName, dst)
        image = cv2.imread("test/"+fileName)
        #color conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        #apply the Canny edge detection
        edges = cv2.Canny(image, 100, 200)
        cv2.imwrite('static/edges.jpg', edges)
        #apply thresholding to segment the image
        retval2,threshold2 = cv2.threshold(gray_image,128,255,cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold.jpg', threshold2)
        # create the sharpening kernel
        kernel_sharpening = np.array([[-1,-1,-1],
                                    [-1, 9,-1],
                                    [-1,-1,-1]])

        # apply the sharpening kernel to the image
        sharpened = cv2.filter2D(image, -1, kernel_sharpening)

        # save the sharpened image
        cv2.imwrite('static/sharpened.jpg', sharpened)


        
        
        verify_dir = 'static/images'
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'Eyedisease-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            for img in os.listdir(verify_dir):
                path = os.path.join(verify_dir, img)
                img_num = img.split('.')[0] # a(2)    .    png
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                verifying_data.append([np.array(img), img_num])  #[[img,2],[img2,3]]
                np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 8, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        fig = plt.figure()
        
        str_label=" "
        accuracy=""
        rem=""
        rem1=""
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            

            if np.argmax(model_out) == 0:
                str_label = "cataract"
                print("The predicted image of the cataract is with a accuracy of {} %".format(model_out[0]*100))
                accuracy="The predicted image of the cataract is with a accuracy of {}%".format(model_out[0]*100)
                rem = "The remedies for  cataract are:\n\n "
                rem1 = ["Surgery: Surgery is the most effective treatment for cataracts. ",  
                "Prescription Eyewear: In the early stages of cataracts, prescription eyewear such as glasses or contact lenses may help improve vision.", 
                "Eye Drops: Some eye drops may be prescribed to help manage symptoms associated with cataracts, such as dry eyes or discomfort. ", 
                "Lifestyle Changes: Making certain lifestyle changes can help slow the progression of cataracts or reduce the risk of developing them."]
                
                

                
            elif np.argmax(model_out) == 1:
                str_label  = "glaucoma"
                print("The predicted image of the glaucoma is with a accuracy of {} %".format(model_out[1]*100))
                accuracy="The predicted image of the glaucoma is with a accuracy of {}%".format(model_out[1]*100)
                rem = "The remedies for  glaucoma are:\n\n "
                rem1 = [" Medications",  
                "Laser Therapy", 
                "Surgery", 
                "Lifestyle Changes"]
                
                
                
            elif np.argmax(model_out) == 2:
                str_label = "Mild"
                print("The predicted image of the Mild is with a accuracy of {} %".format(model_out[2]*100))
                accuracy="The predicted image of the Mild is with a accuracy of {}%".format(model_out[2]*100)
                rem = "The remedies for  Mild are:\n\n "
                rem1 = [" Control Blood Sugar Levels",  
                "Blood Pressure and Cholesterol Control", 
                "Healthy Lifestyle Choices"]
                
                

            elif np.argmax(model_out) == 3:
                str_label = "Moderate"
                print("The predicted image of the Moderate is with a accuracy of {} %".format(model_out[3]*100))
                accuracy="The predicted image of the Moderate is with a accuracy of {}%".format(model_out[3]*100)
                rem = "The remedies for  Moderate are:\n\n "
                rem1 = ["Control Blood Sugar Levels",  
                "Regular Eye Exams", 
                "Laser Treatment or Injections", 
                "Blood Pressure and Cholesterol Management"]
                


            elif np.argmax(model_out) == 4:
                str_label  = "normal"
                print("The predicted image of the normal is with a accuracy of {} %".format(model_out[4]*100))
                accuracy="The predicted image of the normal is with a accuracy of {}%".format(model_out[4]*100)
                
                
                
                
            elif np.argmax(model_out) == 5:
                str_label = "Proliferate_DR"
                print("The predicted image of the normal is with a accuracy of {} %".format(model_out[5]*100))
                accuracy="The predicted image of the normal is with a accuracy of {}%".format(model_out[5]*100)
                rem = "The remedies for  Proliferate_DR are:\n\n "
                rem1 = [" Laser Treatment (Photocoagulation)",  
                "Intravitreal Injections", 
                "Vitrectomy Surgery", 
                "Control of Diabetes and Blood Pressure"]
                
                

            elif np.argmax(model_out) == 6:
                str_label = "daibetic"
                print("The predicted image of the Moderate is with a accuracy of {} %".format(model_out[6]*100))
                accuracy="The predicted image of the Moderate is with a accuracy of {}%".format(model_out[6]*100)
                rem = "The remedies for  daibetic are:\n\n "
                rem1 = [" Regular Eye Exams",  
                "Control Blood Sugar Levels", 
                "Blood Pressure and Cholesterol Control", 
                "Treatment Options: Depending on the severity of diabetic eye disease"]
                

            elif np.argmax(model_out) == 7:
                str_label = "Severe"
                print("The predicted image of the Moderate is with a accuracy of {} %".format(model_out[7]*100))
                accuracy="The predicted image of the Moderate is with a accuracy of {}%".format(model_out[7]*100)
                rem = "The remedies for  Severe are:\n\n "
                rem1 = [" Medication",  
                "Lifestyle modifications", 
                "Vision aids and rehabilitation."]
                


   



                
           
       
        return render_template('userlog.html', status=str_label,accuracy=accuracy,remedie=rem,remedie1=rem1,ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName,ImageDisplay1="http://127.0.0.1:5000/static/gray.jpg",ImageDisplay2="http://127.0.0.1:5000/static/edges.jpg",ImageDisplay3="http://127.0.0.1:5000/static/threshold.jpg",ImageDisplay4="http://127.0.0.1:5000/static/sharpened.jpg")
        
    return render_template('index.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
