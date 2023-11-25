from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import cv2
import tensorflow as tf

app = Flask(__name__)
# Model Saved
model = load_model("CurrenyNote_Model.h5")

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            # Saving the uploaded image temporarily
            image_path = os.path.join("static", "uploaded_image.jpg")
            image_file.save(image_path)

            
            img = image.load_img(image_path, target_size=(180, 180))
          

            
            img = image.img_to_array(img)
            img = img / 255
            img = np.expand_dims(img, axis=0)

            
            predictions = model.predict(img)

            
            score = tf.nn.softmax(predictions)
            class_index = np.argmax(score)
            

            if class_index == 0:
                class_name = 'Five Hundred Rupees Note : 500'
            elif class_index == 1:
                class_name = 'Thousand Rupees Note : 1000'
            elif class_index == 2:
                class_name = 'Five Thousand Rupees Note : 5000'
            else:
                class_name = 'Wrong Input, Give Input Again'

            return render_template("index.html", class_name=class_name)
        
    else:
        return render_template("index.html")



if __name__ == "__main__":
    app.run(debug=True)