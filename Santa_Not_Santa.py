from tensorflow.keras.models import load_model
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

preprocess = imagenet_utils.preprocess_input

model = load_model("Santa_Not-Santa_model_98.h5")

def Make_Prediction(imagePath, MyModel):
    global model
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    image = preprocess(image)
    pred = MyModel.predict(image)
    pred = pred[0]
    not_santa, santa = pred
    if not_santa > santa:
        print("Its not Santa Image....")
    else:
        print("Great, Its Me.. Santa....@@")
    print(pred)
    
    
        
while True:
    imagePath = input("Select Image: ")
    
    if imagePath == "Bye":
        break
    
    ValiedExt = ["jpg", "png", "jpeg"]
    Pext = imagePath.split(".")[-1]
    if Pext not in ValiedExt:
        print("Choose Image Only,...") 
    else:
        Make_Prediction(imagePath, model)
    
    
    
    
    
    
    
    
    
    
    