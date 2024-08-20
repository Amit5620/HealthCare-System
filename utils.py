import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

import brain_tumor as bt
import lung_cancer as lc
import pneumonia as pn

def load_model_and_predict(image_path, disease):
    # Load the appropriate model based on disease type
    # model_paths = {
    #     'lung_cancer': 'models/lung_cancer_model.h5',
    #     'pneumonia': 'models/pneumonia_model.h5',
    #     'brain_tumor': 'models/brain_tumor_model.h5'
    # }
    
    # model = load_model(model_paths[disease])

    # Load and preprocess the image
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if disease=='brain-tumor':
        model = load_model("models/brain_tumor_model.h5")
        campath, prediction = bt.make_prediction(img, model, campath="static/uploads/123.jpeg", view=False)
    
    if disease=='lung-cancer':
        model = load_model("models/lung_cancer_model.h5")
        campath, prediction = lc.make_prediction(img, model, campath="static/uploads/123.jpeg", view=False)
    
    if disease=='pneumonia':
        model = load_model("models/pneumonia_model.h5")
        campath, prediction = pn.make_prediction(img, model, campath="static/uploads/123.jpeg", view=False)
    
    
    # Predict
    # prediction = model.predict(img_array)
    
    # Highlight affected area
    # affected_img_path = highlight_affected_area(image_path, model, disease)
    
    return prediction, campath

# def highlight_affected_area(image_path, model, disease):
#     # Dummy implementation for demonstration purposes
#     # You can use Grad-CAM or other techniques to highlight affected areas
    
#     img = cv2.imread(image_path)
#     heatmap = np.random.random((224, 224))  # Dummy heatmap
#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
#     superimposed_img = heatmap * 0.4 + img
#     affected_img_path = image_path.replace('.jpg', '_affected.jpg')
#     cv2.imwrite(affected_img_path, superimposed_img)
    
#     return affected_img_path
