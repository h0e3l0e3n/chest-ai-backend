import os
import sys

# This tells the computer: "Be quiet and just work!"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# 1. THE SECRET SAUCE: Tell Keras to ignore the 'quantization_config' error
def fix_model_loading():
    # We create a fake version of the 'Dense' layer that ignores the extra talk
    from tensorflow.keras.layers import Dense
    class CustomDense(Dense):
        def __init__(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            super().__init__(*args, **kwargs)
    
    # We tell TensorFlow to use our 'CustomDense' instead of the broken one
    return {'Dense': CustomDense}

# 2. LOAD THE MODEL
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'chexpert_mobilenetv2_fast (2).h5')

try:
    # We use 'custom_objects' to apply our fix
    model = tf.keras.models.load_model(model_path, custom_objects=fix_model_loading(), compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

def predict_image(img_path):
    # Check the size: Colab error said 160x160, let's use that!
    img = image.load_img(img_path, target_size=(160, 160))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0 

    preds = model.predict(x, verbose=0)
    
    # MAP THE ANSWER: 
    # Important: This list must be in the SAME ORDER as your Colab training labels
    labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
              'Emphysema', 'Fibrosis', 'Infiltration', 'Mass', 'No Finding', 
              'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
    
    result_index = np.argmax(preds[0])
    return labels[result_index]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(predict_image(sys.argv[1]))
    else:
        print("Please provide an image path.")