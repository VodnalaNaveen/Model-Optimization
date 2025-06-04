# importing the requrired libraries

import argparse
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow
from tensorflow_model_optimization.python.core.keras.compat import keras
import tensorflow_model_optimization as tfmot
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()



if __name__ == "__main__":

    # creating the object to pass the arguments
    parser = argparse.ArgumentParser()

    # adding argument 1 to the script
    parser.add_argument("--weights_path",type=str,required=True,help="pass the weights path")
    
    #  adding argument 2 to the script
    parser.add_argument('--compressed_model_path',type=str,required=True,help=' where to save the compressed model ?')

    # declaring the arguments
    args = parser.parse_args()

    weights_path = args.weights_path # adding weights path to the variable
    compressed_model_path = args.compressed_model_path # adding the path to save the compressed model

    # Loading the model
    model = tensorflow.keras.models.load_model(weights_path)
    

    converter = tensorflow.lite.TFLiteConverter.from_keras_model(model) # loaded model
    converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()

    # Save the compression model
    with open((f'{compressed_model_path}/compressed.tflite'),mode="wb") as file:
        file.write(tflite_quant_model)
    
   