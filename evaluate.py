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
import time



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--normal_model_path", type=str, required=True, help="Path of the normal .h5 model")
    parser.add_argument("--compressed_model_path", type=str, required=True, help="Path of the compressed .h5 model")
    parser.add_argument("--quantized_model_path", type=str, required=True, help="Path of the quantized .tflite model")
    parser.add_argument("--data_path", type=str, required=True, help="Path of the CSV dataset")
    parser.add_argument("--no_prediction", type=int,required=False,default=100,help='Enter number of predictions')
    
    args = parser.parse_args()
    normal_model_path=args.normal_model_path
    compressed_model_path=args.compressed_model_path
    quantized_model_path=args.quantized_model_path
    data_path=args.data_path
    no_prediction=args.no_prediction

    df = pd.read_csv(data_path)
    df.drop(columns = ["Timestamp","Email Address"],axis=1,inplace=True)
    le = LabelEncoder()
    for column in df.columns:
        le = le.fit(df[column])
        df[column] = le.fit_transform(df[column])

    obj = KMeans(n_clusters=2,n_init=5).fit(df)
    clusters = obj.labels_
    df["labels"] = clusters 
    data = df.drop(columns="labels",axis=1)
    labels = df["labels"] 

    # ------------------ Evaluate normal Model ------------------

    start = time.time()
    normal_model = keras.models.load_model(normal_model_path)
    end = time.time()
    print(f"time taken to load a normal model is {end-start:.4f} seconds")

    c = 0
    time_taken = 0
    predicted_labels = []
    tot_loss=0
    for index in range(len(data)):
        if c == no_prediction:
            break
        observation = data.loc[index]
        observation = np.array(observation).reshape(1,-1)
        start = time.time()
        prediction = normal_model.predict(observation,verbose=0)
        end = time.time()
        t = end - start
        time_taken += t
        c += 1
        if prediction[0][0] < 0.5:
            predicted_labels.append(0)
        else:
            predicted_labels.append(1)
        
    
        loss = -(labels[index]*np.log(prediction[0][0]) + (1-labels[index])*np.log(1-prediction[0][0]))
        tot_loss += loss
        
    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(labels.loc[0:no_prediction-1])
    test_accuracy = (predicted_labels == true_labels).mean()



    print(f"time taken to make a {no_prediction} prediction using normal_model is {time_taken:.4f} seconds")
    print(f"accuracy for {no_prediction} prediction using normal_model is {test_accuracy:.4f}")
    print(f"loss for {no_prediction} prediction using normal_model is {tot_loss/c:.4f} ")

    

    start = time.time()
    compressed_model = tensorflow.lite.Interpreter(model_path=compressed_model_path)
    end = time.time()
    print(f"time taken to load a compressed model is {end-start:.4f} seconds")
    compressed_model.allocate_tensors()
    input_tensor = compressed_model.get_input_details() #--> how to pass the data to the model
    output_tensor = compressed_model.get_output_details() #--> how to calucate the output from the model


    c = 0
    time_taken=0
    predicted_labels = []
    tot_loss=0
    for index in range(len(data)):
        if c == no_prediction:
            break
        observation = data.loc[index].astype(np.float32)
        observation = np.array(observation).reshape(1,-1)
        start = time.time()
        compressed_model.invoke() 
        compressed_model.set_tensor(input_tensor[0]["index"],observation) # passing the data to the model
        output = compressed_model.get_tensor(output_tensor[0]["index"])
        end = time.time()
        t=end-start
        time_taken+=t
        if output[0][0] < 0.5:
            predicted_labels.append(0)
        else:
            predicted_labels.append(1)
        loss = -(labels[index]*np.log(output[0][0]) + (1-labels[index])*np.log(1-output[0][0]))
        tot_loss += loss
        c +=1
    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(labels.loc[0:no_prediction-1])
    test_accuracy = (predicted_labels == true_labels)



    print(f"time taken to make a {no_prediction} prediction using compressed_model is {time_taken:.4f} seconds")
    print(f"accuracy for {no_prediction} prediction using compressed_model is {test_accuracy.mean():.4f} ")
    print(f"loss for {no_prediction} prediction using compressed_model is {tot_loss/no_prediction:.4f} ")



    start = time.time()
    quantized_model = tensorflow.lite.Interpreter(model_path=quantized_model_path) # model_path, model_content
    end = time.time()
    print(f"time taken to load a quantized model is {end-start:.4f} seconds")

    
    quantized_model.allocate_tensors()
    input_tensor = quantized_model.get_input_details() #--> how to pass the data to the model
    output_tensor = quantized_model.get_output_details() #--> how to calucate the output from the model


    c = 0
    time_taken=0
    predicted_labels = []
    tot_loss=0
    for index in range(len(data)):
        if c == no_prediction:
            break
        observation = data.loc[index].astype(np.float32)
        observation = np.array(observation).reshape(1,10)
        start = time.time()
        quantized_model.invoke() 
        quantized_model.set_tensor(input_tensor[0]["index"],observation) # passing the data to the model
        output = quantized_model.get_tensor(output_tensor[0]["index"])
        end = time.time()
        t=end-start
        time_taken+=t
        if output[0][0] < 0.5:
            predicted_labels.append(0)
        else:
            predicted_labels.append(1)
        loss = -(labels[index]*np.log(output[0][0]) + (1-labels[index])*np.log(1-output[0][0]))
        tot_loss += loss
        c +=1
    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(labels.loc[0:no_prediction-1])
    test_accuracy = (predicted_labels == true_labels)



    print(f"time taken to make a {no_prediction} prediction using quantized_model is {time_taken:.4f} seconds")
    print(f"accuracy for {no_prediction} prediction using quantized_model is {test_accuracy.mean():.4f} ")
    print(f"loss for {no_prediction} prediction using quantized_model is {tot_loss/no_prediction:.4f} ")

