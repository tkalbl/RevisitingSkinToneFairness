batch_size=16
lr=1e-5
Fitzpatrick_threshold = 2

import tensorflow as tf
from utils import run_experiment,save_predictions
print(tf.__version__)
#from tensorflow.python.client import device_lib
#device_lib.list_local_devices()
#import pathlib
#import os
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import create_dataset,prepare_df, recalculate_weights
from sklearn.model_selection import train_test_split
print("Succesfully imported all libraries")

TRAINING_PATH = "./data/ISIC2018/ISIC2018_Task3_Training_Input/"
df_train_temp = prepare_df(TRAINING_PATH) # do NOT dropna!
df_Bevan = pd.read_csv("Bevan_corrected.csv")
df_train_temp = pd.merge(df_train_temp, df_Bevan, left_on='image', right_on='image')
print("{} samples are used".format(len(df_train_temp)))### CHANGE!

label_types = ["RP","RP2"]
label_names = ["fitzpatrick_Bevan","fitzpatrick_corrected"]
seeds = [42,5073,8447]
for i in range(len(label_types)):
    label_name = label_names[i]
    label_type = label_types[i]
    for j in range(len(seeds)):
        seed = seeds[j]
        number = i*3 + j+1
        exp_name = f"03DataShift0{number}{label_type}"
        
        df_train=df_train_temp[df_train_temp[label_name]<=Fitzpatrick_threshold]
        df_test = df_train_temp[df_train_temp[label_name]>Fitzpatrick_threshold]
        
        idx_train, idx_valid = train_test_split(df_train.index,stratify=df_train["lesion"],test_size=0.20,random_state=seed,shuffle=True)
        df_valid = df_train.loc[idx_valid,:]
        df_train = df_train.loc[idx_train,:]

        # re-calculate weights
        df_train = recalculate_weights(df_train)
        df_valid = recalculate_weights(df_valid)
        df_test = recalculate_weights(df_test)

        train_data = create_dataset(df_train,TRAINING_PATH,shuffle=True).batch(batch_size)
        valid_data = create_dataset(df_valid,TRAINING_PATH).batch(batch_size)

        test_data = create_dataset(df_test,TRAINING_PATH).batch(batch_size)

        print("Datasets succsefully created and edited")

        model = tf.keras.applications.MobileNetV2(
            input_shape=None,
            alpha=1.0,
            include_top=True,
            weights="imagenet",
            input_tensor=None,
            pooling="avg",
            classes=1000,
            classifier_activation="softmax",
        )

        # adapt the model
        last_layer = model.layers[-2].output

        output = tf.keras.layers.Dense(7, activation='softmax', name='predict_class')(last_layer)
        # building and printing the final model
        model = tf.keras.models.Model(inputs=model.layers[0].output,outputs=output)

        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=opt,
                      weighted_metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        print("Model created and compiled. Start training now")

        run_experiment(model,train_data,valid_data,exp_name=exp_name,patience=30,epochs=60)

        print("Trainnig completed. Saving predictions now.")
        save_predictions(exp_name,df_test,data_path = TRAINING_PATH)
        del opt,last_layer,model,train_data,valid_data,test_data,df_train,df_valid,df_test,idx_train, idx_valid,exp_name,seed,number,output