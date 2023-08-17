

import tensorflow as tf
from utils import visualize_history,run_experiment,save_predictions
print("Succesfully imported all libraries")
print(tf.__version__)
from tensorflow.python.client import device_lib
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import create_dataset,prepare_df,parse_function
from sklearn.model_selection import train_test_split


TRAINING_PATH = "./data/ISIC2018/ISIC2018_Task3_Training_Input/"

df_train = prepare_df(TRAINING_PATH)

idx_train, idx_valid = train_test_split(df_train.index,stratify=df_train["lesion"],test_size=0.20,random_state=42,shuffle=True)

df_valid = df_train.loc[idx_valid,:]
df_train = df_train.loc[idx_train,:]



batch_sizes = [64,48,32,24,16]
exponents = [3,4,5,6]
for batch_size in batch_sizes:
    for exponent in exponents:
        lr=10**(-exponent)
        exp_name=f"00Baseline02Batch{batch_size}Lr{exponent}"
        print(exp_name)

        train_data = create_dataset(df_train,TRAINING_PATH,shuffle=True).batch(batch_size)
        valid_data = create_dataset(df_valid,TRAINING_PATH).batch(batch_size)

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

        run_experiment(model,train_data,valid_data,exp_name=exp_name,patience=60,epochs=90)
        del model,last_layer,output,opt,train_data,valid_data
