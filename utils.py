import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
import numpy as np
import pandas as pd

MODEL_PATH = "./models/"
HISTORY_PATH = "./train_history/"
PREDICTION_PATH = "./predictions/"

def visualize_history(exp_name:str):
    """translates the experiment name in a history path and plots the training history """
    train_hist = pickle.load(open(f"{HISTORY_PATH}train_history_{exp_name}.pkl", "rb"))
    # we plot both, the LOSS and METRICS
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
    fig.suptitle(f'Training history of experiment {exp_name}', fontsize=14, fontweight='bold')

    ax1.plot(train_hist['loss'])
    ax1.plot(train_hist['val_loss'])
    ax1.set(xlabel='epoch', ylabel='LOSS')
    ax1.legend(['train', 'valid'], loc='upper right')

    ax2.plot(train_hist['sparse_categorical_accuracy'])
    ax2.plot(train_hist['val_sparse_categorical_accuracy'])
    ax2.set(xlabel='epoch', ylabel='Cat_ACC')
    ax2.legend(['train', 'valid'], loc='upper right')
    plt.show()

def recalculate_weights(df):
    """takes a dataframe with the columns image and lesion. calculates the weights depending on lesion frequency"""
    df["int_label"] = df["lesion"].replace({"MEL":0,"NV":1,"BCC":2,"AKIEC":3,"BKL":4,"DF":5,"VASC":6})
    occurrence = df["lesion"].value_counts()
    N = len(df)
    smooth = 0
    weight = (N + smooth)/(occurrence +smooth)/7
    some_dict = weight.to_dict()
    df["weights"]=df["lesion"].replace(to_replace=some_dict)
    return df

def prepare_df(path,filename="metadata.csv"):
    df_metadata = pd.read_csv(path+filename)
    df_metadata = recalculate_weights(df_metadata)
    return df_metadata

def parse_function(filename, label, weight):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image = tf.image.resize(image_decoded, [224, 224])
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label, weight

def create_dataset(df,DATAPATH,shuffle=False):
    """ Requires the dataframe with image names"""
    file_list = DATAPATH + df["image"]
    #print(file_list)
    file_list = file_list.to_list()
    N = len(file_list)
    #print(file_list)
    labels = df["int_label"].to_list()
    weights = df["weights"].to_list()
    images = tf.constant(file_list) # or tf.convert_to_tensor(files)
    labels = tf.constant(labels) # or tf.convert_to_tensor(labels)
    weights = tf.constant(weights)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels, weights))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=N)
    dataset = dataset.map(parse_function)
    return dataset


def run_experiment(model_name,Train_dataset, Validation_dataset,exp_name="default", patience=5, epochs=30):
    """model_name: tf.model, compiled!; Train_dataset/Validation_dataset: tf.Dataset (with images, labels, optionally weights); exp_name: str - will be part of saving pathes; patience,epochs,batch_size: int; """
    
    # defining the early stop criteria
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    # save the best model
    mc = tf.keras.callbacks.ModelCheckpoint(f'{MODEL_PATH}/best_model_{exp_name}.h5', monitor='val_loss', mode='min', save_best_only=True)
    # train and save the model
    history = model_name.fit(Train_dataset, validation_data=Validation_dataset, epochs=epochs, verbose=1, callbacks=[es,mc])
    history_name=f'{HISTORY_PATH}train_history_{exp_name}.pkl'
    # Save the model
    with open(history_name, 'wb') as handle:
        pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    visualize_history(exp_name)
    model_path = f'{MODEL_PATH}best_model_{exp_name}.h5'
    return None
    
def save_predictions(exp_name,df_test,data_path,batch_size=32):
    """model_path: str; X_test: batched tf.Dataset: Predicts labels and saves them in a csv"""
    model_path = f'{MODEL_PATH}best_model_{exp_name}.h5'
    predictive_model = tf.keras.models.load_model(model_path)
    X_test = create_dataset(df_test,data_path).batch(batch_size)
    predictions = predictive_model.predict(X_test,verbose=1)
    df_pred = pd.DataFrame(predictions,columns=['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'])
    pred_with_name = pd.DataFrame()
    pred_with_name["image"] = df_test["image"].reset_index(drop=True)
    pred_with_name["predicted_lesion"] = df_pred.idxmax(axis=1)
    pred_with_name[['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']] = df_pred[['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']]
    pred_with_name.to_csv(f'{PREDICTION_PATH}predictions_{exp_name}.csv',index=False)
    return None
