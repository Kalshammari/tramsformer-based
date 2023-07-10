# -*- coding: utf-8 -*-

#Import The Solar Flare Data Set Files
!wget https://www.dropbox.com/s/uy58al2rwf6yn9u/labels_1540_4classes_icmla_21.pck
!wget https://www.dropbox.com/s/4bt5ugb9rimbrgx/mvts_1540_icmla_21.pck

import numpy as np
import pandas as pd
import networkx as nx
import pickle
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import plot_model
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, adjusted_rand_score

#Reading pickle files
def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


Sampled_inputs=load("mvts_1540_icmla_21.pck")

Sampled_labels=load("labels_1540_4classes_icmla_21.pck")

temp=Sampled_inputs[0]
print(temp)
df = pd.DataFrame(temp)
trainData=Sampled_inputs
trainLabel=Sampled_labels
print("trainData.shape: ",trainData.shape)
print("trainLebel.shape: ",trainLabel.shape)

df = pd.DataFrame(trainData[0])
df

print(trainLabel[0])

#standardization/z normalization of the univaraite time series
#-------------------data transform 3D->2D->3D ------------------------------
#Takes 3D array(x,y,z) >> transpose(y,z) >> return (x,z,y)
def GetTransposed2D(arrayFrom):
    toReturn = []
    alen = arrayFrom.shape[0]
    for i in range(0, alen):
        toReturn.append(arrayFrom[i].T)

    return np.array(toReturn)

#Takes 3D array(x,y,z) >> Flatten() >> return (x*y,z)
def Make2D(array3D):
    toReturn = []
    x = array3D.shape[0]
    y = array3D.shape[1]
    for i in range(0, x):
        for j in range(0, y):
            toReturn.append(array3D[i,j])

    return np.array(toReturn)

#Transform instance(92400, 33) into(1540x60x33)
def Get3D_MVTS_from2D(array2D, windowSize):
    arrlen = array2D.shape[0]
    mvts = []
    for i in range(0, arrlen, windowSize):
        mvts.append(array2D[i:i+windowSize])

    return np.array(mvts)




#-------------------data Scaler ------------------------------------------
from sklearn.preprocessing import StandardScaler

#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized

def GetStandardScaler(data2d):
    scaler = StandardScaler()
    scaler = scaler.fit(data2d)
    return scaler

def GetStandardScaledData(data2d):
    scaler = StandardScaler()
    scaler = scaler.fit(data2d)
    #print(scaler.mean_)
    data_scaled = scaler.transform(data2d)
    return data_scaled

def transform_scale_data(data3d, scaler):
    print("original data shape:", data3d.shape)
    trans = GetTransposed2D(data3d)
    print("transposed data shape:", trans.shape)    #(x, 60, 33)
    data2d = Make2D(trans)
    print("2d data shape:", data2d.shape)
    #  scaler = GetStandardScaler(data2d)
    data_scaled = scaler.transform(data2d)
    mvts_scalled = Get3D_MVTS_from2D(data_scaled, data3d.shape[2])#,60)
    print("mvts data shape:", mvts_scalled.shape)
    transBack = GetTransposed2D(mvts_scalled)
    print("transBack data shape:", transBack.shape)
    return transBack

TORCH_SEED = 0
#building standard scaler on train data X

#---------------Node Label Data Scaling-----------
trans = GetTransposed2D(trainData)
data2d = Make2D(trans)
scaler = GetStandardScaler(data2d)

trainData = transform_scale_data(trainData, scaler)
#trainLabel = trainLabel
unique_y_train, counts_y_train = np.unique(trainLabel, return_counts=True)
num_y_class = unique_y_train.shape[0]
print("X_train shape: ", trainData.shape)
print("y_train shape: ", trainLabel.shape)
#y_train_stats = dict(zip(unique_y_train, counts_y_train))
print("unique_y_train: ", unique_y_train)
print("y_train_counts: ", counts_y_train)
print("num_y_class: ", num_y_class)

df = pd.DataFrame(trainData[0])
df

print(trainData)

#Transposing trainData to shape:(1540, 60, 33)
trainDatatemp=np.empty([1540,60, 33])
n=len(trainData)
for l in range(0, n):
  temp=trainData[l]
  temp=temp.T
  trainDatatemp[l,:,:]=temp


trainData=trainDatatemp
print("Transposing trainData shape: ",trainData.shape)

#Taking the first 25 parameters which are based parameters:(1540, 60, 25)
trainDatat1=np.empty([1540,60, 25])
n=len(trainData)
for l in range(0, n):
  temp=trainData[l,:,0:25]
  trainDatat1[l,:,:]=temp


trainData=trainDatat1
print("Transposing trainData shape: ",trainData.shape)

from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split

def transformer_encoder(inputs, head_size, num_heads, ff_dim):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads
    )(x, x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
):
    n_classes=len(unique_y_train)
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

#Train and evaluate
input_shape = trainData.shape[1:]

model = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=10,
    mlp_units=[64],

)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"],
)
model.summary()

callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

for r in range(0,5):

    print("Random_state: ", r)

    X_train, X_test, y_train, y_test = train_test_split(trainData, trainLabel, test_size=0.3, random_state=r, stratify=trainLabel)
    print("X_train.shape y_train.shape y_test.shape ",X_train.shape, y_train.shape)
    print("X_test.shape y_test.shape ",X_test.shape, y_test.shape)#check percentage of examples
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)
    unique_y_train, counts_y_train = np.unique(y_train, return_counts=True)
    y_train_stats = dict(zip(unique_y_train, counts_y_train))
    print("y_train_counts")
    print(y_train_stats)

    unique_y_test, counts_y_test = np.unique(y_test, return_counts=True)
    y_test_stats = dict(zip(unique_y_test, counts_y_test))
    print("y_test_counts")
    print(y_test_stats)


    history= model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=8,
    callbacks=callbacks,)
    #history = model1.fit(train_x, train_y,validation_split = 0.1, epochs=50, batch_size=4)
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    model.evaluate(X_test, y_test, verbose=1)
    # Plot the model
    plot_model(model, to_file='transformer_model.png', show_shapes=True, show_layer_names=True)
    y_pred = model.predict(X_test)
    logits = y_pred - np.max(y_pred, axis = 1, keepdims = True)
    tsne = TSNE(random_state=0)
    tsne_output = tsne.fit_transform(logits)
    show_tsne_representation(tsne_output, y_test)
    y_pred = np.argmax(y_pred, axis = 1)


    n_classes=len(unique_y_train)
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=range(n_classes), yticklabels=range(n_classes))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    target_names=[0,1,2,3]
    report = classification_report(y_test, y_pred, target_names=target_names,output_dict=True)
    # Extract metrics
    precision = [report[label]['precision'] for label in target_names]
    recall = [report[label]['recall'] for label in target_names]
    f1_score = [report[label]['f1-score'] for label in target_names]
    # Plot the metrics
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(target_names))
    width = 0.3
    ax.bar(x - width, precision, width, label='Precision')
    ax.bar(x, recall, width, label='Recall')
    ax.bar(x + width, f1_score, width, label='F1-Score')
    ax.set_xticks(x)
    ax.set_xticklabels(target_names)
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Classification Report Metrics')
    ax.legend()
    plt.tight_layout()
    plt.show()

print(model.metrics_names)

#70% for training and 30% for testing
import numpy as np
A=[0.78,0.82,0.85,0.85,0.84] #batch size 8 head size 256
mean=np.mean(A)
STD=np.std(A)
M=np.max(A)
print("Mean Accuracy for 5 different random states",np.mean(A))
print("Mean Accuracy for 5 different random states with round",round(np.mean(A),2))
print("Max Accuracy for 5 different random states",M)
print("Standatd Deviation Accuracy  for 5 different random states",round(np.std(A),3))
print("Mean +/- Standatd Deviation Accuracy  for 5 different random states",round(mean+STD,2))

#X=0, M=1, BC=2, Q=3
import numpy as np
print("----Precision (X)-------")
print("-------------------------")


A=[0.92,0.95,0.93,0.97,0.98] #batch size 8 head size 64
mean=np.mean(A)
STD=np.std(A)
M=np.max(A)
print("Mean Precision (X) for 5 different random states",round(np.mean(A),2))
print("Max Precision (X)for 5 different random states",M)
print("Standatd Deviation Precision (X) for 5 different random states",round(np.std(A),3))
print("Mean +/- Standatd Deviation Precision (X) for 5 different random states",round(mean+STD,2))


print("----Recall (X)-------")

print("-------------------------")


A=[ 0.97,0.99,0.99,0.98,0.99] #batch size 8 head size 64
mean=np.mean(A)
STD=np.std(A)
M=np.max(A)
print("Mean Recall (X) for 5 different random states",round(np.mean(A),2))
print("Max Recall (X)for 5 different random states",M)
print("Standatd Deviation Recall (X) for 5 different random states",round(np.std(A),3))
print("Mean +/- Standatd Deviation Recall (X) for 5 different random states",round(mean+STD,2))

print("----F1 (X)-------")

print("-------------------------")


A=[0.95,0.97,0.96,0.97,0.99] #batch size 8 head size 64
mean=np.mean(A)
STD=np.std(A)
M=np.max(A)
print("Mean F1 (X) for 5 different random states",round(np.mean(A),2))
print("Max F1 (X)for 5 different random states",M)
print("Standatd Deviation F1 (X) for 5 different random states",round(np.std(A),3))
print("Mean +/- Standatd Deviation F1 (X) for 5 different random states",round(mean+STD,2))

#X=0, M=1, BC=2, Q=3
A=[0.83,0.77,0.76,0.90,0.84] #batch size 8 head size 64

mean=np.mean(A)
STD=np.std(A)
M=np.max(A)

print("-------Precision (M)----------")
print("-------------------------")
print("Mean Precision (M) for 5 different random states",round(np.mean(A),2))
print("Max Precision (M)for 5 different random states",M)
print("Standatd Deviation Precision (M) for 5 different random states",round(np.std(A),3))
print("Mean +/- Standatd Deviation Precision (M) for 5 different random states",round(mean+STD,2))


print("-------Recall (M)----------")
print("-------------------------")
B=[0.75,0.89,0.90,0.78,0.91] #batch size 8 head size 64
mean=np.mean(B)
STD=np.std(B)
M=np.max(B)
print("Mean Recall (M) for 5 different random states",round(np.mean(B),2))
print("Max Recall (M)for 5 different random states",M)
print("Standatd Deviation Recall (M) for 5 different random states",round(np.std(B),3))
print("Mean +/- Standatd Deviation Recall (M) for 5 different random states",round(mean+STD,2))

print("-------F1 (M)----------")
print("-------------------------")
C=[0.79,0.83,0.82,0.84,0.87] #batch size 8 head size 64
mean=np.mean(C)
STD=np.std(C)
M=np.max(C)
print("Mean F1 (M) for 4 different random states",round(np.mean(C),2))
print("Max F1 (M)for 4 different random states",M)
print("Standatd Deviation F1 (M) for 5 different random states",round(np.std(C),3))
print("Mean +/- Standatd Deviation F1 (M) for 5 different random states",round(mean+STD,2))

#X=0, M=1, BC=2, Q=3
A=[0.63,0.69,0.80,0.71,0.70] #batch size 8 head size 64

mean=np.mean(A)
STD=np.std(A)
M=np.max(A)

print("-------Precision (BC)----------")
print("Mean Precision (BC) for 5 different random states",round(np.mean(A),2))
print("Max Precision (BC)for 5 different random states",M)
print("Standatd Deviation Precision (BC) for 5 different random states",round(np.std(A),3))
print("Mean +/- Standatd Deviation Precision (BC) for 5 different random states",round(mean+STD,2))


print("-------Recall (BC)----------")

B=[0.60,0.67,0.71,0.79,0.75] #batch size 8 head size 64
mean=np.mean(B)
STD=np.std(B)
M=np.max(B)
print("Mean Recall (BC) for 5 different random states",round(np.mean(B),2))
print("Max Recall (BC)for 5 different random states",M)
print("Standatd Deviation Recall (BC) for 5 different random states",round(np.std(B),3))
print("Mean +/- Standatd Deviation Recall (M) for 5 different random states",round(mean+STD,2))

print("-------F1 (BC)----------")

C=[0.61,0.68,0.75,0.75,0.72] #batch size 8 head size 64
mean=np.mean(C)
STD=np.std(C)
M=np.max(C)
print("Mean F1 (BC) for 5 different random states",round(np.mean(C),2))
print("Max F1 (BC)for 5 different random states",M)
print("Standatd Deviation F1 (BC) for 5 different random states",round(np.std(C),3))
print("Mean +/- Standatd Deviation F1 (BC) for 5 different random states",round(mean+STD,2))

#X=0, M=1, BC=2, Q=3
A=[0.75,0.86,0.92,0.85,0.88] #batch size 8 head size 64
mean=np.mean(A)
STD=np.std(A)
M=np.max(A)

print("-------Precision (Q)----------")
print("Mean Precision (Q) for 5 different random states",round(np.mean(A),2))
print("Max Precision (Q)for 5 different random states",M)
print("Standatd Deviation Precision (Q) for 5 different random states",round(np.std(A),3))
print("Mean +/- Standatd Deviation Precision (Q) for 5 different random states",round(mean+STD,2))


print("-------Recall (Q)----------")

B=[0.81,0.72,0.79,0.84,0.72] #batch size 8 head size 64
mean=np.mean(B)
STD=np.std(B)
M=np.max(B)
print("Mean Recall (Q) for 5 different random states",round(np.mean(B),2))
print("Max Recall (Q)for 5 different random states",M)
print("Standatd Deviation Recall (Q) for 5 different random states",round(np.std(B),3))
print("Mean +/- Standatd Deviation Recall (Q) for 5 different random states",round(mean+STD,2))

print("-------F1 (Q)----------")
C=[0.78,0.78,0.85,0.85,0.79] #batch size 8 head size 64
mean=np.mean(C)
STD=np.std(C)
M=np.max(C)
print("Mean F1 (Q) for 5 different random states",round(np.mean(C),2))
print("Max F1 (Q)for 5 different random states",M)
print("Standatd Deviation F1 (Q) for 5 different random states",round(np.std(C),3))
print("Mean +/- Standatd Deviation F1 (Q) for 5 different random states",round(mean+STD,2))
