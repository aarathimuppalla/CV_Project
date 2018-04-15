# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, merge, ActivityRegularization
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD, Adam, RMSprop
from keras import backend as K
from keras.layers import Lambda
import csv
import tensorflow as tf

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='latin1')
    return dict

def get_similarities(N,labels):
    out = [];
    for i in range(0,N):
        a=np.floor(np.random.rand(2)*len(data))
        b=int(np.array_equal(labels[int(a[1])],labels[int(a[0])]))
        if not ((b==0) & (i%10 < 8)):
            out.append([int(a[1]),int(a[0]),(1-b)])
    return out

def generate_batch1(out,batch_size,shuffle=False):
    while True:
        if shuffle:
            indices = np.random.permutation(np.arange(len(out)))
        else:
            indices = np.arange(len(out))
        shuffled_triples = [out[ix] for ix in indices]
        num_batches = len(shuffled_triples) // batch_size    
        for j in range(num_batches):
            i1, i2, label1 = [], [], []
            batch = out[j * batch_size : (j + 1) * batch_size]
            for i in range(0,len(batch)):
                i1.append(data[batch[i][0]]);
                i2.append(data[batch[i][1]]);
                label1.append(batch[i][2]);
            X1 = np.array(i1)
            X2 = np.array(i2)
            Y1 = np.array(label1)
            yield ([X1, X2], Y1)
            
            
def generate_batch(out,batch_size,shuffle=False):
    while True:
        if shuffle:
            indices = np.random.permutation(np.arange(len(out)))
        else:
            indices = np.arange(len(out))
        shuffled_triples = [out[ix] for ix in indices]
        num_batches = len(shuffled_triples) // batch_size    
        for j in range(num_batches):
            i1, i2, label1,label2 = [], [], [], []
            batch = out[j * batch_size : (j + 1) * batch_size]
    
            for i in range(0,len(batch)):
                i1.append(data[batch[i][0]]);
                i2.append(data[batch[i][1]]);
                label1.append(batch[i][2]);
                label2.append(batch[i][2]*24);
            X1 = np.array(i1)
            X2 = np.array(i2)
            Y1 = np.array(label1)
            Y2 = np.array(label2)
            yield ([X1, X2], [Y1, Y2])
            
def generate_test_data(out):
    while True:       
        i1, i2, label1 = [], [], []
        for i in range(0,len(out)):
            i1.append(data[out[i][0]]);
            i2.append(data[out[i][1]]);
            label1.append(out[i][2]);
        X1 = np.array(i1)
        X2 = np.array(i2)
        Y1 = np.array(label1)
        yield ([X1, X2], Y1)
            
def Margin_Loss(y_true, y_pred):
    m=24
    loss = 0.5*(1-y_true)*y_pred + 0.5*y_true*K.maximum(0.0, m - y_pred)
    return loss



def euc_dist(x):
    'Merge function: euclidean_distance(u,v)'

    
    a = tf.cast(x[0] + 0.5, tf.float32)
    b = tf.cast(x[1] + 0.5, tf.float32)
    s = a - b;
    output = K.sum((s ** 2))
    return output

def euc_dist_shape(input_shape):
    'Merge output shape'
    shape = list(input_shape)
    outshape = (shape[0][0],1)
    return tuple(outshape)



data_folder = "F:\Aarathi\IIIT\Computer Vision\Project\cifar-10-batches-py"
test_file = "F:\Aarathi\IIIT\Computer Vision\Project\cifar-10-batches-py\test_batch"

#Read image data
for file in os.listdir(data_folder):
    if file.endswith(".meta"):
        meta_file = os.path.join(data_folder, file)
    elif "data_batch_1" in file:
        data_batch = os.path.join(data_folder, file)
        a = unpickle(data_batch)
        data = a["data"]
        #print(data.shape)
        labels = a["labels"]
    elif "data_batch" in file:
        data_batch = os.path.join(data_folder, file)
        a = unpickle(data_batch)
        data = np.concatenate((data,a["data"]),axis=0)
        labels = np.concatenate((labels,a['labels']),axis=0)
        #print(data.shape)
        #print(labels.shape)

#Read Labels
b = unpickle(meta_file)
label_names = b["label_names"]
data=data.reshape((data.shape[0],3,32,32))/255.0;
labels = np_utils.to_categorical(labels, 10);

##for visualization
#print(label_names)
#print(labels[1]);
#im=data[1,:];
#im=im.reshape((3,32,32));
#plt.imshow(im.T)
#plt.show()
#####

# Create the model
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3), strides=1))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
model.add(Conv2D(32, (5, 5), activation='relu', padding='same', kernel_constraint=maxnorm(3),strides=1))
model.add(AveragePooling2D(pool_size=(3, 3), strides=2,dim_ordering="th"))
model.add(Conv2D(64, (5, 5), activation='relu', padding='same', kernel_constraint=maxnorm(3),strides=1))
model.add(AveragePooling2D(pool_size=(3, 3), strides=2,dim_ordering="th"))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
#BatchNormalization()
model.add(Dense(12, activation='sigmoid'))


input_shape=(3, 32, 32);
i1 = Input(input_shape)
i2 = Input(input_shape)
b1 = model(i1);
b2 = model(i2);  
 

merge = merge([b1,b2], mode = euc_dist, output_shape=euc_dist_shape)
pred = ActivityRegularization(l1=0.01)(merge)
sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
sim_model = Model(input=[i1,i2], outputs=[merge])
sim_model.compile( loss=Margin_Loss, optimizer="adam")
#sim_model.compile( loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])

#Generate pair of images with similarity labels
data_sim = get_similarities(200000,labels) #increase for better results

BATCH_SIZE = 200

split_point1 = int(len(data_sim) * 0.8) ;
#split_point2 = int(len(data_sim) * 0.9) 
#data_sim_train, data_sim_val, data_sim_test = data_sim[0:split_point1], data_sim[split_point1:split_point2], data_sim[split_point2:]
data_sim_train, data_sim_val= data_sim[0:split_point1], data_sim[split_point1:]


NUM_EPOCHS=30 #increase for better results
train_gen = generate_batch1(data_sim_train, BATCH_SIZE,shuffle=True)
val_gen = generate_batch1(data_sim_val, BATCH_SIZE,shuffle=False)

num_train_steps = len(data_sim_train) // BATCH_SIZE
num_val_steps = len(data_sim_val) // BATCH_SIZE

model_out = sim_model.fit_generator(train_gen,
                              steps_per_epoch=num_train_steps,
                              epochs=NUM_EPOCHS,
                              validation_data=val_gen,
                              validation_steps=num_val_steps)

print(model.summary())
print(sim_model.summary())


#Getting binary values for all images in the dataset and storing in a file
out_list=[]
for i in range(0,len(data)):
    a=data[i].reshape(1,3,32,32)
    b = model.predict(a);b[b<0.5]=0; b[b>=0.5]=1
    out_list.append(b)


csvfile = "binary.csv"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in out_list:
        writer.writerow(np.ndarray.tolist(val[0])) 
        
csvfile = "labels.csv"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in labels:
        writer.writerow(val) 
        
#loss vs epoch graph
plt.title("Loss")
plt.plot(model_out.history["loss"], color="r", label="train")
plt.plot(model_out.history["val_loss"], color="b", label="validation")
plt.legend(loc="best")
plt.tight_layout()
plt.show()



