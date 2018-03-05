import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, merge
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD, Adam
from keras import backend as K


def my_init(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)

#from keras.utils import np_utils


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='latin1')
    return dict

def get_similarities(N,labels):
    out = [];
    for i in range(0,N):
        a=np.floor(np.random.rand(2)*len(data))
        b=np.array_equal(labels[int(a[1])],labels[int(a[0])])
        out.append([int(a[1]),int(a[0]),b])
    return out

def generate_batch(out,batch_size,shuffle=False):
    while True:
        if shuffle:
            indices = np.random.permutation(np.arange(len(out)))
        else:
            indices = np.arange(len(out))
        shuffled_triples = [out[ix] for ix in indices]
        num_batches = len(shuffled_triples) // batch_size    
        for j in range(num_batches):
            i1, i2,b1,b2, label = [], [], [],[], []
            batch = out[j * batch_size : (j + 1) * batch_size]
    
            for i in range(0,len(batch)):
                i1.append(data[batch[i][0]]);
                i2.append(data[batch[i][1]]);
                #b1.append(model.predict(data[batch[i][0]].reshape(1,3,32,32)));
                #b1.append(model.predict(data[batch[i][1]].reshape(1,3,32,32)));
                
                b1.append(get_3rd_layer_output([data[batch[i][0]].reshape(1,3,32,32)])[0])
                b2.append(get_3rd_layer_output([data[batch[i][1]].reshape(1,3,32,32)])[0])
                label.append(batch[i][2]);
            X1 = np.array(i1)
            X2 = np.array(i2)
            B1 = np.array(b1)
            B2 = np.array(b2)
            Y = np_utils.to_categorical(np.array(label), num_classes=2)
            yield ([X1, X2], [B1,B2,Y])

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
model.add(Conv2D(32, (5, 5), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3), strides=1, bias_initializer='glorot_normal'))
#model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

model.add(Conv2D(32, (5, 5), activation='relu', padding='same', kernel_constraint=maxnorm(3),strides=1))
model.add(AveragePooling2D(pool_size=(3, 3), strides=2,dim_ordering="th"))

model.add(Conv2D(64, (5, 5), activation='relu', padding='same', kernel_constraint=maxnorm(3),strides=1))
model.add(AveragePooling2D(pool_size=(3, 3), strides=2,dim_ordering="th"))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
BatchNormalization()
#model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))


input_shape=(3, 32, 32);
i1 = Input(input_shape)
i2 = Input(input_shape)
b1 = model(i1);
b2 = model(i2);
B1 = Dense(10, activation='relu')(b1)
B2 = Dense(10, activation='relu')(b1)

L1_distance = lambda x: K.abs(x[0]-x[1])
merge = merge([B1,B1], mode = L1_distance, output_shape=lambda x: x[0])
pred = Dense(2,activation='sigmoid')(merge)
sim_model = Model(input=[i1,i2],output=[B1,B2,pred])

get_3rd_layer_output = K.function([sim_model.layers[0].input],
                                  [sim_model.layers[2].output])


def loss_function(y_true,y_pred):
    alpha = 0.5;
    m=0.5;
    b1=y_true[0];
    b2=y_true[1];
    #loss1=Softmax(y_true,y_pred)
    euc_dist=((np.dot(b1,b2)))
    #loss2 =  0.5*(1-y_pred)*euc_dist+0.5*y_pred*K.max(m-euc_dist,0)+alpha*K.abs(K.sum(K.subtract(K.abs(b1),1)))+alpha*(K.sum(K.subtract(K.abs(b2),1)))
    loss1 =  0.5*(1-y_pred)*euc_dist;
    loss2=0.5*y_pred*np.max(m-euc_dist,0);
    #loss3=alpha*np.abs(np.linalg.norm(np.add(np.abs(b1),-1)))+alpha*np.abs(np.linalg.norm(np.add(np.abs(b2),-1)))
    
    return (loss1+loss2)

#sim_model.compile(loss='binary_crossentropy',optimizer=Adam(0.0006),metrics=["accuracy"])

##Loss function given in Paper
sim_model.compile(loss=loss_function,optimizer=Adam(),metrics=["accuracy"])

data_sim = get_similarities(20000,labels) #increase for better results

BATCH_SIZE = 30

split_point = int(len(data_sim) * 0.7) 
data_sim_train, data_sim_test = data_sim[0:split_point], data_sim[split_point:]


NUM_EPOCHS=10 #increase for better results
train_gen = generate_batch(data_sim_train, BATCH_SIZE,shuffle=True)
val_gen = generate_batch(data_sim_test, BATCH_SIZE,shuffle=False)

num_train_steps = len(data_sim_train) // BATCH_SIZE
num_val_steps = len(data_sim_test) // BATCH_SIZE

model_out = sim_model.fit_generator(train_gen,
                              steps_per_epoch=num_train_steps,
                              epochs=NUM_EPOCHS,
                              validation_data=val_gen,
                              validation_steps=num_val_steps)


plt.title("Loss")
plt.plot(model_out.history["loss"], color="r", label="train")
plt.plot(model_out.history["val_loss"], color="b", label="validation")
plt.legend(loc="best")
plt.tight_layout()
plt.show()


plt.title("Accuracy")
plt.plot(model_out.history["acc"], color="r", label="train")
plt.plot(model_out.history["val_acc"], color="b", label="validation")
plt.legend(loc="best")

plt.tight_layout()
plt.show()


