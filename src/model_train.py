import gc
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from tqdm import tqdm
import networkx as nx
from sklearn.model_selection import train_test_split

class ThetaR(tf.keras.layers.Layer):
    def __init__(self, fps=7, ms= 256):
        super(ThetaR, self).__init__()
        self.w = self.add_weight("kernel",
                               shape=(fps,),
                               regularizer=tf.keras.regularizers.l1(0.000)
                                )
        
    def call(self, inputs):
        return tf.tensordot(inputs,self.w,1)

class UnregularisedThetaNN(tf.keras.layers.Layer):
    def __init__(self, fps=7, ms= 256):
        super(UnregularisedThetaNN, self).__init__()
        self.ms = ms
        self.theta = tf.keras.models.Sequential([
            tf.keras.Input(shape=(fps,)),
            tf.keras.layers.Dense(12,activation=tf.keras.activations.linear),
            tf.keras.layers.Dense(12,activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(12,activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(12,activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(12,activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(6 ,activation=tf.keras.activations.linear),
            tf.keras.layers.Dense(1)
            ])
    def call(self, inputs):
        ans = []
        for i in range(self.ms):
            o = self.theta(inputs[:,i])
            ans.append(o)
        ans = tf.keras.layers.Concatenate()(ans)
        return ans


class save_best_weights(tf.keras.callbacks.Callback):
    best_weights=[]   
    def __init__(self):
        super(save_best_weights, self).__init__()
        self.best = 0
    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        accuracy=logs.get('val_categorical_accuracy')* 100
        if np.greater(accuracy, self.best):
            self.best = accuracy            
            save_best_weights.best_weights=self.model.get_weights()
            self.model.layers[2].theta.save("./data/model_build_outputs/nn.h5")
            print('\nSaving weights validation loss= {0:6.4f}  validation accuracy= {1:6.3f} %\n'.format(current_loss, accuracy),end='\r') 
class save_best_weights_linear(tf.keras.callbacks.Callback):
    best_weights=[]   
    def __init__(self):
        super(save_best_weights_linear, self).__init__()
        self.best = 0
    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        accuracy=logs.get('val_categorical_accuracy')* 100
        if np.greater(accuracy, self.best):
            self.best = accuracy            
            save_best_weights_linear.best_weights=self.model.get_weights()
            np.save("./data/model_build_outputs/linear.npy",self.model.weights[0].numpy())
            print('\nSaving weights validation loss= {0:6.4f}  validation accuracy= {1:6.3f} %\n'.format(current_loss, accuracy),end='\r')  


if __name__ == "__main__":

    X = np.load("./data/model_build_outputs/X.npy")


    y = np.load("./data/model_build_outputs/y.npy")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    features_per_stop = X.shape[2]
    maxstops = X.shape[1]

    del X,y

    inputs = tf.keras.Input((maxstops,features_per_stop),dtype=tf.float32)
    util = ThetaR(features_per_stop)(inputs)

    outputs = tf.keras.activations.softmax(
        util, axis=-1
    )
    nnutil = UnregularisedThetaNN(features_per_stop,maxstops)(inputs)

    simple = tf.keras.Model(inputs=inputs, outputs=outputs, name="mnl_model")

    simple.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalAccuracy()])

    simple.summary()

    bestweights = save_best_weights_linear()

    simple.fit(X_train,y_train,epochs=50,batch_size=32,validation_data=(X_test,y_test),callbacks=[bestweights],verbose=0)
    simple.fit(X_train,y_train,epochs=50,batch_size=256,validation_data=(X_test,y_test),callbacks=[bestweights],verbose=0)
    simple.fit(X_train,y_train,epochs=75,batch_size=1024,validation_data=(X_test,y_test),callbacks=[bestweights],verbose=0)

    simple.set_weights(bestweights.best_weights)

    simple.trainable = False
    
    combinedinput =tf.keras.layers.Add()([util, nnutil])

    combinedoutputs = tf.keras.activations.softmax(
        combinedinput, axis=-1
    )

    numnl = tf.keras.Model(inputs=inputs, outputs=combinedoutputs, name="numnl")
    numnl.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalAccuracy()])
    numnl.summary()

    bestweights = save_best_weights()
    
    numnl.fit(X_train,y_train, epochs= 20, batch_size=32, validation_data=(X_test,y_test), callbacks=[bestweights], verbose=0)

    numnl.fit(X_train,y_train, epochs= 30, batch_size=256, validation_data=(X_test,y_test), callbacks=[bestweights], verbose=0)

    numnl.fit(X_train,y_train, epochs= 50, batch_size=1024, validation_data=(X_test,y_test), callbacks=[bestweights], verbose=0)


    numnl.set_weights(bestweights.best_weights)

    linear = numnl.layers[1].weights[0].numpy()

    np.save("./data/model_build_outputs/linear.npy",linear)
    numnl.layers[2].theta.save("./data/model_build_outputs/nn.h5")

