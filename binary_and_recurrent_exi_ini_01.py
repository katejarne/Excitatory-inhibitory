##########################################################
#                 Author C. Jarne                        #
#            binary_and_recurrent_main.py  (ver 2.0)     #                       
#  Based on a Keras-Cog task from Alexander Atanasov     #
#  An "and" task (low edge triggered)                    #                
#                                                        #
# MIT LICENCE                                            #
##########################################################

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import keras.backend as K
import gc
from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint, Callback#, warnings
from keras.layers.recurrent import SimpleRNN
from keras.layers import TimeDistributed, Dense, Activation, Dropout, GaussianNoise
from keras.utils import plot_model
from keras import metrics,  activations, initializers, constraints
from keras import optimizers
from keras import regularizers
from keras.engine.topology import Layer, InputSpec
from keras.utils.generic_utils import get_custom_objects
from keras.initializers import Initializer
from keras.regularizers import l1,l2
# Para coustomizar el constraint!!!!
from keras.constraints import Constraint
 
import keras

# taking dataset from function
from generate_data_set_and import *
#from generate_data_set_xor import *
#from generate_data_set_or import *
#from generate_data_set_ff import *

import tensorflow as tf
#start_time = time.time()


os.environ["CUDA_VISIBLE_DEVICES"]="-1"  

####

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.009, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value   = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print(" Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

def and_fun(t,N_rec,base,base_plot):
    ''' 
    class NonNegLast(Constraint):
        def __call__(self, w):
          
            last_rowss_01= w[:, 0:N_rec/5]*K.cast(K.greater_equal(w[:,0:N_rec/5], 0.), K.floatx())
            last_rowss_02= w[:, N_rec/5:2*N_rec/5]*K.cast(K.less_equal(w[:,  N_rec/5:2*N_rec/5], 0.), K.floatx())
            last_rowss_03= w[:, 2*N_rec/5:int(3*N_rec/5-0.5*N_rec/5)]*K.cast(K.greater_equal(w[:, 2*N_rec/5:int(3*N_rec/5-0.5*N_rec/5)], 0.), K.floatx())
            last_rowss_04= w[:, int(3*N_rec/5-0.5*N_rec/5):3*N_rec/5]*K.cast(K.less_equal(w[:,int(3*N_rec/5-0.5*N_rec/5):3*N_rec/5], 0.), K.floatx())
            last_rowss_05= w[:, 3*N_rec/5:4*N_rec/5]*K.cast(K.greater_equal(w[:, 3*N_rec/5:4*N_rec/5], 0.), K.floatx())
            last_rowss_06= w[:,4*N_rec/5:N_rec]*K.cast(K.less_equal(w[:, 4*N_rec/5:N_rec], 0.), K.floatx())
            full_w      = K.concatenate([last_rowss_01,last_rowss_02,last_rowss_03,last_rowss_04,last_rowss_05,last_rowss_06],1)
            return full_w
    
    '''
    '''
    class NonNegLast(Constraint):
        def __call__(self, w):
          
            last_rowss_01= w[:, 0:int(N_rec/4)]*K.cast(K.greater_equal(w[:,0:int(N_rec/4)], 0.), K.floatx())
            last_rowss_02= w[:, int(N_rec/4):2*int(N_rec/4)]*K.cast(K.greater_equal(w[:,  int(N_rec/4):2*int(N_rec/4)], 0.), K.floatx())
            last_rowss_03= w[:, 2*int(N_rec/4):int(3*N_rec/4-0.5*N_rec/4)]*K.cast(K.less_equal(w[:, 2*int(N_rec/4):int(3*N_rec/4-0.5*N_rec/4)], 0.), K.floatx())
            last_rowss_04= w[:, int(3*N_rec/4-0.5*N_rec/4):N_rec]*K.cast(K.greater_equal(w[:,int(3*N_rec/4-0.5*N_rec/4):N_rec], 0.), K.floatx())
                              
            full_w      = K.concatenate([last_rowss_01,last_rowss_02,last_rowss_03,last_rowss_04],1)
            return full_w
    '''
    '''
    class NonNegLast(Constraint):
        def __call__(self, w):
            last_rowss_00= w[ 0:2,:]*K.cast(K.less_equal(w[0:2,:], 0.001), K.floatx())
            last_rowss_01= w[2:int(N_rec/4),:]*K.cast(K.greater_equal(w[2:int(N_rec/4),:], 0.001), K.floatx())
            last_rowss_02= w[int(N_rec/4):2*int(N_rec/4),:]*K.cast(K.greater_equal(w[int(N_rec/4):2*int(N_rec/4),:], 0.001), K.floatx())
            last_rowss_03= w[2*int(N_rec/4):int(3*N_rec/4-0.5*N_rec/4),:]*K.cast(K.less_equal(w[2*int(N_rec/4):int(3*N_rec/4-0.5*N_rec/4),:], -0.001), K.floatx())
            last_rowss_04= w[int(3*N_rec/4-0.5*N_rec/4):N_rec-1,:]*K.cast(K.greater_equal(w[int(3*N_rec/4-0.5*N_rec/4):N_rec-1,:], 0.), K.floatx())
            last_rowss_05= w[N_rec-1:N_rec,:]*K.cast(K.less_equal(w[N_rec-1:N_rec,:], 0.), K.floatx())
                    
            full_shufle = K.concatenate([last_rowss_00,w[2:int(N_rec/4),:],last_rowss_02,last_rowss_03,last_rowss_04,last_rowss_05],0)
            return full_shufle

    '''

    '''
    class NonNegLast(Constraint):
        def __call__(self, w):
            first_cols= w[0:int(N_rec/2),:]*K.cast(K.less_equal(w[0:int(N_rec/2),:], 0.0), K.floatx())
            last_cols= w[int(N_rec/2):int(N_rec),:]*K.cast(K.greater_equal(w[int(N_rec/2):int(N_rec),:], 0.0), K.floatx())                                
            full_matrix = K.concatenate([first_cols,last_cols],0)
            return full_matrix

    '''
    '''
    class NonNegLast_input(Constraint):
        def __call__(self, w):
          
            last_rowss_01= w[:, 0:N_rec]*K.cast(K.greater_equal(w[:,0:N_rec], 0.), K.floatx())             
            full_w_      = last_rowss_01
            return full_w_
    '''

    '''
    class NonNegLast(Constraint):
        def __call__(self, w):
            first_cols= w[:,0:int(N_rec/4)]*K.cast(K.less_equal(w[:,0:int(N_rec/4)], 0.0), K.floatx())
            last_cols= w[:,int(N_rec/4):int(N_rec)]*K.cast(K.greater_equal(w[:,int(N_rec/4):int(N_rec)], 0.0), K.floatx())                                
            full_matrix = K.concatenate([first_cols,last_cols],1)
            return full_matrix
    
    
    '''
    class NonNegLast(Constraint):
        def __call__(self, w):
            first_cols= w[:,0:int(N_rec/2)]*K.cast(K.less_equal(w[:,0:int(N_rec/2)], 0.0), K.floatx())
            last_cols= w[:,int(N_rec/2):int(N_rec)]*K.cast(K.greater_equal(w[:,int(N_rec/2):int(N_rec)], 0.0), K.floatx())                                
            full_matrix = K.concatenate([first_cols,last_cols],1)
            return full_matrix
    
    
    class NonNegLast_input(Constraint):
        def __call__(self, w):
          
            last_rowss_01= w[:, 0:N_rec]*K.cast(K.greater_equal(w[:,0:N_rec], 0.), K.floatx())             
            full_w_      = last_rowss_01
            return full_w_

    lista_distancia=[]
    #Parameters

    sample_size      = 15050#6*15050
    epochs           = 20
    #N_rec            = 50 #100
    p_connect        = 0.9

    #to be used in the Simple rnn redefined (not yet implemented)
    dale_ratio       = 0.8
    tau              = 100
    mem_gap          = t

    g=1
 
    def my_init_exi_ini(shape, dtype=None):
        #def __call__(self, shape, dtype=None):
        mu_ex=0.05#0.15
        mu_in=-0.05#-0.15
        sigma=np.sqrt(1/(N_rec))
        exi= g*np.random.normal(mu_ex, sigma, (int(N_rec),int(N_rec/2)))
        ini= g*np.random.normal(mu_in, sigma, (int(N_rec),int(N_rec/2)))        
        #exi= g*np.random.normal(mu_ex, sigma, (int(N_rec),int(3*N_rec/4)))
        #ini= g*np.random.normal(mu_in, sigma, (int(N_rec),int(N_rec/4)))
        #shape      = np.concatenate((ini,exi), axis=1)
        shape      = np.concatenate((exi,ini), axis=1)
        return K.variable( shape, dtype=dtype )
    '''
    def my_init_rec(shape, name=None,dtype=tf.float32):
        sigma=np.sqrt(1/(N_rec))
        mu_ex=0.05#0.15
        mu_in=-0.05#-0.15
        sigma=np.sqrt(1/(N_rec))
        exi= g*np.random.normal(mu_ex, sigma, (int(N_rec),int(N_rec/2)))
        ini= g*np.random.normal(mu_in, sigma, (int(N_rec),int(N_rec/2)))        
        shape      = np.concatenate((exi,ini), axis=1)
        return K.variable(shape, name=name, dtype=dtype)   

    '''
    '''
    def my_init_rec(shape, name=None,dtype=tf.float32):

        exi= np.ones((int(N_rec),int(N_rec/2)))
        ini= -1*np.ones((int(N_rec),int(N_rec/2)))        
        shape      = np.concatenate((exi,ini), axis=1)
        return K.variable(shape, name=name, dtype=dtype)
    '''
    '''
    def my_init_rec(shape, name=None,dtype=tf.float32):
        sigma=np.sqrt(1/(N_rec))
        mu=0
        exi= np.ones((int(N_rec),1))
        medio=g*np.random.normal(mu, sigma, (int(N_rec),int(N_rec)-2))#np.zeros((int(N_rec),int(N_rec)-2))
        ini= np.ones((int(N_rec),1))        
        shape      = np.concatenate((exi,medio,ini), axis=1)
        return K.variable(shape, name=name, dtype=dtype)
    '''
    '''
    def my_init_rec(shape, name=None,dtype=tf.float32):
        sigma=np.sqrt(1/(N_rec))
        mu=0
        exi=np.eye(N_rec,int(N_rec/2),k=0) # -1*np.eye(N_rec,int(N_rec/2),k=0)#np.full((N_rec,int(N_rec/2)),g*np.random.normal(mu, sigma,(int(N_rec),1)))
        ini=np.eye(N_rec,int(N_rec/2),k=0) # np.eye(N_rec,int(N_rec/2),k=0)   #np.full((N_rec,int(N_rec/2)),g*np.random.normal(mu, sigma,(int(N_rec),1)))
        shape      = np.concatenate((ini,exi), axis=1)
        print("shape",shape)
        #time.sleep(5)
        return K.variable(shape, name=name, dtype=dtype)
    '''
    def my_init_rec(shape, name=None,dtype=tf.float32):
        shape      = 1*np.identity(N_rec)
        #print("shape",shape)
        return K.variable(shape, name=name, dtype=dtype)


    pepe= keras.initializers.RandomNormal(mean=0.0, stddev=np.sqrt(float(2)/float((N_rec))), seed=None)


    x_train,y_train, mask,seq_dur = generate_trials(sample_size,mem_gap)
    #x_train,y_train, mask,seq_dur = generate_trials(sample_size,mem_gap,1)
    #Network model construction
    seed(None)# cambie el seed    
    model = Sequential()
    model.add(SimpleRNN(units=N_rec,return_sequences=True,use_bias=True, kernel_constraint=NonNegLast_input() ,input_shape=(None, 2), kernel_initializer='glorot_uniform', recurrent_initializer="orthogonal",bias_initializer="zeros" ,activation='tanh', recurrent_constraint=NonNegLast()))#orthogonal
    #model.add(SimpleRNN(units=N_rec,return_sequences=True,use_bias=False, kernel_constraint=NonNegLast_input() ,input_shape=(None, 2), kernel_initializer='glorot_uniform', recurrent_initializer="orthogonal",bias_initializer="zeros" , activation= None))#, recurrent_constraint=NonNegLast()))#orthogonal
  
    #model.add(SimpleRNN(units=N_rec,return_sequences=True,use_bias=False, kernel_constraint=NonNegLast_input() ,input_shape=(None, 2), kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', activation='tanh', recurrent_constraint=NonNegLast()))
    model.add(Dense(units=1,input_dim=N_rec))#,activation="linear"softplus#orthogonal"orthogonal"#,W_regularizer=l2(0.01)
    model.save(base+'/00_initial.hdf5')

    # Model Compiling:
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    ADAM           = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,epsilon=1e-08, decay=0.0001,clipnorm=0.5)
    ADA            = optimizers.Adagrad(learning_rate=0.01)
    model.compile(loss = 'mse', optimizer=ADAM, sample_weight_mode="temporal")

    # Saving weigths
    filepath       = base+'/and_weights-{epoch:02d}.hdf5'
    #checkpoint    = ModelCheckpoint(filepath, monitor='accuracy')
    #checkpoint     = ModelCheckpoint(filepath)
    callbacks      = [EarlyStoppingByLossVal(monitor='loss', value=0.00009, verbose=1), ModelCheckpoint(filepath, monitor='val_loss', save_best_only=False, verbose=1),]


    #from keras.callbacks import TensorBoard
    #tensorboard = TensorBoard(log_di r='./logs', histogram_freq=0, write_graph=True, write_images=False)

    history      = model.fit(x_train[50:sample_size,:,:], y_train[50:sample_size,:,:], epochs=epochs, batch_size=64, callbacks = callbacks,     sample_weight=None,shuffle=True, )

    #callbacks=[tensorboard]
    #callbacks = [checkpoint]

    # Model Testing: 
    x_pred = x_train[0:50,:,:]
    y_pred = model.predict(x_pred)

    print("x_train shape:\n",x_train.shape)
    print("x_pred shape\n",x_pred.shape)
    print("y_train shape\n",y_train.shape)

    fig     = plt.figure(figsize=(6,8))
    fig.suptitle("\"And\" Data Set Trainined Output \n (amplitude in arb. units time in mSec)",fontsize = 20)
    for ii in np.arange(10):
        plt.subplot(5, 2, ii + 1)
        
        #plt.plot(x_train[ii, :, 1],color='r',label="Context: \n1=pulse memory \n -1=Oscilatory")    
        plt.plot(x_train[ii, :, 0],color='g',label="Input A")
        plt.plot(x_train[ii, :, 1],color='pink',label="Input B")
        plt.plot(y_train[ii, :, 0],color='grey',linewidth=3,label="Desierd output")
        plt.plot(y_pred[ii, :, 0], color='r',label="Predicted Output")
        plt.ylim([-2.5, 2.5])
        plt.legend(fontsize= 5,loc=3)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        a=y_train[ii, :, 0]
        b=y_pred[ii, :, 0]
        a_min_b = np.linalg.norm(a-b)      
        lista_distancia.append(a_min_b)
    figname =  base_plot+"/data_set_sample_trained.png" 
    #figname = "plots_and/data_set_and_sample_trained.png" 
    plt.savefig(figname,dpi=200)
    #plt.close()
    #plt.show()

    print(model.summary())
    #plot_model(model, to_file=base_plot+'/model.png')

    print ("history keys",(history.history.keys()))

    #print("--- %s to train the network seconds ---" % (time.time() - start_time))

    fig     = plt.figure(figsize=(8,6))
    plt.grid(True)
    plt.plot(history.history['loss'])
    plt.title('Model loss during training')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    #figname = "plots_and/model_loss.png" 
    figname = base_plot+"/model_loss.png" 
    plt.savefig(figname,dpi=200)

    '''
    plt.figure()  
    plt.grid(True)
    plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_loss'])
    plt.title('accuracy')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    figname = "plots/accuracy.png" 
    plt.savefig(figname,dpi=200)
    '''
    #plt.show()
    K.clear_session()
    gc.collect()
    return lista_distancia



