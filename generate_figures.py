import os
import time
import fnmatch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import grid
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pylab import grid
from scipy.stats import norm
from scipy.stats import norm, skew, kurtosis
from numpy import linalg as LA
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D


from numpy.linalg import matrix_rank
from keras.models import Sequential#,load_model
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
from keras.layers.recurrent import SimpleRNN
from keras.layers import TimeDistributed, Dense, Activation, Dropout
from keras.utils import plot_model, CustomObjectScope
from keras import metrics, optimizers, regularizers, initializers
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras.constraints import Constraint
import keras
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="-1"  

from tensorflow.keras.models import load_model


config = tf.compat.v1.ConfigProto()  
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU  
config.log_device_placement = True  # to log device placement (on which device the operation ran)  
                    


# taking dataset from function:

#from generate_data_set import *
from generate_data_set_and import *
#from generate_data_set_or import *
#from generate_data_set_xor import *
#from generate_data_set_not import *
#from generate_data_set_and_rand_place import *
#from generate_data_set_ff import *

from keras.initializers import Initializer

from keras.utils import CustomObjectScope

# Para coustomizar el constraint!!!!
from keras.constraints import Constraint

# To print network status
#from print_status_2_inputs_paper import *
from print_status_2_inputs_paper_exc_inh import *

#from print_figures_activity_paper_ok import *


#########################

#Path with network/s

#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/weights/2020_02-03_no_exi_inh"
#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/weights/2021-08-24/AND-30-70/RANDOMNORMAL-AND/weights_20_N_100_gap_0"
#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/weights/2021-08-24/AND-50-50/ORTHOGONAL-AND"
#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/weights/2021-08-24/AND-50-50/RANDOMNORMAL-AND"
#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/weights/2021-08-24/AND-30-70/ORTHOGONAL_AND"
#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/weights/2021-08-24/AND-30-70/ORTHOGONAL_AND/weights_20_N_100_gap_1"

#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/weights/2021-08-24/XOR-50-50/XOR.50-50-ORTHOGONAL"
#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/weights/2021-08-24/XOR-50-50/XOR-50-50-Randon-Normal"
#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/weights/2021-08-24/XOR-30-70/RAND_NORMAL_XOR"
#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/weights/2021-08-24/XOR-30-70/ORTHOGONAL_XOR"


#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/weights/2021-08-24/OR-50-50/RANDOMNORMAL-OR"
#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/weights/2021-08-24/OR-50-50/ORTHOGONAL-OR"
#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/weights/2021-08-24/OR-30-70/RANDOMNORMAL-OR"
#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/weights/2021-08-24/OR-30-70/ORTHOGONAL-OR"


#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/weights/2021-08-24/flip-flop-ran-nor-50-50"
#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/weights/2021-08-24/flip-lop-ran-nor-30-70"
#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/weights/2021-08-24/filip-flop-orthogonal-50-50/si"
#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/weights/2021-08-24/flip-flop-orthogonal-30-70/si"

#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/weights/2021-09-01_AND"
#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/weights/2021-09-01_AND/weights_20_N_100_gap_0"

## Neuro tasks ##

#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/neuro_tasks/perceptual_decision_making/weights/2021-08-24/orthogonal-30-70"
#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/neuro_tasks/perceptual_decision_making/weights/2021-08-24/orthogonal-50-50"
#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/neuro_tasks/perceptual_decision_making/weights/2021-08-24/random-normal-30-70"
#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/neuro_tasks/perceptual_decision_making/weights/2021-08-24/random-normal-50-50"

#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/neuro_tasks/parametric_working_memory/weights/2021-08-24/50-50-ran-normal-120-epocs/sin_bias"

#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/neuro_tasks/delay_match_to_sample/weights/2021-08-24/50-50-ran-nor-2.5/si-50-50-ran-nor-2.5"

#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/neuro_tasks/context_dependent_decision_making/weights/2021-08-24/orthogonal-50-50/orthogonal-50-50"

############################

#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/neuro_tasks/perceptual_decision_making/weights/2021-08-24/orthogonal-30-70"

#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/weights/drive-download-20220201T211952Z-001"
#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/weights/70-30_no_bias_g_1.7_100_epocs-20220204T115944Z-001/70-30_no_bias_g_1.7_100_epocs/weights_20_N_100_gap_1"

r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/weights/si/450"

#Parameters:
sample_size_3       = 4#15#4
mem_gap             = 20
sample_size         = 15 # Data set to print some results
lista_distancia_all =[]
lista_freq_sample_net=[]
H_number_list=[]
radios=[]
# Generate a data Set to study the Network properties:

x_train,y_train, mask,seq_dur  = generate_trials(sample_size,mem_gap)#,1) 
test                           = x_train[0:1,:,:] # Here you select from the generated data set which is used for test status
test_set                       = x_train[0:20,:,:]
y_test_set                     = y_train[0:20,:,0]
full_eigen_list                =[]
j2_full_eigen_list             =[]

dist_par_i                     =[]
dist_par_mu                    =[]
dist_par_sigma                 =[]
dist_par_pdf_kurtosis          =[]
dist_par_pdf_skew              =[]


net_freq=[]
def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)




def get_states(model):
    return [K.get_value(s) for s,_ in model.state_updates]

g=1

def my_init(shape, dtype=None):
    mu=0
    sigma=np.sqrt(1/(N_rec))
    #value = np.random.random(shape)
    shape= g*np.random.normal(mu, sigma, shape)          
    return K.variable(value=shape, dtype=dtype)

'''
def my_init_exi_ini(shape, dtype=None):
    mu_ex=0
    mu_in=0
    sigma=np.sqrt(1/(N_rec))        
    exi= g*np.random.normal(mu_ex, sigma, (int(N_rec/2),int(N_rec)))
    ini= g*np.random.normal(mu_in, sigma, (int(N_rec/2),int(N_rec)))
    shape      = np.concatenate((exi, ini), axis=0)

    return K.variable(value=shape, dtype=dtype)
'''
'''
class CustomInitializer:
    def __call__(self, shape, dtype=None):
        return my_init_exi_ini(shape, dtype=dtype)

get_custom_objects().update({'my_init_exi_ini': CustomInitializer})
'''

class NonNegLast(Constraint):
    def __call__(self, w):
        first_cols= w[:,0:int(N_rec/2)]*K.cast(K.less_equal(w[:,0:int(N_rec/2)], 0.0), K.floatx())
        last_cols= w[:,int(N_rec/2):int(N_rec)]*K.cast(K.greater_equal(w[:,int(N_rec/2):int(N_rec)], 0.0), K.floatx())                                
        full_matrix = K.concatenate([first_cols,last_cols],1)
        return full_matrix


'''
class NonNegLast(Constraint):
    def __call__(self, w):
          
        last_rowss_01= w[:, 0:int(N_rec/4)]*K.cast(K.greater_equal(w[:,0:int(N_rec/4)], 0.), K.floatx())
        last_rowss_02= w[:, int(N_rec/4):2*int(N_rec/4)]*K.cast(K.greater_equal(w[:,  int(N_rec/4):2*int(N_rec/4)], 0.), K.floatx())
        last_rowss_03= w[:, 2*int(N_rec/4):int(3*N_rec/4-0.5*N_rec/4)]*K.cast(K.less_equal(w[:, 2*int(N_rec/4):int(3*N_rec/4-0.5*N_rec/4)], 0.), K.floatx())
        last_rowss_04= w[:, int(3*N_rec/4-0.5*N_rec/4):N_rec]*K.cast(K.greater_equal(w[:,int(3*N_rec/4-0.5*N_rec/4):N_rec], 0.), K.floatx())                      
        full_w      = K.concatenate([last_rowss_01,last_rowss_02,last_rowss_04,last_rowss_03],1)
        return full_w
'''
    
class NonNegLast_input(Constraint):
    def __call__(self, w):
          
        last_rowss_01= w[:, 0:N_rec]*K.cast(K.greater_equal(w[:,0:N_rec], 0.), K.floatx())             
        full_w_      = last_rowss_01
        return full_w_


'''    
class my_init_exi_ini:
    def __call__(self,shape, dtype=None):
        return my_init_exi_ini(shape, dtype=dtype)
'''

class my_init_exi_ini( Initializer ):
    def __call__(self, shape, dtype=None):
        mu_ex=0.15
        mu_in=-0.15
        sigma=np.sqrt(1/(N_rec))        
        exi= g*np.random.normal(mu_ex, sigma, (int(N_rec),int(N_rec/2)))
        ini= g*np.random.normal(mu_in, sigma, (int(N_rec),int(N_rec/2)))
        #exi= g*np.random.normal(mu_ex, sigma, (int(3*N_rec/4),int(N_rec)))
        #ini= g*np.random.normal(mu_in, sigma, (int(N_rec/4),int(N_rec)))
        shape      = np.concatenate((ini,exi), axis=1)
        return K.variable( shape, dtype=dtype )

class my_init_rec( Initializer ):
    def __call__(self, shape, dtype=None):
        #def my_init_rec(shape, name=None,dtype=tf.float32):
        shape      = 1*np.identity(N_rec)
        #print("shape",shape)
        return K.variable(shape, dtype=dtype)


N_rec=100
pepe= keras.initializers.RandomNormal(mean=0.0, stddev=np.sqrt(float(1)/float((N_rec))), seed=None)
#x_train,y_train, mask,seq_dur = generate_trials(sample_size,mem_gap)

plot_dir="plots"

lista_neg     =[]
lista_pos     =[]
total         =[]
lista_neg_porc=[]
lista_pos_porc=[]
lista_tot_porc=[]

string_name_list=[]

for root, sub, files in os.walk(r_dir):
    files = sorted(files)
    
    for i,f in enumerate(files):
        print(f)
        if   fnmatch.fnmatch(f, '*20.hdf5'):# fnmatch.fnmatch(f, '*initial.hdf5') :#19
           print("file: ",f)
           r_dir=root
           string_name=root[-10:]
           print("string_name",string_name)
           string_name_list.append(string_name)
           print("r_dir",r_dir)

           #General network model construction:
           model = Sequential()
          # model = load_model(r_dir+"/"+f)   #custom_objects={'NonNegLast':NonNegLast})
           model = load_model(r_dir+"/"+f,  custom_objects={'NonNegLast':NonNegLast, 'NonNegLast_input':NonNegLast_input, 'my_init_exi_ini' : my_init_exi_ini,'my_init_rec':my_init_rec},compile = False)
           # Compiling model for each file:
           #model.compile(loss = 'mse', optimizer='Adam', sample_weight_mode="temporal")

           print("-------------",i)
           #Weights!!!
           for jj, layer in enumerate(model.layers):
               print("i-esima capa: ",jj)
               print(layer.get_config(), layer.get_weights())

           pesos     = model.layers[0].get_weights()
           pesos__   = model.layers[0].get_weights()[0]
           pesos_in  = pesos[0]
           pesos_out = model.layers[1].get_weights()
           pesos     = model.layers[0].get_weights()[1] 

           biases   = model.layers[0].get_weights()[2]

           N_rec                          =len(pesos_in[0])  # it has to match the value of the recorded trained network
           neurons                        = N_rec
           colors                         = cm.rainbow(np.linspace(0, 1, neurons+1))


           print( "h",model.layers[0].states[0])

           print("-------------\n-------------")   
           print("pesos:\n:",pesos)
           print("-------------\n-------------")
           print("N_REC:",N_rec)
           #unidades        = np.arange(len(biases)) #np.arange(len(pesos))
           unidades        = np.arange(len(pesos))
           conection       = pesos

           
           print("array: ",np.arange(len(pesos)))       
           
           

           #print("biases: ",biases)
           
           fig =plt.figure(figsize=cm2inch(12,6))
           #plt.scatter(unidades,biases, label="Bias value for each unit",color="green",s=2)
           plt.hlines(y=0.01, xmin=-1, xmax=N_rec +1,linestyle='--', alpha=.25, color="dimgrey")
           plt.hlines(y=0., xmin=-1, xmax=N_rec +1,linestyle='--', alpha=.25, color="salmon")
           plt.hlines(y=-0.01, xmin=-1, xmax=N_rec +1,linestyle='--', alpha=.25, color="dimgrey")
           plt.xlim([-1,N_rec +1])
           plt.legend(fontsize= 'small',loc=1)
           plt.xlabel('Units',fontsize = 15)
           plt.ylim([-0.065,0.065])
           plt.savefig(plot_dir+"/bias_"+str(i)+"_"+str(f)+"_"+str(string_name)+"_.png",dpi=200)

           print("##########################\n ##########################")
           print("conection",conection)       
           print("##########################\n ##########################")

           histo_lista    =[]
           array_red_list =[]
           histo_lista_pos=[]
           histo_lista_neg=[]

           conection_usar =conection   

           conection_usar=list(conection)
           conection_usar=np.asarray(conection_usar)

           peso_mask=0.0001
           peso_mask_2=-0.0001

           conection_usar[(conection_usar < peso_mask)&(conection_usar > peso_mask_2)] = 0
           #np.fill_diagonal(conection_usar, 0.0)
           model.layers[0].set_weights([1*pesos_in,1*conection_usar, 1*biases])
           #model.layers[0].set_weights([pesos_in,conection_usar])
           conection_sym  =0.5*(conection+tf.transpose(conection))

           

           w, v = LA.eig(conection_usar)
           normas= LA.norm(v, axis=0)
           
           print("Eigenvalues:\n", w)
           print("Eigenvectors:\n",v)
           print("Distance:", np.sqrt(w.real*w.real+w.imag*w.imag))

           lista_dist  = np.c_[w,w.real]
           lista_dist_2= np.c_[w,abs(w.real)]
           dist__= np.sqrt(w.real*w.real+w.imag*w.imag)
           dist__radio_reduced=sorted(dist__)
           
           radio_int_max= dist__radio_reduced[-7]
           
           maximo      = max(lista_dist, key=lambda item: item[1])
           minimo_1    = min(dist__, key=lambda item: item)
           maximo_2    = max(lista_dist_2, key=lambda item: item[1])
           marcar      = maximo[0]
           minimisimo  = minimo_1
           marcar_2    = maximo_2[0]
           dist_min_cuad=np.sqrt(minimisimo.real*minimisimo.real+minimisimo.imag*minimisimo.imag)
           #radios.append([string_name,dist_min_cuad])
           radios.append([string_name, radio_int_max, dist_min_cuad])
           print("First Element",maximo)
           print("Last Element",marcar)

           frecuency   =0
           if marcar_2.imag==0:
               frecuency =0
           else: 
               frecuency =abs(float(marcar_2.imag)/(3.14159*float(marcar_2.real)))

           print( "frecuency",frecuency)

           lista_modulos_    =np.sqrt(w.real*w.real+w.imag*w.imag)
           lista_freq_       =1000*np.absolute(w.imag/(3.14159*w.real))
           w_2               =list(w)

           list_dist_ordered =sorted(w_2, key=lambda x: abs(x.imag) )
           print("List sorted", list_dist_ordered)

           j2 = [i for i in w_2 if abs(i.real*i.real+i.imag*i.imag) > 1 and i.imag!=0]
          
           #print (j2)

           if len(j2)>0:
               ultimo= max(j2,key= np.abs)   #np.imag)
           else:
               ultimo =marcar_2
          # else:
          #    j2 = [i for i in w_2 if abs(i.real*i.real+i.imag*i.imag) >1]
          #    ultimo= max(j2,key= np.abs)

           #Debugging prints:
           #print("modulos",lista_modulos_)
           #print("j2 ",j2 )
           #print("j2 ultimo",ultimo )

           
           frecuency_ultimo =1000*abs(float(ultimo.imag)/(2*3.14159*float(ultimo.real)))
           net_freq.append([string_name,frecuency_ultimo])
           frecuency_ultimo_="%.2f" % frecuency_ultimo
           lista_modulos_cuad=  [i**2 for i in lista_modulos_]

           #Henrici’s departure from normality
           H_number=np.sqrt(np.power(np.linalg.norm(conection_usar),2)-sum(lista_modulos_cuad))/np.linalg.norm(conection_usar)
           H_number_list.append([string_name,H_number])

           #Symetric part

 
           w_s, v_s      = LA.eig(conection_sym)
           lista_dist_s  = np.c_[w_s,w_s.real]
           lista_dist_2_s= np.c_[w_s,abs(w_s.real)]
           maximo_s      = max(lista_dist_s, key=lambda item: item[1])

           maximo_2_s    = max(lista_dist_2_s, key=lambda item: item[1])
           marcar_s      = maximo_s[0]
           marcar_2_s    = maximo_2_s[0]

           ################ Fig Eigenvalues ########################

           fig=plt.figure(figsize=cm2inch(6.5,6) )
           ax = fig.add_axes([0, 0, 1, 1])         
           a = np.linspace(0, 2*np.pi, 500)
           cx,cy = np.cos(a), np.sin(a)
           plt.text(-1, 1.4, 'a)', va='center', fontsize=10)
           cx2,cy2 =  dist_min_cuad*np.cos(a),  dist_min_cuad*np.sin(a)
           cx3,cy3 = radio_int_max*np.cos(a),radio_int_max*np.sin(a)
           plt.plot(cx, cy,'--', alpha=.25, color="salmon") # draw unit circle line
           plt.plot(cx2, cy2,'--', alpha=.25, color="dimgrey") 
           plt.plot(cx3, cy3,'--', alpha=.25, color="dimgrey") 
           plt.axvline(x=1,color="salmon",alpha=.25,linestyle='--')
           #plt.plot([0,marcar.real],[0,marcar.imag],'-',alpha=.15,color="grey")
           #plt.plot([0,ultimo.real],[0,ultimo.imag],'-',alpha=.15,color="grey")

           t=w.real
           plt.scatter(w.real,w.imag,c=t,cmap='Spectral',s=2,alpha=.75)
           #plt.scatter(ultimo.real,ultimo.imag,color="blue",label="Max Eigenvalue Comp.\n "+str(ultimo),s=5)
           #plt.scatter(marcar.real,marcar.imag,color="red", label="Max Eigenvalue Real \n" +str(marcar_2)+"\n"+" Freq: "+str(frecuency_ultimo_),s=5)
           #plt.scatter(minimisimo.real,minimisimo.imag, color="green", label="Rad minimo "+str(dist_min_cuad), s=10)     
           plt.xticks(fontsize=6)
           plt.yticks(fontsize=6)
           #plt.ylim([-1.3, 2])
           #plt.xlim([-1.5, 2])
           plt.ylim([-1.3, 1.7])
           plt.xlim([-1.3, 1.7])
           plt.xlabel(r'$Re( \lambda)$',fontsize = 11)
           plt.ylabel(r'$Im( \lambda)$',fontsize = 11) 

           ax.spines['top'].set_visible(False)
           ax.spines['right'].set_visible(False)
           #ax.spines['bottom'].set_visible(False)
           #ax.spines['left'].set_visible(False)
           #ax.get_xaxis().set_ticks([])
           ax.get_yaxis().set_ticks([])
           
           leg = plt.legend(fontsize= 5,loc=1)
           leg.get_frame().set_linewidth(0.0)
           #leg = plt.legend()
           #leg.get_frame().set_linewidth(0.0)    
           plt.savefig(plot_dir+"/autoval_"+str(i)+"_"+str(f)+"_"+str(string_name)+".png",dpi=300, bbox_inches = 'tight')
           plt.close()
           ########################################################
           eje_x_normas=np.arange(len(normas))
           fig=plt.figure(figsize=cm2inch(7.1,6) )
           ax = fig.add_axes([0, 0, 1, 1])   
           plt.scatter(eje_x_normas,normas,color="grey",s=2, label=" Norm")
          
           plt.scatter(eje_x_normas,v[:,0],color="g",s=2,label="first component amplitude")
           plt.ylim([-1.5, 1.5])
           plt.axhline(y=0,color="dimgrey",alpha=.25,linestyle='--')
           leg = plt.legend(fontsize= 5,loc=1)
           leg.get_frame().set_linewidth(0.0)
           plt.savefig(plot_dir+"/normas_"+str(i)+"_"+str(f)+"_"+str(string_name)+".png",dpi=300, bbox_inches = 'tight')
           plt.close()
           #########################################################
           
           fig=plt.figure(figsize=cm2inch(7.1,6))
           ax = fig.add_axes([0, 0, 1, 1])     
           t_=w_s.real
           plt.scatter(w_s.real,w_s.imag,c=t_,cmap='viridis',label="Eigenvalue Sym spectrum\n Max: "+str(marcar_2_s),s=2)
           a = np.linspace(0, 2*np.pi, 500)
           cx,cy = np.cos(a), np.sin(a)
           plt.plot(cx, cy,'--', alpha=.25, color="dimgrey")           
           plt.scatter(marcar_s.real,marcar_s.imag,color="red", label="Eigenvalue maximum real part",s=5)
           plt.plot([0,marcar_s.real],[0,marcar_s.imag],'-',color="grey",alpha=.15)           
           plt.axvline(x=1,color="salmon",alpha=.25,linestyle='--')
           plt.xticks(fontsize=6)
           plt.yticks(fontsize=6)
           plt.ylim([-1.3, 1.6])
           plt.xlim([-1.5, 1.6])
           plt.xlabel(r'$Re( \lambda)$',fontsize = 11)
           plt.ylabel(r'$Im( \lambda)$',fontsize = 11)
           #plt.legend(fontsize= 8,loc=1)            
           leg = plt.legend(fontsize= 7,loc=1)
           leg.get_frame().set_linewidth(0.0)
           #leg = plt.legend()
           #leg.get_frame().set_linewidth(0.0)
           ax.spines['top'].set_visible(False)
           ax.spines['right'].set_visible(False)
           ax.spines['bottom'].set_visible(False)
           ax.spines['left'].set_visible(False)
           ax.get_xaxis().set_ticks([])
           ax.get_yaxis().set_ticks([])
    
           plt.savefig(plot_dir+"/autoval_sym_"+str(i)+"_"+str(f)+"_"+str(string_name)+".png",dpi=300, bbox_inches = 'tight')
           plt.close()
   
           #################################################################
           # Histogram estimations:

           fig=plt.figure(figsize=cm2inch(7,5.5))
           ax = fig.add_axes([0, 0, 1, 1])     
           for ii in unidades:
               histo_lista.extend(pesos[ii])

           media= np.average(histo_lista)


           # best fit of data
           (mu, sigma) = norm.fit(histo_lista)
           # the histogram of the data

           #n, bins, patches = plt.hist(histo_lista, 200, normed=1, facecolor='green', alpha=0.75)
           n, bins, patches = plt.hist(histo_lista, 200, facecolor='green', alpha=0.45, stacked=True, density = True)
           y =  norm.pdf( bins, mu, sigma)
           
           pdf_kurtosis = kurtosis(y)
           pdf_skew     = skew(y)

           dist_par_i.append(string_name)
           dist_par_mu.append(mu)
           dist_par_sigma.append(sigma)
           dist_par_pdf_kurtosis.append(pdf_kurtosis)
           dist_par_pdf_skew.append(pdf_skew)

           mu_="%.4f" % mu
           sigma_= "%.4f" % sigma
           pdf_skew_= "%.4f" % pdf_skew
           pdf_kurtosis_="%.4f" % pdf_kurtosis
                     
           #if i==0:
           #    plt.title('Initial Histogram Weights', fontsize = 18)
           #else:
           #plt.title('Histogram Weights after \"AND\" learning', fontsize = 12)
           #plt.hist(histo_lista, bins=200,color="mistyrose",normed=1,label="Weight Value \n Mu= "+str(mu)+"\n Sigma= "+str(sigma))
           plt.axvline(mu, color='r', linestyle='dashed', linewidth=1, label=r'$W^{Rec}$ Distribution Parameter'+ "\n\n Skew= " +str(pdf_skew_)+"\n Kurtosis= "+str(pdf_kurtosis_)+"\n"+r'$\mu$= '+str(mu_))

           plt.plot(bins, y, 'r-', linewidth=1)
           plt.vlines(x=sigma, ymin=0, ymax=900, linewidth=1,linestyle='dashed', color='grey',alpha=.35, label='$\sigma= $ '+str(sigma_))
           plt.vlines(x=-sigma,ymin=0, ymax=900, linewidth=1,linestyle='dashed', color='grey',alpha=.35)
           
           n = n.astype('int') # it MUST be integer
           # Good old loop. Choose colormap of your taste
           for i_pa in range(len(patches)):
               patches[i_pa].set_facecolor(plt.cm.Spectral(n[i_pa]/max(n)))

           plt.xlabel('Weight strength [arb. units]',fontsize = 10)
           plt.ylim([0,5])
           plt.xlim([-0.5,0.5])
           #plt.legend(fontsize= 8,loc=1)
           leg = plt.legend(fontsize= 4,loc=1)
           ax.spines['top'].set_visible(False)
           ax.spines['right'].set_visible(False)
           #ax.spines['bottom'].set_visible(False)
           ax.spines['left'].set_visible(False)
           #ax.get_xaxis().set_ticks([])
           ax.get_yaxis().set_ticks([])

           #leg.get_frame().set_linewidth(0.0)
           #plt.legend(fontsize= 8,loc=1)
           
           #plt.legend( bbox_to_anchor = (1.05, 0.6), fontsize= 10)
           plt.savefig(plot_dir+"/weight_histo_"+str(string_name)+"_"+str(f)+"_"+str(string_name)+"_.png",dpi=300, bbox_inches = 'tight')
           plt.close()


           ##############################
           fig=plt.figure(figsize=cm2inch(7,5.5))
           #histo positivo
        
           for ii in unidades:
              histo_lista_pos.extend(pesos[ii])
              
                  
           new_nums = list(filter(lambda x: x >0.005,histo_lista_pos))       
           
           media= np.average(new_nums)


           # best fit of data
           (mu2, sigma2) = norm.fit(new_nums)
           # the histogram of the data

           #n, bins, patches = plt.hist(histo_lista_pos, 200, normed=1, facecolor='green', alpha=0.75)
           n2, bins2, patches2 = plt.hist(new_nums, 100, facecolor='green', alpha=0.75, stacked=True, density = True)
           y2 =  norm.pdf( bins2,mu2, sigma2)
           
           pdf_kurtosis2 = kurtosis(y2)
           pdf_skew2     = skew(y2)


           mu2_="%.4f" % mu2
           sigma2_= "%.4f" % sigma2
           pdf_skew2_= "%.4f" % pdf_skew2
           pdf_kurtosis2_="%.4f" % pdf_kurtosis2
                     
           #if i==0:
           #    plt.title('Initial Histogram Weights', fontsize = 18)
           #else:
           #plt.title('Histogram Weights after \"AND\" learning', fontsize = 12)
           #plt.hist(histo_lista_pos, bins=200,color="mistyrose",normed=1,label="Weight Value \n Mu= "+str(mu)+"\n Sigma= "+str(sigma))
           plt.axvline(mu2, color='r', linestyle='dashed', linewidth=1, label=r'$W^{Rec}$ Values'+'\n\n'+ r'$\mu$= '+str(mu2_)+'\n $\sigma= $ '+str(sigma2_)+"\n Skew= " +str(pdf_skew2_)+"\n Kurtosis= "+str(pdf_kurtosis2_))

           plt.plot(bins2, y2, 'r-', linewidth=1)
           plt.vlines(x=sigma, ymin=0, ymax=900, linewidth=1, color='grey',alpha=.15)
           plt.vlines(x=-sigma,ymin=0, ymax=900, linewidth=1, color='grey',alpha=.15)
           
           n = n.astype('int') # it MUST be integer
           # Good old loop. Choose colormap of your taste
           for i_pa in range(len(patches2)):
               patches2[i_pa].set_facecolor(plt.cm.jet(n[i_pa]/max(n)))

           plt.xlabel('Weight strength [arb. units]',fontsize = 15)
           plt.ylim([0,4.5])
           plt.xlim([-0.5,0.5])
           #plt.legend(fontsize= 8,loc=1)
           leg = plt.legend(fontsize= 5,loc=1)
           ax.spines['top'].set_visible(False)
           ax.spines['right'].set_visible(False)
           ax.spines['bottom'].set_visible(False)
           ax.spines['left'].set_visible(False)
           ax.get_xaxis().set_ticks([])
           ax.get_yaxis().set_ticks([])

           leg.get_frame().set_linewidth(0.0)
           #plt.legend(fontsize= 8,loc=1)
           
           plt.legend( bbox_to_anchor = (1.05, 0.6), fontsize= 12)
           plt.savefig(plot_dir+"/weight_histo_pos"+str(string_name)+"_"+str(f)+"_"+str(string_name)+"_.png",dpi=300, bbox_inches = 'tight')
           plt.close()
           #------------------------------
                  
           
           ##############################
           fig=plt.figure(figsize=cm2inch(7,5.5))
           #histo negativo
         
                 
           new_nums2 = list(filter(lambda x: x <0.005,histo_lista_pos))       
           
           media= np.average(new_nums2)


           # best fit of data
           (mu3, sigma3) = norm.fit(new_nums2)
           # the histogram of the data

           #n, bins, patches = plt.hist(histo_lista_pos, 200, normed=1, facecolor='green', alpha=0.75)
           n3, bins3, patches3 = plt.hist(new_nums2, 100, facecolor='green', alpha=0.75, stacked=True, density = True)
           y3 =  norm.pdf( bins3,mu3, sigma3)
           
           pdf_kurtosis3 = kurtosis(y3)
           pdf_skew3     = skew(y3)


           mu3_="%.4f" % mu3
           sigma3_= "%.4f" % sigma3
           pdf_skew3_= "%.4f" % pdf_skew3
           pdf_kurtosis3_="%.4f" % pdf_kurtosis3
                     
           #if i==0:
           #    plt.title('Initial Histogram Weights', fontsize = 18)
           #else:
           #plt.title('Histogram Weights after \"AND\" learning', fontsize = 12)
           #plt.hist(histo_lista_pos, bins=200,color="mistyrose",normed=1,label="Weight Value \n Mu= "+str(mu)+"\n Sigma= "+str(sigma))
           plt.axvline(mu3, color='r', linestyle='dashed', linewidth=1, label=r'$W^{Rec}$ Values'+'\n\n'+ r'$\mu$= '+str(mu3_)+'\n $\sigma= $ '+str(sigma3_)+"\n Skew= " +str(pdf_skew3_)+"\n Kurtosis= "+str(pdf_kurtosis3_))

           plt.plot(bins3, y3, 'r-', linewidth=1)
           plt.vlines(x=sigma, ymin=0, ymax=900, linewidth=1, color='grey',alpha=.15)
           plt.vlines(x=-sigma,ymin=0, ymax=900, linewidth=1, color='grey',alpha=.15)
           
           n = n.astype('int') # it MUST be integer
           # Good old loop. Choose colormap of your taste
           for i_pa in range(len(patches3)):
               patches3[i_pa].set_facecolor(plt.cm.viridis(n[i_pa]/max(n)))

           plt.xlabel('Weight strength [arb. units]',fontsize = 15)
           plt.ylim([0,4.5])
           plt.xlim([-0.5,0.5])
           #plt.legend(fontsize= 8,loc=1)
           leg = plt.legend(fontsize= 5,loc=1)
           ax.spines['top'].set_visible(False)
           ax.spines['right'].set_visible(False)
           ax.spines['bottom'].set_visible(False)
           ax.spines['left'].set_visible(False)
           ax.get_xaxis().set_ticks([])
           ax.get_yaxis().set_ticks([])

           leg.get_frame().set_linewidth(0.0)
           #plt.legend(fontsize= 8,loc=1)
           
           plt.legend( bbox_to_anchor = (1.05, 0.6), fontsize= 12)
           plt.savefig(plot_dir+"/weight_histo_neg"+str(string_name)+"_"+str(f)+"_"+str(string_name)+"_.png",dpi=300, bbox_inches = 'tight')
           plt.close()       
                  
                  

           ################################################################################
           fig=plt.figure(figsize=cm2inch(8.5,6.5))
           #plt.title('Histogram Weights in after \"AND\" learning', fontsize = 11)
           ax = fig.add_axes([0, 0, 1, 1])     
           plt.hist(pesos_in[0], bins=200,label="Weight")
           plt.xlabel('Weight strength [arb. units]',fontsize = 20)
           leg = plt.legend(fontsize= 6,loc=1)
           ax.spines['top'].set_visible(False)
           ax.spines['right'].set_visible(False)
           ax.spines['bottom'].set_visible(False)
           ax.spines['left'].set_visible(False)
           leg.get_frame().set_linewidth(0.0)
           plt.savefig(plot_dir+"/in_weight_histo_"+str(i)+"_"+str(f)+"_"+str(string_name)+"_.png",dpi=300, bbox_inches = 'tight')
           plt.close()

           ##############################################################################     
           rank=matrix_rank(conection_usar)
           deter=np.linalg.det(conection_usar)
           print("rank",rank)
           #################Conectivity matrix: positive or excitatory weights ##########
           #0.1
           fig2= plt.figure(figsize=(15,5)) 
           
           conection_pos  = np.ma.masked_where(abs(conection) < peso_mask, conection)
           import matplotlib.colors as clr
           from matplotlib.colors import BoundaryNorm
           ax=[]
           
           cbar_max  = 1#0.75
           cbar_min  =-1#-0.75
           cbar_step =0.025 #0.012#0.025
           out=pesos_out[0]
           #out_rev= out[::-1]
          
             
           # define the colormap
           cmap = plt.get_cmap('Spectral')
           #cmap = plt.get_cmap('viridis')
           # extract all colors from the .jet map
           cmaplist = [cmap(i) for i in range(cmap.N)]
           # create the new map
           cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

           # define the bins and normalize and forcing 0 to be part of the colorbar!
           bounds = np.arange(np.min(conection_pos),np.max(conection_pos),.05)
           idx=np.searchsorted(bounds,0)
           bounds=np.insert(bounds,idx,0)
           norm_ = BoundaryNorm(bounds, cmap.N)  
           
        
           #cmap = clr.LinearSegmentedColormap.from_list('custom blue', ['#244162','#DCE6F1'], N=256)
         
           cmap.set_bad(color='white')
           
           
           
           ## W in
           ax.append(fig2.add_subplot(1,3,1))
           plt.title(r'$W^{in}$', fontsize = 20)
           im1=plt.imshow(pesos_in.T, cmap=cmap,interpolation="none",label=r'$W^{in}$',extent=[0,2,0,100],aspect='0.2',vmin =np.min(conection_pos), vmax = np.max(conection_pos))
           #plt.colorbar(im1, orientation='vertical')
           #plt.xlim([-1,N_rec +1])
           #plt.ylim([-1,N_rec +1])
       
           plt.xticks(np.arange(0,3, 1))
           plt.yticks(np.arange(0,N_rec +1, 5))
           
           
           ## Wrec
           ax.append(fig2.add_subplot(1,3,2))
           plt.title(r'$W^{Rec}$', fontsize = 20)

           
       
           im=plt.imshow(conection_pos,interpolation='none',norm=norm_,cmap=cmap,label='Conection matrix with rank'+str(rank), aspect="auto",vmin =np.min(conection_pos), vmax = np.max(conection_pos))
           #plt.colorbar(im, orientation='vertical')
           
           
       
           plt.xlim([-1,N_rec +1])
           plt.ylim([-1,N_rec +1])
       
           plt.xticks(np.arange(0,N_rec +1, 5))
           plt.yticks(np.arange(0,N_rec +1, 5))
           #plt.ylabel('Unit [i]',fontsize = 16)
           #plt.xlabel('Unit [j]',fontsize = 16)
           #plt.text(5, 5, 'Conection matrix with rank: '+str(rank)+'\n Det: '+str(deter), bbox={'facecolor': 'white', 'pad': 10})
           #plt.legend(fontsize= 'medium',loc=1)
           
           plt.ylabel('Post-synaptic',fontsize = 15) #Post-synaptic
           plt.xlabel('Pre-synaptic',fontsize = 15)
           plt.colorbar(im, orientation='vertical')
           
           ### W out
           ax.append(fig2.add_subplot(1,3,3))
           plt.title(r'$W^{out}$', fontsize = 20)
           im3=plt.imshow(out, cmap=cmap,interpolation="none",label= r'$W^{out}$',extent=[0,1,0,100],aspect='0.2',vmin =np.min(conection_pos), vmax = np.max(conection_pos))
           #plt.colorbar(im3, orientation='vertical')
           plt.xticks(np.arange(0,2, 1))
           plt.yticks(np.arange(0,N_rec +1, 5))
           
          
           
           fig2.tight_layout()
           
           plt.legend(fontsize= 'small',loc=1)
           plt.savefig(plot_dir+"/conection_matrix_P_"+str(i)+"_"+str(f)+"_"+str(string_name)+"_.png",dpi=200)
           plt.close()
      
           #################Conectivity matrix: positive or excitatory weights ##########
           #0.1
           plt.figure(figsize=(14,12)) 
           ax = fig.add_axes([0, 0, 1, 1]) 
           plt.title('Connectivity matrix', fontsize = 40)#, weight="bold")
           grid(True)
           #peso_mask=0.001
           
           
           cmap           = plt.cm.gist_ncar # Tipos de mapeo #OrRd # 'gist_ncar'#"plasma"
           conection_pos  = np.ma.masked_where(abs(conection) < peso_mask, conection)
           #conection_pos = conection_usar#np.ma.masked_where(conection_usar < peso_mask, conection)
           import matplotlib.colors as clr
           from matplotlib.colors import BoundaryNorm
           #cmap = clr.LinearSegmentedColormap.from_list('custom blue', ['#244162','#DCE6F1'], N=256)
           
           cmap.set_bad(color='white')
           
           
           # define the colormap
           cmap = plt.get_cmap('Spectral')
           #cmap = plt.get_cmap('viridis')
           # extract all colors from the .jet map
           cmaplist = [cmap(i) for i in range(cmap.N)]
           # create the new map
           cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

           # define the bins and normalize and forcing 0 to be part of the colorbar!
           bounds = np.arange(np.min(conection_pos),np.max(conection_pos),.05)
           idx=np.searchsorted(bounds,0)
           bounds=np.insert(bounds,idx,0)
           norm_ = BoundaryNorm(bounds, cmap.N)

           
           im=plt.imshow(conection_pos,interpolation='none',norm=norm_,cmap=cmap,label='Conection matrix with rank'+str(rank))
           #cbar=plt.colorbar(ticks=np.arange(cbar_min, cbar_max+cbar_step, cbar_step))
           #cmap = clr.LinearSegmentedColormap.from_list('custom blue', ['#244162','#DCE6F1'], N=256)
           #cbar.ax.set_ylabel('Weights [arbitrary unit]', fontsize = 40, weight="bold")
           plt.colorbar(im, orientation='vertical')
           
           #cbar=plt.colorbar(ticks=np.arange(cbar_min, cbar_max+cbar_step, cbar_step))
           plt.xlim([-1,N_rec +1])
           plt.ylim([-1,N_rec +1])
       
           plt.xticks(np.arange(0,N_rec +1, 5))
           plt.yticks(np.arange(0,N_rec +1, 5))
           plt.ylabel('Post-synaptic',fontsize = 40) #Post-synaptic
           plt.xlabel('Pre-synaptic',fontsize = 40)
           #plt.text(5, 5, 'Conection matrix with rank: '+str(rank)+'\n Det: '+str(deter), bbox={'facecolor': 'white', 'pad': 10})
           #plt.legend(fontsize= 'medium',loc=1)
           plt.savefig(plot_dir+"/conection_matrix_"+str(i)+"_"+str(f)+"_"+str(string_name)+"_.png",dpi=200)
           plt.close()
           
           

           ########## Here we plot iner state of the network with the desierd stimuli: 
           '''
           for sample_number in np.arange(sample_size_3):
               #if sample_number==2:#2#3xor
               print ("sample_number",sample_number)
               print_sample = plot_sample(sample_number,2,neurons,x_train,y_train,model,seq_dur,i,plot_dir,f,string_name)
               lista_freq_sample_net.append([string_name,sample_number,print_sample])
           #print("print_sample",print_sample)
           #time.sleep(5)
           '''
           ########## 
           '''
           '''
           # Model Testing: 
           x_pred = x_train[0:10,:,:]
           y_pred = model.predict(x_pred)

           print("x_train shape:\n",x_train.shape)
           print("x_pred shape\n",x_pred.shape)
           print( "y_train shape\n",y_train.shape)

           lista_distancia=[]
           
           #################################################

           for ii in np.arange(10):
               a=y_train[ii, :, 0]
               b=y_pred[ii, :, 0]          
               a_min_b = np.linalg.norm(a-b)      
               lista_distancia.append(a_min_b)

       
           lista_distancia.insert(0,N_rec)
           lista_distancia_all.append(lista_distancia)       

           
           #########################################
           fig= plt.figure(figsize=cm2inch(10.5,10))
           #fig.suptitle("\"And\" Data Set Trainined Output \n (amplitude in arb. units time in mSec)",fontsize = 20)
           for ii in np.arange(6):

               a=y_train[ii, :, 0]
               b=y_pred[ii, :, 0]
               a_min_b = np.linalg.norm(a-b)  
               a_min_b_="%.4f" %a_min_b 
               #lista_distancia.append(a_min_b)
               plt.subplot(3, 2, ii + 1)                
               plt.plot(x_train[ii, :, 0],color='g',label="Input A")
               plt.plot(x_train[ii, :, 1],color='pink',label="Input B")
               plt.plot(y_train[ii, :, 0],color='gray',linewidth=2,label="Expected Output")
               plt.plot(y_pred[ii, :, 0], color='r',linewidth=1,label="Predicted Output\n Distance= "+str(a_min_b_))
               #plt.ylim([-2, 1.6])
               plt.ylim([-2.5, 2])
               #plt.xlim([0, 205])
               #plt.xlim([0, 205])
               plt.xticks(np.arange(0,205,100),fontsize = 8)
               #plt.xticks(np.arange(0,405,100),fontsize = 8)
               #plt.legend(fontsize= 4.75,loc=3)
               leg = plt.legend(fontsize= 3.5,loc=3)
               leg.get_frame().set_linewidth(0.0)
               #plt.xticks([])
               plt.yticks([])
               plt.xticks(fontsize=5)
               plt.yticks(fontsize=5)
           fig.text(0.5, 0.03, 'time [mS]',fontsize=14, ha='center')
           fig.text(0.1, 0.5, 'Amplitude [Arb. Units]', va='center', ha='center', rotation='vertical', fontsize=14)
           figname =plot_dir+"/data_set_"+str(f)+"_"+str(string_name)+"_.png"       
           plt.savefig(figname,dpi=300, bbox_inches = 'tight')
           plt.close()
           K.clear_session()

todo_2       = np.c_[lista_distancia_all]

#np.savetxt(plot_dir+'/distance_sample.txt',todo_2,fmt='%f %f %f %f %f %f %f %f %f %f %f',delimiter=' ',header="Nrec #S1 #S2 #S3 #S4 #S5 #S6 #S7 #S8 #S9 #S10")
#np.savetxt(plot_dir+'/freq_net.txt',net_freq,fmt='%s %s',delimiter='\t',header="Net-id #freq ")
#np.savetxt(plot_dir+'/freq_net_units.txt',lista_freq_sample_net,fmt='%s',delimiter='\t',header="Net-id #sample #freq ")
np.savetxt(plot_dir+'/radios.txt',radios, fmt='%s',delimiter='\t',header="Net-id #radio int max #radio int min ")


todo=np.c_[dist_par_i, dist_par_mu, dist_par_sigma, dist_par_pdf_kurtosis, dist_par_pdf_skew]
#print("todo",todo)
np.savetxt(plot_dir+'/distribution_parameters.txt' ,todo,fmt='%s %s %s %s %s',delimiter=' ',header='Net-id, #mu #sigma #kurtosis #skew')
np.savetxt(plot_dir+'/H_number.txt',H_number_list,fmt='%s %s',delimiter='\t',header="Name #H number ")

print("v",v)
#print("v[0,:]",v[0,:])
print("v[:,0]",v[:,0])
#print ("distancias",todo_2)
#print("freq de las redes",net_freq)
#print("freq unidades",lista_freq_sample_net)

