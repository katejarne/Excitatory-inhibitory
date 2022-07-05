##########################################################
#                 Author C. Jarne                        #
#               call loop  (ver 1.0)                     #                       
# MIT LICENCE                                            #
##########################################################

import os
import time
from binary_and_recurrent_exi_ini_01 import * #two input tasks
#from binary_and_recurrent_exi_ini_02 import *#one input tasks


config = tf.compat.v1.ConfigProto()  
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU  
config.log_device_placement = True  # to log device placement (on which device the operation ran)                      


start_time = time.time()
time_vector=[450]


f          ='weights'
f_plot     ='plots'

#f          ='weights_or'
#f_plot     ='plots_or'


#f          ='weights_xor'
#f_plot     ='plots_xor'

#f          ='weights_ff'
#f_plot     ='plots_ff'

#f          ='weights_pulse'
#f_plot     ='plots_pulse'

#f          ='weights_osc'
#f_plot     ='plots_osc'

#f          ='weights_not'
#f_plot     ='plots_not'

distancias = []

for t in time_vector:
    for i in np.arange(5,10,1):
        mem_gap = 20 #20 boolean /200 osc.
        N_rec   =t
        base= f+'/'+  os.path.basename(f+'_'+str(mem_gap)+'_N_'+str(N_rec)+'_gap_'+str(i))
        base_plot= f_plot+'/'+  os.path.basename(f_plot+'_'+str(t)+'_N_'+str(i))
        dir = str(base)
        if not os.path.exists(dir):
           os.mkdir(base)
        print(str(dir))

        dir = str(base_plot)
        if not os.path.exists(dir):
           os.mkdir(base_plot)        
        print(str(dir))
    
        pepe    =and_fun(mem_gap,N_rec,base,base_plot)
        distancias.append(pepe)
print('-------------------------')
print (distancias)
print("--- %s to train the network seconds ---" % (time.time() - start_time))
