#########################################
#                                       #
#     An "Or" data set generator        #
#            of samples                 #
#     with adjutable parameters         #
#                                       #
#   Mit License C. Jarne V. 1.0 2018    #
#########################################

import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.random import seed

start_time = time.time()

semillita=2
seed(semillita)
#def generate_trials(size): #No time loop
def generate_trials(size, mem_gap):#time loop
    seed(3)

    first_in         = 60 #-25#time to start the first stimulus   #30 #60
    stim_dur         = 20 #stimulus duration #20 #30
    stim_noise       =0.1# 0.1 #noise
    var_delay_length = 50#change for a variable length stimulus
    out_gap          = 140#300#140#+25#130 #how much lenth add to the sequence duration    #140 #100
    sample_size      = size # sample size
    rec_noise        = 0
    seed(3)

    #or_seed         = np.array([[1, 1],[0, 1],[1, 0],[0, 0]])
    or_seed_A = np.array([[0],[1],[0],[1]])
    or_seed_B = np.array([[0],[0],[1],[1]])
    or_y            = np.array([0,1,1,1])
    seq_dur          = first_in+stim_dur+mem_gap+var_delay_length+out_gap #Sequence duration

    if var_delay_length == 0:
        var_delay = np.zeros(sample_size, dtype=np.int)
    else:
        var_delay = np.random.randint(var_delay_length, size=sample_size) + 1
    second_in = first_in + stim_dur + mem_gap

    out_t       = mem_gap+ first_in+stim_dur
    trial_types = np.random.randint(4, size=sample_size)
    x_train     = np.zeros((sample_size, seq_dur, 2))
    y_train     = 0.01 * np.ones((sample_size, seq_dur, 1))

    for ii in np.arange(sample_size):
        x_train[ii, first_in+var_delay[ii]:first_in + stim_dur+var_delay[ii], 0] = or_seed_A[trial_types[ii], 0]
        x_train[ii, first_in+var_delay[ii]:first_in + stim_dur+var_delay[ii], 1] = or_seed_B[trial_types[ii], 0]
        y_train[ii, out_t + var_delay[ii]:, 0]       = or_y[trial_types[ii]]

    mask = np.zeros((sample_size, seq_dur))
    for sample in np.arange(sample_size):
        mask[sample,:] = [1 for y in y_train[sample,:,:]]

    x_train = x_train#+ stim_noise * np.random.randn(sample_size, seq_dur, 2)
    print("--- %s seconds to generate Or dataset---" % (time.time() - start_time))
    return (x_train, y_train, mask,seq_dur)

#To see how is the training data set uncoment these lines

sample_size=10
mem_gap=20
x_train,y_train, mask,seq_dur = generate_trials(sample_size,mem_gap) 

#print ("x",x_train)
#print ("y",y_train)

fig     = plt.figure(figsize=(6,8))
fig.suptitle("\"And\" Data Set Training Sample\n (amplitude in arb. units time in mSec)",fontsize = 20)
for ii in np.arange(10):
    plt.subplot(5, 2, ii + 1)
    plt.plot(x_train[ii, :, 0],color='g',label="Input A")
    plt.plot(x_train[ii, :, 1],color='pink',label="Input B")
    plt.plot(y_train[ii, :, 0],color='grey',label="Output")
    plt.ylim([-2.5, 2.5])
    plt.legend(fontsize= 5,loc=3)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
figname = "data_set_or_sample_seed_"+str(semillita)+".png"
#figname = base+"/data_set_and_sample.png" 
plt.savefig(figname,dpi=200)
plt.show()




