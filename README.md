# Excitatory-inhibitory
Code for the generation and Analysis of RNNs of paper: Effect in the spectra of eigenvalues and dynamics of RNN trained with Excitatory-Inhibitory constraint

Simulations were generated using python 3.6.9 Tensorflow version 2.0 and Keras 2.3.1 Following the procedure described previously in https://arxiv.org/abs/1906.01094

To generate your sim configure and run loop_to_call.py, which allows you to choose the task for traning, the number of networks to train and additional details.
To change Training parameters you shold edit: 

binary_and_recurrent_exi_ini_01.py

The trained networks are saved in hdf5 format and can be opened using.

generate_figures.py

to generate all the figures corresponding to one particular realization using the corresponding testing set generator.

For example to test the "and" task, use generate_data_set_and.py
