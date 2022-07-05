import os
import time
import fnmatch

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.cm as cm


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)



#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/plots/2022-02-07_estudio_radios_y_parametros/DM"

r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/plots/2022-06-studio_size_network"
#r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/plots/2022-02-07_estudio_radios_y_parametros/ran_normal_dm"


   #r_dir="/home/kathy/Desktop/Neural_Networks/my_code_2020-2021/extitatorias_inhibitorias/plots/2022-02-07_estudio_radios_y_parametros/work"
########### Fig a) ############
#colors = cm.rainbow(np.linspace(0, 1, 31))

colors=plt.cm.Spectral(np.linspace(0, 1, 20))

fig=plt.figure(figsize=cm2inch(13,8) )
#ax = fig.add_axes([0, 0, 1, 1])  
cmap='viridis'

'''
import matplotlib
for name, hex in matplotlib.colors.cnames.items():
    print(name, hex)
    colors.append(name)
'''    
n=0

lista_mean=[]
lista_n=[]
lista_error=[]

for root, sub, files in os.walk(r_dir):
    
    files = sorted(files)
    print("root",root)
    label=str(root)
    label_list=label.split('/')
    va=label_list[-1]
    print(va)
    print (label)
    for i,f in enumerate(files):
        print(f)
        if fnmatch.fnmatch(f, 'radios*.txt'):#fnmatch.fnmatch(f, '*initial.hdf5') :#
           n=n+1
           r_dir=root
           pepe    =np.genfromtxt(r_dir+"/"+f,delimiter='\t')
           #print("pepe",pepe)
           w       = pepe.T
           histo_lista    =[]
           histo_lista.extend(w[1])
           #print(histo_lista)
           media= np.average(histo_lista)
           standar_dev=np.std(histo_lista, ddof=1) / np.sqrt(np.size(histo_lista))
           lista_mean.append(media)
           media= "%.4f" % media
           lista_n.append(float(va))
           lista_error.append(standar_dev)
           #n, bins, patches = plt.hist(histo_lista, 10, normed=1, facecolor='green', alpha=0.75)
           plt.hist( histo_lista,15, alpha = 0.8,label="R-mean: "+str(media)+" "+va, color=colors[n])#, color=colors[len(histo_lista)])
           
           #plt.xticks(np.arange(4.9, 1.1,0.25))#,fontsize = 5)
           plt.legend(fontsize= 5,loc=1)
           
    '''
    for i,f in enumerate(files):
        #print(f)
        if fnmatch.fnmatch(f, 'H_number.txt'):#fnmatch.fnmatch(f, '*initial.hdf5') :#
           print("file: ",f)
           r_dir=root
           pepe    =np.genfromtxt(r_dir+"/"+f,delimiter=' ')
           print(pepe)
           w       = pepe.T
           histo_lista    =[]
           histo_lista.extend(w[1])
           print(histo_lista)
           media= np.average(histo_lista)
           media= "%.4f" % media
           #n, bins, patches = plt.hist(histo_lista, 10, normed=1, facecolor='green', alpha=0.75)
           plt.hist( histo_lista, alpha = 0.5,label="H Ini mean "+str(media))
           plt.legend(fontsize= 5,loc=1)
           #plt.xticks(np.arange(4.9, 1.1,0.25))#,fontsize = 5)

   '''       
 
#plt.xlabel('H number',fontsize = 8)
#plt.axvline(x=0, linewidth=1, color='grey',alpha=.15)
#plt.axvline(x=1, linewidth=1, color='grey',alpha=.15)
#plt.xticks([])
plt.yticks([])
plt.xlim(0.0,2)

#plt.ylim([min(w[1])-0.01,max(w[1])+0.01])
#plt.xticks(np.arange(0.0, 2,0.2),fontsize = 5)#ortho
#plt.xticks(np.arange(0.4, 1.2,0.2),fontsize = 5)#ortho

#plt.xticks(np.arange(0.0, 1,0.2),fontsize = 5)#normal
#plt.yticks(np.arange(0,8.1,2),fontsize = 5)
plt.savefig("plots/radio_histo.png",dpi=300, bbox_inches = 'tight')
#plt.show()
plt.close()

mean_line= np.average(lista_mean)
sigma_mas= np.average(lista_mean)+np.std(lista_mean)
sigma_menos=np.average(lista_mean)-np.std(lista_mean)

fig=plt.figure(figsize=cm2inch(13,8) )
plt.ylim(0.0,1.32)
plt.axhline(y=sigma_mas, linewidth=1, color='grey',linestyle='--',alpha=.15)
plt.axhline(y=sigma_menos, linewidth=1, color='grey',linestyle='--',alpha=.15)

plt.axhline(y=mean_line, linewidth=1, color='grey',linestyle='--',alpha=.15)
plt.yticks(np.arange(0.0, 1.31,0.2),fontsize = 5)
#plt.scatter(lista_n, lista_mean, label="radio vs size",color="green",s=2)
plt.errorbar(lista_n, lista_mean, yerr=lista_error, fmt='|'  , label="Mean radio vs Network size",color=colors[2]) #\n Orthogonal Initial condition
plt.legend(fontsize= 5,loc=1)

lista_n.append(0)
lista_n.append(550)
lista_n.sort()
x = np.array(lista_n)
print(x)

plt.xlim(50,520)
plt.fill_between(x,sigma_menos,sigma_mas,facecolor='lightblue')

plt.xlabel('Network size',fontsize = 8)
plt.ylabel("Radio")
plt.savefig("plots/radio_vs_size.png",dpi=300, bbox_inches = 'tight')
#plt.show()
plt.close()



