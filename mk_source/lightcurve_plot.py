import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os


goodness_parameter = 'chi2_4c'

par=pd.read_csv('source/params/models_4c_NRinfo.csv')

if os.path.exists('magnitude_plots/output_plots') == False:
    os.mkdir('magnitude_plots/output_plots')


# list of bands you want to plot
bands = ('g', 'z', 'Ks')


l=len( os.listdir('source/mkn_output') )
treshold=3e4  #plot only lightcurves with chi3 lower than...


for m in range(l):

    fig = plt.figure()

    if par[goodness_parameter][m] <= treshold:

        filepath = 'magnitude_plots/output_plots/' + str(par['name'][m]) 
        if os.path.exists(filepath) == False:
            os.mkdir(filepath)


        model=pd.read_csv('source/mkn_output/mkn_model' + str(m) +'.txt', engine='python', sep=' ')
        t=model['time[s]'].to_numpy()/(3600*24)

        for q in range( len(bands) ):

            band = np.zeros( len(t) )
            n=0

            for u in model.columns:

                if u[0:2].replace('_','') == bands[q]:
                    n += 1
                    band += model[u]


            band = band/n


            #loading data from telescopes

            data=pd.read_csv('source/data_output/data', engine='python', sep=' ')

            for l in bands:
                for k in range(len(data['time'].to_numpy())):
                    if data['name'][k][0:2].replace('_','')==l:
                        data['name'][k]=l


            data={'time' : data[data['name']==bands[q]]['time'].to_numpy(),
                  'mag'  : -data[data['name']==bands[q]]['magnitude'].to_numpy(),
                  'err'  : data[data['name']==bands[q]]['err'].to_numpy()}


            plt.errorbar( data['time'], data['mag'], xerr=0., yerr=data['err'], color='C'+str(q), linestyle='None', linewidth=1.4 )

            plt.plot( t, -band, color='C'+str(q), label=bands[q], linestyle='--', linewidth=0.5)

        #plt.title( par['name'][m] )
        print( str(m) , str( par[goodness_parameter][m] ) )


        #plt.xscale('log')
        plt.xlabel( 'time (days)' )
        plt.ylabel( 'AB magnitude' )
        plt.legend(loc='lower left')
        plt.savefig( 'magnitude_plots/output_plots/'+  str(par['name'][m]) + '/' + str( par['name'][m] ) + str( bands ) )
        #plt.show()
        plt.close()
