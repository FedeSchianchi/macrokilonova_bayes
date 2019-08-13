import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os


goodness_parameter = 'chi2'

par  = pd.read_csv('source/params/models.csv')
data = pd.read_csv('source/data_output/data', engine='python', sep=' ')

if os.path.exists('magnitude_plots/output_plots') == False:
    os.mkdir('magnitude_plots/output_plots')


# list of bands you want to plot
bands = {}

bands[0] = ('U', 'B', 'V')
bands[1] = ('r', 'i', 'z')
#bands[2] = ('y', 'J', 'H', 'Ks')


l=len( par['name'] )


for m in range(l):

    fig, ax = plt.subplots(len(bands), 1, figsize=(5, 8), sharex=True)
    fig.subplots_adjust(hspace=0, left=0.15)

    model=pd.read_csv('source/mkn_output/mkn_model' + str(m) +'.txt', engine='python', sep=' ')
    t=model['time[s]'].to_numpy()/(3600*24)

    for p in range(len(bands)):
        for q in range( len(bands[p]) ):

            band = np.zeros( len(t) )
            n=0


            for u in model.columns:

                if u[0:2].replace('_','') == bands[p][q]:
                    n += 1
                    band += model[u]


            band = band/n

            #loading data from telescopes
            data = pd.read_csv('source/data_output/data', engine='python', sep=' ')
            for l in bands[p]:
                for k in range( len(data['time'].to_numpy()) ):
                    if data['name'][k][0:2].replace('_','')==l:
                        data['name'][k]=l


            data={'time' :  data[ data['name']==bands[p][q] ]['time'].to_numpy(),
                  'mag'  : -data[ data['name']==bands[p][q] ]['magnitude'].to_numpy(),
                  'err'  :  data[ data['name']==bands[p][q] ]['err'].to_numpy()}


            ax[p].errorbar( data['time'], data['mag'], xerr=0., yerr=data['err'], color='C'+str(q), linestyle='None', linewidth=1.4 )

            ax[p].plot( t, -band, color='C'+str(q), label=bands[p][q], linestyle='--', linewidth=0.5)

            ax[p].legend(loc='lower left')


    print( par['name'][m] , str( par[goodness_parameter][m] ) )
    plt.suptitle(par['name'][m])
    plt.xlabel( 'time (days)' )
    fig.text(0.04, 0.5, 'AB magnitude', va='center', rotation='vertical')
    if par[goodness_parameter][m] >= 0 or par[goodness_parameter][m] < 0:
        plt.savefig( 'magnitude_plots/output_plots/' + par['name'][m] + 'chi2=' + str( int(par[goodness_parameter][m]) ) )
    else:
        plt.savefig('magnitude_plots/output_plots/' + par['name'][m] + 'chi2=' + str(par[goodness_parameter][m]))
    plt.show()