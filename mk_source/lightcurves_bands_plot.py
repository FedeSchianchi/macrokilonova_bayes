import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

par=pd.read_csv('source/params/bands_allcomps.csv')

# list of 3 bands you want to plot
bands = ('g', 'z', 'Ks')
l=len( par['name'] )
l=l/9


for m in range( int(l) ):

    fig, ax = plt.subplots(1, 3, figsize=(9, 5), sharey=False)
    fig.subplots_adjust(wspace=0.25, hspace=0, left=0.15)
    fig.suptitle(par['name'][9*m])

    for q in range(3):

        n_min=9*m+3*q
        n=1+9*m+3*q
        n_max=2+9*m+3*q

        model=pd.read_csv('source/mkn_output/mkn_model' + str(n) +'.txt', engine='python', sep=' ')
        model_max=pd.read_csv('source/mkn_output/mkn_model' + str(n_max) + '.txt', engine='python', sep=' ')
        model_min=pd.read_csv('source/mkn_output/mkn_model' + str(n_min) + '.txt', engine='python', sep=' ')
        t=model['time[s]'].to_numpy()/(3600*24)


        a_band = np.zeros( len(t) )
        b_band = np.zeros( len(t) )
        c_band = np.zeros( len(t) )

        a_band_max = np.zeros( len(t) )
        b_band_max = np.zeros( len(t) )
        c_band_max = np.zeros( len(t) )

        a_band_min = np.zeros( len(t) )
        b_band_min = np.zeros( len(t) )
        c_band_min = np.zeros( len(t) )

        na=0
        nb=0
        nc=0


        for u in model.columns:

            if u[0:2].replace('_','') == bands[0]:
                na += 1
                a_band += model[u]
                a_band_max += model_max[u]
                a_band_min += model_min[u]

            elif u[0:2].replace('_','') == bands[1]:
                nb += 1
                b_band += model[u]
                b_band_max += model_max[u]
                b_band_min += model_min[u]

            elif u[0:2].replace('_','') == bands[2]:
                nc += 1
                c_band += model[u]
                c_band_max += model_max[u]
                c_band_min += model_min[u]


        a_band=a_band/na
        a_band_min=a_band_min/na
        a_band_max=a_band_max/na

        b_band=b_band/nb
        b_band_min=b_band_min/nb
        b_band_max=b_band_max/nb

        c_band=c_band/nc
        c_band_min=c_band_min/nc
        c_band_max=c_band_max/nc


        #loading data from telescopes

        data=pd.read_csv('source/data_output/data', engine='python', sep=' ')

        for l in bands:
            for k in range(len(data['time'].to_numpy())):
                if data['name'][k][0:2].replace('_','')==l:
                    data['name'][k]=l


        a_data={'a_time' : data[data['name']==bands[0]]['time'].to_numpy(),
                'a_mag' : -data[data['name']==bands[0]]['magnitude'].to_numpy(),
                'a_err' : data[data['name']==bands[0]]['err'].to_numpy()}


        b_data={'b_time' : data[data['name']==bands[1]]['time'].to_numpy(),
                'b_mag' : -data[data['name']==bands[1]]['magnitude'].to_numpy(),
                'b_err' : data[data['name']==bands[1]]['err'].to_numpy()}


        c_data={'c_time' : data[data['name']==bands[2]]['time'].to_numpy(),
                 'c_mag' : -data[data['name']==bands[2]]['magnitude'].to_numpy(),
                 'c_err' : data[data['name']==bands[2]]['err'].to_numpy()}


        ax[q].errorbar( a_data['a_time'], a_data['a_mag'], xerr=0., yerr=a_data['a_err'], color='blue', linestyle='None', linewidth=1. )
        ax[q].errorbar( b_data['b_time'], b_data['b_mag'], xerr=0., yerr=b_data['b_err'], color='red', linestyle='None', linewidth=1. )
        ax[q].errorbar( c_data['c_time'], c_data['c_mag'], xerr=0., yerr=c_data['c_err'], color='black', linestyle='None', linewidth=1. )


        ax[q].plot( t, -a_band, color='blue', label=bands[0].replace('_',''), linestyle='--', linewidth=0.5)
        ax[q].fill_between(t, -a_band_min, -a_band_max, color='blue', alpha=0.5)

        ax[q].plot( t, -b_band, color='red', label=bands[1].replace('_',''), linestyle='--', linewidth=0.5)
        ax[q].fill_between(t,  -b_band_min, -b_band_max, color='red', alpha=0.5 )

        ax[q].plot( t, -c_band, color='black', label=bands[2].replace('_',''), linestyle='--', linewidth=0.5)
        ax[q].fill_between(t, -c_band_min, -c_band_max, color='black', alpha=0.5 )


        ax[q].set_title( par['geometry'][1+3*q+9*m] )
        print(2+3*q+9*m)


    ax[1].set_xlabel( 'time (days)' )
    ax[0].set_ylabel( 'AB magnitude' )
    ax[1].legend(loc='lower left')
    plt.savefig( 'magnitude_plots/output_plots/' + par['name'][9*m] + ' ' + bands[0] + bands[1] + bands[2] )
    plt.show()