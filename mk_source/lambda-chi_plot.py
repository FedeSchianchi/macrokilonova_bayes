import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_old = pd.read_csv('source/params/summary.csv')
data_new = pd.read_csv('source/params/models_3c.csv')

xparam='Lambda'

# extracting Lambda from old simulations
x_BHBlp_old = data_old[data_old['EOS'] == 'BHBlp'][xparam].to_numpy()
x_DD2_old = data_old[data_old['EOS'] == 'DD2'][xparam].to_numpy()
x_LS220_old = data_old[data_old['EOS'] == 'LS220'][xparam].to_numpy()
x_SFHo_old = data_old[data_old['EOS'] == 'SFHo'][xparam].to_numpy()
x_SLy4_old = data_old[data_old['EOS'] == 'SLy4'][xparam].to_numpy()

# extracting Lambda from new simulations
x_DD2_new = data_new[data_new['EOS'] == 'DD2'][xparam].to_numpy()
x_LS220_new = data_new[data_new['EOS'] == 'LS220'][xparam].to_numpy()
x_SFHo_new = data_new[data_new['EOS'] == 'SFHo'][xparam].to_numpy()
x_SLy4_new = data_new[data_new['EOS'] == 'SLy4'][xparam].to_numpy()

weight=1.

for i in (0,1):

    if i == 0:
        yparam = 'errw_simDisk'
        weight = 0.3
        labels=(None, None, None, None, None)
    else:
        yparam = 'errw_fitDisk'
        weight = 1.
        labels = ('DD2', 'LS220', 'SFHo', 'SLy4', 'BHBlp')


    #extracting Mdisk from old simulations
    y_BHBlp_old = data_old[ data_old['EOS'] == 'BHBlp' ][ yparam ].to_numpy()
    y_DD2_old = data_old[ data_old['EOS'] =='DD2' ][ yparam ].to_numpy()
    y_LS220_old = data_old[ data_old['EOS'] =='LS220' ][ yparam ].to_numpy()
    y_SFHo_old = data_old[ data_old['EOS'] =='SFHo' ][ yparam ].to_numpy()
    y_SLy4_old = data_old[ data_old['EOS'] == 'SLy4' ][ yparam ].to_numpy()

    #extracting Mdisk from new simulations
    y_DD2_new = data_new[ data_new['EOS'] =='DD2' ][ yparam ].to_numpy()
    y_LS220_new = data_new[ data_new['EOS'] =='LS220' ][ yparam ].to_numpy()
    y_SFHo_new = data_new[ data_new['EOS'] =='SFHo' ][ yparam ].to_numpy()
    y_SLy4_new = data_new[ data_new['EOS'] == 'SLy4' ][ yparam ].to_numpy()


    #plotting old data
    plt.scatter( x_BHBlp_old, y_BHBlp_old, marker='o', label=labels[4], color='gray', alpha=weight )
    plt.scatter( x_DD2_old, y_DD2_old, marker='P', color='gray', norm=weight, alpha=weight )
    plt.scatter( x_LS220_old, y_LS220_old, marker='v', color='gray', norm=weight, alpha=weight )
    plt.scatter( x_SFHo_old, y_SFHo_old, marker='h', color='gray', norm=weight, alpha=weight )
    plt.scatter( x_SLy4_old, y_SLy4_old, marker='*', color='gray', norm=weight, alpha=weight )

    #plotting new data
    plt.scatter( x_DD2_new, y_DD2_new, marker='P', label=labels[0], color='b', alpha=weight )
    plt.scatter( x_LS220_new, y_LS220_new, marker='v', label=labels[1], color='y', alpha=weight )
    plt.scatter( x_SFHo_new, y_SFHo_new, marker='h', label=labels[2], color='r', alpha=weight )
    plt.scatter( x_SLy4_new, y_SLy4_new, marker='*', label=labels[3], color='g', alpha=weight )

    yticks = np.logspace(2e4, 1e5, 5)
    xticks = ( 0, 0.2, 0.4, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2.0,)

plt.fill_betweenx( (0,1e30), 800, 2000, alpha=0.1, label='excluded by GW170817' )

plt.xlabel( '$\widetilde{\Lambda}$' )
plt.ylabel( '$\chi^2$' )
#plt.yticks( yticks )
#plt.xticks( xticks )
plt.ylim( 0, 0.05 )
plt.xlim(0 , 1500 )
#plt.yscale( 'log' )
plt.legend( loc=0 )

plt.savefig('lambda-chi_plots/lambda-errw')

plt.show()
