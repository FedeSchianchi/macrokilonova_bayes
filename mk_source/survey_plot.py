import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mt
import matplotlib.cm as matcm
import matplotlib as mpl


Lambda = np.loadtxt('source/surveys/Lambda')
xi_w = np.loadtxt('source/surveys/xi_w')
xi_s = np.loadtxt('source/surveys/xi_s')
mdisk = np.loadtxt('source/surveys/mdisk')


min_list = []
max_list = []


if np.shape(Lambda)==():

    chi2 = np.loadtxt('source/surveys/Lambda='+str(Lambda))
    max_list.append( np.ndarray.max(chi2) )
    min_list.append( np.ndarray.min(chi2) )


else:
    for i in Lambda:

        chi2 = np.loadtxt('source/surveys/Lambda='+str(i))
        max_list.append( np.ndarray.max(chi2) )
        min_list.append( np.ndarray.min(chi2) )

#v_max = max(max_list)
#v_min = min(min_list)

v_min = 2e4
v_max = 1e5

i = 0

if np.shape(Lambda)==():

    chi2 = np.loadtxt('source/surveys/Lambda=' + str(Lambda))
    plt.imshow( chi2, interpolation=None, cmap='jet', vmin=v_min, vmax=v_max, extent=[np.amin(xi_s), np.amax(xi_s), np.amin(xi_w), np.amax(xi_w)] )
    plt.title('$\widetilde{\Lambda}$=' + str(int(Lambda)))
    #plt.scatter(0.4, 0.2, color='red')
    plt.xlabel('$ xi_{s} $')
    plt.ylabel('$ xi_{w} $')

    cax = plt.axes([0.80, 0.09, 0.01, 0.8])
    cmap = plt.get_cmap('jet', 100)
    norm = mpl.colors.LogNorm(vmin=v_min, vmax=v_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('$\chi^2$')

else:

    fig, ax = plt.subplots(3, 2, figsize=(5.3, 8), sharey=True, sharex=True)
    fig.subplots_adjust(wspace=0.01, hspace=0.4, left=0.1, right=0.90)

    for a in ax:
        for b in a:

            chi2 = np.loadtxt('source/surveys/Lambda='+str(Lambda[i]))

            b.set_title('$\widetilde{\Lambda}$='+str(int(Lambda[i])))
            b.imshow( chi2, origin='lower', interpolation=None, cmap='jet', vmin=v_min, vmax=v_max, extent=[np.amin(xi_s), np.amax(xi_s), np.amin(xi_w), np.amax(xi_w)] )
            b.scatter(0.4, 0.2, color='red')

            i+=1

    fig.text(0.02, 0.5, r'$ \xi_{w} $', va='center', rotation='vertical')
    fig.text(0.5, 0.04, r'$ \xi_{s} $', va='center', rotation='horizontal')
    fig.text(0.96, 0.5, '$ \chi^2 $', va='center', rotation='horizontal')

    cax = plt.axes([0.87, 0.09, 0.01, 0.8])
    cmap = plt.get_cmap('jet', 100)
    norm = mpl.colors.LogNorm(vmin=v_min, vmax=v_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax = cax)

plt.savefig('source/surveys/survey_plot')

plt.show()