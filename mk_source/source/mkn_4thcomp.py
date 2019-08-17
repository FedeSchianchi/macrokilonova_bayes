import shutil
import os
import angular_distribution as ad
import ejecta as ej
import filters as ft
import math
import matplotlib.pyplot as plt
import numpy as np
import observer_projection as op
import source_properties as sp
import units
import pandas as pd
import bayes_prob as bayes
import scipy.integrate as integrate


#fitting function
def f(x, alpha, beta, gamma, delta):
    return np.amax(np.array([1e-3, alpha+beta*np.tanh((x-gamma)/delta)]), axis=0)


class MKN(object):
    """
    Driver for the kilonova model
    """

    def __init__(self,
                 # number of different components of the ejecta
                 glob_params,
                 # dictionary of global parameters defining basic properties of the ejecta to be sampled
                 glob_vars,
                 # dictionary of ejecta parameters defining its composition and geometry
                 ejecta_params,
                 # dictionary of shell parameters defining basic properties of the shell
                 ejecta_vars,
                 source_name,
                 **kwargs):

        super(MKN,self).__init__(**kwargs)

        #initializing the global properties of the source
        #print('I am initializing the global properties of the source')
        SP = sp.SourceProperties(source_name)
        self.D = SP.D
        self.view_angle = SP.view_angle

        #initialize the angular distribution
        #print('I am initializing the angles')
        self.n_slices = glob_params['n slices']
        self.dist_slices = glob_params['dist slices']
        self.AD = ad.AngularDistribution(self.dist_slices,self.n_slices)
        self.angular_distribution, self.omega_distribution = self.AD(self.n_slices/2,glob_params['omega frac'])   # due to the symmetry abount the equatorial plane, the number of independent slices is half

        #initialize the filters
        #print('I am initializing the filters')
        if (source_name == 'default'):
            self.FT = ft.Filters("properties")
        else:
            self.FT = ft.Filters("measures")
        self.dic_filt, self.lambda_vec, self.mag = self.FT(SP.filter_data_folder, glob_params["time min"]*units.sec2day+SP.t0, glob_params["time max"]*units.sec2day+SP.t0)


        #initialize the time
        #print('I am initializing the global time')
        self.time_min = glob_params['time min']
        self.time_max = glob_params['time max']
        self.n_time   = glob_params['n time']
        self.tscale   = glob_params['scale for t']
        self.t0 = SP.t0
        if (self.tscale == "measures" and source_name=='default'):
            print('')
            print("no measures available to set the time (default option)")
            print("please use linear or log scale")
            exit()

        self.time = SP.init_time(self.tscale, self.time_min, self.time_max, self.n_time, self.mag)

        #initialize the observer location
        #print('I am initializing the observer orientation')
        self.FF = op.ObserverProjection(self.n_slices, self.dist_slices)
#        self.flux_factor = self.FF(SP.view_angle)

        # register the geometry of the ejecta and create an Ejecta object
        self.glob_params   = glob_params
        self.glob_vars     = glob_vars
        self.ejecta_params = ejecta_params
        self.ejecta_vars   = ejecta_vars
#        self.ejecta       = ej.Ejecta(len(self.ejecta_params),self.ejecta_params.keys(),self.ejecta_params)

        #print('I am initializing the components')
        self.E = ej.Ejecta(len(self.ejecta_params.keys()), self.ejecta_params.keys(), self.ejecta_params)

#############################
#  LIKELIHOOD CALCULATION  #
#############################

    def compute_log_likelihood(self,residuals, errs):
        # compute the likelihood
        chi2 = 0.
        err = 0.
        ndata = 0.
        for ilambda in residuals.keys():
            for res in residuals[ilambda]:
                chi2 += res*res
            for error in errs[ilambda]:
                err += error
                ndata += 1
        err = err / ndata
        return (chi2, err, ndata)

    def log_likelihood(self,r_ph,T_eff):

        self.flux_factor = self.FF(self.view_angle)

        # compute the residuals
        if (source_name != 'default'):
            #print('i.e., I am computing residuals')
            (residuals, errs) = ft.calc_all_residuals(  self.flux_factor,  self.time,  r_ph,T_eff,  self.lambda_vec,  self.dic_filt,  self.D, self.t0,  self.mag, 'const' ) #'const' or '1/t'

        # compute the likelihood
            #print('and then I am computing the likelihood')
            (chi2, err, ndata) = self.compute_log_likelihood(residuals, errs)

            #print('logL:',logL)

        else:
            logL = 0.

        return (chi2, err, ndata)

#########################
# write out the output  #
#########################

    def write_output(self, r_ph,T_eff, L_bol, i):

        # compute the bolometric luminosity
        self.model_lum = ft.calc_lum_iso(L_bol, self.flux_factor)

        # compute the magnitudes
        self.model_mag = ft.calc_magnitudes(self.flux_factor, self.time, r_ph,T_eff, self.lambda_vec, self.dic_filt, self.D, self.t0)

        # open the output file
        file_output = 'mkn_model'+ i +'.txt'

        g = open('mkn_output/'+file_output,'w')

        #print('file name:',file_output)

        # write the header
        g.write('time[s]'+' '+'luminosity[erg/s]'+' ')
        for ilambda in self.dic_filt.keys():
            if (self.dic_filt[ilambda]['active'] !=1):
                continue
            g.write((self.dic_filt[ilambda]['name']))
            g.write(' ')
        g.write('\n')
        g.write('\n')

        # write profiles of the physical quantities
        for i in range(len(self.time)):
            g.write(str(self.time[i])+' '+str(self.model_lum[i])+' ')
            for ilambda in self.dic_filt.keys():
                if (self.dic_filt[ilambda]['active'] !=1):
                    continue
                g.write(str(self.model_mag[ilambda][i]))
                g.write(' ')
            g.write('\n')

        g.close()

################################
# plot some of the lightcurves #
################################


    def write_data(self):

        p = open('data_output/data', 'w')
        p.write('name' + ' ' + 'time' + ' ' + 'magnitude' + ' ' + 'err')
        p.write('\n')
        p.write('\n')


        for ilambda in self.mag.keys():
            if (source_name != 'default'):
                if( self.dic_filt[ilambda]['plot'] == 1):
                    for l in range(len(self.mag[ilambda]['time'])):
                        p.write( self.mag[ilambda]['name'] + ' ' + str(self.mag[ilambda]['time'][l]-self.t0) + ' ' + str(self.mag[ilambda]['mag'][l])
                                 + ' ' + str(self.mag[ilambda]['sigma'][l]) )
                        p.write('\n')

        p.close()


if __name__=='__main__':

    shutil.rmtree('data_output')
    shutil.rmtree('mkn_output')

    os.mkdir('data_output')
    os.mkdir('mkn_output')

    filename = 'params/models_4c_NRinfo.csv'   #tab containing data to be imported

    params= pd.read_csv(filename)

    N = len(params['name'].to_numpy())

    write_output = True   #set True if you whant save the light curves
    bern = 'Tab'  #'fit' for use m_bernoulli from Lambda by the fit
    fit_disk = 'new'  #'new' for new fit, 'old' for Radice's fit


    if fit_disk == 'new':
        # parameters from new fit
        alpha = 9.035807380707361e-15
        beta = 0.0998368736646533
        gamma = 295.0367164875689
        delta = 162.49822267566785

    elif fit_disk == 'old':
        # Radice's parameters
        alpha = 0.084
        beta = 0.127
        gamma = 567.1
        delta = 405.14

    if bern == 'fit':
        # m_bern fit parameters
        a_bern = 1.1567870e-05
        b_bern = -1.7261556e-03
        m_bern = params['Lambda'].to_numpy()*a_bern+b_bern

    # evaluating opacity and Mdisk arrays from data in the tab
    opacity = {}
    opacity_bern = {}
    mdisk = {}

    if 'Yee' in params.keys():
        for k in range(N):
            if params['Yeej'][k] >= 0.25:
                opacity[k] = 1.
            else:
                opacity[k] = 30.

    if 'Yeej_bern' in params.keys():
        for k in range(N):
            if params['Yeej_bern'][k] <= 0.25:
                opacity_bern[k] = 30.
            else:
                opacity_bern[k] = 1.

    #evaluate Mdisk from the fit when we don't have it
    if ( 'Lambda' in params.keys() and 'Mdisk' in params.keys() ):
        for k in range(N):
            if math.isnan(params['Mdisk'][k]):
                mdisk[k] = f(params['Lambda'][k], alpha, beta, gamma, delta)
            else:
                mdisk[k] = params['Mdisk'][k]

    # evaluating an array containing only Mdisk from the fitting formula
    if 'Lambda' in params.keys():
        mdisk_fit = np.zeros(N)
        for r in range(N):
            mdisk_fit[r] = f(params['Lambda'].to_numpy()[r], alpha, beta, gamma, delta)


    for i in range(N):

    #dictionary with the global parameters of the model
        glob_params = {'lc model'   :'grossman',    # model for the lightcurve (grossman or villar)
                       'mkn model'  :'aniso4comp',    # possible choices: iso1comp, iso2comp, iso3comp, iso4comp, aniso1comp, aniso2comp, aniso3comp, aniso4comp
                       'omega frac' :1.0,           # fraction of the solid angle filled by the ejecta
                       'rad shell'  :True,          # exclude the free streaming part
                       'v_min'      :1.e-7,         # minimal velocity for the Grossman model
                       'n_v'        :400,           # number of points for the Grossman model
                       'vscale'     :'linear',      # scale for the velocity in the Grossman model
                       'sigma0'     :0.11,          # parameter for the nuclear heating rate
                       'alpha'      :1.3,           # parameter for the nuclear heating rate
                       't0eps'      :1.3,           # parameter for the nuclear heating rate
                       'cnst_eff'   :0.3333,        # parameter for the constant heating efficiency
                       'n slices'   :30,            # number for the number of slices along the polar angle [12,18,24,30]
                       'dist slices':'cos_uniform', # discretization law for the polar angle [uniform or cos_uniform]
                       'time min'   :3600.,         # minimum time [s]
                       'time max'   :1036800./2,       # maximum time [s]
                       'n time'     :200,           # integer number of bins in time
                       'scale for t':'linear',    # kind of spacing in time [log - linear - measures]
                       'NR_data'    :True,         # use (True) or not use (False) NR profiles
                       'NR_filename':'../example_NR_data/' + str( params['name'][i] )          # path of the NR profiles, necessary if NR_data is True
                       }

        source_name = 'AT2017gfo'   # name of the source or "default"
        #source_name = 'default'   # name of the source or "default"

        # dictionary for the global variables
        glob_vars = {'m_disk':  mdisk_fit[i],        # mass of the disk [Msun], useful if the ejecta is expressed as a fraction of the disk mass
                     'eps0':1.2e19,        # prefactor of the nuclear heating rate [erg/s/g]
                     'T_floor_LA':1000.,   # floor temperature for Lanthanides [K]
                     'T_floor_Ni':3500.,   # floor temperature for Nikel [K]
                     'a_eps_nuc':0.5,      # variation of the heating rate due to weak r-process heating: first parameter
                     'b_eps_nuc':2.5,      # variation of the heating rate due to weak r-process heating: second parameter
                     't_eps_nuc':1.0}      # variation of the heating rate due to weak r-process heating: time scale [days]

    ###############################
    # Template for isotropic case #
    ###############################

        # hardcoded ejecta geometric and thermal parameters for the spherical case
        ejecta_params_iso = {}
        ejecta_params_iso['dynamics']   = {'mass_dist':'uniform', 'vel_dist':'uniform', 'op_dist':'uniform', 'therm_model':'BKWM', 'eps_ye_dep':True, 'v_law':'poly'}
        ejecta_params_iso['wind']       = {'mass_dist':'uniform', 'vel_dist':'uniform', 'op_dist':'uniform', 'therm_model':'BKWM', 'eps_ye_dep':True, 'v_law':'poly'}
        ejecta_params_iso['secular']    = {'mass_dist':'uniform', 'vel_dist':'uniform', 'op_dist':'uniform', 'therm_model':'BKWM', 'eps_ye_dep':True, 'v_law':'poly'}
        ejecta_params_iso['bernoulli']  = {'mass_dist':'uniform', 'vel_dist':'uniform', 'op_dist':'uniform', 'therm_model':'BKWM', 'eps_ye_dep':True, 'v_law':'poly'}

        # set of shell parameters to be sampled on
        ejecta_vars_iso={}

        ejecta_vars_iso['dynamics'] = {'xi_disk'       :None,
                                      'm_ej'           :None,#params['Mej'][i],
                                      'step_angle_mass':None,
                                      'high_lat_flag'  :None,
                                      'central_vel'    :None,#params['vej'][i],
                                      'high_lat_vel'   :None,
                                      'low_lat_vel'    :None,
                                      'step_angle_vel' :None,
                                      'central_op'     :1.,
                                      'high_lat_op'    :None,
                                      'low_lat_op'     :None,
                                      'step_angle_op'  :None}

        ejecta_vars_iso['secular'] = {'xi_disk'       :0.4,
                                     'm_ej'           :None,
                                     'step_angle_mass':None,
                                     'high_lat_flag'  :None,
                                     'central_vel'    :0.04,
                                     'high_lat_vel'   :None,
                                     'low_lat_vel'    :None,
                                     'step_angle_vel' :None,
                                     'central_op'     :5.0,
                                     'low_lat_op'     :None,
                                     'high_lat_op'    :None,
                                     'step_angle_op'  :None}

        ejecta_vars_iso['wind'] = {'xi_disk'       :0.2,
                                  'm_ej'           :None,
                                  'step_angle_mass':None,
                                  'high_lat_flag'  :True,
                                  'central_vel'    :0.067,
                                  'high_lat_vel'   :None,
                                  'step_angle_vel' :math.radians(60.),
                                  'central_op'     :None,
                                  'high_lat_op'    :0.5,
                                  'low_lat_op'     :5.0,
                                  'step_angle_op'  :np.pi/4.}

        ejecta_vars_iso['bernoulli'] = {'xi_disk'       :None,
                                       'm_ej'           :0.00324,
                                       'step_angle_mass':None,
                                       'high_lat_flag'  :None,
                                       'central_vel'    :0.15073161990537148,
                                       'high_lat_vel'   :None,
                                       'low_lat_vel'    :None,
                                       'step_angle_vel' :None,
                                       'central_op'     :1.,
                                       'high_lat_op'    :None,
                                       'low_lat_op'     :None,
                                       'step_angle_op'  :None}


    #################################
    # Template for anisotropic case #
    #################################

        # hardcoded ejecta geometric and thermal parameters for the aspherical case
        ejecta_params_aniso = {}
        ejecta_params_aniso['dynamics']   = {'mass_dist':'sin2', 'vel_dist':'uniform', 'op_dist':'step',    'therm_model':'BKWM', 'eps_ye_dep':True, 'v_law':'poly'}
        ejecta_params_aniso['wind']       = {'mass_dist':'step', 'vel_dist':'uniform', 'op_dist':'step',    'therm_model':'BKWM', 'eps_ye_dep':True, 'v_law':'poly'}
        ejecta_params_aniso['secular']    = {'mass_dist':'sin2', 'vel_dist':'uniform', 'op_dist':'uniform', 'therm_model':'BKWM', 'eps_ye_dep':True, 'v_law':'poly'}
        ejecta_params_aniso['bernoulli']  = {'mass_dist':'sin2', 'vel_dist':'uniform', 'op_dist':'uniform', 'therm_model':'BKWM', 'eps_ye_dep':True, 'v_law':'poly'}

        # set of shell parameters to be sampled on
        ejecta_vars_aniso={}

        ejecta_vars_aniso['dynamics'] = {'xi_disk'       :None,
                                        'm_ej'           :params['Mej'][i],
                                        'step_angle_mass':None,
                                        'high_lat_flag'  :None,
                                        'central_vel'    :params['vej'][i],
                                        'min_vel'        :None,
                                        'max_vel'        :None,
                                        'step_angle_vel' :None,
                                        'central_op'     :None,
                                        'high_lat_op'    :1.0,
                                        'low_lat_op'     :30.,
                                        'step_angle_op'  :np.pi/6}

        ejecta_vars_aniso['secular'] = {'xi_disk'       :0.4,
                                       'm_ej'           :None,
                                       'step_angle_mass':None,
                                       'high_lat_flag'  :None,
                                       'central_vel'    :0.04,
                                       'min__vel'       :None,
                                       'max_vel'        :None,
                                       'step_angle_vel' :None,
                                       'central_op'     :5.0,
                                       'low_lat_op'     :None,
                                       'high_lat_op'    :None,
                                       'step_angle_op'  :None}

        ejecta_vars_aniso['wind'] = {'xi_disk'       :0.2,
                                    'm_ej'           :None,
                                    'step_angle_mass':np.pi/3,
                                    'high_lat_flag'  :True,
                                    'central_vel'    :0.067,
                                    'high_lat_vel'   :None,
                                    'low_lat_vel'    :None,
                                    'step_angle_vel' :None,
                                    'central_op'     :None,
                                    'high_lat_op'    :0.5,
                                    'low_lat_op'     :5.0,
                                    'step_angle_op'  :np.pi/4.}

        ejecta_vars_aniso['bernoulli'] = {'xi_disk'      :None,
                                        'm_ej'           :0.003024, #m_bern[i],#params['Mej_bern'][i],
                                        'step_angle_mass':None,
                                        'high_lat_flag'  :None,
                                        'central_vel'    :0.15073161990537148,
                                        'min_vel'        :1e-5,
                                        'max_vel'        :None,#params['vej_bern'][i],
                                        'step_angle_vel' :None,#math.radians(params['ThetaRMS_bern'][i]),
                                        'central_op'     :1.,#opacity_bern[i],
                                        'high_lat_op'    :30.,
                                        'low_lat_op'     :1.,
                                        'step_angle_op'  :0.8}


    ##########################################################
    # choose the appropriate set of parameters and variables #
    ##########################################################

        if (glob_params['mkn model'] == 'iso1comp'):
            ejecta_params = {}
            ejecta_vars = {}
            ejecta_params['dynamics'] = ejecta_params_iso['dynamics']
            ejecta_vars['dynamics']   = ejecta_vars_iso['dynamics']

        elif (glob_params['mkn model'] == 'iso2comp'):
            ejecta_params = {}
            ejecta_vars = {}
            ejecta_params['dynamics'] = ejecta_params_iso['dynamics']
            ejecta_vars['dynamics'] = ejecta_vars_iso['dynamics']
            ejecta_params['secular'] = ejecta_params_iso['secular']
            ejecta_vars['secular'] = ejecta_vars_iso['secular']

        elif (glob_params['mkn model'] == 'iso3comp'):
            ejecta_params = {}
            ejecta_vars = {}
            ejecta_params['dynamics'] = ejecta_params_iso['dynamics']
            ejecta_vars['dynamics'] = ejecta_vars_iso['dynamics']
            ejecta_params['secular'] = ejecta_params_iso['secular']
            ejecta_vars['secular'] = ejecta_vars_iso['secular']
            ejecta_params['wind'] = ejecta_params_iso['wind']
            ejecta_vars['wind'] = ejecta_vars_iso['wind']

        elif (glob_params['mkn model'] == 'iso4comp'):
            ejecta_params = ejecta_params_iso
            ejecta_vars = ejecta_vars_iso

        elif (glob_params['mkn model'] == 'aniso1comp'):
            ejecta_params = {}
            ejecta_vars = {}
            ejecta_params['dynamics'] = ejecta_params_aniso['dynamics']
            ejecta_vars['dynamics']    = ejecta_vars_aniso['dynamics']

        elif (glob_params['mkn model'] == 'aniso2comp'):
            ejecta_params = {}
            ejecta_vars = {}
            ejecta_params['dynamics'] = ejecta_params_aniso['dynamics']
            ejecta_vars['dynamics']    = ejecta_vars_aniso['dynamics']
            ejecta_params['secular'] = ejecta_params_aniso['secular']
            ejecta_vars['secular']    = ejecta_vars_aniso['secular']

        elif (glob_params['mkn model'] == 'aniso3comp'):
            ejecta_params = {}
            ejecta_vars = {}
            ejecta_params['dynamics'] = ejecta_params_aniso['dynamics']
            ejecta_vars['dynamics']    = ejecta_vars_aniso['dynamics']
            ejecta_params['secular'] = ejecta_params_aniso['secular']
            ejecta_vars['secular']    = ejecta_vars_aniso['secular']
            ejecta_params['wind'] = ejecta_params_aniso['wind']
            ejecta_vars['wind'] = ejecta_vars_aniso['wind']

        elif (glob_params['mkn model'] == 'aniso4comp'):
            ejecta_params = ejecta_params_aniso
            ejecta_vars = ejecta_vars_aniso


        #print('I am initializing the model')
        model = MKN(glob_params, glob_vars,ejecta_params, ejecta_vars, source_name)

        #print('I am computing the light curves')
    #    r_ph,L_bol,T_eff = model.lightcurve(ejecta_vars,glob_params['NR_data'],glob_params['NR_filename'])
        r_ph, L_bol, T_eff = model.E.lightcurve(model.angular_distribution,
                                              model.omega_distribution,
                                              model.time,
                                              model.ejecta_vars,
                                              model.ejecta_params,
                                              model.glob_vars,
                                              model.glob_params)

        #print('I am computing the likelihood')
        (chi2, err, ndata) =  model.log_likelihood(r_ph,T_eff)


        if (write_output):
            #print('I am printing out the output')
            model.write_output(r_ph,T_eff,L_bol, str(i))

        #p = 1. - integrate.quad( bayes.density_of_probability1, 0.1, chi2, args=(ndata) )[0]

        params['chi2'][i] = chi2
        print(params['name'][i], chi2)

    model.write_data()
    params = params.drop(columns=['Unnamed: 0'])
    params.to_csv(filename)
