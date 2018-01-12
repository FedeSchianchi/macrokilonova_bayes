import numpy as np
import angular_distribution
import mass_angular_distribution
import thermalization
import velocity_angular_distribution
import opacity_angular_distribution
import initialize_components
import units
import initialize_components
import observer_projection as op
import nuclear_heat
from expansion_model_single_spherical import ExpansionModelSingleSpherical

def T_eff_calc(Lum,dOmega,r_ph):
    return (Lum/(dOmega*r_ph**2*units.sigma_SB))**(1./4.)

class Shell(object):
    """
    Ejecta shell class
    Required arguments are:
    name,          # name of the shell     (dynamical, wind, secular)
    mass_dist,     # mass distribution     (step, uniform, continuos)
    vel_dist,      # velocity distribution (step, uniform, continuos)
    op_dist,       # opacity distribution  (step, uniform, continuos)
    therm_model,   # thermal model         (BKWM, cnst)
    eps_ye_dep,    # nuclear heating model (Ye depedence True, False)
    """
    def __init__(self, name, params, **kwargs):
        assert len(params) >= 5
        self.name            = name
        self.ET              = thermalization.Thermalization(params['therm_model'])
        self.EPSNUC          = nuclear_heat.NuclearHeat(params['eps_ye_dep'])
        self.mass_dist       = mass_angular_distribution.MassAngularDistribution(params['mass_dist'])
        self.vel_dist        = velocity_angular_distribution.VelocityAngularDistribution(params['vel_dist'])
        self.op_dist         = opacity_angular_distribution.OpacityAngularDistribution(params['op_dist'])
        self.expansion_model = ExpansionModelSingleSpherical('GK')
    
    def update(self, m_tot, angular_distribution, **kwargs):
        x = self.mass_dist(m_tot,angular_distribution,**kwargs)
        y = self.vel_dist(angular_distribution,**kwargs)
        z = self.op_dist(angular_distribution,**kwargs)
        return x, y, z

    def expansion_angular_distribution(self,
                                       angular_distribution,
                                       omega_distribution,
                                       m_tot,
                                       time,
                                       glob_params,
                                       glob_model_params,
                                       **kwargs):
                                       
# assign the global model variables
        v_min    = glob_model_params['v_min']
        n_v      = glob_model_params['n_v']
        vscale   = glob_model_params['vscale']
        sigma0   = glob_model_params['sigma0']
        alpha    = glob_model_params['alpha']
        t0eps    = glob_model_params['t0eps']
        cnst_eff = glob_model_params['cnst_eff']

# assign the global variables
        eps0      = glob_params['eps0']
        a_eps_nuc = glob_params['a_eps_nuc']
        b_eps_nuc = glob_params['b_eps_nuc']
        t_eps_nuc = glob_params['t_eps_nuc']

        self.ejected_mass,self.velocity_rms,self.opacity = self.update(m_tot,angular_distribution,**kwargs)
        self.physical_radius = []
        self.Lbol = []
        
        for omega,m_ej,v_rms,kappa in zip(omega_distribution,self.ejected_mass,self.velocity_rms,self.opacity):

            vel,m_vel,t_diff,t_fs = self.expansion_model(omega,m_ej,v_rms,v_min,n_v,vscale,kappa)
            v_diff  = np.interp(time, t_diff[::-1], vel[::-1])
            v_fs    = np.interp(time, t_fs[::-1], vel[::-1])
            mv_diff = np.interp(time, t_diff[::-1], m_vel[::-1])
            mv_fs   = np.interp(time, t_fs[::-1], m_vel[::-1])
            m_rad = mv_diff-mv_fs

            self.physical_radius.append(self.r_ph_calc(v_fs, time))

            self.Lbol.append([self.bolometric_luminosity(m,
                                                    t,
                                                    t0eps,
                                                    sigma0,
                                                    eps0,
                                                    a_eps_nuc,
                                                    b_eps_nuc,
                                                    t_eps_nuc,
                                                    m_ej,
                                                    omega,
                                                    v_rms,
                                                    kappa,
                                                    cnst_eff,
                                                    alpha) for t,m in zip(time,m_rad)])
        self.physical_radius = np.array(self.physical_radius)
        self.Lbol = np.array(self.Lbol)
        tmp = []

        for k in range(len(angular_distribution)):
            tmp.append(np.array([T_eff_calc(L,omega_distribution[k],R) for L,R in zip(self.Lbol[k,:],self.physical_radius[k,:])]))
            self.Teff = np.asarray(tmp)
        return self.physical_radius, self.Lbol, self.Teff

    def bolometric_luminosity(self, m_rad, time,
                              t0eps,sigma0,eps0,
                              a_eps_nuc,b_eps_nuc,t_eps_nuc,
                              m_ej,
                              omega,v_rms,kappa,
                              cnst_eff,alpha):
        
        eps = self.EPSNUC(alpha,time,t0eps,sigma0,
                          eps0,self.ET,m_ej,omega,v_rms,
                          cnst_eff,
                          cnst_a_eps_nuc=a_eps_nuc,
                          cnst_b_eps_nuc=b_eps_nuc,
                          cnst_t_eps_nuc=t_eps_nuc,
                          opacity=kappa)
        return m_rad*eps

    def r_ph_calc(self, vel, t):
        return (vel*units.c)*t

#####################

if __name__=="__main__":
    params = {'mass_dist':'uniform', 'vel_dist':'step', 'op_dist':'step', 'therm_model':'BKWM', 'eps_ye_dep':True}
    S = Shell('wind', params)
    angular_distribution = [(0,1),(1,2),(2,3.1415)]
    omega_distribution = [0.01,0.2,0.5]
    time_min = 36000.    #
    time_max = 172800.   #
    n_time = 200
    m_tot = 0.1
    v_min = 1e-7
    n_v = 100
    vscale = 'linear'
    eps0 = 1e19
    sigma0 = 1.0
    alpha = 0.1
    t0eps = 1.0
    cnst_eff = 1.0
    a_eps_nuc = 1.0
    b_eps_nuc = 1.0
    t_eps_nuc = 1.0
    time = np.linspace(time_min,time_max,n_time)
    t, r_ph, L_bol, Teff = S.expansion_angular_distribution(angular_distribution,
                                                   omega_distribution,
                                                   m_tot,
                                                   time,
                                                   v_min,
                                                   n_v,
                                                   vscale,
                                                   eps0,
                                                   sigma0,
                                                   alpha,
                                                   t0eps,
                                                   cnst_eff,
                                                   a_eps_nuc,
                                                   b_eps_nuc,
                                                   t_eps_nuc,
                                                   low_lat_vel=0.2,
                                                   high_lat_vel=0.001,
                                                   step_angle_vel=1.0,
                                                   low_lat_op=0.2,
                                                   high_lat_op=0.001,
                                                   step_angle_op=1.0)
    import matplotlib.pyplot as plt
    print np.shape(r_ph), np.shape(L_bol)
    for k in range(r_ph.shape[0]): plt.plot(r_ph[k, :], L_bol[k, :],'.')
    plt.show()
