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
import filters as ft
import observer_projection as op

from expansion_model_single_spherical import ExpansionModelSingleSpherical
from shell import Shell

def T_eff_calc(Lum,dOmega,r_ph):
    return (Lum/(dOmega*r_ph**2*units.sigma_SB))**(1./4.)

class Ejecta(object):

    def __init__(self, n_shells, names, params, *args, **kwargs):
        assert len(names) == n_shells
        self.ncomponents = n_shells
        self.components = [Shell(n, params[n], **kwargs) for n in names]
    
    def lightcurve(self,
                   angular_distribution,
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
                   **kwargs):
        
        photospheric_radius = []
        self.bolometric_luminosity = []
        for c in self.components:
            t, r, Lb, Tc = c.expansion_angular_distribution(angular_distribution,
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
                                                     **kwargs)
            photospheric_radius.append(r)
            self.bolometric_luminosity.append(Lb)
            self.time = time
        self.photospheric_radius = photospheric_radius[0]
        for k in np.arange(1,len(photospheric_radius)): self.photospheric_radius = np.maximum(self.photospheric_radius,r[k])

        self.total_bolometric_luminosity = None
        for b in self.bolometric_luminosity:
            if self.total_bolometric_luminosity is None: self.total_bolometric_luminosity = b
            else: self.total_bolometric_luminosity += b

        tmp = []
        for k in range(len(angular_distribution)):
            tmp.append(np.array([T_eff_calc(L,omega_distribution[k],R) for L,R in zip(self.total_bolometric_luminosity[k,:],self.photospheric_radius[k,:])]))
            self.T_eff_tot = np.asarray(tmp)
        return self.time, np.array(self.photospheric_radius), np.array(self.bolometric_luminosity), self.T_eff_tot

if __name__=="__main__":
    params = {}
    params['wind'] = {'mass_dist':'uniform', 'vel_dist':'step', 'op_dist':'step', 'therm_model':'BKWM', 'eps_ye_dep':True}
    params['secular'] = {'mass_dist':'uniform', 'vel_dist':'step', 'op_dist':'step', 'therm_model':'BKWM', 'eps_ye_dep':True}
    E = Ejecta(2, params.keys(), params)
    angular_distribution = [(0,1),(1,2),(2,3.1415)]
    omega_distribution = [0.01,0.2,0.5]
    time_min = 36000.      #
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
    time, r_ph, L_bol, Teff = E.lightcurve(angular_distribution,
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
    for j in range(E.ncomponents):
        for k in range(L_bol.shape[1]): plt.plot(time, L_bol[j, 0, :],'.')
    plt.show()
