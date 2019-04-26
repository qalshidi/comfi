"""Functions and stuff for ease of use on MHD results"""
import numpy as np
import scipy.constants
import h5py

def kenergy(velx, velz, velp=0, dens=1):
    """ Calculate kinetic energy """
    return 0.5*(np.multiply(velx, velx) + np.multiply(velz, velz) + np.multiply(velp, velp)) / dens

def benergy(magx, magz, magp=0):
    """Calculate magnetic energy density"""
    return 0.5*(np.multiply(magx, magx) + np.multiply(magz, magz) + np.multiply(magp, magp))

def nuin(dens, temp):
    """Calculate collision frequency"""
    return 7.4e-11 * dens * 1e20 * 1e-6 * 9.62771 * np.sqrt(temp)

def load(filename, xsize, zsize):
    """load result binary file into matrix with correct orientation"""
    mat = np.fromfile(filename)
    mat.resize(xsize, zsize)
    mat = np.flipud(mat.transpose())
    return mat

class Results:
    """Gets all the results in one go"""
    def __init__(self, params, filename='mhdsim.h5'):
        self.constants = {'n_0': params['n_0'],
                          'l_0': params['l_0'],
                          'B_0': params['B_0'],
                          'width': params['width'],
                          'height': params['height'],
                          'm_e': scipy.constants.m_e/scipy.constants.m_p,
                          'V_0': params['B_0']/np.sqrt(scipy.constants.mu_0*params['n_0']*scipy.constants.m_p),
                          'P_0': params['B_0']*params['B_0']/scipy.constants.mu_0}
        self.constants['t_0'] = self.constants['l_0']/self.constants['V_0']
        self.constants['T_0'] = self.constants['P_0']/(self.constants['n_0']*scipy.constants.k)
        self.constants['e_0'] = self.constants['B_0']/(scipy.constants.mu_0*params['n_0']*self.constants['V_0']*params['l_0'])
        self.constants['q'] = scipy.constants.e / self.constants['e_0']
        self.constants['j_0'] = self.constants['e_0']*params['n_0']*self.constants['V_0']
        self.file = h5py.File(filename, 'r')
    def get_var(self, varname, step):
        return np.array(self.file['/'+str(step)+'/'+varname])
    def get_time_vector(self):
        return np.array(self.file['t'][0, :])

    def nu_in(self, step):
        """ Returns the ion neutral collision frequency """
        sigma_in = 1.16e-18 # m-2
        m_in = scipy.constants.m_p*scipy.constants.m_n/(scipy.constants.m_p+scipy.constants.m_n)
        in_coeff = sigma_in * np.sqrt(8.0*scipy.constants.k/(scipy.constants.pi*m_in))
        return self.constants['n_0']*self.constants['t_0']*self.get_var('Nn', step)*in_coeff*np.sqrt(np.abs(0.5*(self.get_var('Tp', step)+self.get_var('Tn', step))*self.constants['T_0']))
    def v_balfven(self, step):
        return np.sqrt((self.get_var('Bx', step)**2+self.get_var('Bz', step)**2+self.get_var('Bp', step)**2)/(self.get_var('Np', step)+self.get_var('Nn', step)))
    def v_alfven(self, step):
        return np.sqrt((self.get_var('Bx', step)**2+self.get_var('Bz', step)**2+self.get_var('Bp', step)**2)/self.get_var('Np', step))
    def dv2(self, step):
        return (self.get_var('NVx', step)/self.get_var('Np', step)-self.get_var('NUx', step)/self.get_var('Nn', step))**2
    def frictional_heating(self, step):
        return self.nu_in(step)*self.get_var('Np', step)*self.dv2(step)/self.get_var('Nn', step)
    
        """

        self.data['Jx'], self.data['Jz'], self.data['Jp'] = self.current()
        self.data['Vx'], self.data['Vz'], self.data['Vp'] = self.velocity_p()
        self.data['Ux'], self.data['Uz'], self.data['Up'] = self.velocity_n()
        self.data['nuin'] = nu_in(self.data['Nn'], 0.5*(self.data['Tp']+self.data['Tn']))
        self.data['frictionalheating'] = self.frictionalheating()
        self.data['frictionalheatingpn'] = self.data['frictionalheating']/self.data['Nn']
        self.data['resistivity'] = resistivity(self.data['Np'], self.data['Nn'], self.data['Tp'], self.data['Tn'])
        self.data['J2'] = self.data['Jx']**2 + self.data['Jz']**2 + self.data['Jp']**2
        self.data['ohmicheating'] = self.data['resistivity']*self.data['J2']
        self.data['ohmicheatingpn'] = self.data['resistivity']*self.data['J2']/self.data['Nn']
        self.data['dV2'] = (self.data['Vx']-self.data['Ux'])**2 + (self.data['Vz']-self.data['Uz'])**2 + (self.data['Vp']-self.data['Up'])**2
        self.data['pB'] = 0.5*(self.data['Bx']**2+self.data['Bz']**2+self.data['Bp']**2)
        self.data['pTp'] = self.data['Np']*self.data['Tp']
        self.data['pTn'] = self.data['Nn']*self.data['Tn']
        self.data['pTtot'] = 0.5*(self.data['Nn']+self.data['Np'])*(self.data['Tn']+self.data['Tp'])
        self.data['beta'] = self.data['pTtot']/self.data['pB']
        self.data['betap'] = self.data['pTp']/self.data['pB']
        """

"""
    def velocity_p(self):
        return self.data['NVx']/self.data['Np'], self.data['NVz']/self.data['Np'], self.data['NVp']/self.data['Np']

    def velocity_n(self):
        return self.data['NUx']/self.data['Nn'], self.data['NUz']/self.data['Nn'], self.data['NUp']/self.data['Nn']

    def frictionalheating(self):
        vx, vz, vp = self.velocity_p()
        ux, uz, up = self.velocity_n()
        dv2 = (vx-ux)**2 + (vz-uz)**2 + (vp-up)**2
        np = self.data['Np']
        nn = self.data['Nn']
        tp = self.data['Tp']
        tn = self.data['Tn']
        return np*nu_in(nn, 0.5*(tp+tn))*dv2

    def frictionalheatingpn(self):
        vx, vz, vp = self.velocity_p()
        ux, uz, up = self.velocity_n()
        dv2 = (vx-ux)**2 + (vz-uz)**2 + (vp-up)**2
        np = self.data['Np']
        nn = self.data['Nn']
        tp = self.data['Tp']
        tn = self.data['Tn']
        return nu_in(nn, 0.5*(tp+tn))*dv2*np/nn
    
    def current(self):
        xpart = self.c.l_0*(jp1_neumann(self.data['Bp'])-jm1_neumann(self.data['Bp']))/self.c.dx
        zpart = self.c.l_0*(im1_neumann(self.data['Bp'])-ip1_neumann(self.data['Bp']))/self.c.dz
        ppart = self.c.l_0*(ip1_neumann(self.data['Bz'])-im1_neumann(self.data['Bz']))/self.c.dx
        ppart -= self.c.l_0*(jp1_neumann(self.data['Bx'])-jm1_dirichlet(self.data['Bx'], np.zeros(self.nx)))/self.c.dz
        return xpart, zpart, ppart
"""


def nu_ii(ni, temp):
    return 9.62771 * 0.9e-6 * 1.e20 * np.power(temp*5.764e6, -1.5) * ni 

def nu_en(nn, temp):
    return 1.95e-10 * 1.e14 * 9.62771 * nn * np.sqrt(temp*5.764e6)

def nu_ei(ne, te):
    return 10 * 1.e14 * 9.62771 * 3.759e-6 * ne * np.power(te*5.764e6, -1.5)

#def resistivity(np, nn, tp, tn):
    #c = Const()
    #return c.m_e/(c.q*c.q) * (nu_ei(np, tp)+nu_en(nn, 0.5*(tp+tn)))/np

def gyrofreq_e(mag):
    return 1.76e11 * mag * 0.1 * 9.62771 / (2*np.pi)

def gyrofreq_p(mag):
    return 9.58e7 * mag * 9.62771 * 0.1 / (2*np.pi)

def ip1_neumann(unknown):
    result = np.zeros_like(unknown)
    result[:, :-1] = unknown[:, 1:]
    result[:, -1] = unknown[:, -1]
    return result

def ip1_mirror(unknown):
    result = np.zeros_like(unknown)
    result[:, :-1] = unknown[:, 1:]
    result[:, -1] = -unknown[:, -1]
    return result

def im1_neumann(unknown):
    result = np.zeros_like(unknown)
    result[:, 1:] = unknown[:, :-1]
    result[:, 0] = unknown[:, 0]
    return result

def im1_mirror(unknown):
    result = np.zeros_like(unknown)
    result[:, 1:] = unknown[:, :-1]
    result[:, 0] = -unknown[:, 0]
    return result

def jp1_neumann(unknown):
    result = np.zeros_like(unknown)
    result[1:, :] = unknown[:-1, :]
    result[0, :] = unknown[0, :]
    return result

def jm1_neumann(unknown):
    result = np.zeros_like(unknown)
    result[:-1, :] = unknown[1:, :]
    result[-1, :] = unknown[-1, :]
    return result

def jm1_dirichlet(unknown, bc):
    result = np.zeros_like(unknown)
    result[:-1, :] = unknown[1:, :]
    result[-1, :] = bc
    return result

def nu_in(nn, T, t_0=1, n_0=1, T_0=1):
    sigma = 1.16e-18
    m_in = 1.007276466879*1.007825/(1.007276466879+1.007825)
    in_coeff = sigma * np.sqrt(8.0*scipy.constants.k/scipy.constants.pi/m_in)
    return in_coeff * nn * n_0 * np.sqrt(np.abs(T*T_0)) * t_0

