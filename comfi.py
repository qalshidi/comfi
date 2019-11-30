"""Functions and stuff for ease of use on MHD results"""
import numpy as np
import h5py
#import scipy.constants

def nuin(dens, temp):
    """Calculate collision frequency"""
    return 7.4e-11 * dens * 1e20 * 1e-6 * 9.62771 * np.sqrt(temp)

class Results:
    """Class to get and plot results from a CoMFi simulation

    Usage:
        Results()
    """
    def __init__(self, params, filename='mhdsim.h5'):
        self.h5file = h5py.File(filename, 'r')
        self.params = params

    def var(self, varindex, time_step, dataset="unknowns"):
        """ Returns a matrix of the unknown.

        Paramters:
            varindex: Index of the unknown.
            time_step: Time step.

        Returns: np.array of size (nx, nz)
        """
        unknowns = np.array(self.h5file['/'+str(time_step)+'/'+dataset])
        var = unknowns[:, varindex]
        return var.reshape(self.params["nx"], self.params["nz"])

    def unknowns(self, time_step):
        """ Returns a dictionary of the unknowns """
        unknowns = {"Bx": self.var(0, time_step),
                    "Bz": self.var(1, time_step),
                    "Bperp": self.var(2, time_step),
                    "n_n": self.var(3, time_step),
                    "Ux": self.var(4, time_step),
                    "Uz": self.var(5, time_step),
                    "Uperp": self.var(6, time_step),
                    "n_p": self.var(7, time_step),
                    "Vx": self.var(8, time_step),
                    "Vz": self.var(9, time_step),
                    "Vperp": self.var(10, time_step),
                    "E_n": self.var(11, time_step),
                    "E_p": self.var(12, time_step),
                    "GLM": self.var(13, time_step)
                    }
        return unknowns

    def raw_matrix(self, time_step, dataset="unknowns"):
        """Returns raw matrix from hdf5 file"""
        return np.array(self.h5file['/'+str(time_step)+'/'+dataset])


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

