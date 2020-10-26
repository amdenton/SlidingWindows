import zipfile
import pickle
import os

import numpy as np
import affine

class Dem_data:

    # TODO set to float?
    def __init__(self, z, num_aggre=0, xz=None, yz=None, xxz=None, yyz=None, xyz=None):
        self.num_aggre = num_aggre
        self._z = z.astype(float)
        
        shape = z.shape
        for array in [xz, yz, xxz, yyz, xyz]:
            if (array is None):
                array = np.zeros(shape).astype(float)
            else:
                array = array.astype(float)
                if (array.shape != shape):
                    raise ValueError('All arrays must have the same shape')
        
        self.set_arrays(z, xz, yz, xxz, yyz, xyz)

    def arrays(self):
        return self._z, self._xz, self._yz, self._xxz, self._yyz, self._xyz

    def z(self):
        return self._z
    
    def xz(self):
        return self._xz
    
    def yz(self):
        return self._yz
                    
    def xxz(self):
        return self._xxz
    
    def yyz(self):
        return self._yyz

    def xyz(self):
        return self._xyz

    def set_arrays(self, z, xz, yz, xxz, yyz, xyz):
        shape = z.shape
        dtype = z.dtype

        for array in [xz, yz, xxz, yyz, xyz]:
            if (array.shape != shape):
                raise ValueError('All arrays must have the same shape')
            if (array.dtype != dtype):
                raise ValueError('All arrays must have the same data type')

        self._z = z
        self._xz = xz
        self._yz = yz
        self._xxz = xxz
        self._yyz = yyz
        self._xyz = xyz

    @staticmethod
    def from_import(file_name):
        with np.load(file_name) as npz:
            z = npz['z']
            xz = npz['xz']
            yz = npz['yz']
            xxz = npz['xxz']
            yyz = npz['yyz']
            xyz = npz['xyz']
            num_aggre = npz['num_aggre']

        return Dem_data(z, num_aggre, xz, yz, xxz, yyz, xyz)
    
    def export(self, file_name):
        np.savez(
            file_name,
            z=self._z,
            xz=self._xz,
            yz=self._yz,
            xxz=self._xxz,
            yyz=self._yyz,
            xyz=self._xyz,
            num_aggre=self.num_aggre
        )
