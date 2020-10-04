import numpy as np
import zipfile
import pickle
import os
import affine

class Dem_data:

    # TODO set to float?
    def __init__(self, profile, z, xz=None, yz=None, xxz=None, yyz=None, xyz=None, num_aggre=0, num_aggre_updated=0):
        self.profile = profile

        self.num_aggre = num_aggre
        self._num_aggre_updated = num_aggre_updated

        self._z = z.astype(float)
        shape = z.shape

        for array in [xz, yz, xxz, yyz, xyz]:
            if (array is None):
                array = np.zeros(shape).astype(float)
            else:
                if (array.shape != shape):
                    raise ValueError('All arrays must have the same shape')
        
        self.set_arrays(z, xz, yz, xxz, yyz, xyz)

    def get_arrays(self):
        return self._z, self._xz, self._yz, self._xxz, self._yyz, self._xyz

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

    def update_img_tsfm(self, transform):
        num_trunc = 2**self.num_aggre - 2**self._num_aggre_updated
        img_offset = num_trunc / 2

        x = transform[2] + ((transform[0] + transform[1]) * img_offset)
        y = transform[5] + ((transform[3] + transform[4]) * img_offset)
        transform = affine.Affine(transform[0], transform[1], x, transform[3] , transform[4], y)

        self._num_aggre_updated += self.num_aggre

    def img_tsfm_is_updated(self):
        return (self.num_aggre == self._num_aggre_updated)

    @staticmethod
    def from_import(file_name):
        with zipfile.ZipFile(file_name) as zip:
            with zip.open('profile.pickle') as profile_pkl:
                profile = pickle.load(profile_pkl)

        with np.load(file_name) as npz:
            z = npz['z']
            xz = npz['xz']
            yz = npz['yz']
            xxz = npz['xxz']
            yyz = npz['yyz']
            xyz = npz['xyz']
            num_aggre = npz['num_aggre']
            num_aggre_updated = npz['num_aggre_updated']

        return Dem_data(profile, z, xz, yz, xxz, yyz, xyz, num_aggre, num_aggre_updated)
    
    def export(self, file_name):
        np.savez(
            file_name,
            z=self._z,
            xz=self._xz,
            yz=self._yz,
            xxz=self._xxz,
            yyz=self._yyz,
            xyz=self._xyz,
            num_aggre=self.num_aggre,
            num_aggre_updated = self._num_aggre_updated
        )

        with open('profile.pickle', 'wb') as profile_pkl:
            pickle.dump(self.profile, profile_pkl)

        with zipfile.ZipFile(file_name, 'a') as zip:
            zip.write('profile.pickle')

        os.remove('profile.pickle')

