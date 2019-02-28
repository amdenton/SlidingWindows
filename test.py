import unittest
import hashlib
import WindowFractalCode as wfrac

class TestWindowFractalCode(unittest.TestCase):

    def test_image_output(self):
        wfrac.main()
        expected = 'e0a230836666691a6be3cb6cad8785b10af41ba8073e17bd4d10abfb662881f7'
        result = self.hash_image('result.tif')
        #print(f'expected: {expected}')
        #print(f'  result: {result}')
        self.assertEqual(expected, result)


    def hash_image(self, filename):
        """Image def borrowed from
        https://www.pythoncentral.io/hashing-files-with-python/"""
        BLOCKSIZE=65536
        hasher = hashlib.sha256()
        with open(filename, 'rb') as afile:
            buf = afile.read(BLOCKSIZE)
            while len(buf) > 0:
                hasher.update(buf)
                buf = afile.read(BLOCKSIZE)
        
        return(hasher.hexdigest())

if __name__ == '__main__':
    unittest.main()
