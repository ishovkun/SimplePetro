import unittest
from import_path import import_path
import numpy as np
import scipy as sp
from copy import deepcopy
import_path("../lib2d/WellHandler.py")
from WellHandler import WellHandler

class WellHandlerTest(unittest.TestCase):
    """
    Tests WellHandler class
    """
    def setUp(self):
        pass

    def test__init__(self):
        nx = 80
        ny = 75
        self.assertRaises(ValueError,
                          WellHandler,
                          'nx', ny)
        self.assertRaises(ValueError,
                          WellHandler,
                          nx, 16.5)
        # just run it
        WellHandler(nx, ny)

    def test_locatewells(self):
        xSize = 6000.
        ySize = 7500.
        nx = 80
        ny = 75
        dx = xSize/nx
        dy = ySize/ny
        # dz = np.ones(nx*ny)
        x_centers = np.linspace(dx/2, xSize-dx/2, nx)
        y_centers = np.linspace(dy/2, ySize-dy/2, ny)
        wells = {
            '1': {'heel': [4184., 1569.], 'rad': 0.25, 'dir': 3},
            '2': {'heel': [4968.5, 2510.4], 'rad': 0.25, 'dir': 3},
            '3': {'heel': [3294.9, 2928.8], 'rad': 0.25, 'dir': 3},
            '4': {'heel': [2562.7, 4393.2], 'rad': 0.25, 'dir': 3},
            '5': {'heel': [1307.5, 2824.2], 'rad': 0.25, 'dir': 3},
            '6': {'heel': [890., 895.], 'len': 225, 'rad': 0.25, 'dir': 1},
        }

        wh = WellHandler(nx, ny)
        # check for wrong input handling
        self.assertRaises(ValueError,
                          wh.locateWells,
                          1, y_centers,
                          wells)
        self.assertRaises(ValueError,
                          wh.locateWells,
                          x_centers, {13:5},
                          wells)
        self.assertRaises(AssertionError,
                          wh.locateWells,
                          x_centers, np.ones(nx),
                          wells)
        self.assertRaises(ValueError,
                          wh.locateWells,
                          x_centers, y_centers,
                          [wells])
        w = deepcopy(wells)
        w['1']['heel'] = {}
        self.assertRaises(AssertionError,
                          wh.locateWells,
                          x_centers, y_centers,
                          w)

        wh.locateWells(x_centers, y_centers, wells)
        assert wells['1']['cells'][0] + 1 == 1256
        assert wells['2']['cells'][0] + 1 == 2067
        assert wells['3']['cells'][0] + 1 == 2364
        assert wells['4']['cells'][0] + 1 == 3475
        assert wells['5']['cells'][0] + 1 == 2258
        assert wells['6']['cells'][0] + 1 == 652
        # test for wells not in the domain
        w = deepcopy(wells)
        w['1']['heel'] = [720000., 1.]
        self.assertRaises(AssertionError,
                          wh.locateWells,
                          x_centers, y_centers,
                          w)
        # test for too long hor wells ending beyond the domain
        w = deepcopy(wells)
        w['6']['len'] = 1e16
        self.assertRaises(AssertionError,
                          wh.locateWells,
                          x_centers, y_centers, w)

    def testProductivities(self):
        xSize = 3.
        ySize = 3.
        nx = 3
        ny = 3
        dx = xSize/nx
        dy = ySize/ny
        dz = np.ones(nx*ny)
        x_centers = np.linspace(dx/2, xSize-dx/2, nx)
        y_centers = np.linspace(dy/2, ySize-dy/2, ny)
        permx = np.ones(nx*ny)
        permy = np.ones(nx*ny)
        permz = np.ones(nx*ny)
        fvolf = np.ones(nx*ny)
        visc = np.ones(nx*ny)
        wells = {
            '1': {'heel': [0.5, 0.5], 'rad': 0.5, 'dir': 3},
            '2': {'heel': [1.5, 1.5], 'rad': 0.5, 'dir': 3},
        }

        current_schedule = {'1': [1, -1000.0, 0.0],
                            '2': [2, 1500.0, -0.75]}

        wh = WellHandler(nx, ny)
        wh.locateWells(x_centers, y_centers, wells)
        # wrong itput
        self.assertRaises(ValueError,
                          wh.getProductivities,
                          1, 2, 3,
                          permx, permy, permz,
                          fvolf, visc,
                          wells, current_schedule)
        self.assertRaises(AssertionError,
                          wh.getProductivities,
                          1., 2., np.ones(3),
                          permx, permy, permz,
                          fvolf, visc,
                          wells, current_schedule)
        # test for impermeable cells
        permx[:] = 0
        wh.getProductivities(dx, dy, dz,
                             permx, permy, permz,
                             fvolf, visc, 
                             wells, current_schedule)
        dx = 100.; dy = 100.; dz = np.ones(nx*ny)*100
        permx[:] = 100.; permy = permx; permz = permx
        wh.getProductivities(dx, dy, dz,
                             permx, permy, permz,
                             fvolf, visc, 
                             wells, current_schedule)
        pi_computed = wells['2']['prod']
        # from balhoff's examples
        pi_desired = 135.3
        # should be approximately equal (balhoff's units 
        # conversions are innacurate)
        np.testing.assert_approx_equal(pi_computed, pi_desired,
                                       significant=3)

    def testWelldata(self):
        pass


if __name__ == '__main__':
    unittest.main()
    # whSuite = unittest.TestLoader().loadTestsFromTestCase(WellHandlerTest)
    # testRunner = unittest.TextTestRunner()
    # testRunner.run(whSuite)


