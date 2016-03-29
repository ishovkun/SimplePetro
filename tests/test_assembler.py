import unittest 
from import_path import import_path
import numpy as np
import scipy as sp
from copy import deepcopy
import_path("../Assembler.py")
from Assembler import Assembler2D


class AssembleTest(unittest.TestCase):
    """
    Tests Assembler class
    """
    def setUp(self):
        # define an instance of Assembler
        # self.a = Assembler2D()
        pass

    def testSummationMatrices1x5(self):
        a = Assembler2D([1, 5])
        sx = a.sumx_matrix
        sy = a.sumy_matrix
        assert isinstance(sx, sp.sparse.csr.csr_matrix), \
            "summation matrix is not sparse csr"
        assert isinstance(sy, sp.sparse.csr.csr_matrix), \
            "summation matrix is not sparse csr"
        # print sx.toarray()
        sx_right = np.array([
             [2,  0,  0,  0,  0],
             [2,  0,  0,  0,  0],
             [0,  2,  0,  0,  0],
             [0,  2,  0,  0,  0],
             [0,  0,  2,  0,  0],
             [0,  0,  2,  0,  0],
             [0,  0,  0,  2,  0],
             [0,  0,  0,  2,  0],
             [0,  0,  0,  0,  2],
             [0,  0,  0,  0,  2],
          ])
        assert np.array_equal(sx.toarray(), sx_right), \
            "summation matrix along x is wrong"

    def test__init__(self):
        '''
        Test the __init__ method of Assembler
        '''
        try: Assembler2D((1, 5, 4))
        except AssertionError: pass

        try: Assembler2D((1, 5.5))
        except AssertionError: pass

    def testFaceValuesX(self):
        '''
        test method that computes interblock transmissibilities in x direction
        '''
        a = Assembler2D((3, 3))
        vx = np.ones([9, 3])
        
        # test for invalid method
        self.assertRaises(AssertionError,
                          a.getFaceValuesX,
                          vx, method='babashka')
        
        # test for wrong type of first argument
        nvx = [1, 2, 3]
        self.assertRaises(ValueError,
                          a.getFaceValuesX, nvx)

        # test for vector dimensions
        self.assertRaises(AssertionError,
                          a.getFaceValuesX, vx)

        # test for vector length
        vx = np.ones(3*4)
        self.assertRaises(AssertionError,
                          a.getFaceValuesX, vx)

        # test output
        vx = np.ones(3*3); vx[0] = 0
        vxf_correct = np.array([
             [0,  0,  1,  1],
             [1,  1,  1,  1],
             [1,  1,  1,  1],
            ])
        vxf = a.getFaceValuesX(vx)
        assert np.array_equal(vxf, vxf_correct)

    def testFaceValuesY(self):
        '''
        test method that computes interblock transmissibilities in y direction
        '''
        a = Assembler2D((3, 3))
        vy = np.ones([9, 3])
        
        # test for invalid method
        self.assertRaises(AssertionError,
                          a.getFaceValuesY,
                          vy, "donkey")

        # test for vector type
        nvy = (15, 16, 17)
        self.assertRaises(ValueError,
                          a.getFaceValuesY, nvy)

        # test for vector dimensions
        self.assertRaises(AssertionError,
                          a.getFaceValuesY, vy)

        # test for vector length
        vy = np.ones(3*4)
        self.assertRaises(AssertionError,
                          a.getFaceValuesY, vy)

        # test output
        vy = np.ones(3*3); vy[0] = 0
        vyf = a.getFaceValuesY(vy)
        vyc = np.array([
             [0,  1,  1],
             [0,  1,  1],
             [1,  1,  1],
             [1,  1,  1],
            ])
        assert np.array_equal(vyf, vyc)

    def testT_Matrix(self):
        """
        Test function that assembles transmissibility matrix
        """
        nx = 3; ny = 3
        a = Assembler2D([nx, ny])
        # transmissibilities in cell centers
        tx = np.ones(nx*ny)
        ty = np.ones(ny*ny)
        # transmissibilities in cell faces
        txf = a.getFaceValuesX(tx)
        tyf = a.getFaceValuesY(ty)
        constraints = (0, 0, 1, 0)
        # check for invalid input type
        self.assertRaises(ValueError,
                          a.getTransmissibilityMatrix,
                          [1, 2, 3], tyf, constraints)
        self.assertRaises(ValueError,
                          a.getTransmissibilityMatrix,
                          txf, [1, 2, 3], constraints)
        self.assertRaises(ValueError,
                          a.getTransmissibilityMatrix,
                          txf, tyf, 1)
        # check for invalid input shapes
        self.assertRaises(AssertionError,
                          a.getTransmissibilityMatrix,
                          np.ones(3*4), tyf, constraints)
        self.assertRaises(AssertionError,
                          a.getTransmissibilityMatrix,
                          txf, np.ones([3, 4]), constraints)
        self.assertRaises(AssertionError,
                          a.getTransmissibilityMatrix,
                          txf, tyf, [1, 2, 3])

        # 3x3 example from balhoff's presentation
        t_matrix = a.getTransmissibilityMatrix(txf, tyf, constraints).toarray()
        right_matrix = np.array([
             [ 2, -1,  0, -1,  0,  0,  0,  0,  0],
             [-1,  3, -1,  0, -1,  0,  0,  0,  0],
             [ 0, -1,  4,  0,  0, -1,  0,  0,  0],
             [-1,  0,  0,  3, -1,  0, -1,  0,  0],
             [ 0, -1,  0, -1,  4, -1,  0, -1,  0],
             [ 0,  0, -1,  0, -1,  5,  0,  0, -1],
             [ 0,  0,  0, -1,  0,  0,  2, -1,  0],
             [ 0,  0,  0,  0, -1,  0, -1,  3, -1],
             [ 0,  0,  0,  0,  0, -1,  0, -1,  4],
            ])
        assert np.array_equal(t_matrix, right_matrix)

    def testSourceVector(self):
        nx = 3; ny = 3;
        a = Assembler2D([nx, ny])
        # transmissibilities in cell centers
        tx = np.ones(nx*ny)
        ty = np.ones(ny*ny)
        # transmissibilities in cell faces
        txf = a.getFaceValuesX(tx)
        tyf = a.getFaceValuesY(ty)
        constraints = (0, 0, 1, 0)
        bc_values = [0, 0, 100., 0]
        # correct well dict
        wells = {
            '1': {'cells': [4], 'rad': 0.5, 'dir': 3, 
                  'prod': np.array([0.8322916])},
            '2': {'cells': [8], 'rad': 0.5, 'dir': 3,
                  'prod': np.array([0.8322916])},
        }

        #   time wellname control value skin
        current_schedule = {'1': [1, -1000.0, 0.0],
                            '2': [2, 1500.0, -0.75]}

        # check input argument types
        self.assertRaises(ValueError,
                          a.getSourceVector,
                          [1, 2, 3], tyf,
                          constraints, bc_values,
                          wells, current_schedule)
        self.assertRaises(ValueError,
                          a.getSourceVector,
                          txf, {},
                          constraints, bc_values,
                          wells, current_schedule)
        self.assertRaises(ValueError,
                          a.getSourceVector,
                          txf, tyf,
                          1., bc_values,
                          wells, current_schedule)
        self.assertRaises(ValueError,
                          a.getSourceVector,
                          txf, tyf,
                          constraints, 'abc',
                          wells, current_schedule)
        self.assertRaises(ValueError,
                          a.getSourceVector,
                          txf, tyf,
                          constraints, bc_values,
                          [], current_schedule)
        self.assertRaises(ValueError,
                          a.getSourceVector,
                          txf, tyf,
                          constraints, bc_values,
                          wells, 'obama')
        self.assertRaises(AssertionError,
                          a.getSourceVector,
                          np.ones([3, 5]), tyf,
                          constraints, bc_values,
                          wells, current_schedule)
        self.assertRaises(AssertionError,
                          a.getSourceVector,
                          txf, np.ones([3, 5]),
                          constraints, bc_values,
                          wells, current_schedule)
        self.assertRaises(AssertionError,
                          a.getSourceVector,
                          txf, tyf,
                          [1, 2, 3], bc_values,
                          wells, current_schedule)
        self.assertRaises(AssertionError,
                          a.getSourceVector,
                          txf, tyf,
                          constraints, [1, 2, 3],
                          wells, current_schedule)
        self.assertRaises(AssertionError,
                          a.getSourceVector,
                          txf, tyf,
                          [1, 0, 1, 3], bc_values,
                          wells, current_schedule)
        self.assertRaises(AssertionError,
                          a.getSourceVector,
                          txf, tyf,
                          constraints, bc_values,
                          {1: 2}, current_schedule)
        self.assertRaises(AssertionError,
                          a.getSourceVector,
                          txf, tyf,
                          constraints, bc_values,
                          {'1': 2}, current_schedule)

        # changable (to test for wrong input)
        wells1 = deepcopy(wells)
        del wells1['1']['rad']
        self.assertRaises(AssertionError,
                          a.getSourceVector,
                          txf, tyf,
                          constraints, bc_values,
                          wells1, current_schedule)
        wells1 = deepcopy(wells)
        wells1['1']['prod'] = [1, 2, 3]
        self.assertRaises(AssertionError,
                          a.getSourceVector,
                          txf, tyf,
                          constraints, bc_values,
                          wells1, current_schedule)
        wells1 = deepcopy(wells)
        wells1['1']['rad'] = 'a'
        self.assertRaises(AssertionError,
                          a.getSourceVector,
                          txf, tyf,
                          constraints, bc_values,
                          wells1, current_schedule)
        sch = deepcopy(current_schedule)
        sch['1'] = [1, 2]
        self.assertRaises(AssertionError,
                          a.getSourceVector,
                          txf, tyf,
                          constraints, bc_values,
                          wells1, sch)
        sch = deepcopy(current_schedule)
        sch['99'] = [1.2, 2]
        self.assertRaises(AssertionError,
                          a.getSourceVector,
                          txf, tyf,
                          constraints, bc_values,
                          wells1, sch)

        q = a.getSourceVector(txf, tyf, constraints, bc_values,
                              wells, current_schedule)
        q_correct = np.zeros(nx*ny)
        q_correct[[2, 5, 8]] = 2*tx[0]*bc_values[2]
        q_correct[4] = current_schedule['1'][1]
        q_correct[8] += wells['2']['prod']*current_schedule['2'][1]
        assert np.array_equal(q, q_correct)

    def tearDown(self):
        pass  


if __name__ == '__main__':
    unittest.main()
