import numpy as np
from scipy.sparse import diags, block_diag


class Assembler2D(object):
    """
    Assembles transmissibility and accumulation matrices
    and imposes boundary conditions
    """
    def __init__(self, size):
        super(Assembler2D, self).__init__()
        assert len(size) == 2, \
            "size should be a [int, int] list or tuple"
        assert int(size[0]) == size[0], \
            "entries of size should be integers"
        assert int(size[1]) == size[1], \
            "entries of size should be integers"
        self.nx = int(size[0])
        self.ny = int(size[1])
        self.__computeSummationMatrices()

    def __computeSummationMatrices(self):
        nx = self.nx; ny = self.ny
        # create summation matrix along x
        # with size nx*ny + ny, nx*ny
        # create block sparse matrix
        diag1 = np.ones(nx)
        diag1[nx-1] = 2
        diag2 = np.ones(nx)
        diag2[0] = 2
        block = diags([diag1, diag2], [-1, 0],
                      shape=(nx+1, nx))

        sumx_matrix = block_diag([block]*ny)
        self.sumx_matrix = sumx_matrix.tocsr()

        # create summation matrix along y
        # with size nx*ny + nx, nx*ny
        # it's simply 2-diagonal
        diag1 = np.ones(nx*ny)
        diag1[nx*ny-nx:] = 2
        diag2 = np.ones(nx*ny)
        diag2[:nx] = 2
        self.sumy_matrix = diags([diag1, diag2], [-nx, 0],
                                 shape=(nx*(ny+1), nx*ny),
                                 format='csr')

        # print self.sumx_matrix.toarray()
        # print self.sumy_matrix.toarray()
        # print self.sumx_matrix.shape
        # print self.sumy_matrix.shape

    def getFaceValuesX(self, vectorx_cells, method="harmonic"):
        '''
        computes values on cell faces from the values in
        cell centers as an average along x
        Input:
            vectorx_cells: np.array(nx*ny)
                array of values in cell centers
            method: string "harmonic" or"arithmetic"
                method of taking average
        Returns:
            vectorx_faces: np.array(ny, nx+1)
                values on cell faces (average along x)
        '''
        # check method
        assert(method in ["harmonic", "arithmetic"])
        # check input vector
        if not isinstance(vectorx_cells, np.ndarray):
            raise ValueError("wrong type of vectorx_cells")

        # check for dimensions
        vshape = vectorx_cells.shape
        assert len(vshape) == 1, \
            "vectorx_cells should be a 1D vector."
        assert self.nx*self.ny == vshape[0], \
            "Vector of invalid size %d. Shold be %d." % (vshape[0], self.nx*self.ny)

        if method == "arithmetic": order = 1
        else: order = -1

        if (order == -1):
            # ignore division-by-zero warnings (inactive cells have zero perm)
            with np.errstate(divide='ignore'):
                vectorx_inv = 1./vectorx_cells
            vectorx_faces = 1./(0.5*self.sumx_matrix*vectorx_inv)
        else:
            vectorx_faces = 0.5*self.sumx_matrix*vectorx_cells
        # print vectorx_faces/vectorx_cells[0]
        return vectorx_faces.reshape(self.ny, self.nx+1)

    def getFaceValuesY(self, vectory_cells, method="harmonic"):
        '''
        computes values on cell faces from the values in
        cell centers as an average along y
        Input:
            vectory_cells: np.array(nx*ny)
                array of values in cell centers
            method: string "harmonic" or"arithmetic"
                method of taking average
        Returns:
            vectory_faces: np.array(ny+1, nx)
                values on cell faces (average along y)
        '''
        if not isinstance(vectory_cells, np.ndarray):
            raise ValueError("Wrong type of vectory_cells")
        # check for dimensions
        vshape = vectory_cells.shape
        assert self.nx*self.ny == vshape[0], \
            "Wrong length of vectory_cells."
        assert len(vshape) == 1, \
            "vectory_cells must be a 1D array."

        assert(method in ["harmonic", "arithmetic"])

        if method == "arithmetic": order = 1
        else: order = -1

        if (order == -1):
            # ignore division-by-zero warnings (inactive cells have zero perm)
            with np.errstate(divide='ignore'):
                vectory_inv = 1./vectory_cells
            vectory_faces = 1./(0.5*self.sumy_matrix*vectory_inv)
        elif (order == 1):
            vectory_faces = 0.5*self.sumy_matrix*vectory_cells
        # print vectory_faces/vectory_cells[0]
        # print vectory_faces.reshape(self.ny+1, self.nx)/vectory_cells[0]
        return vectory_faces.reshape(self.ny+1, self.nx)

    def getTransmissibilityMatrix(self, tx_faces, ty_faces,
                                  constraints):
        '''
        returns transmissibility matrix for 2nd order accurate
        central differences
        Input:
            tx_faces: np.array(ny, nx+1)
                x-component of transmissibility values on cell faces
            ty_faces: np.array(ny+1, nx)
                y-component of transmissibility values on cell faces
                (technically, in bottommost cells and intercell faces)
            constraints: list [bool, bool, bool, bool]
                type of boundary conditions (False - neumann, True - dirichlet)
                in a conterclockwise manner (Left, Bottom, Right, Top)
            t_matrix: csr matrix(nx*ny, nx*ny), optional
        Returns:
            t_matrix: scipy.sparce csr matrix (nx*ny, nx*ny)
                system transmissibility matrix without boundary conditions
        '''
        nx = self.nx; ny = self. ny
        # check for arguments type
        if not isinstance(tx_faces, np.ndarray):
            raise ValueError("tx_faces must be a numpy array")
        if not isinstance(ty_faces, np.ndarray):
            raise ValueError("ty_faces must be a numpy array")
        if not isinstance(constraints, (list, tuple)):
            raise ValueError("constraints must be a list or tuple")

        # check for transmissibilities shapes
        assert(tx_faces.shape == (ny, nx+1))
        assert(ty_faces.shape == (ny+1, nx))
        assert(len(constraints) == 4)

        # check entries of constraint
        for i in constraints:
            assert i in [0, 1], \
                "constraint values must be either 0 or 1"

        # create diagonals of transmissibility matrix
        # -nx-th full diagonal
        d1 = -ty_faces[1:, :].flatten()
        # bc's on the bottom
        if constraints[1]: d1[:nx] *= 2
        else: d1[:nx] = 0
        # -1st diagonal
        d2 = -tx_faces[:, :nx].flatten()
        d2[0::nx] = 0
        # 1st diagonal
        d4 = -tx_faces[:, 1:].flatten()
        d4[nx-1::nx] = 0
        # print d4/tx_faces[0, 0]
        # nx-th diagonal
        d5 = -ty_faces[:ny, :].flatten()
        # impose bc's on the top
        # print d5/ty_faces[0, 0]
        if constraints[3]: d5[nx*(ny-1):nx*ny] *= 2
        else: d5[nx*(ny-1):nx*ny] = 0
        # print d5/ty_faces[0, 0]
        # 0th (main) diagonal
        d3 = -(d1 + d2 + d4 + d5)
        d3[::nx] += 2*tx_faces[:, 0]*constraints[0]
        d3[nx-1::nx] += 2*tx_faces[:, nx]*constraints[2]

        if (ny > 1):
            t_matrix = diags([d1[nx:], d2[1:], d3, d4[:-1], d5[:-nx]],
                             [-nx, -1, 0, 1, nx],
                             shape=(nx*ny, nx*ny), format='csr')

        else:   # 1d case (ny == 1)
            t_matrix = diags([d2[1:], d3, d4[:-1], ],
                             [-1, 0, 1],
                             shape=(nx*ny, nx*ny), format='csr')

        np.set_printoptions(edgeitems=1000,
                            linewidth=1000)
        # print t_matrix.todense()/tx_faces[0, 0]

        return t_matrix

    def getSourceVector(self, tx_faces, ty_faces,
                        constraints, bc_values,
                        wells, current_schedule):
        '''
        Assembles source vector
        Input:
            Input:
            tx_faces: np.array(ny, nx+1)
                x-component of transmissibility values on cell faces
            ty_faces: np.array(ny+1, nx)
                y-component of transmissibility values on cell faces
                (technically, in bottommost cells and intercell faces)
            constraints: list [bool, bool, bool, bool]
                type of boundary conditions (False - neumann, True - dirichlet)
                in a conterclockwise manner (Left, Bottom, Right, Top)
            bc_values: list [float, float, float, float]
                values imposed on boundaries
            wells: dictionary
                data on well types and locations
                'well_name': 'cells': np.array, 'rad': float, 'dir': int,
                             'prod;: np.array
            current schedule: dictionary with well_name keys and values
                lists of length 3 [int control, float value, float skin]
        '''
        nx = self.nx; ny = self.ny
        nx = self.nx; ny = self.ny
        # check for arguments types
        if not isinstance(tx_faces, np.ndarray):
            raise ValueError("tx_faces must be a numpy array")
        if not isinstance(ty_faces, np.ndarray):
            raise ValueError("ty_faces must be a numpy array")
        if not isinstance(constraints, (list, tuple, np.ndarray)):
            raise ValueError("constraints must be a list or tuple")
        if not isinstance(bc_values, (list, tuple, np.ndarray)):
            raise ValueError("bc_values must be a list or tuple")
        if not isinstance(wells, dict):
            raise ValueError("wells must be a dictionary")
        if not isinstance(current_schedule, dict):
            raise ValueError("current_schedule must be a dictionary")

        # check the shapes of arrays
        assert tx_faces.shape == (ny, nx+1), \
            "Invalid shape of tx_faces (should be (%d, %d))" % (ny, nx+1)
        assert ty_faces.shape == (ny+1, nx), \
            "Invalid shape of ty_faces (should be (%d, %d))" % (ny+1, nx)
        assert len(constraints) == 4, \
            "Invalid length of constraints (should be 4)"
        assert len(bc_values) == 4, \
            "Invalid length of bc_values (should be 4)"

        # check entries of constraint
        for i in constraints: assert i in [0, 1], \
            "constraint values must be either 0 or 1"

        # check entries of wells
        for w in wells.keys():
            assert isinstance(w, str), \
                "wells keys should be strings"
            well = wells[w]
            assert isinstance(well, dict), \
                "well[%s] must be a dictionary" % w
            assert 'prod' in well.keys(), \
                "Specify well productivity index well['prod']"
            assert 'cells' in well.keys(), \
                "Specify well location well['cells']"
            assert 'rad' in well.keys(), \
                "Specify well radius well['rad']"
            assert isinstance(well["cells"], (np.ndarray, list, tuple)), \
                "well['cells'] should be a numpy array or list/tuple"
            assert isinstance(well["prod"], (np.ndarray, list)), \
                "Well productivity well['prod'] should be a numpy array or list"
            assert isinstance(well["rad"], (float, int)), \
                "Well radius well['rad'] should be a number"
            assert len(well['prod']) == len(well['cells']), \
                "lengths of well['prod'] and well['cells'] should be the same"

        # check current_schedule
        for w in current_schedule.keys():
            assert w in wells.keys(), \
                "Well %s unknown (not in wells dict)" % w
            control = current_schedule[w]
            assert isinstance(control, (np.ndarray, list)), \
                "current_schedule values should be numpy arrays or lists"
            assert len(control) == 3, \
                "well controls well[%s] should be of a length 3" % w
            assert control[0] in [1, 2], \
                "well control can be only 1  or 2 (flow or pressure)"

        q_vector = np.zeros(nx*ny)

        # BOUNDARY CONDITIONS
        # bottom
        if constraints[1]:
            q_vector[:nx] = 2*ty_faces[0, :]*bc_values[1]
        else:
            q_vector[:nx] = bc_values[1]
        # top
        if constraints[3]:
            q_vector[-nx:] = 2*ty_faces[ny+1, :]*bc_values[3]
        else:
            q_vector[-nx:] = bc_values[3]
        # left
        if constraints[0]:
            q_vector[::nx] = 2*tx_faces[:, 0]*bc_values[0]
        else:
            q_vector[::nx] = bc_values[0]
        # right
        if constraints[2]:
            q_vector[nx-1::nx] = 2*tx_faces[:, nx]*bc_values[2]
        else:
            q_vector[::nx] = bc_values[0]

        # WELLS
        for w in current_schedule:
            control = current_schedule[w]
            well = wells[w]
            # print control
            # print well
            if (control[0] == 1):
                j_sum = np.array(well['prod']).sum()
                for i in xrange(len(well['cells'])):
                    j = well['prod'][i]
                    cell = well['cells'][i]
                    q_vector[cell] += control[1]*j/j_sum
            elif (control[0] == 2):
                for i in xrange(len(well['cells'])):
                    j = well['prod'][i]
                    pwf = control[1]
                    cells = well['cells'][i]
                    q_vector[cells] += pwf*j

        return q_vector

    def getProductivityMatrix(self, wells, current_schedule):
        nx = self.nx; ny = self.ny
        j_centers = np.zeros(nx*ny)
        for w in current_schedule:
            well = wells[w]
            control = current_schedule[w][0]
            if (control == 2):
                for i in xrange(len(well['cells'])):
                    l = well['cells'][i]
                    j_l = well['prod'][i]
                    j_centers[l] += j_l

        j_matrix = diags(j_centers, 0, (nx*ny, nx*ny), format='csr')
        return j_matrix

    def getJacobian(self, x0, res0, compute_res, args, dx=1e-2):
        '''
        Assembles a Jacobian matrix. Computes the Jacobian
        numerically.
        Input:
            x0: np.array(nx*ny)
                solution from the previous iteration
            res0: np.array(nx*ny)
                residual value corresponding to x0
            compute_res: function
                function that computes residual. Must be taking x0 as
                a first argument.
            args: tuple?
                additional arguments passed to the compute_res function
            dx: float
                x increment. the derivatives are taken as
                (compute_res(x0+dx) - res0)/dx
        Returns:
            jacobian: scipy.sparse matrix (csr?)
                the jacobian of the system
        '''
        
        print compute_res(x0, *args)


if __name__ == '__main__':
    size = [4, 5]
    assembler = Assembler2D([size[0], size[1]])
    k_faces = np.ones([size[0]*size[1]])
    tx_faces = assembler.getFaceValuesX(k_faces)
    ty_faces = assembler.getFaceValuesY(k_faces)
    assembler.getTransmissibilityMatrix(tx_faces, ty_faces,
                                        [0, 0, 1, 0])
    # assembler.getSourceVector(tx_faces, ty_faces,
    #     [0, 0, 1, 0], [15, 0, 300, 0],
    #     np.array([6]), [1], np.array([333]))
