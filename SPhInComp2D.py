import numpy as np
from scipy.sparse import diags
import scipy.sparse.linalg as sparse_solvers
import scipy
from Units import Units
from lib2d.WellHandler import WellHandler
from lib2d.Assembler import Assembler2D
from lib2d.Simulator2D import Simulator2D


class SPhInComp2D(Simulator2D):
    """
    Solves incompressible single phase fluid flow
    in a 2d rectangular domain with gravity
    """
    def __init__(self, **input_data):
        super(SPhInComp2D, self).__init__(**input_data)
        self._readInput()

        self.units = Units("Oilfield")
        self.wHandler = WellHandler(self.nx, self.ny)
        self.assembler = Assembler2D([self.nx, self.ny])
        self.preconditioner = None

    def _readInput(self):
        super(SPhInComp2D, self)._readInput()
        super(SPhInComp2D, self)._getActiveCells()

    def _readSchedule(self, current_time, dt):
        return super(SPhInComp2D, self)._readSchedule(current_time, dt)

    def _saveFieldData(self, data, time):
        super(SPhInComp2D, self)._saveFieldData(data, time)

    def _computePreconditioner(self, system_matrix):
        '''
        Create a preconditioner base on
        an incomplete LU decomposition
        '''
        # convert to compressed column storage format (efficiency)
        sm_csc = system_matrix.tocsc()
        lu = sparse_solvers.spilu(sm_csc)
        m_operator = lambda x: lu.solve(x)
        shape = (self.nx*self.ny, self.nx*self.ny)
        self.preconditioner = sparse_solvers.LinearOperator(shape, m_operator)

    def solve(self):
        # abbreviations
        u = self.units
        dx = self.dx; dy = self.dy; dz = self.dz
        dt = self.dt; fvolf = self.fvolf; visc = self.visc
        nx = self.nx; ny = self.ny  # ; n_cells = self.n_cells
        permx = self.permx; permy = self.permy
        permz = self.permz
        # matrices be initialized in assembler during first time step
        t_matrix = None
        j_matrix = None
        b_matrix = None

        p = self.p_init
        t = 0.
        print "time", "\tp_mean", "\tcum production"
        # while (t < dt):
        tx_centers = permx*dz*dy/(visc*fvolf*dx) * u.transmissibility()
        ty_centers = permy*dz*dx/(visc*fvolf*dy) * u.transmissibility()
        tx_faces = self.assembler.getFaceValuesX(tx_centers)
        ty_faces = self.assembler.getFaceValuesY(ty_centers)
        # print ty_faces
        t_matrix = self.assembler.getTransmissibilityMatrix(tx_faces, ty_faces,
                                                            self.constraints)
        rhog = self.dens/fvolf*u.gravity

        g_vector = t_matrix*(rhog*self.z_centers)
        b_centers = dx*dy*dz*self.rcomp*self.poro/self.fvolf * u.accumulation()
        # print b_centers

        # these changes are overridden later.
        # made to preserve the sparsity pattern
        if (len(self.inactive_cells) > 0):
            b_centers[self.inactive_cells] = 1.

        b_matrix = diags(b_centers, 0, shape=(nx*ny, nx*ny), format='csr')

        while (t < self.reptimes[-1]):
            p_old = p

            # store original time step (it can change due to schedule)
            dtemp = dt
            current_schedule, dt = self._readSchedule(t, dt)
            self.wHandler.getProductivities(self.dx, self.dy, self.dz,
                                            permx, permy, permz,
                                            fvolf, visc,
                                            self.wells, current_schedule)

            q_vector = self.assembler.getSourceVector(tx_faces, ty_faces,
                                                      self.constraints, self.bc_values,
                                                      self.wells, current_schedule)

            j_matrix = self.assembler.getProductivityMatrix(self.wells, current_schedule)

            system_matrix = t_matrix + b_matrix/dt + j_matrix
            rhs_vector = b_matrix/dt*p_old + q_vector + g_vector

            # treat inactive cells
            if (len(self.inactive_cells) > 0):
                inact = self.inactive_cells
                system_matrix[inact, inact] = np.ones(len(inact))
                rhs_vector[inact] = p_old[inact]

            # CG with a preconditioner
            # create a preconditioner using incomplete LU decomposition
            if self.preconditioner is None:
                self._computePreconditioner(system_matrix)

            # solve a system with a preconditioner
            p, _ = sparse_solvers.cg(system_matrix, rhs_vector,
                                     x0=p_old, M=self.preconditioner)
            t += dt

            # OUTPUT
            p_mean = p[self.active_cells].mean()
            well_data = self.wHandler.getWellData(p, self.wells, 
                                                  current_schedule)
            # write
            self._saveProductionData(well_data, p_mean, t, dt)

            # stdout output
            print t, p_mean, self.cum_production

            # write field report
            if (t in self.reptimes):
                self._saveFieldData({'p (psi)': p}, t)

            dt = dtemp                   # restore original time step


# Example run
def main():
    nx = 3
    ny = 3
    dx = 300.
    dy = 300.
    permx = 100.*np.ones(nx*ny)
    permy = 100.*np.ones(nx*ny)
    permz = 100.*np.ones(nx*ny)
    poro = 0.2*np.ones(nx*ny)
    p_init = 1000.*np.ones(nx*ny)
    rcomp = 5e-6*np.ones(nx*ny)
    visc = 1*np.ones(nx*ny)
    dens = 62.4*np.ones(nx*ny)
    dz = 1*np.ones(nx*ny)
    fvolf = 1.*np.ones(nx*ny)
    xSize = 300; ySize = 300
    # x_centers = np.linspace(dx/2, xSize-dx/2, nx)
    # y_centers = np.linspace(dy/2, ySize-dy/2, ny)
    z_centers = np.zeros(nx*ny)

    constraints = [0, 0, 1, 0]
    bc_values = [0, 0, 0, 0]

    #  name       locations           radius      direction
    wells = {
        '1': {'cells': [4], 'rad': 0.5, 'dir': 3},
        '2': {'cells': [1, 2], 'rad': 0.5, 'dir': 2},
    }

    #   time wellname control value skin
    schedule = [
        # first column should be sorted
        # control: 0-inactive, 1-rate, 2-pressure
        [0., '1', 1, -1000., 0.],
        [0., '2', 2, 1500., -0.75],
    ]

    input_data = {
        "CASE": "sim1",     # name of the simulation case
        "DIMS": [600., 600.],      # dimensions of reservoir
        # "XCELLS": x_centers,    # y-coordinates of the cells
        # "YCELLS": y_centers,    # x-coordinates of the cells
        "ZCELLS": z_centers,      # z-coordinates of the cells
        "NX": nx,           # number of grid blocks along x
        "NY": ny,           # number of grid blocks along y
        "DX": dx,           # cell width
        "DY": dy,           # cell length
        "DZ": dz,           # cell thicknesses
        "PERMX": permx,     # permeability in x direction
        "PERMY": permy,     # permeability in y direction
        "PERMZ": permz,     # permeability in z direction
        "PORO": poro,       # rock porosity
        "PINIT": p_init,    # initial reservoir pressure distribution
        "RCOMP": rcomp,     # reservoir compressibility
        "FVOLF": fvolf,     # formation volume factor
        "DENS": dens,
        "VISC": visc,       # fluid viscosity

        "MAXITER": 50,      # max number of newton iterations
        "EPS": 1e-6,        # solver accuracy
        "DT": 1.,         # time step

        "BTYPE": constraints,         # type of boundary condition (0-neumann)
        "BVALUES": bc_values,         # boundary condition values

        "WELLS": wells,
        "SCHEDULE": schedule,
        "REPTIMES": [1., 100., 200.],
    }

    problem = SPhInComp2D(**input_data)
    # print problem.schedule
    # print problem._readSchedule(99.6, 1.)
    problem.solve()

if __name__ == '__main__':
    main()