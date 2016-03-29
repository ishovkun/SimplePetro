import numpy as np
from scipy.sparse import diags
import scipy.sparse.linalg as sparse_solvers
import scipy
from lib2d.Units import Units
from lib2d.WellHandler import WellHandler
from lib2d.Assembler import Assembler2D
from lib2d.Simulator2D import Simulator2D



class SPhInComp2D_NR(Simulator2D):
    """
    Solves incompressible single phase fluid flow
    in a 2d rectangular domain with gravity.
    Uses Newton-Rapson algorithm. (This is a slower method,
    but good for testing the algorithm).
    """
    def __init__(self, **input_data):
        super(SPhInComp2D, self).__init__(**input_data)
        self._readInput()

        self.units = Units("Oilfield")
        self.wHandler = WellHandler(self.nx, self.ny)
        self.assembler = Assembler2D([self.nx, self.ny])
        self.preconditioner = None
        self.newt_tol = 1e-5

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
        # print "time", "\tp_mean", "\tcum production"
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

        p_mean = self.p_init.mean()
        while (t < dt):      # time loop
        # while (t < self.reptimes[-1]):      # time loop
            p_old = p

            # store original time step (it can change due to schedule)
            dtemp = dt
            current_schedule, dt = self._readSchedule(t, dt)
            self.wHandler.getProductivities(self.dx, self.dy, self.dz,
                                            permx, permy, permz,
                                            fvolf, visc,
                                            self.wells, current_schedule)
            j_matrix = self.assembler.getProductivityMatrix(self.wells, current_schedule)

            q_vector = self.assembler.getSourceVector(tx_faces, ty_faces,
                                                      self.constraints, self.bc_values,
                                                      self.wells, current_schedule)
            error = self.newt_tol*2.
            niter = 1
            while (error > self.newt_tol and niter < 10):     # Newton loop

                residual = self.computeResidual(p, p_old, dt,
                                                t_matrix, b_matrix, j_matrix,
                                                q_vector, g_vector)
                args = (p_old, dt, t_matrix, b_matrix, j_matrix,
                        q_vector, g_vector)
                jacobian = self.assembler.getJacobian(p, residual,
                                                      self.computeResidual, args)
                dp = sparse_solvers.spsolve(jacobian, -residual)
                p = p + dp
                # norm of pressure increment (normalize by average pressure)
                error1 = np.linalg.norm(dp)/p_mean
                # norm or the residual (normalize by pore volume)
                error2 = np.linalg.norm(residual)  #/(self.poro*self.area*self.dx)
                error = error1 + error2
                # print niter
                niter += 1

            p_mean = p.mean()
            # print p_mean
            print p
            t += dt
            dt = dtemp                   # restore original time step

    def computeResidual(self, p, p_old, dt,
                        t_matrix, b_matrix, j_matrix,
                        q_vector, g_vector):
        system_matrix = t_matrix + b_matrix/dt + j_matrix
        rhs_vector = b_matrix/dt*p_old + q_vector + g_vector
        # treat inactive cells
        if (len(self.inactive_cells) > 0):
            inact = self.inactive_cells
            system_matrix[inact, inact] = np.ones(len(inact))
            rhs_vector[inact] = p_old[inact]

        residual = system_matrix*p - rhs_vector
        return residual


# Example run
def main():
    xSize = 300.; ySize = 300.
    nx = 3
    ny = 3
    dx = xSize/nx; dy = ySize/ny
    dz = 100.*np.ones(nx*ny)
    permx = 100.*np.ones(nx*ny);
    permy = 100.*np.ones(nx*ny);
    permz = 100.*np.ones(nx*ny)
    poro = 0.2*np.ones(nx*ny);
    p_init = 1000.*np.ones(nx*ny)
    rcomp = 5e-6*np.ones(nx*ny)
    visc = 1.*np.ones(nx*ny)
    dens = 62.4*np.ones(nx*ny)
    fvolf = 1.*np.ones(nx*ny)
    x_centers = np.linspace(dx/2, xSize-dx/2, nx)
    y_centers = np.linspace(dy/2, ySize-dy/2, ny)
    z_centers = 1.*np.ones(nx*ny)

    constraints = [0, 0, 1, 0]
    bc_values = [0, 0, 1200., 0]

    #  name       locations           radius      direction
    wells = {
        '1': {'cells': [4], 'rad': 0.5, 'dir': 3},
        '2': {'cells': [8], 'rad': 0.5, 'dir': 3},
    }

    #   time wellname control value skin
    schedule = [
        # first column should be sorted
        # 3rd column: control (0-inactive, 1-rate, 2-pressure)
        [0., '1', 1, -1000., -0.75],
        [0., '2', 2, 1500., -0.75],
    ]

    input_data = {
        "CASE": "3x3_problem",     # name of the simulation case
        "DIMS": [xSize, ySize],      # dimensions of reservoir
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

        "DT": 0.01,         # time step

        "BTYPE": constraints,         # type of boundary condition (0-neumann)
        "BVALUES": bc_values,         # boundary condition values

        "WELLS": wells,
        "SCHEDULE": schedule,
        "REPTIMES": [0.1],
    }

    problem = SPhInComp2D(**input_data)
    # print problem.schedule
    # print problem._readSchedule(99.6, 1.)
    problem.solve()

if __name__ == '__main__':
    main()