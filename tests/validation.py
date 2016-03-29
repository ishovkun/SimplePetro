import numpy as np
from .. import SPhInComp2D
from analytical import analytical_rate
import matplotlib.pyplot as plt
from ..Units import Units


def main():
    xSize = 1000.; ySize = 1.
    nx = 51
    ny = 1
    dx = xSize/nx; dy = ySize/ny
    dz = 1.*np.ones(nx*ny)
    permx = 10.*np.ones(nx*ny); permx[0] = 5000.
    permy = 10.*np.ones(nx*ny)
    permz = 10.*np.ones(nx*ny)
    poro = 0.2*np.ones(nx*ny)
    p_init = 1000.*np.ones(nx*ny)
    rcomp = 5e-6*np.ones(nx*ny)
    visc = 1.*np.ones(nx*ny)
    dens = 62.4*np.ones(nx*ny)
    fvolf = 1.*np.ones(nx*ny)
    x_centers = np.linspace(dx/2, xSize-dx/2, nx)
    y_centers = np.linspace(dy/2, ySize-dy/2, ny)
    z_centers = np.zeros(nx*ny)

    constraints = [0, 0, 0, 0]
    bc_values = [0, 0, 0, 0]

    #  name       locations           radius      direction
    wells = {
        '1': {'cells': [0], 'rad': 0.5, 'dir': 3},
    }

    #   time wellname control value skin
    schedule = [
        # first column should be sorted
        # 3rd column: control (0-inactive, 1-rate, 2-pressure)
        [0., '1', 2, 200., 0.],
    ]

    input_data = {
        "CASE": "Problem1",     # name of the simulation case
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

        "MAXITER": 50,      # max number of newton iterations
        "EPS": 1e-6,        # solver accuracy
        "DT": 0.01,         # time step

        "BTYPE": constraints,         # type of boundary condition (0-neumann)
        "BVALUES": bc_values,         # boundary condition values

        "WELLS": wells,
        "SCHEDULE": schedule,
        "REPTIMES": [50],
    }

    problem = SPhInComp2D(**input_data)
    problem.solve()

    data = np.loadtxt("Problem1.prd", skiprows=1)
    t = data[:, 0]
    q = data[:, 2]
    xe = xSize; xwf = x_centers[0]
    u = Units()
    i = 5  # point with not increased permeability
    td = permx[i]*t/(poro[i]*visc[i]*rcomp[i])/(xwf - xe)**2
    td_units = u.perm/u.visc
    td = td*td_units
    pf = schedule[0][3]; pi = p_init[0]
    qd = -(xwf-xe)/(pf-pi)*q*visc[i]/permx[i]/dy/dz[i]
    qd = qd*u.visc/u.perm
    qd_an = analytical_rate(td)
    plt.loglog(td, qd, 'ks', label='numerical')
    plt.loglog(td, qd_an, label='analytical')
    plt.xlabel(r"$t_d$", fontsize=16)
    plt.ylabel(r"$q_d$", fontsize=16)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
