# Remove when running not from examples/NecheliK
from import_path import import_path
import_path("../../SPhInComp2D.py")
import_path("../../WellHandler.py")
#
import numpy as np
from Units import Units
from WellHandler import WellHandler
from SPhInComp2D import SPhInComp2D


def main():
    xSize = 6000.
    ySize = 7500.
    nx = 80
    ny = 75
    dx = xSize/nx
    dy = ySize/ny
    x_centers = np.linspace(dx/2, xSize-dx/2, nx)
    y_centers = np.linspace(dy/2, ySize-dy/2, ny)
    z_centers = -np.loadtxt("PJ1-Depth.txt").flatten()
    permx = np.loadtxt("PJ1-Permeability.txt").flatten()
    permy = permx*0.1
    permz = permx
    dz = np.loadtxt("PJ1-Thickness.txt").flatten()
    poro = np.loadtxt("PJ1-Porosity.txt").flatten()
    rcomp = 1e-5*np.ones(nx*ny)
    visc = 1.95*np.ones(nx*ny)
    fvolf = 1.0*np.ones(nx*ny)
    dens = 62.4*np.ones(nx*ny)
    u = Units()
    rhog = dens*u.gravity
    p_init = 3500. + rhog*(z_centers-z_centers.max())
    constraints = [0, 0, 0, 0]
    bc_values = [0, 0, 0, 0]
    wells = {
        '1': {'heel': [4184., 1569.], 'rad': 0.25, 'dir': 3},
        '2': {'heel': [4968.5, 2510.4], 'rad': 0.25, 'dir': 3},
        '3': {'heel': [3294.9, 2928.8], 'rad': 0.25, 'dir': 3},
        '4': {'heel': [2562.7, 4393.2], 'rad': 0.25, 'dir': 3},
        '5': {'heel': [1307.5, 2824.2], 'rad': 0.25, 'dir': 3},
        '6': {'heel': [890., 895.], 'len': 225, 'rad': 0.25, 'dir': 1},
    }

    schedule = [
        [0., '1', 1, -125., 6.0],
        [0., '2', 1, -175., 0.0],
        [0., '3', 1, -750., 0.0],
        [0., '4', 2, 1200., 0.0],
        [0., '5', 2, 1200., 6.0],
        [0., '6', 1, -1000., 0.0],
        [300., '1', 2, 1500., 6.0],
        [300., '2', 2, 1500., 0.0],
        [300., '3', 2, 1500., 0.0],
        [300., '6', 2, 2000., 0.0],
    ]

    wh = WellHandler(nx, ny)
    wh.locateWells(x_centers, y_centers, wells)
    # for w in wells:
    #     cells = wells[w]['cells']
    #     print w, cells

    input_data = {
        "CASE": "PJ1",     # name of the simulation case
        "DIMS": [xSize, ySize],      # dimensions of reservoir
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

        # "MAXITER": 50,      # max number of newton iterations
        # "EPS": 1e-6,        # solver accuracy
        "DT": 1.,         # time step

        "BTYPE": constraints,         # type of boundary condition (0-neumann)
        "BVALUES": bc_values,         # boundary condition values

        "WELLS": wells,
        "SCHEDULE": schedule,
        "REPTIMES": [1., 10., 100., 1000.],
    }

    problem = SPhInComp2D(**input_data)
    problem.solve()


if __name__ == '__main__':
    main()
