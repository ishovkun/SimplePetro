from import_path import import_path
import_path("../../SPhInComp2D.py")
import_path("../../WellHandler.py")
import numpy as np
from Units import Units
from WellHandler import WellHandler
from SPhInComp2D import SPhInComp2D
from scipy.optimize import minimize
import os

def error(fname1, fname2):
    # load simulation results and history
    simdata = np.loadtxt(fname1, skiprows=1)
    history = np.loadtxt(fname2, skiprows=2)
    htime = history[:, 0]
    history = history[:, 1:]
    sim_time = simdata[:, 0]
    # pick columns with well rates
    sim_wellcolumns = np.array([2, 7, 12, 17, 22, 27])
    # allocate array with periodic production
    simperiodic = np.zeros([len(htime), len(sim_wellcolumns)])
    # iterate through months
    for i in xrange(len(htime)):
    # for i in range(2):
        # determine endpoints of months in simulation data
        if (i == 0): i_start = 0
        else: i_start = np.nonzero(sim_time == htime[i-1])[0][0] + 2
        i_end = np.nonzero(sim_time == htime[i])[0][0] + 1
        # compute monthly production for simulation results
        simperiodic[i] = -simdata[i_start:i_end, sim_wellcolumns].sum(axis=0)
    # compute error
    return np.linalg.norm(simperiodic - history)

def compute_error(params, input_data):
    input_data["PERMY"] = input_data["PERMX"]/params[0]
    nx = input_data["NX"]; ny = input_data["NY"];
    rcomp = params[1]*np.ones(nx*ny)
    input_data["RCOMP"] = rcomp
    problem = SPhInComp2D(**input_data)
    problem.solve()
    err = error("PJ1_optimization.prd", "history.txt")
    with open("opt.txt", 'a') as f:
        f.write("%f\t%e:\t%f\n" % (params[0], params[1], err))
    return err

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
    # z_centers.fill(0)
    # z_centers = -z_centers + z_centers.max()
    # print z_centers
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
    # print z_centers-z_centers.max()
    # print p_init
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

    input_data = {
        "CASE": "PJ1_optimization",     # name of the simulation case
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
        "REPTIMES": [1000.],
    }

    import time
    start_time = time.time()
    bnd = ((0.1, 10), (8e-7, 8e-5))
    with open("opt.txt", 'w') as f:
        f.write("kx/ky\tcomp\terror\n")
    # res = minimize(compute_error, [10., 1e-6], input_data, method='L-BFGS-B',
                   # bounds=bnd, tol=1e-6)
    res = minimize(compute_error, x0=[10., 1e-5], args=(input_data,),
                   bounds=bnd, tol=1e-5)
    print res
    end_time = time.time()
    with open("opt.txt", 'a') as f:
        f.write(str(res))
        f.write("\nComputation time = %f seconds" % (end_time - start_time))

    # clean-up
    os.remove("PJ1_optimization.prd")
    os.remove("PJ1_optimization1.rep")



if __name__ == '__main__':
    main()
