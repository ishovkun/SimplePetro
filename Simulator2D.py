import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import cg, spsolve
# import matplotlib.pyplot as plt
from Assembler import Assembler2D


class Simulator2D(object):
    """
    Basic simulator class with that processes input data.
    All other classes inherit from this.
    """
    def __init__(self, **input_data):
        super(Simulator2D, self).__init__()
        self.input = input_data

        # self.xSize = None       # size of the domain along x
        # self.ySize = None       # size of the domain along y
        self.dims = None        # dimensions [float, float]
        self.nx = None          # number of cells in x direction
        self.ny = None          # number of cells in y direction
        self.n_cells = None     # total numer of cells; inferred from array sizes
        self.dx = None          # (float) cell x-size
        self.dy = None          # (float) cell y-size
        self.dz = None          # (array) cell z-size
        # self.x_centers = None   # x coordinates of cell centers
        # self.y_centers = None   # y coordinates of cell centers
        self.z_centers = None   # z coordinates of cell centers
        self.permx = None       # permeability in x direction
        self.permy = None       # permeability in y direction
        self.permz = None       # permeability in z direction
        self.poro = None        # reservoir porosity
        self.rcomp = None       # reservoir compressibility
        self.visc = None        # fluid viscosity
        self.dens = None        # fluid density
        self.p_init = None
        self.dt = None
        self.constraints = None
        self.bc_values = None
        self.reptimes = None
        self.wells = None
        self.schedule = None
        self.cgmaxiter = 50
        self.cgeps = 1e-8

        self.n_report_file = 1      # number of the field report file
        self.cum_production = 0     # total production for all wells
        self.cum_injection = 0
        self.cum_prod_wells = None
        self.cum_inj_wells = None
        self.production_header_written = False  # flag

        self.inactive_cells = None
        self.active_cells = None

        # self.wHandler = WellHandler(self.wells,
        #                             self.x_centers, self.y_centers, self.z_centers,
        #                             self.dx, self.dy, self.dz)

    def _readInput(self):
        """
        read, check input and assign it to class attributes
        Input:
        input_data: dict
        dictionary that contains all the data for simulation with keys:
            CASE: string,
                name of the case
            DIMS: list [float, float]
                x and y sizes of reservoir
            NX: int
                number of cells in x direction
            NY: int
                number of cells in y direction
            ZCELLS: np.array(nx*ny)
                z-coordinates of cell centers
            DX: float
                x size cells (constant)
            DY: float
                y size cells (constant)
            DZ: np.array(nx*ny)
                thicknesses of the cells (not constant)
            PERMX: np.array(nx*ny)
                x-component of cells' permeability
            PERMY: np.array(nx*ny)
                y-component of cells' permeability
            PERMZ: np.array(nx*ny)
                z-component of cells' permeability (to compute horizontal
                well productivity)
            PORO: np.array(nx*ny)
                porosity in cells
            PINIT: np.array(nx*ny)
                initial pressure in cells
            RCOMP: np.array(nx*ny)
                total compressibility in cells
            FVOLF: np.array(nx*ny)
                formation volume factor
            VISC: np.array(nx*ny)
                fluid viscosity in cells
            DENS: np.array(nx*ny)
                fluid density in cells
            DT: float
                time step. time step changes if DT is larger
                than the time to the next schedule time
                or next report time
            BTYPE: list [bool, bool, bool, bool]
                Types of boundary conditions given in a counterclockwise manner
                starting from the left boundary (0 - neumann, 1 - dirichlet).
            BVALUES: list [float, float, float, float]
                values imposed on the boundaries  in a counterclockwise manner
                starting from the left boundary.
            WELLS: dict with the keys of well names
                specifies well locations, radii, and directions with the folotting values:
                'cells': list(n_cells)
                    numbers of the cells the well intersects.
                    there should not be more than 1 well in one cell.
                'rad': float
                    well radius
                'dir': int 1, 2, or 3
                    directions of the well (1-x, 2-y, 3-z)
            SCHEDULE: list [float, string, int, float, float]
                list of well controls of the following format:
                time: should be not larger than the largest time in REPTIMES
                well name: should be one from the WELLS keywords
                control type: 0-inactive, 1-rate, 2-pressure
                rate or BHP: negative rate - producer
                skin: optional (don't need for rate control)
            CGEPS: float, optional (default 1e-8)
                precision of CG solver
            CGMAXITER: int, optional (default 50)
                maximum number of iterations of CG solver
        """
        needed_keys = [
            "CASE", "DIMS", "NX", "NY",         # dimensions
            "ZCELLS",                           # cell depth
            "DX", "DY", "DZ",                   # cell sizes
            "PERMX", "PERMY", "PERMZ",          # permeability
            "PORO", "PINIT", "RCOMP", "FVOLF",  # reservoir props
            "VISC",                             # fluid visc
            "DENS",                             # fluid density
            "DT",                               # solver settings
            "BTYPE", "BVALUES",                 # boundary conditions
            "WELLS", "SCHEDULE",                # well data
            "REPTIMES",                         # report times
        ]

        attr_names = {
            "CASE": 'caseName',
            "DIMS": 'dims',
            "NX": 'nx', "NY": 'ny',
            "DX": "dx", "DY": 'dy', "DZ": 'dz',
            "ZCELLS": 'z_centers',
            "PERMX": 'permx', "PERMY": 'permy', "PERMZ": 'permz',
            "PORO": 'poro', "PINIT": 'p_init', "RCOMP": 'rcomp',
            "FVOLF": 'fvolf',
            "VISC": 'visc',
            "DENS": 'dens',
            "DT": 'dt',
            "BTYPE": 'constraints',
            "BVALUES": 'bc_values',
            "WELLS": 'wells',
            "SCHEDULE": 'schedule',
            "REPTIMES": 'reptimes',
            "CGEPS": "cgeps",
            "CGMAXITER": "cgmaxiter"
        }

        optional_keys = ["CGEPS", "CGMAXITER", "NEWTTOL"]

        n_needed_keys = len(needed_keys)
        for k in xrange(n_needed_keys):
            key = needed_keys[k]
            if key in self.input:
                value = self._checkKeyWord(key)
                attr = attr_names[key]
                self.__dict__[attr] = value
            else:
                raise IOError("Please, specify %s" % (key))

        for k in xrange(len(optional_keys)):
            key = optional_keys[k]
            if key in self.input:
                value = self._checkKeyWord(key)
                self.__dict__[attr] = value

        # n_cells is inferred from array sizes
        # check for consistency
        if self.n_cells is not None:
            assert(self.n_cells == self.nx*self.ny)
        else: self.n_cells = self.nx*self.ny

        self.cum_prod_wells = np.zeros(len(self.wells))
        self.cum_inj_wells = np.zeros(len(self.wells))
        # list empty attributes
        # for key in self.__dict__:
        #     if self.__dict__[key] is None:
        #         print key, 'is None'

    def _checkKeyWord(self, key):
        value = self.input[key]
        inttypes = ["NX", "NY", "CGMAXITER", ]

        floattypes = [
            "DT", "CGEPS",
            "DX", "DY",
        ]

        arrtypes = [
            "PERMX", "PERMY", "PERMZ",
            "PORO", "PINIT", "RCOMP",
            "DZ", "ZCELLS", "VISC",
            "FVOLF", "DENS",
            # "XCELLS", "YCELLS",
        ]

        strtype = ["CASE"]

        simple_key_exception = IOError("Wrong value encountered in %s" % (key))

        if key in inttypes:
            try: return int(value)
            except: raise simple_key_exception

        elif key in floattypes:
            try: return float(value)
            except: raise simple_key_exception

        elif key in arrtypes:
            assert type(value) == np.ndarray, "%s is not an array" % key
            if (self.nx is not None and self.ny is not None):
                assert value.shape[0] == self.nx*self.ny, "Wrong shape in %s" % key
            elif (self.n_cells is not None):
                assert(value.shape[0] == self.n_cells)
            else:
                self.n_cells = value.shape[0]
            return value

        elif key in strtype:
            try: return str(value)
            except: raise simple_key_exception

        elif key == "BTYPE":
            assert type(value) == list, "%s is not a list" % key
            assert len(value) == 4, "Wrong size of %s" % key
            try: return [bool(v) for v in value]
            except: raise simple_key_exception
            return value

        elif key == "BVALUES":
            assert type(value) == list, "%s is not a list" % key
            assert len(value) == 4, "Wrong size of %s" % key
            try: return [float(v) for v in value]
            except: raise simple_key_exception
            return value

        elif key == "DIMS":
            assert type(value) == list, "%s is not a list" % key
            assert len(value) == 2, "Wrong size of %s" % key
            try: return [float(v) for v in value]
            except: raise simple_key_exception
            return value

        elif key == "REPTIMES":
            value = self.input[key]
            assert type(value) == list, "%s is not a list" % key
            try: return [float(v) for v in value]
            except: raise simple_key_exception
            return value

        elif key == "WELLS":
            for w in value.keys():
                well = value[w]
                try:
                    cells = well['cells']
                    assert type(cells) == list
                    try: [float(c) for c in cells]
                    except: raise simple_key_exception
                except:
                    raise IOError("No well location specified")
            return value

        elif key == "SCHEDULE":
            assert type(value) == list, simple_key_exception
            # check if times sorted
            times = [stage[0] for stage in value]
            assert times == sorted(times), "SCHEDULE must be ordered by time"
            for stage in value:
                assert len(stage) > 4, simple_key_exception
                assert stage[1] in self.wells, simple_key_exception
                assert stage[2] in [0, 1, 2]
                if len(stage) == 4: stage.append(0.)    # if no skin
            return value

    def _readSchedule(self, current_time, dt):
        '''
        read schedule and determine current well controls
        also determines time step not to skip well control
        or report time
        '''
        current_schedule = {}
        for i in xrange(len(self.schedule)):
            stage = self.schedule[i]
            if (current_time >= stage[0]):
                current_schedule[stage[1]] = stage[2:]
            if (stage[0] > current_time):
                dt_max = stage[0] - current_time
                dt = min(dt, dt_max)
                break

        for i in xrange(len(self.reptimes)):
            reptime = self.reptimes[i]
            if (reptime <= current_time):
                continue
            if (reptime > current_time):
                dt_max = reptime - current_time
                dt = min(dt, dt_max)
                break

        return current_schedule, dt

    def _saveFieldData(self, data, time):
        '''
        Generates a report file with full field data
        Input:
            data: dictionary of the type:
                'property_name': np.array(nx*ny)
            time: float
                current time
        '''
        filename = self.caseName + str(self.n_report_file) + ".rep"
        print "Saving report for %.02f days" % time
        keys = data.keys()
        with open(filename, "w") as f:
            f.write("Report for %04.02f days\n" % time)
            f.write("Cell_i\tCell_j")
            # loop through keys to create a header
            for key in keys:
                f.write("\t" + key)
            f.write("\n")
            # loop through cells
            for j in xrange(self.ny):
                for i in xrange(self.nx):
                    cell = j*self.nx + i
                    f.write(str(i+1)+"\t"+str(j+1))
                    for key in keys:
                        f.write("\t%f" % (data[key][cell]))
                    f.write("\n")

        self.n_report_file += 1

    def _saveProductionData(self, well_data, p_mean, t, dt):
        '''
        Saves wellbore data such as rates and pressures
        '''
        filename = self.caseName + ".prd"
        if not self.production_header_written:    # header
            header = "Time (days)\tP_mean (psi)"
            for w in sorted(well_data):
                header += "\t" + w + "_rate (scf/d)"
                header += "\t" + w + "_bhp (psi)"
                header += "\t" + w + "_cumprod (scf)"
                header += "\t" + w + "_cuminj (scf)"
                header += "\t" + w + "_PI (scf/day-psi)"
            header += "\tCUMPROD(scf)"
            header += "\tCUMINJ(scf)"
            with open(filename, 'w') as f:
                f.write(header + "\n")
            self.production_header_written = True

        with open(filename, 'a') as f:
            f.write(str(t))
            f.write("\t%.04f" % p_mean)
            counter = 0
            for w in sorted(well_data):
                data = well_data[w]
                if (data['rate'] < 0):
                    self.cum_production -= data['rate']*dt
                    self.cum_prod_wells[counter] -= data['rate']*dt
                else:
                    self.cum_inj_wells[counter] += data['rate']*dt
                    self.cum_injection += data['rate']*dt
                # get well productivity index (could not exist for inactive wells)
                j_prod = data['PI']
                f.write('\t%.04f' % data['rate'])
                f.write('\t%.04f' % data['BHP'])
                f.write('\t%.04f' % self.cum_prod_wells[counter])
                f.write('\t%.04f' % self.cum_inj_wells[counter])
                f.write('\t%.04f' % j_prod)
                counter += 1
            f.write('\t%.04f' % self.cum_production)
            f.write('\t%.04f\n' % self.cum_injection)

    def _getActiveCells(self):
        self.inactive_cells = np.nonzero(self.poro <= 1e-4)[0]
        self.active_cells = np.nonzero(self.poro > 1e-4)[0]


def main():
    nx = 3
    ny = 3
    dx = 300.
    dy = 300.
    permx = 100.*np.ones(nx*ny)
    permy = 100.*np.ones(nx*ny)
    permz = 100.*np.ones(nx*ny)
    poro = 0.2*np.ones(nx*ny); poro[3:5] = 0
    p_init = 1000.*np.ones(nx*ny)
    rcomp = 5e-6*np.ones(nx*ny)
    visc = 1*np.ones(nx*ny)
    dens = 62.4*np.ones(nx*ny)
    dz = 1*np.ones(nx*ny)
    fvolf = 1.*np.ones(nx*ny)
    xSize = 300; ySize = 300
    x_centers = np.linspace(dx/2, xSize-dx/2, nx)
    y_centers = np.linspace(dy/2, ySize-dy/2, ny)
    z_centers = np.zeros(nx*ny)

    constraints = [0, 0, 1, 0]
    bc_values = [0, 0, 0, 0,]

    #  name       locations           radius      direction
    wells = {
        '1': {'cells': [4], 'rad': 0.5, 'dir': 3},
        '2': {'cells': [8], 'rad': 0.5, 'dir': 3},
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
        "VISC": visc,       # fluid viscosity
        "DENS": dens,       # fluid density

        "MAXITER": 50,      # max number of newton iterations
        "EPS": 1e-6,        # solver accuracy
        "DT": 0.01,         # time step

        "BTYPE": constraints,         # type of boundary condition (0-neumann)
        "BVALUES": bc_values,         # boundary condition values

        "WELLS": wells,
        "SCHEDULE": schedule,
        "REPTIMES": [1., 100., 200.],
    }

    problem = Simulator2D(**input_data)
    problem._readInput()
    # print problem._readSchedule(99.6, 1.)
    problem._getActiveCells()

if __name__ == '__main__':
    main()