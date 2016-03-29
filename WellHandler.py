import numpy as np
from Units import Units

class WellHandler(object):
    """Class that handles wells"""
    def __init__(self, nx, ny):
        '''
        Initialize well handler.
        Input:
            dx: float
                cell x-size
            dy: float
                cell y-size
            dz: np.array(nx*ny)
                cell thicknesses
        '''
        # super(WellHandler, self).__init__()
        if not isinstance(nx, (int, long)):
            raise ValueError("nx should be an integer")
        if not isinstance(ny, (int, long)):
            raise ValueError("ny should be an integer")

        self.nx = nx
        self.ny = ny
        self.units = Units()

    def locateWells(self, x_centers, y_centers, wells):
        '''
        Compute the numbers of the cells the wells are located.
        writes into the self.wells dictionary
        Input:
            x_centers: np.array(nx)
                x-coordinates of cell centers
            y_centers: np.array(ny)
                y-coordinates of cell centers
            wells: dict with entries:
                heel: list [float, float]
                    coordinates of the heel of the well
                len: float
                    length of the well (meaningful for horizontal wells with dir in [1, 2])
                'rad': float
                    well radius
                'dir': int 1, 2, or 3
                    directions of the well (1-x, 2-y, 3-z)
                extra entries not prohibited
        '''
        nx = self.nx; ny = self.ny
        # CHECKINPUT
        if not isinstance(x_centers, np.ndarray):
            raise ValueError("x_centers is not a numpy array")
        if not isinstance(y_centers, np.ndarray):
            raise ValueError("y_centers is not a numpy array")
        xcshape = x_centers.shape; ycshape = y_centers.shape
        assert len(xcshape) == 1, \
            "x_centers must be a 1D array"
        assert len(ycshape) == 1, \
            "y_centers must be a 1D array"
        assert xcshape[0] == nx and ycshape[0] == ny, \
            "wrong size of x_centers or y_centers"
        dx = x_centers[1] - x_centers[0]
        dy = y_centers[1] - x_centers[0]
        # check well dictionary
        if not isinstance(wells, dict):
            raise ValueError("wells should be a dictionary")
        for w in wells.keys():
            well = wells[w]
            assert isinstance(well, dict), \
                "wells[%s] should be a dictionary" % w
            assert 'heel' in well.keys(), \
                "Specify wells[%s]['heel']" % w
            assert 'rad' in well.keys(), \
                "Specify wells[%s]['rad']" % w
            assert 'dir' in well.keys(), \
                "Specify wells[%s]['rad']" % w
            d = well['dir']
            assert d in [1, 2, 3], \
                "wells[%s]['dir'] is not in [1, 2, 3]" % w
            if d != 3:
                assert 'len' in well.keys(), \
                    "Specify wells[%s]['len']" % w
            assert isinstance(well['heel'], (list, tuple)), \
                "wells[%s]['heel'] is not a list/tuple"
            assert len(well['heel']) == 2, \
                "heel coordinates should have 2 entries"
            assert isinstance(well['heel'][0], float)
            assert isinstance(well['heel'][1], float)
            assert well['heel'][0] <= x_centers.max() + dx/2, \
                "well beyond the domain"
            assert well['heel'][1] <= y_centers.max() + dy/2, \
                "well beyond the domain"
        # ENDCHECK
        # iterate over wells
        for well in wells.values():
                well['cells'] = []
                heel = well['heel']
                i_heel = np.argmin(abs(x_centers-heel[0]))
                j_heel = np.argmin(abs(y_centers-heel[1]))
                l_heel = j_heel*nx + i_heel

                if (well['dir'] == 3):
                    well['cells'].append(l_heel)

                elif (well['dir'] == 1):
                    toe = heel; toe[0] = heel[0] + well['len']
                    if (toe[0] > x_centers.max() + dx/2):
                        raise AssertionError("well is beyond the domain")
                    i_toe = np.argmin(abs(x_centers-toe[0]))
                    j_toe = j_heel
                    l_toe = j_toe*nx + i_toe
                    well['cells'] = range(l_heel, l_toe)

                elif (well['dir'] == 2):
                    toe = heel; toe[1] = heel[1] + well['len']
                    if (toe[1] > y_centers.max() + dy/2):
                        raise AssertionError("well is beyond the domain")
                    j_toe = np.argmin(abs(y_centers - toe[1]))
                    for j in xrange(j_heel, j_toe):
                        l = j*nx + i_toe
                        well['cells'].append(l)

    def getProductivities(self, 
                          dx, dy, dz,
                          permx, permy, permz,
                          fvolf, visc, 
                          wells, current_schedule):
        """
        Adds 'prod' entries into the wells dict
        Input:
        dx: float
            cell x-size
        dy: float
            cell y-size
        dz: np.array(self.nx*self.ny)
            cell thicknesses
        permx: np.array(nx*ny)
            x-component of permeability in cell centers
        permy: np.array(nx*ny)
            y-component of permeability in cell centers
        permz: np.array(nx*ny)
            z-component of permeability in cell centers
        fvolf: np.array(nx*ny)
            formation volume factor in cell centers
        fvisc: np.array(nx*ny)
            fluid viscosity volume factor in cell centers
        wells: dict with wellname keys and dict values of the format:
            cells: np.array
                cell numbers where the well is located
            'rad': float
                well radius
            'dir': int 1, 2, or 3
                directions of the well (1-x, 2-y, 3-z)
            extra entries not prohibited
                current schedule: dictionary of the format
                'well_name': [int, float, float]
                [control (1 or 2), value of rate/bhp, skin]
        Returns:
            None (Modifies wells dictionary)
        """
        nx = self.nx; ny = self.ny
        # INUTCHECK
        if not isinstance(dx, float):
            raise ValueError("dx is not float")
        if not isinstance(dy, float):
            raise ValueError("dy is not float")
        if not isinstance(dz, np.ndarray):
            raise ValueError("dz is not a numpy array")
        
        if not isinstance(permx, np.ndarray):
            raise ValueError("permx is not a numpy array")
        if not isinstance(permy, np.ndarray):
            raise ValueError("permy is not a numpy array")
        if not isinstance(permz, np.ndarray):
            raise ValueError("permz is not a numpy array")
        if not isinstance(fvolf, np.ndarray):
            raise ValueError("fvolf is not a numpy array")
        if not isinstance(visc, np.ndarray):
            raise ValueError("visc is not a numpy array")
        # check dimensions
        assert dz.shape == permx.shape == permy.shape == permz.shape == (nx*ny,), \
            "Input dimensions are inconsistent (should be [1, %dx*%d])" % (nx, ny)
        assert visc.shape == fvolf.shape == (nx*ny,), \
            "Input dimensions are inconsistent (should be [1, %dx*%d])" % (nx, ny)

        # check wells
        for w in wells.keys():
            well = wells[w]
            assert isinstance(well, dict), \
                "wells[%s] is not a dictionary" % w
            assert 'cells' in well.keys(), \
                "Specify wells[%s]['cells']" % w
            assert 'rad' in well.keys(), \
                "Specify wells[%s]['rad']" % w
            assert 'dir' in well.keys(), \
                "Specify wells[%s]['dir']" % w
            assert isinstance(well['cells'], (list, np.ndarray)), \
                "wells[%s]['cells'] is not a numpy array or a list" % w
            assert isinstance(well['rad'], float), \
                "wells[%s]['rad'] is not a float" % w
            assert well['dir'] in [1, 2, 3], \
                "wells[%s]['dir'] is not in [1, 2, 3]" % w
        # check current_schedule
        for w in current_schedule.keys():
            well = current_schedule[w]
            assert w in wells.keys(), \
                "unknown well %s" % w
            assert isinstance(well, list), \
                "current_schedule[%s] is not a list" % w
            assert len(well) == 3, \
                "length of current_schedule[%s] must be = 3" % w
            assert well[0] in [1, 2], \
                "welll controls can be either 1 or 2"
        # ENDCHECK

        for w in wells:
            if (w not in current_schedule):
                continue
            else:
                well = wells[w]
                s = current_schedule[w][2]      # skin
                rw = well['rad']                # well radius
                prod = np.zeros(len(well['cells']))                       # productivity
                counter = 0
                for i in well['cells']:
                    kx = permx[i]
                    ky = permy[i]
                    kz = permz[i]
                    mu = visc[i]
                    bw = fvolf[i]
                    if well['dir'] == 3:        # vertical
                        if kx != 0:
                            req = 0.28*((ky/kx)**0.5*dx**2 + (kx/ky)**0.5*dy**2)**0.5
                            req = req/((ky/kx)**0.25 + (kx/ky)**0.25)
                            j = 2.*np.pi*dz[i]*(kx*ky)**0.5/(mu*bw)/(np.log(req/rw)+s)
                        else: j = 0

                    elif well['dir'] == 1:        # horizontal along x
                        if ky != 0:
                            req = 0.28*((kz/ky)**0.5*dy**2 + (ky/kz)**0.5*dz[i]**2)**0.5
                            req = req/((kz/ky)**0.25 + (ky/kz)**0.25)
                            j = 2.*np.pi*dx*(ky*kz)**0.5/(mu*bw)/(np.log(req/rw)+s)
                        else: j = 0

                    elif well['dir'] == 2:        # horizontal along y
                        if kz != 0:
                            req = 0.28*((kz/kx)**0.5*dx**2 + (kx/kz)**0.5*dz[i]**2)**0.5
                            req = req/((kz/kx)**0.25 + (kx/kz)**0.25)
                            j = 2.*np.pi*dy*(kx*kz)**0.5/(mu*bw)/(np.log(req/rw)+s)
                        else: j = 0

                    j = j*self.units.transmissibility()
                    prod[counter] = j
                    counter += 1

                well['prod'] = prod
        # print wells['6']['prod']

    def getWellData(self, p, wells, current_schedule):
        '''
        returns a dictionary with BHP and rates of all the wells:
        Input:
            p: np.arrat(n_cells)
                pressure after solution
            current_schedule: listx
                current rate controls
        Returns:
            well_data: dict with the following content:
                key: wellname
                values: {"BHP": float, "rate": float}
        '''
        well_data = {}
        for w in wells:
            well = wells[w]
            well_data[w] = {}
            if w in current_schedule:
                pe = p[well['cells']]           # get reservoir pressure in well cells
                # print current_schedule[w]
                control = current_schedule[w][0]
                value = current_schedule[w][1]

                if (control == 1):              # rate control
                    well_data[w]['rate'] = value
                    j = well['prod']  # get productivities in well cells
                    pw = np.mean(pe + value/j)
                    well_data[w]['BHP'] = pw
                    well_data[w]['PI'] = j.mean()

                elif (control == 2):            # pressure control
                    j = well['prod']  # get productivities in well cells
                    # print j
                    pw = value                  # BHP
                    q = -np.sum(j*(pe - pw))
                    well_data[w]['rate'] = q
                    well_data[w]['BHP'] = pw
                    well_data[w]['PI'] = j.mean()

                # elif (control == 0):          # inactive are not in current schedule

            else:
                pe = p[well['cells']]           # get reservoir pressure in well cells
                well_data[w]['rate'] = 0
                pw = pe.mean()
                well_data[w]['BHP'] = pw
                well_data[w]['PI'] = 0

        # print well_data
        return well_data
