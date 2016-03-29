
class Units:
    # these are some constants in SI
    ft = 0.3048
    mDa = 9.869233e-13*1e-3;    # mili darcy
    cP = 1e-3                   # centipoise
    day = 86400                 # days to seconds
    psi = 6894.76               # pounds per square inch to Pa
    stb = 0.159                 # stock tank barrel to m^3
    lbs = 0.453592              # pound mass to kg
    inch = 2.54e-2              # inches to m
    g = 9.80665                 # gravity

    def __init__(self, system="Oilfield"):
        assert(system in ["Oilfield", "SI"])
        if (system == "Oilfield"): self.__setOilUnits()
        elif (system == "SI"): self.__setSIUnits()
        else: raise ValueError("unit system unknown!!!")
        self.system = system

    def __setOilUnits(self):
        '''
        time - days
        length - feet
        viscosity - psi*day
        mass - lbs
        compessibility - 1/psi
        gravity - ft/day^2 (32.174 ft/s2)
        '''
        self.length = 1.                      # length
        self.perm = self.mDa/self.ft**2        # permeability
        self.time = self.day       # time
        self.pres = self.psi       # pressure
        self.comp = 1.             # compressibility
        self.visc = self.cP/(self.psi*self.day)        # viscosity [psi*day]
        self.mass = self.lbs       # mass
        self.dens = 1.             # density [lbs/dt^3]
        # [g] = N/kg = N*m^2/(kg*m^2) = Pa*m^2/kg =
        # = Pa/psi * m^2/ft^2 * lbs/kg
        self.gravity = self.g/self.psi/self.ft**2*self.lbs     # psi*ft/lbm
        self.R_constant = 10.73     # gas constant [psi*ft^3/(lbmole*degR)]
        self.gas_constant = 10.73
        self.P_sc = 14.7            # atmospheric pressure [psi]
        self.T_sc = 32. + 459.67    # standard temperature [degR]

    def __setSIUnits(self):
        self.length = 1.
        self.perm = self.mDa*1000./self.length   # Da in Si
        self.time = 1.
        self.pres = 1.
        self.visc = 1./(self.pres*self.time)              # pa-s
        self.comp = 1.
        self.dens = 1.
        self.gravity = self.g     # m/s^2
        self.P_sc = 101325.        # atmospheric pressure [Pa]
        self.T_sc = 273.15        # standard temperature [K]

    def absoluteTemperature(self, T):
        if (self.system == "SI"):
            return T + 273.15   # Kelvin
        else:
            return T + 459.67   # Rankine

    def transmissibility(self):
        '''
        oilfield: ft^2/(day-psi)
        '''
        # return self.perm*self.length**2/(self.visc*self.length);
        return self.perm/self.visc

    def accumulation(self):
        return 1.;

    def cfl(self):
        '''dimensionless Current number'''
        return self.perm/self.visc


if __name__ == '__main__':
    u = Units("Oilfield")
    print u.transmissibility()
    # eta = u.mDa/u.ft**2 / (u.cP/u.psi/u.day)
    # print u.cfl()
    # print u.gravity
    # print u.gravity
