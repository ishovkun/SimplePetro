Simple Reservoir Simulator written in Python. The project started out and is inspired by PGE392K Reservoir Simulation at the University of Texas at Austin.

Current State:

Now the code can only solve 2D single-phase fluid flow with heterogeneities and gravity. The code allows the placement of horizontal and vertical wells as well and the changes in their schedule.

Project implemented in Python 2.7

Additional libraries:
- numpy, scipy, matplotlib

Models:

- SPhInComp2D.py - stands for Single Phase Incompressible fluid. Class that solves the problem.

- SPhInComp2D_NewtonRapson.py - solves Single Phase Incompressible fluid with
Newton's method (jacobian computed numerically). It is slower but is good for testing the jacobian assembler.


Examples:


Simulation produce:
- .prd files - reports with well pressures, rates, and cumulative injections and productions for each time step
- .rep files - reports with full field data at given times

Our Team:
    Igor Shovkun. Who wants to participate??? :-)