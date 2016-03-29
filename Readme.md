Simple Reservoir Simulator written in Python. The project started out and is inspired by PGE392K Reservoir Simulation at the University of Texas at Austin.

Current State:
    Now the code can only solve 2D single-phase fluid flow with heterogeneities and gravity. The code allows the placement of horizontal and vertical wells as well and the changes in their schedule.

Project implemented in Python 2.7

Additional libraries:
- numpy, scipy, matplotlib

Source files:

    SPhInComp2D.py - stands for Single Phase Incompressible fluid. Class that solves the problem.

    Simulator2D.py - dummy class that handles the input.

    Assembler.py - class that computes inter-block transmissibilities, T matrix, J matrix, and Q vector.

    WellHandler.py - class that handles wells.

    Units - class that handles units and constants.

Examples:


Auxiliary files:
    .prd files - reports with well pressures, rates, and cumulative injections and productions for each time step
    .rep files - reports with full field data at given times

Our Team:
    Igor Shovkun. Who wants to participate??? :-)