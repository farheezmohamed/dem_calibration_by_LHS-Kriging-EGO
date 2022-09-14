# dem_calibration_by_LHS-Kriging-EGO
An optimal calibration procedure for the Discrete Element Method (DEM) by LHS-Kriging-EGO

calibrage.py is the main script which will run the DEM simulations by calling LIGGGHTS.

The LIGGGHTS scripts are numbered in the order of execution.

To run the scripts you have to install the SMT package (https://smt.readthedocs.io/en/latest/index.html): 
pip3 install smt

Follow this procedure to use LIGGGHTS with Python:
https://www.cfdem.com/media/DEM/docu/Section_python.html#installing-the-python-wrapper-into-python

Before running the calibration procedure: 
    
Give the right experimental values and useful variables

If the number of calibrated parameters change
    - Adjust xlimits matrix
    - Adjust the fix commands in the LIGGGHTS scripts

Choose the right number of LHS samples (~10 times the number of calibrated parameters)

Choose the EGO iterations n_iter

Adjust the cpu variable

0 : angle of repose
1 : filling the test tube
2 : packing
