"""
Created on Wed Apr 21 14:07:01 2021

@author: Farheez Mohamed
"""
import datetime

import math
import numpy as np

from scipy import stats

#import multiprocessing

from multiprocessing import Pool

from smt.sampling_methods import LHS

from liggghts import liggghts

from smt.surrogate_models import KRG

from smt.applications import EGO
from smt.applications.mixed_integer import (
    MixedIntegerContext,
    FLOAT,
)

# Taking the time 
begin_time = datetime.datetime.now()

"""
VARIABLES

"""

# Experimental values
global AoRexp
global D_ratio_exp
AoRexp = 25 #deg
D_ratio_exp = 1.05

# Particle radius
r = 0.002

# Number of calibrated parameters
n_var = 4

# Parameters boundaries
xlimits = np.array([[0.0, 0.5], [0.0, 1], [0.0501, 1], [0.0, 1]])

# Number of floats = number of parameters
xtypes = np.repeat(FLOAT, n_var)

# Number of samples
num = 40

# Number of EGO iterations
n_iter = 100

# Number of CPU
cpu = 20   # multiprocessing.cpu_count() : max available cpu

"""
END VARIABLES

"""

# LHS
sampling = LHS(xlimits=xlimits)
x = sampling(num)

# Erase results file
effacer = open("results.txt", "w")
effacer.close()

# Matrix called in the loop
lmp0 = np.repeat(liggghts(), num)
lmp1 = np.repeat(liggghts(), num)
lmp2 = np.repeat(liggghts(), num)

# Parameters values to read in the LIGGGHTS scripts
for i in range(num):
    lmp0[i] = liggghts()
    lmp1[i] = liggghts()
    lmp2[i] = liggghts()
    
    lmp0[i].command ('variable poisson equal' + ' ' + str(x[i,0]))
    lmp0[i].command ('variable CoF equal' + ' ' + str(x[i,1]))
    lmp0[i].command ('variable CoRest equal' + ' ' + str(x[i,2]))
    lmp0[i].command ('variable CoRF equal' + ' ' + str(x[i,3]))
    
    lmp1[i].command ('variable poisson equal' + ' ' + str(x[i,0]))
    lmp1[i].command ('variable CoF equal' + ' ' + str(x[i,1]))
    lmp1[i].command ('variable CoRest equal' + ' ' + str(x[i,2]))
    lmp1[i].command ('variable CoRF equal' + ' ' + str(x[i,3]))
    
    lmp2[i].command ('variable poisson equal' + ' ' + str(x[i,0]))
    lmp2[i].command ('variable CoF equal' + ' ' + str(x[i,1]))
    lmp2[i].command ('variable CoRest equal' + ' ' + str(x[i,2]))
    lmp2[i].command ('variable CoRF equal' + ' ' + str(x[i,3]))
    
    
# Angle of repose
print('SIMULATION ANGLE OF REPOSE')
def angle(i):
    lmp0[i].file("in.11_angle_fill")
    
    numAtoms = lmp0[i].extract_global("natoms", 0) # number of particles
    pos = lmp0[i].extract_atom("x",3) # z positions of particles
    
    # Matrix of X and Z positions
    Z = []
    X = []
    for k in range(0, numAtoms):
        Z.append(pos[k][2])
        X.append(pos[k][0])
        
    X = np.array(X)
    Z = np.array(Z)

    pos = np.array([X, Z]).T
    
    # Linear regression and calculation of the angle of repose
    Z_max = []
    X_max = []
   
    try:
        for i in range(0, 4):
            condition = (pos[:, 0] > -0.10 + i * 0.01) & (pos[:, 0] < -0.09 + i * 0.01)
            data = pos[condition, :]
            Z_max.append(max(data[:, 1]))
            Z_max_i = np.argmax(data[:, 1])
            X_max.append(data[Z_max_i, 0])
        
        Z_max = np.array(Z_max)
        X_max = np.array(X_max)
        pos_max = np.array([X_max, Z_max]).T
    
        pente, intercept, r_value, p_value, std_err = stats.linregress(pos_max[:,0], pos_max[:,1])
    
        AoR = math.atan(pente)*180/math.pi

        
    except ValueError:
        print('ANGLE OF REPOSE TOO SMALL')
        AoR = 0

    print('AoR = ', AoR)
    return AoR


# Multiprocessing
p_angle = Pool(processes = cpu)
results_angle = p_angle.map(angle, [k for k in range(num)])
p_angle.close()
print(results_angle)
print(' ')

# Test tube filling
print('SIMULATION TUBE TEST FILLING')
def tube(i):
    lmp1[i].file("in.21_packing_fill")
    
    num1_packing = lmp1[i].extract_global("natoms", 0)
    pos1_packing = lmp1[i].extract_atom("x",3)
    
    Z1 = []
    for m in range(0, num1_packing):
        Z1.append(pos1_packing[m][2])
        
    Z1.sort(reverse=True)
    Z1sub = Z1[0:10]
        
    # Highest particle
    h_aeree = np.mean(Z1sub)+r
    
    # Writing restart file
    lmp1[i].command('write_restart restart/packing'+str(i)+'.restart')
    lmp1[i].close()
    return h_aeree

p_tube = Pool(processes = cpu)
results_tube = p_tube.map(tube, [k for k in range(num)])
p_tube.close()
print(results_tube)
print(' ')

# Packing
print('SIMULATION PACKING')
def packing(i):
    # Reading restart file
    lmp2[i].command('newton off')
    lmp2[i].command('communicate single vel yes')
    lmp2[i].command('units si')
    lmp2[i].command('neighbor 0.002 bin')
    lmp2[i].command('neigh_modify delay 0')
    lmp2[i].command('read_restart restart/packing'+str(i)+'.restart')
    
    lmp2[i].file("in.24_packing_packing")
    
    num2_tassement = lmp2[i].extract_global("natoms", 0)
    pos2_tassement = lmp2[i].extract_atom("x",3)
    
    Z2 = []
    for n in range(0, num2_tassement):
        Z2.append(pos2_tassement[n][2])
        
    Z2.sort(reverse=True)
    Z2sub = Z2[0:10]
        
    # Highest particle
    h_tassee = np.mean(Z2sub)+r
    
    return h_tassee

p_packing = Pool(processes = cpu)
results_packing = p_packing.map(packing, [k for k in range(num)])
p_packing.close()
print(results_packing)
print(' ')

results_densityratio = [a/b for a, b in zip(results_tube, results_packing)]

# Input-Output Matrix
paramX1 = np.array([x[i,0] for i in range(0,num)])
paramX2 = np.array([x[i,1] for i in range(0,num)])
paramX3 = np.array([x[i,2] for i in range(0,num)])
paramX4 = np.array([x[i,3] for i in range(0,num)])
data = np.array([paramX1, paramX2, paramX3, paramX4, results_angle, results_densityratio]).T 
print("Input-output matrix [Parametres, AoR, Density_ration =")
print(data)
print(' ')

# Save matrix in the results file
results = open("results.txt", "a")
results.write('Input-Output Matrix =' + '\n' + str(data))
results.write(' ' + '\n')
results.close()

# Begin the optimization process
begin_optimization = datetime.datetime.now()

# Krigeage and EGO
y_aor = data[:,4]  
y_density_ratio= data[:,-1] # -1 takes the last column

aor = KRG(theta0=[1e-2])
density_ratio = KRG(theta0=[1e-2])


def function(parametres):
    # Kriging 2 functions
    aor.set_training_values(x, y_aor)
    aor.train()
    AoRnum = aor.predict_values(parametres)
    
    density_ratio.set_training_values(x, y_density_ratio)
    density_ratio.train()
    D_ratio_num = density_ratio.predict_values(parametres)
    
    
    # Objective function
    y = abs(AoRnum-AoRexp)/AoRexp + abs(D_ratio_num-D_ratio_exp)/D_ratio_exp
    return y


# EGO
criterion = "EI"  #'EI' or 'SBO' or 'UCB'
sm = KRG(print_global=False)
mixint = MixedIntegerContext(xtypes, xlimits)
sampling = mixint.build_sampling_method(LHS, criterion="ese", random_state=42)

ego = EGO(
    n_iter=n_iter,
    criterion=criterion,
    xlimits=xlimits,
    surrogate=sm,

)

# Saving calibrated parameters in the results file
x_opt, y_opt, _, _, y_data = ego.optimize(fun=function)
print("Minimum in [Poisson, CoF, CoRest, CoRF]={} with Difference={:.1f}".format(x_opt, float(y_opt)))

paramoptim = open("results.txt", "a")
paramoptim.write('Calibrated parameters : [Poisson, CoF, CoRest, CoRF] = ' + str(x_opt) + '\n')
paramoptim.write('Minimal difference = ' + str(y_opt) + '\n')
paramoptim.write(' ' + '\n')
paramoptim.close()

# Matrix of calibrated parameters
param_calibrated = np.array([[x_opt[0], x_opt[1], x_opt[2], x_opt[3]]])

# Calculation of AoR and D_ratio with the kriging functions
AoR_kri = aor.predict_values(param_calibrated)
D_ratio_kri = density_ratio.predict_values(param_calibrated)

print('AoR_calibrated = ', AoR_kri)
print('D_ratio_calibrated = ', D_ratio_kri)

# End of optimization
optimization_time = datetime.datetime.now() - begin_optimization

# Saving the results in the results file
verification = open("results.txt", "a")
verification.write('Results with Kriging :' + '\n')
verification.write('AoR_kri = ' + str(AoR_kri) + '\n')
verification.write('D_ratio_kri = ' + str(D_ratio_kri) + '\n')
verification.write('Temps du Krigeage + EGO = ' + str(optimization_time))
verification.write(' ' + '\n')
verification.close()



"""
EXECUTING THE SIMULATIONS WITH THE CALIBRATED PARAMETERS
"""
lgt0 = liggghts()
lgt1 = liggghts()
lgt2 = liggghts()

lgt0.command ('variable poisson equal' + ' ' + str(x_opt[0]))
lgt0.command ('variable CoF equal' + ' ' + str(x_opt[1]))
lgt0.command ('variable CoRest equal' + ' ' + str(x_opt[2]))
lgt0.command ('variable CoRF equal' + ' ' + str(x_opt[3]))

lgt1.command ('variable poisson equal' + ' ' + str(x_opt[0]))
lgt1.command ('variable CoF equal' + ' ' + str(x_opt[1]))
lgt1.command ('variable CoRest equal' + ' ' + str(x_opt[2]))
lgt1.command ('variable CoRF equal' + ' ' + str(x_opt[3]))

lgt2.command ('variable poisson equal' + ' ' + str(x_opt[0]))
lgt2.command ('variable CoF equal' + ' ' + str(x_opt[1]))
lgt2.command ('variable CoRest equal' + ' ' + str(x_opt[2]))
lgt2.command ('variable CoRF equal' + ' ' + str(x_opt[3]))

# Angle of repose
lgt0.file("in.31_angle_calibrated")
numAtoms_lgt0 = lgt0.extract_global("natoms", 0) 
pos_lgt0 = lgt0.extract_atom("x",3) 
    
Z_lgt0 = []
X_lgt0 = []
for k in range(0, numAtoms_lgt0):
    Z_lgt0.append(pos_lgt0[k][2])
    X_lgt0.append(pos_lgt0[k][0])

X_lgt0 = np.array(X_lgt0)
Z_lgt0 = np.array(Z_lgt0)

pos = np.array([X_lgt0, Z_lgt0]).T

Z_max = []
X_max = []

try:
    for i in range(0, 4):
        condition = (pos[:, 0] > -0.10 + i * 0.01) & (pos[:, 0] < -0.09 + i * 0.01)
        data = pos[condition, :]
        Z_max.append(max(data[:, 1]))
        Z_max_i = np.argmax(data[:, 1])
        X_max.append(data[Z_max_i, 0])

    Z_max0 = np.array(Z_max)
    X_max0 = np.array(X_max)
    pos_max = np.array([X_max0, Z_max0]).T

    pente, intercept, r_value, p_value, std_err = stats.linregress(pos_max[:,0], pos_max[:,1])

    AoR_lgt0 = math.atan(pente)*180/math.pi
    
except ValueError:
    print('AoR too small there is probably an issue')
    AoR_lgt0 = 0

if AoR_lgt0 == 0:
    # EGO
    criterion = "EI"  #'EI' or 'SBO' or 'UCB'
    sm = KRG(print_global=False)
    mixint = MixedIntegerContext(xtypes, xlimits)
    sampling = mixint.build_sampling_method(LHS, criterion="ese", random_state=42)

    ego = EGO(
        n_iter=n_iter,
        criterion=criterion,
        xlimits=xlimits,
        surrogate=sm,

    )

    # Saving calibrated parameters in the results file
    x_opt, y_opt, _, _, y_data = ego.optimize(fun=function)
    print("Minimum in [Poisson, CoF, CoRest, CoRF]={} with Difference={:.1f}".format(x_opt, float(y_opt)))
    
    paramoptim = open("results.txt", "a")
    paramoptim.write('OPTIMIZATION HAD TO BE EXECUTED AGAIN' + '\n')
    paramoptim.write('Calibrated parameters : [Poisson, CoF, CoRest, CoRF] = ' + str(x_opt) + '\n')
    paramoptim.write('Minimal difference = ' + str(y_opt) + '\n')
    paramoptim.write(' ' + '\n')
    paramoptim.close()

    # Matrix of calibrated parameters
    param_calibrated = np.array([[x_opt[0], x_opt[1], x_opt[2], x_opt[3]]])
    
    # Calculation of AoR and D_ratio with the kriging functions
    AoR_kri = aor.predict_values(param_calibrated)
    D_ratio_kri = density_ratio.predict_values(param_calibrated)

    print('AoR_calibrated = ', AoR_kri)
    print('D_ratio_calibrated = ', D_ratio_kri)

    # End of optimization
    optimization_time = datetime.datetime.now() - begin_optimization

    # Saving the results in the results file
    verification = open("results.txt", "a")
    verification.write('Results with Kriging :' + '\n')
    verification.write('AoR_kri = ' + str(AoR_kri) + '\n')
    verification.write('D_ratio_kri = ' + str(D_ratio_kri) + '\n')
    verification.write('Temps du Krigeage + EGO = ' + str(optimization_time))
    verification.write(' ' + '\n')
    verification.close()
    
    lgt0.command ('variable poisson equal' + ' ' + str(x_opt[0]))
    lgt0.command ('variable CoF equal' + ' ' + str(x_opt[1]))
    lgt0.command ('variable CoRest equal' + ' ' + str(x_opt[2]))
    lgt0.command ('variable CoRF equal' + ' ' + str(x_opt[3]))
    
    lgt1.command ('variable poisson equal' + ' ' + str(x_opt[0]))
    lgt1.command ('variable CoF equal' + ' ' + str(x_opt[1]))
    lgt1.command ('variable CoRest equal' + ' ' + str(x_opt[2]))
    lgt1.command ('variable CoRF equal' + ' ' + str(x_opt[3]))

    lgt2.command ('variable poisson equal' + ' ' + str(x_opt[0]))
    lgt2.command ('variable CoF equal' + ' ' + str(x_opt[1]))
    lgt2.command ('variable CoRest equal' + ' ' + str(x_opt[2]))
    lgt2.command ('variable CoRF equal' + ' ' + str(x_opt[3]))
    
    # Angle of repose
    lgt0.file("in.31_angle_calibrated")
    numAtoms_lgt0 = lgt0.extract_global("natoms", 0) 
    pos_lgt0 = lgt0.extract_atom("x",3) 
    
    Z_lgt0 = []
    X_lgt0 = []
    for k in range(0, numAtoms_lgt0):
        Z_lgt0.append(pos_lgt0[k][2])
        X_lgt0.append(pos_lgt0[k][0])

    X_lgt0 = np.array(X_lgt0)
    Z_lgt0 = np.array(Z_lgt0)

    pos = np.array([X_lgt0, Z_lgt0]).T

    Z_max = []
    X_max = []

    try:
        for i in range(0, 4):
            condition = (pos[:, 0] > -0.10 + i * 0.01) & (pos[:, 0] < -0.09 + i * 0.01)
            data = pos[condition, :]
            Z_max.append(max(data[:, 1]))
            Z_max_i = np.argmax(data[:, 1])
            X_max.append(data[Z_max_i, 0])

        Z_max0 = np.array(Z_max)
        X_max0 = np.array(X_max)
        pos_max = np.array([X_max0, Z_max0]).T

        pente, intercept, r_value, p_value, std_err = stats.linregress(pos_max[:,0], pos_max[:,1])

        AoR_lgt0 = math.atan(pente)*180/math.pi
        
    except ValueError:
        print('AoR too small there is probably an issue')
        AoR_lgt0 = 0
        
else:
    print('AoR_lgt0 =/= 0, optimization was ok')


# Test tube filling
lgt1.file("in.41_packing_fill_calibrated")
numAtoms_lgt1 = lgt1.extract_global("natoms", 0)
pos_lgt1 = lgt1.extract_atom("x",3)
    
Z_lgt1 = []
for m in range(0, numAtoms_lgt1):
    Z_lgt1.append(pos_lgt1[m][2])
    
Z_lgt1.sort(reverse=True)
Z_lgt1_sub = Z_lgt1[0:10]
        
h_lgt1 = np.mean(Z_lgt1_sub)+r
    
lgt1.command('write_restart restart/packing.restart')
lgt1.close()

# Packing
lgt2.file("in.43_packing_packing_calibrated")
numAtoms_lgt2 = lgt2.extract_global("natoms", 0)
pos_lgt2 = lgt2.extract_atom("x",3)
    
Z_lgt2 = []
for n in range(0, numAtoms_lgt2):
    Z_lgt2.append(pos_lgt2[n][2])
    
Z_lgt2.sort(reverse=True)
Z_lgt2_sub = Z_lgt2[0:10]
        
h_lgt2 = np.mean(Z_lgt2_sub)+r

Density_ratio = h_lgt1 / h_lgt2

# DEM results with calibrated parameters
calibration = open("results.txt", "a")
calibration.write('DEM Results with calibrated parameters :' + '\n')
calibration.write('Coefficient Poisson = ' + str(x_opt[0]) + '\n')
calibration.write('Coefficient Friction = ' + str(x_opt[1]) + '\n')
calibration.write('Coefficient Restitution = ' + str(x_opt[2]) + '\n')
calibration.write('Coefficient Rolling Friction = ' + str(x_opt[3]) + '\n')
calibration.write('AoR = ' + str(AoR_lgt0) + '\n')
calibration.write('Density_ratio = ' + str(Density_ratio) + '\n')
calibration.write(' ' + '\n')
calibration.close()

# Calibration time
calibration_time = datetime.datetime.now() - begin_time
print('Calibration time [h, min, s] = ', calibration_time)
duration = open("results.txt", "a")
duration.write('Duree du script [h, min, s] = ' + str(calibration_time))
duration.close()
