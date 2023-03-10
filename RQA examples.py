""" RQA implementation """

"""I lorenz example """
import os
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# load lorenz sequence from R
fpath = os.getcwd() + '/data/lorData.txt'
lorData = np.loadtxt(fpath, comments="#", delimiter=",", skiprows=1)
x = lorData[:, 1]
y = lorData[:, 2]
z = lorData[:, 3]

# ---- AMI ----- #
import teaspoon.parameter_selection.MI_delay as AMI
d = AMI.MI_for_delay(x, plotting=False, method='kraskov 1', h_method='sqrt', k=2, ranking=True)
# 8 instead of 9

# ---- FNN ----- #
import teaspoon.parameter_selection.FNN_n as FNN
m = FNN.FNN_n(ts=x, tau=d, maxDim=10, plotting=False)[1]
# 2 instead of 3


""" based on Analyzing Multivariate Dynamics Using Cross-Recurrence Quantification Analysis (CRQA),
 Diagonal-Cross-Recurrence Profiles (DCRP), and Multidimensional Recurrence Quantification Analysis (MdRQA)
  – A Tutorial in R
Sebastian Wallot1* and Giuseppe Leonardi2 """




""" alternative lorenz sequence"""
# generate a sequence of data points of the Lorenz-system dynamics

# Create an image of the Lorenz attractor.
# The maths behind this code is described in the scipython blog article
# at https://scipython.com/blog/the-lorenz-attractor/
# Christian Hill, January 2016.
# Updated, January 2021 to use scipy.integrate.solve_ivp.

# Lorenz paramters and initial conditions.
sigma, beta, rho = 10, 2.667, 28
u0, v0, w0 = -13, -14, 47

# Maximum time point and total number of time points.
tmax, n = 20, 1000

def lorenz(t, X, sigma, beta, rho):
    """The Lorenz equations."""
    u, v, w = X
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    return up, vp, wp

# Integrate the Lorenz equations.
soln = solve_ivp(lorenz, (0, tmax), (u0, v0, w0), args=(sigma, beta, rho),
                 dense_output=True)

# Interpolate solution onto the time grid, t.
t = np.linspace(0, tmax, n)
x, y, z = soln.sol(t)  # doesnt match R results


def lorenz(xyz, *, s=10, r=28, b=2.667):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])

dt = 0.02
num_steps = 1000

xyzs = np.empty((num_steps + 1, 3))  # Need one more for the initial values
xyzs[0] = (-13, -14, 47)  # Set initial values
# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps):
    xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt
