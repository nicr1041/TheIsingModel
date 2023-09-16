'''
Description of the program:
---------------------------
This scripts performs a gradient descent to calculate the critical exponents gamma and nu of the ferromagnetic Ising Model. Hereby, the critical temperature has to be given.
It is necessary to have datasets for different lattice sizes, i.e. with different numbers of sites N. All the necessary data have to be stored in the "_reprocessed_data_wolf.csv" file.
This is the same file that has already been used for the calculation of the partition function.
The list of discrete beta values must be the same for each dataset, i.e. for each N, as it has already been the case for the calculation of the partition functions.
Furthermore, there is also one "Z_beta_N=##.csv" for each lattice size N file, as calculated in the "partition_function.py" script.


Required data:
--------------
-   "_reprocessed_data_wolf.csv" (the same file that was used for the calculation of the partition function. This includes all measurements for all beta and all different N)
    This file can be created with the "combine_csv_reprocessed.py" script using the raw data files
-   "Z_beta_N=##.csv" file for each number of lattice sites N as was calculated in the "partition_function.py" (there is a dummy file for this in the repository)

Structure of the "_reprocessed_ising_wolf.csv" file:
----------------------------------------------------
n;interpol_E;interpol_M;beta;J;B;N

Hereby, J is the spin coupling of the Ising Model, B the magnetic field, N the number of sites, n the iteration step of the Monte Carlo algorithm,
interpol_E and interpol_M are the measured values during the Monte Carlo simulation that serve as basis for the interpolation.


Literature:
-----------
M.E.J. Newman and G.T. Barkema. Monte Carlo Methods in Statistical Physics. Clarendon Press Oxford, 1999.
(There is an entire chapter on the multiple histogram method.)
'''

import torch
import numpy as np
import pandas as pd
import numba
from matplotlib import pyplot as plt


# Some global variables:
# Make sure that the "_reprocessed_ising_wolf.csv" file is in the correct directory.
df=pd.read_csv('_reprocessed_ising_wolf.csv', sep=';')

# Alter the elements of the list so that it suits the list of beta values that the Monte Carlo algorithm was run for.
# THE LIST OF beta VALUES MUST BE THE SAME AS FOR PARTITION FUNCTION CALCULATION.
_beta=np.array([0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6])
_N=df['N'].unique()
_beta_len=(_beta.shape[0])
_t=np.zeros(_beta_len)

# n_block=number of iterations per beta - cut off (thermalization); this equals the number of values p for u and m
n_block=9800
_N_len=(_N.shape[0])

#initializing the arrays
meta=np.ones((_beta_len,3))     # Stores the number of measurements, partition function Z, beta.
meta_tot=np.ones((_N_len,_beta_len,3))      # Stores the number of measurements, partition function Z, beta for each number of sites N.
energy=np.zeros((_N_len,_beta_len,n_block))
magnetization=np.zeros((_N_len,_beta_len,n_block))


_beta_interpolated=20   # To scale the resolution of the interpolated parameter.

# Defining the critical temperature of the system (here: ferromagnetic Ising Model on the square lattice).
T_c=torch.tensor([2.269], requires_grad=False, dtype=torch.float32)

# Initializing the critical exponents with initial values.
# Instead of using gamma and nu, 1/nu and gamma/nu are used to make them independent in the algorithm.
exp1=torch.tensor([1.0], requires_grad=True, dtype=torch.float32)       # this is 1/nu
exp2=torch.tensor([1.0], requires_grad=True, dtype=torch.float32)       # this is gamma/nu


# main function
def main():
    initializing_arrays()
    loading_partition_functions()
    gradient_descend()


#---------------------------------------------
# Actual gradient descend calculation
#---------------------------------------------

# Loss function as proposed by Newman et al. on p. 236.
# In order to understand the following part, to have a look at Newman et al.. The loss function in latex format is:
# \sigma^2= \frac{1}{x_{max}-x_{min}}\int_{x_{min}}^{x_{max}}\sum_L \Tilde{\chi}_L^2(x) - \Bigl ( \sum_L \Tilde{\chi}_L(x) \Bigr)^2 dx
# x is the rescaled variable on the horizontal axis. The rescaling dependes on the critical temeprature, the linear lattice size L, and 1/nu.
def loss(lowlim, uplim, epoch):
    colors=['gray','green','blue','red','orange']
    chi_x=torch.zeros(_N_len,_beta_interpolated)        # Stores the rescaled interpolated values.
    x=np.linspace(lowlim,uplim,_beta_interpolated)
    h=(uplim-lowlim)/(_beta_interpolated)
    for i in range(0,_N_len):
        L=np.sqrt(_N[i])
        print(L)
        tmp=((torch.from_numpy(x)*L**(-exp1)+1)*T_c)**(-1)      # Get beta values out of x.
        chi_x[i,:]=(L**(-exp2))*calculate_chi_beta(tmp, i ,L**2)
    # Calculate the integral of the loss function:
    integrand=torch.var(chi_x,axis=0)
    integral=torch.tensor([0.0])
    for i in range(0,_beta_interpolated-1):
        integral+=h/2*(integrand[i]+integrand[i+1])
    print(f'chi_x**2 {(chi_x**2).sum(axis=0)}')
    print(f'chi_x {chi_x}')
    print(f'chi_x**2 {chi_x**2}')
    # Plot the result of the collpase with the given values of gamma and nu:
    plt.clf()
    for i in range(0,_N_len):
        tmp_chi_x=chi_x[i,:]
        plt.scatter(x,tmp_chi_x.detach().numpy(),color=colors[i])
    plt.xlim(-2,4)
    # Save the plot with the current critical exponents to see the collapse.
    plt.savefig('tmp_plots/'+str(epoch)+'.png', dpi=300)
    return 1/(uplim-lowlim)*integral


# Main function of the gradient descent from which all necessary subfunctions are started.
def gradient_descend():
    n_iterations=20
    learning_rate=0.1   # Step size of the gradient descent.
    lowlim=-0.1         # Setting up the range of the data collpase
    uplim=0.1
    print('total number of iteration: ',n_iterations, '   learning rate: ',learning_rate, 'x-interval: ', lowlim,uplim, 'integral steps: ', _beta_interpolated)
    # Iterations of the gradient descent:
    for epoch in range(n_iterations):
        print('-------------------------------')
        print('gradient descend     ',epoch/n_iterations*100,'%')
        # Calculate the current value of the loss function.
        l=loss(lowlim,uplim, epoch)
        l.backward()
        # Calculate the gradients of the loss function.
        dexp1=exp1.grad
        dexp2=exp2.grad
        print('dexp1=     ',dexp1)
        print('dexp2=     ',dexp2)
        # Move towards an assumed minimum.
        with torch.no_grad():
            globals()['exp2']-= learning_rate*dexp2
            globals()['exp1']-= learning_rate*dexp1
        # Zero the gradients to perform a new iteration step.
        exp2.grad.zero_()
        exp1.grad.zero_()

        # Output of the calculated critical exponents
        print(f'epoch {epoch+1}: \n loss={l}, \n exp1={exp1.item()},\n exp2={exp2.item()},\n T_c={T_c.item()}')



#---------------------------------------------
# Functions used to interpolate the chi values
#---------------------------------------------


# loading the data from the _reprocessed_ising_metropolis.csv
def initializing_arrays():
    print('Initializing arrays')
    print('Number of different lattices in data set: ',_N_len)
    print('Number of different beta per lattice in data set: ',_beta_len)

    j=-1
    # Initializing the  arrays that are needed for the approximation of the partition functions.
    for N in _N:    # Running over all the different lattice sizes.
        j=j+1
        i=-1
        # Preparing the meta_tot array.
        for beta in _beta:
            i=i+1
            # Preparing the meta_tot array.
            # meta stores the row values of meta_tot.
            meta[i,2]=beta
            tmp=df.loc[(df['beta']==beta) & (df['N']==N)]   # Get the number of independet measurements.
            meta[i,0]=tmp.shape[0]
            meta[i,1]=1                      # Assign initial value of Z.
            # Preparing the energy array:
            tmp_energy=df.loc[(df['beta']==beta) & (df['N']==N)]
            np_tmp_energy=tmp_energy['interpol_E'].to_numpy()

            # Preparing chi array:
            tmp_magnetization=df.loc[(df['beta']==beta) & (df['N']==N)]
            np_tmp_magnetization=tmp_magnetization['interpol_M'].to_numpy()

            # Writing energy and chi in to the energy and chi array:
            for z in range (0,n_block):
                energy[j,i,z]=np_tmp_energy[z]*N  # Multiplication with to make it extensive, necessary to do the interpolation.
                magnetization[j,i,z]=np_tmp_magnetization[z]*N
        meta_tot[j]=meta

# Loading the partition functions.
def loading_partition_functions():
    N=-1
    for l in _N:
        N=N+1
        df2=pd.read_csv('partition_function/Z_beta_N='+str(l)+'.csv',sep=';')
        _beta=df2['beta'].unique()
        i=-1
        for beta in _beta:
            i=i+1
            tmp=df2.loc[(df2['beta']==beta), ['Z']].to_numpy()
            meta_tot[N,i,1]=tmp    # Write the partition function value.
        print('Loaded data of lattice:      ', l)


#----------------------------------
# Calculate interpolated quantities
#----------------------------------

# Calculate the value of Z for any beta:
@numba.njit
def calculate_Z_beta(x,N):
    tmp=np.array([0.0])
    for i in range(0,_beta_len):        # Looping over all simulations = beta values
        for s in range(0,n_block):      # Looping ofer all states.
            denominator=np.array([0.0])
            for j in range(0,_beta_len):
                denominator=denominator+meta_tot[N,j,0]*(1/meta_tot[N,j,1])*np.exp((x-meta_tot[N,j,2])*energy[N,i,s])
            tmp=tmp+1/denominator
    Z_beta=tmp
    return Z_beta

# Calculate interpolated value of the internal energy:
def calculate_E_beta(x,N):
    tmp=0

    for i in range(0,_beta_len):        # Looping over all simulations = beta values
        print(i/_beta_len*100,' %')
        for s in range(0,n_block):      # Looping ofer all states.
            denominator=0
            for j in range(0,_beta_len):
                denominator=denominator+meta_tot[N,j,0]*(1/meta_tot[N,j,1])*np.exp((x-meta_tot[N,j,2])*energy[N,i,s])
            tmp=tmp+energy[N,i,s]/denominator
    E_beta=1/calculate_Z_beta(x,N)*tmp
    return E_beta



# Calculate interpolated value of M:
@numba.njit
def calculate_M_beta(x,N):
    tmp=np.array([0.0])
    for i in range(0,_beta_len):            # Looping over all simulations = beta values
            for s in range(0,n_block):      # Looping ofer all states.
                denominator=np.array([0.0])
                for j in range(0,_beta_len):
                    denominator=denominator+meta_tot[N,j,0]*(1/meta_tot[N,j,1])*np.exp((x-meta_tot[N,j,2])*energy[N,i,s])
                tmp=tmp+magnetization[N,i,s]/denominator
    M_beta=1/calculate_Z_beta(x,N)*tmp
    return np.abs(M_beta)


# Calculate interpolated value of M^2 for calulating the susceptibility chi.
@numba.njit
def calculate_Msquared_beta(x,N):
    tmp=np.array([0.0])
    for i in range(0,_beta_len):        # Looping over all simulations = beta values
        for s in range(0,n_block):      # Looping ofer all states.
            denominator=np.array([0.0])
            for j in range(0,_beta_len):
                denominator=denominator+meta_tot[N,j,0]*(1/meta_tot[N,j,1])*np.exp((x-meta_tot[N,j,2])*energy[N,i,s])
            tmp=tmp+magnetization[N,i,s]**2/denominator
    Msquared_beta=1/calculate_Z_beta(x,N)*tmp
    return Msquared_beta

# Calculate chi.
# the M values are extensive!!!
def calculate_chi_beta(x,N,N_sites):
    y=x.detach().numpy()
    print(y)
    M=calculate_M_beta(y,N)
    M_sqr=calculate_Msquared_beta(y,N)
    chi=x/N_sites*(torch.from_numpy(M_sqr)-torch.from_numpy(M**2))
    return chi


# Calculate a range of interpolated chi.
def interpolate_chi(N,N_sites):
    beta_range=torch.linspace(0.22,0.6, 100)
    chi_interpolated=calculate_chi_beta(beta_range,N,N_sites).numpy()
    df_chi=pd.DataFrame(columns=['beta','chi_interpolated'])
    df_chi['beta']=beta_range.tolist()
    df_chi['chi_interpolated']=chi_interpolated.tolist()
    df_chi.to_csv('interpolated_values/chi_interpolated_lattice='+str(N_sites)+'.csv', sep=';')

# Calculate a range of interpolated internal energies.
def interpolate_E(N,N_sites):
    beta_range=np.linspace(0.22,0.6, 100)
    E_interpolated=calculate_E_beta(beta_range,N)
    df_E=pd.DataFrame(columns=['beta','E_interpolated'])
    df_E['beta']=beta_range.tolist()
    df_E['E_interpolated']=E_interpolated.tolist()
    print(df_E)
    #df_E.to_csv('interpolated_values/E_interpolated_lattice='+str(N_sites)+'.csv', sep=';')

if __name__ == '__main__':
    main()
