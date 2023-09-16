'''
Description of the program:
---------------------------
This scripts interpolates the internal energy u, the magnetization m and the magnetic susceptibility chi based on the Monte Carlo simulation data sets for the Ising Model.
Basically, this program takes the same data as the "partition_function.py" script, i.e. the "_reprocessed_data_wolf.csv" file, including the Z_beta_N=##.csv" files for each lattice size.

Required data:
--------------
- "_reprocessed_data_wolf.csv" (the same file that was used for the calculation of the dicrete Z_i corresponding to the beta_i).
- "Z_beta_N=##.csv" file for each number of lattice sites N as was calculated in the "partition_function.py" and manually saved (there is a dummy file for this in the repository).


Literature:
-----------
M.E.J. Newman and G.T. Barkema. Monte Carlo Methods in Statistical Physics. Clarendon Press Oxford, 1999.
(There is an entire chapter on the multiple histogram method.)
'''


import numpy as np
import pandas as pd
import os
import numba
import torch

# Define the source directory. This is the directory where the reprocessed data ("_reprocessed_data_wolf.csv") file is stored.
src=""
df=pd.read_csv(src+'_reprocessed_ising_wolf.csv', sep=';')
# Define the source directory where the "Z_beta_N=##.csv" files are to be found. Again, there is one such file for each N.
src_part=""


#------------------------------------------------------------
# Loading data from csv and creating array as described above
#------------------------------------------------------------
# This part is nearly the same as in "partition_function.py".

# Alter the elements of the list so that it suits the list of beta values that the Monte Carlo algorithm was run for.
# THE LIST OF beta VALUES MUST BE THE SAME AS FOR PARTITION FUNCTION CALCULATION.
_beta=np.array([0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6])
# Provide the number of lattice sites that Monte Carlo simulations were run on (as in the csv file).
# THE LIST OF N VALUES MUST BE THE SAME AS FOR PARTITION FUNCTION CALCULATION.
_N=np.array([25,100,225,400,900])
print(_N)
_beta_len=(_beta.shape[0])

n_block=9800
_N_len=(_N.shape[0])
print('Number of different lattices in data set: ',_N_len)
print('Number of different beta per lattice in data set: ',_beta_len)

# Create the arrays necessary to store all values. These are the same in structure and name as used in the "partition_function.py" script.
meta=np.ones((_beta_len,3))
meta_tot=np.ones((_N_len,_beta_len,3))
energy=np.zeros((_N_len,_beta_len,n_block))
magnetization=np.zeros((_N_len,_beta_len,n_block))

j=-1
A=1
# Initializing the  arrays that are needed for the approximation of the partition functions
for N in _N:    # Running over all the different lattice sizes.
    j=j+1
    i=-1

    for beta in _beta:  # Running over all inverse temperatures beta.
        i=i+1
        # Preparing the meta_tot array.
        # meta stores the row values of meta_tot.
        meta[i,2]=beta
        tmp=df.loc[(df['beta']==beta) & (df['N']==N)]   # Get the number of independet measurements.
        meta[i,0]=tmp.shape[0]
        meta[i,1]=1                      # Assign initial value of Z.
        # Preparing the energy array:
        tmp_energy=df.loc[(df['beta']==beta) & (df['N']==N)]
        np_tmp_energy=tmp_energy['mu_E'].to_numpy()

        # Preparing chi array:
        tmp_magnetization=df.loc[(df['beta']==beta) & (df['N']==N)]
        np_tmp_magnetization=tmp_magnetization['mu_M'].to_numpy()

        # Writing energy and chi in to the energy and chi array:
        for z in range (0,n_block):
            energy[j,i,z]=np_tmp_energy[z]*N  # Multiplication with to make it extensive, necessary to do the interpolation.
            magnetization[j,i,z]=np.abs(np_tmp_magnetization[z])*N
    meta_tot[j]=meta
print(meta_tot)



#-------------------------------------------------------------------------
# Read the partition function data from csv if a "Z_beta_N=##.csv" exists.
#-------------------------------------------------------------------------


N=-1
for l in _N:    # Loop over all lattice sizes.
    N=N+1
    if os.path.exists(src_part+'partition_function/Z_beta_N='+str(l)+'.csv'):
        df2=pd.read_csv(src_part+'partition_function/Z_beta_N='+str(l)+'.csv',sep=';')
        _beta=df2['beta'].unique()
        i=-1
        for beta in _beta:
            i=i+1
            tmp=df2.loc[(df2['beta']==beta), ['Z']].to_numpy()
            meta_tot[N,i,1]=tmp    #write the partition function value
        print('Loaded data of lattice', l)
    else:
        print('please calculate the partition functions first')

#------------------------------------------------------------
# Calculate the value of Z_beta
#------------------------------------------------------------

# Calculate the value of Z for any desired beta.
@numba.njit
def calculate_Z_beta(x,N):
    tmp=np.array([0.0])
    for i in range(0,_beta_len):     # Looping over all simulations = beta values
        for s in range(0,n_block):   # Looping ofer all states.
            denominator=np.array([0.0])
            for j in range(0,_beta_len):
                denominator=denominator+meta_tot[N,j,0]*(1/meta_tot[N,j,1])*np.exp((x-meta_tot[N,j,2])*energy[N,i,s])
            tmp=tmp+1/denominator
    Z_beta=tmp
    return Z_beta


# Calculate interpolated value of the internal energy.
@numba.njit
def calculate_E_beta(x,N):
    tmp=np.array([0.0])

    for i in range(0,_beta_len):        # Looping over all simulations = beta values
        for s in range(0,n_block):      # Looping ofer all states.
            denominator=np.array([0.0])
            for j in range(0,_beta_len):
                denominator=denominator+meta_tot[N,j,0]*(1/meta_tot[N,j,1])*np.exp((x-meta_tot[N,j,2])*energy[N,i,s])
            tmp=tmp+energy[N,i,s]/denominator
    E_beta=1/calculate_Z_beta(x,N)*tmp
    return E_beta



# Calculate interpolated value of M.
@numba.njit
def calculate_M_beta(x,N):
    tmp=np.array([0.0])
    for i in range(0,_beta_len):        # Looping over all simulations = beta values
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
    M=calculate_M_beta(y,N)
    M_sqr=calculate_Msquared_beta(y,N)
    chi=x/N_sites*(torch.from_numpy(M_sqr)-torch.from_numpy(M**2))
    return chi


# Calculate a range of interpolated chi.
def interpolate_chi(N,N_sites):
    beta_range=torch.linspace(0.0,0.8, 100)
    chi_interpolated=calculate_chi_beta(beta_range,N,N_sites).numpy()
    df_chi=pd.DataFrame(columns=['beta','chi_interpolated'])
    df_chi['beta']=beta_range.tolist()
    df_chi['chi_interpolated']=chi_interpolated.tolist()
    df_chi.to_csv(src_part+'interpolated_values/chi_interpolated_lattice='+str(N_sites)+'.csv', sep=';')

# Calculate a range of interpolated internal energies.
def interpolate_E(N,N_sites):
    beta_range=np.linspace(0.0,0.8, 100)
    E_interpolated=calculate_E_beta(beta_range,N)
    df_E=pd.DataFrame(columns=['beta','E_interpolated'])
    df_E['beta']=beta_range.tolist()
    df_E['E_interpolated']=E_interpolated.tolist()
    print(df_E)
    df_E.to_csv(src_part+'interpolated_values/E_interpolated_lattice='+str(N_sites)+'.csv', sep=';')
