'''
Description of the program:
---------------------------
This program performs the final data manipulation of the output data of the Metropolis and the Wolff algorithm to calculate mean values and errors of:
- internal energy u
- specific heat c
- magnetization m
- magnetic susceptibility chi
Hereby it uses the so-called blocking method (parameters to be adjusted manually: cut_off and n_block)

The blocking method:
--------------------
The blocking method is used to uncorrelate the measured data of the internal energy and the magnetization.
The values are divided into blocks. Then mean values of each block are calculated. Finally, the mean value of the block mean values and the standard deviations
are calcualted to have an error estimation. For a detailed description see Newmann et al.

Hamiltonian of the ferromagnetic Ising Model:
---------------------------------------------
H=- \sum_{i, j} J s^z_i s^z_j - B_z \sum_i s^z_i
(for the algorithm provided, J=1 and k_B=1)

Literature:
-----------
M.E.J. Newman and G.T. Barkema. Monte Carlo Methods in Statistical Physics. Clarendon Press Oxford, 1999.
(There is an entire chapter on the Metropolis algorithm.)

Additional requirements:
------------------------
- creation of needed subdirectories ("raw_data_csv" and "processed_data_csv")

'''
import sys
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
import glob

#define parameters for the blocking method
cut_off=200
n_block=20
path="raw_data_csv/"


filenames=glob.glob(path+"/*.csv")
print(filenames)
#create a pandas dataframe to process the data that were generated by the Monte Carlo simulation
df=pd.DataFrame(columns=['mu_E','sigma_E','mu_M','sigma_M','beta','J','B','c','sigma_c', 'chi','sigma_chi','N'])

#data manipulation to be executed on every file
for file in filenames:
    #initializing major variables
    beta=0.0    #inverse temperature
    J=0         #spin coupling
    B=0         #external magnetic field
    N=0         #number of sites
    #read the csv file that was yielded by the Monte Carlo simulation.
    tmp_df=pd.read_csv(file,sep=';')
    #assign the values to the correct variables and create numpy array to calculate with the data
    beta=float(tmp_df.at[1, "beta"])
    B=tmp_df.at[0, "B"]
    J=tmp_df.at[0, "J"]
    N=tmp_df.at[0, "N"]
    tmp_arr_n=tmp_df['n'].to_numpy()
    tmp_arr_mu_E=tmp_df['mu_E'].to_numpy() # per site
    tmp_arr_mu_M=tmp_df['mu_M'].to_numpy() # per site

    #cut-off the thermalization values that both the Wolff and the Metropolis algorithm produce until they come to equilibrium sampling with the Boltzmann distribution
    mc_steps=tmp_arr_n[cut_off:]
    energy=tmp_arr_mu_E[cut_off:]
    magnetization=np.abs(tmp_arr_mu_M[cut_off:])

    #using the blocking method:
    #divide the values into sets and calculate the statistical error and the mean value
    tmp_block_energy=np.array_split(energy, n_block)
    block_energy=np.array(tmp_block_energy)
    tmp_block_magnetization=np.array_split(magnetization,n_block)
    block_magnetization=np.array(tmp_block_magnetization)


    # calculate the specific heat c  for each block, derived by eq. 3.15 (Newmann et al.); the factor N necessary, since I load the energy per site of the csv
    block_c=N*beta**2*(np.array(np.mean(block_energy**2, axis=1))-np.array(np.mean(block_energy, axis=1)**2)) #per site

    #calculate mean value and standard deviation of the specific heat
    c=np.mean(block_c, axis=0)
    std_c=1/(np.sqrt(n_block-1))*np.std(block_c) #calculate the error

    #calculate the magnetic susceptibility chi for each block
    block_chi=beta*N*(np.array(np.mean(block_magnetization**2, axis=1))-np.array(np.mean(block_magnetization, axis=1)**2)) #per site

    #calculate mean value and standard deviation of the magnetic susceptibility
    chi=np.mean(block_chi, axis=0)
    std_chi=1/(np.sqrt(n_block-1))*np.std(block_chi) #calculate the error

    #calculate mean value and standard deviation of the magnetization
    mean_magnetization=np.mean(np.array(np.mean(block_magnetization, axis=1))) #calculate mean of mean value of each block
    std_magnetization=1/(np.sqrt(n_block-1))*np.std(np.array(np.mean(block_magnetization, axis=1)), axis=0) #calculate the error

    #calculate mean value and standard deviation of the energy
    mean_energy=np.mean(np.array(np.mean(block_energy, axis=1)))
    std_energy=1/(np.sqrt(n_block-1))*np.std(np.array(np.mean(block_energy, axis=1)), axis=0)

    #store the calculated mean values ans standard deviations
    add=pd.DataFrame([[mean_energy,std_energy, mean_magnetization, std_magnetization,beta, J, B,c,std_c, chi, std_chi,N]],columns=['mu_E','sigma_E','mu_M','sigma_M','beta', 'J', 'B', 'c', 'sigma_c','chi','sigma_chi','N'])
    df=pd.concat([df,add], sort=False)

#write mean value data frame to the a file
df.to_csv("processed_data_csv/ising_monte_carlo_results.csv", sep=';', encoding='utf-8', mode='a', header=True)

#write a readme file that contains the relevant informatiion concerning the simulated data
with open("processed_data_csv/readme.txt", "w") as f:
    f.write("Used simulation model: ferromagnetic Ising Model with ???? (add the algorithm) algorithm for nearest neighbours\n")
    f.write("Initial state: parallel spins\n")
    f.write("B="+str(B)+"\n")
    f.write("J="+str(J)+"\n")
    f.write("k_b=1"+"\n")
    f.write("cut_off="+str(cut_off)+"\n")
    f.write("n_block="+str(n_block)+"\n")
    f.write("The data evaluation has been done by using the blocking method")
