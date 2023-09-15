# This script creates a list of nearest neighbours for the 2D square lattice or the triangular lattice (2D).
# The list of nearest neighbours of the triangular lattice is generated based in the square lattice.
import numpy as np
import pandas as pd
import sys

# linear lattice size
L=int(sys.argv[1])
# create nearest neighbours list for square or triangular lattice
lattice_type=sys.argv[2]
if lattice_type=='triangular':
     no_near_neigh=6
if lattice_type=='square':
     no_near_neigh=4

# initialize lattice and nearest neighours array
lattice=np.zeros((L,L), dtype=np.int32)
nn_list=np.zeros((L**2,no_near_neigh), dtype=np.int32)

#main function
def main():
    generate_sq_lat()
    if lattice_type=='triangular':
        convert_to_triangular_lat()
    write_nn_list_to_file(lattice_type);
    #up to now list of nn for a square lattice:

#generate for square lattice
def generate_sq_lat():
    for i in range(0,L):
        for j in range(0,L):
            lattice[i,j]=i*L+j
    k=0
    for i in range(0,L):
        for j in range(0,L):
            add1=lattice[(i+1+L)%L,j]
            add2=lattice[i, (j+1+L)%L]
            add3=lattice[(i-1+L)%L,j]
            add4=lattice[i, (j-1+L)%L]
            if no_near_neigh==6:
                tmp=np.array([add1, add3, add2, add4, -1,-1]) #up, down, right, left
            if no_near_neigh==4:
                tmp=np.array([add1, add3, add2, add4]) #up, down, right, left
            nn_list[k]=tmp
            k+=1

# generate for triangular lattice
# the generation of these triangular lattices only works for even levels
def convert_to_triangular_lat():

    for i in range(0,L):  # looping over all the lines in the list of nn.
        for k in range(0,L):
            if i%2==0:  # do the following for every second line.
                index=L*i+k  #index of the lattice site to be considered.
                up1=int(nn_list[index,0]) #get the up nn in the list.
                print(f'up1 {up1}')
                down1=int(nn_list[index,1]) # get the down nn in the list.
                print(f'down1 {down1}')
                jup=i+1
                jdown=i-1
                if i==0:
                    jup=i+1
                    jdown=L-1
                if i==(L-1):
                    jup=0
                    jdown=i-1
                up2=((up1%L)+L-1)%L+L*jup
                print(f'up2 {up2}')
                down2=((down1%L)+L-1)%L+L*jdown
                print(f' down2 {down2}')
                nn_list[index, 4]=up2
                nn_list[index, 5]=down2
                nn_list[up2, 4]=index
                nn_list[down2, 5]=index


# frite the list of nearrest neighbours to csv file
def write_nn_list_to_file(lattice_type):
    tmpdf=pd.DataFrame(data=[nn_list[0]])
    for k in range(1,L**2):
        tmp=pd.DataFrame(data=[nn_list[k]])
        tmpdf=pd.concat([tmpdf, tmp], sort=False)
    tmpdf.to_csv(str(L)+'x'+str(L)+lattice_type+'.csv', sep=';', encoding='utf-8', header=False, index=False)

if __name__=='__main__':
    main()
