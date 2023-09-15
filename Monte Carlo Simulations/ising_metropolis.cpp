/*
Description of the program:
---------------------------
This program performs a Markov Chain Monte Carlo simulation using the METROPOLIS algorithm for the ferromagnetic Ising Model (see Hamiltonian).
The sampling is done according to the Boltzmann distribution as necessary in equilibrium physics.
The program provides values of the internal energy u and the magnetization m of the system for the simulated samples for exactly one inverse temperature beta.
These values are output to a csv file (one file for each beta) and can be averaged over using a separate script (e.g. python script). Doing the averaging, the mean values of u and m can be obtained for the concrete beta value.
This program runs on the one, two or three dimensions (aequidistant chain, square lattice and cubic lattice).

In this program periodic boundary conditions and parallel spins as initial configuration were chosen.

The METROPOLIS algorithm:
--------------------
The Metropolis algorithm is a single spin flip algorithm. The overall idea is to choose one spin randomly and flip it with a certain probability depending on beta and the energy difference of both states.
Hereby one iteration consists of one sweep. One sweeps means that if the number of sites equals N, N spins have to be considered before one sweep is finished.
The number of iterations being the number of sweeps to be performed can be chosen individually.

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
- creation of need subdirectories ("raw_data_csv" and "snapshots")
- a script to calculate the mean values of u and m for each beta
*/

#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <string>


//some global variables:
const int J=1;              // spin coupling constant
int B=0;	 		              // external magnetic field
const int seed=32;		      // seed for the random number generator
double r=0.0;               // variable for the random numbers
double beta=0.0;            // inverse temperature beta=1/T with k_B=1
int intn_sites=0;           // number of sites

//global declaration of the rng using the mersenne twister
std::mt19937_64 eng(seed);
std::uniform_real_distribution<double> dist(0.0, 1.0);

//the n_sites vector stores the number of sites in each direction of the lattice
std::vector<int> n_sites; //n_i,n_j,n_k

//description: initialize the lattice with parallel aligned spins
//params:  reference to the lattice vector
//returns: /
void init_lattice(std::vector<std::vector<std::vector<int>>>& lattice)
{
	lattice.resize(n_sites.at(0));
	//looping over all the three dimensions of the nested lattice vector
	for (int i=0; i<n_sites.at(0); i++)
	{
		lattice.at(i).resize(n_sites.at(1));
		for (int j=0; j<n_sites.at(1); j++)
		{
			lattice.at(i).at(j).resize(n_sites.at(2));
			for (int k=0; k<n_sites.at(2); k++)
			{
				//setting the spin value of the specific lattice site
				lattice.at(i).at(j).at(k)=1;
			}
		}
	}
}


//description:	calculate the internal energy of the initial state in order to simply add Delta_E for calculating E
//params:	reference to the lattice
//returns:	initial energy
int initial_energy(std::vector<std::vector<std::vector<int>>>& lattice)
{
	int E_initial=0;
	int E_initial_B=0;

	//looping over all the lattice sites
	for (int i=0; i<n_sites.at(0); i++)
	{
		for (int j=0; j<n_sites.at(1); j++)
		{
			for (int k=0; k<n_sites.at(2); k++)
			{

				//positions of each next neighbour with periodic boundary conditions
				int k_plus_i=(i+lattice.size()+1)%lattice.size();
				int k_plus_j=(j+lattice.at(i).size()+1)%lattice.at(i).size();
				int k_plus_k=(k+lattice.at(i).at(j).size()+1)%lattice.at(i).at(j).size();

				if (lattice.size()>1)
					E_initial+=lattice.at(i).at(j).at(k)*lattice.at(k_plus_i).at(j).at(k);
				if (lattice.at(i).size()>1)
					E_initial+=lattice.at(i).at(j).at(k)*lattice.at(i).at(k_plus_j).at(k);
				if (lattice.at(i).at(j).size()>1)
					E_initial+=lattice.at(i).at(j).at(k)*lattice.at(i).at(j).at(k_plus_k);
				E_initial_B+=B*lattice.at(i).at(j).at(k);
			}
		}
	}

	std::cout << "Energy of the initial state mu_init= " <<  -1*(E_initial+E_initial_B) << std::endl;
	return (-1*(E_initial+E_initial_B));
}


//description:	calculate the magnetization of the initial state
//params:	reference to the lattice
//returns:	initial magnetization
int initial_magnetization(std::vector<std::vector<std::vector<int>>>& lattice)
{
	int M_initial=0;
	for (int i=0; i<n_sites.at(0); i++)		//looping over all lattice sites
	{
		for (int j=0; j<n_sites.at(1); j++)
		{
			for (int k=0; k<n_sites.at(2); k++)
			{
				M_initial+=lattice.at(i).at(j).at(k);
			}
		}
	}
	return(M_initial);
}

//description:	print the lattice in the terminal
//params:	reference to the lattice
//returns: /
void print(std::vector<std::vector<std::vector<int>>>& lattice)
{
	for (int i=0; i<n_sites.at(0); i++)
	{
		for (int j=0; j<n_sites.at(1); j++)
		{
			for (int k=0; k<n_sites.at(2); k++)
			{
				std::cout<<lattice.at(i).at(j).at(k) << ";";
			}
		}
	}
}


//description:	randomly select a lattice site to considered for spin flip
//params:	reference to the lattice
//returns:	lattice site as vector
std::vector<int> select_site(std::vector<std::vector<std::vector<int>>>& lattice)
{
	std::vector<int> selected_site;
	selected_site.resize(3);
	for (int m=0; m<3; m++)
	{
		r=dist(eng);
		selected_site.at(m)=floor(r*n_sites.at(m));
	}
	return selected_site;
}



//description:	calculate Delta_E for an assumed spin flip of the randomly selected site using periodic boundary conditions
//params:	reference to the lattice, randomly selected lattice site
//returns:	energy difference between flipped and unflipped state Delta_E
int Delta_E(std::vector<std::vector<std::vector<int>>>& lattice, std::vector<int> selected_site)
{
	int tmp_Delta_E=0;
	//positions of each nearest neighbour with periodic boundary conditions
	int i=selected_site.at(0);
	int j=selected_site.at(1);
	int k=selected_site.at(2);

	int k_plus_i=(i+lattice.size()+1)%lattice.size();
	int k_minus_i=(i+lattice.size()-1)%lattice.size();

	int k_plus_j=(j+lattice.at(i).size()+1)%lattice.at(i).size();
	int k_minus_j=(j+lattice.at(i).size()-1)%lattice.at(i).size();

	int k_plus_k=(k+lattice.at(i).at(j).size()+1)%lattice.at(i).at(j).size();
	int k_minus_k=(k+lattice.at(i).at(j).size()-1)%lattice.at(i).at(j).size();

	//calculate the differnce in energy
	if (lattice.size()>1)
		tmp_Delta_E+=2*J*lattice.at(i).at(j).at(k)*(lattice.at(k_plus_i).at(j).at(k)+lattice.at(k_minus_i).at(j).at(k));
	if (lattice.at(i).size()>1)
		tmp_Delta_E+=2*J*lattice.at(i).at(j).at(k)*(lattice.at(i).at(k_plus_j).at(k)+lattice.at(i).at(k_minus_j).at(k));
	if (lattice.at(i).at(j).size()>1)
		tmp_Delta_E+=2*J*lattice.at(i).at(j).at(k)*(lattice.at(i).at(j).at(k_plus_k)+lattice.at(i).at(j).at(k_minus_k));
	tmp_Delta_E+=(-2)*B*(-1)*lattice.at(i).at(j).at(k);
	return tmp_Delta_E;
}

//description:	accept or reject the spin flip according to a defined acceptance rule (Metropolis)
//params:	energy difference between flipped and unflipped state Delta_E, beta
//returns:	true for acceptance of spin flip
bool accept(int Delta_E, double beta)
{
	double A=0.0;
        r=dist(eng);
	//if the energy difference is negative, accept the spin flip, otherwise only accept it with a certain probability
	if (Delta_E <=0)
          return true;
        else
        {
          A=exp(-beta*Delta_E);
          if (r<A)
           return true;
          else
            return false;
        }

}


//description:	Writes data to a csv file.
//params:	reference to the lattice, vector of all calculated system energies, number of lattice sites, number of iterations, vector of all calculated magnetizations
//returns:	true if data were successfully written to the csv file
bool publish_data(std::vector<std::vector<std::vector<int>>>& lattice, std::vector<int>& energy, int intn_sites, int n_iterations, std::vector<int>& magnetization)
{
	bool ret=true;
	std::string filename="";
	filename="raw_data_csv/lattice_"+std::to_string(n_sites.at(0))+"_"+std::to_string(n_sites.at(1))+"_"+std::to_string(n_sites.at(2))+"_iterations_"+std::to_string(n_iterations)+"_"+std::to_string(beta)+"_energy.csv";
	try
	{
		std::ofstream file(filename, std::ios::trunc);

		//writing a header file that contains the value of beta and all parameters that appear in the Hamiltonian
		file << "n;mu_E;mu_M;beta;J;B;N"  << std::endl;
		for (int i=0; i<energy.size(); i++)
		{
			file << i << ";" << (double)energy.at(i)/(double)intn_sites << ";" << (double)magnetization.at(i)/(double)intn_sites <<";" << beta << ";" << J << ";" << B << ";" << intn_sites << std::endl;
		}
		file.close();
		throw (filename);
	}
	catch (std::string filename)
	{
		 ret=false;
	}
	return ret;
}

//description:	main function containing the the main logic of the Metropolis algorithm as described in the header
//params: number of lattice sites in direction 1, number of lattice sites in direction 2, number of lattice sites in direction 3, number of iterations, beta, external magnetic field
//returns: /
int main(int argc, char *argv[])
{
	//variable declations and initializing
	//nested lattice vector storing the spins
	std::vector<std::vector<std::vector<int>>> lattice;
	//vector storing the coordinates of the randomly selected site
	std::vector<int> tmp_selected_site;
	//vector storing the energy and magnetization values to be written to a file
	std::vector<int> energy;
	std::vector<int> magnetization;

	//initializing variables with the terminal input
	n_sites.resize(3);
	tmp_selected_site.resize(3);
	int n_iterations=std::stoi(argv[4]);
	beta=std::stod(argv[5]);
	B=std::stoi(argv[6]);
	for (int i=1; i<=3; i++)
	{
		n_sites.at(i-1)=std::stoi(argv[i]);
	}
	intn_sites=n_sites.at(0)*n_sites.at(1)*n_sites.at(2);

	//initialization of the lattice
	init_lattice(lattice);
	//initialize and calculate the initial internal energy and the initial magnetization
	int tmp_energy=initial_energy(lattice);
	int tmp_magnetization=initial_magnetization(lattice);

	std::cout << n_iterations << " , " << n_sites.at(1) << " , " << beta << std::endl;


	//******************************//
  //actual metropolis calculations//
  //******************************//

	//counting the accepted spin flips
	int tmp_count=0;
	//looping over n_iterations sweeps
	for (int j=1;j<=n_iterations; j++)
  {
		//performing the sweeps
		for (int k=1; k<=intn_sites; k++) //intn_sites
    	{
				//randomly select one lattice site
      	tmp_selected_site=select_site(lattice);
				int tmp=0;
				//calculate the differnce in energy of both states the current state and the subsequent state with flipped spin
      	tmp=Delta_E(lattice, tmp_selected_site);
				//perform the acceptance rule of the Metropolis algorithm
				if (accept(tmp, beta))
     			{
						tmp_count+=1;
						//flip the specific spin
       			lattice.at(tmp_selected_site.at(0)).at(tmp_selected_site.at(1)).at(tmp_selected_site.at(2))*=-1;
						//calculate the energy of the created state
						tmp_energy+=tmp;
						//calculate the magnetization of the created state
						tmp_magnetization+=2*lattice.at(tmp_selected_site.at(0)).at(tmp_selected_site.at(1)).at(tmp_selected_site.at(2));
        	}
      }
		//save the energy and the magnetization after each sweep
		energy.push_back(tmp_energy);
		magnetization.push_back(tmp_magnetization);
	}
	//write the date to the csv file
	publish_data(lattice, energy, intn_sites, n_iterations, magnetization);

	return 0;
}
