/*
Description of the program:
---------------------------
This program performs a Markov Chain Monte Carlo simulation using the WOLFF algorithm for the ferromagnetic Ising Model (see Hamiltonian).
The sampling is done according to the Boltzmann distribution as necessary in equilibrium physics.
The program provides values of the internal energy u and the magnetization m of the system for the simulated samples for exactly one inverse temperature beta.
These values are output to a csv file (one file for each beta) and can be averaged over using a separate script (e.g. python script). Doing the averaging, the mean values of u and m can be obtained for the concrete beta value.
This program can be run for any lattice geometry in any dimension provided that there is a list of nearest neighbours (see "lattice_generator.py" and the dummy files "30x30....csv").

In this program periodic boundary conditions and parallel spins as initial configuration were chosen.

The WOLFF algorithm:
--------------------
The Wolff algorithm is a cluster spin flip algorithm. The overall idea is selecting a spin on the lattice randomly and creating a cluster of parallel aligned spin around this previously selected spin.
The cluster is created using a certain probabilty to add parallel spins to the cluster. When the cluster creation is finished, the entire spins of the cluster are flipped. This process is one iteration step.


Hamiltonian of the ferromagnetic Ising Model:
---------------------------------------------
$H=- \sum_{i, j} J s^z_i s^z_j - B_z \sum_i s^z_i$
(for the algorithm provided, J=1 and k_B=1)

Literature:
-----------
M.E.J. Newman and G.T. Barkema. Monte Carlo Methods in Statistical Physics. Clarendon Press Oxford, 1999.
(There is an entire chapter on the Wolff algorithm.)

Additional requirements:
------------------------
- list of nearest neighbours for the specific lattice of interst
- creation of need subdirectories ("raw_data_csv" and "snapshots")
- a script to calculate the mean values of u and m for each beta

*/

#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>

//some global variables:
const int J=1;              // spin coupling constant
int B=0;	 		              // external magnetic field
const int seed=7204;		    // seed for the random number generator
double r=0.0;               // variable for the random numbers
double beta=0.0;            // inverse temperature beta=1/T with k_B=1
int intn_sites=0;           // number of sites
std::vector<int> lattice;   // vector storing the spins on the lattice spin \in {-1,1}
std::vector<int> cluster;   // list storing the tuple sites in the cluster
std::vector< std::vector<int> > nearest_neighbours; //list of nearest neighbours dim1=pos of site in lattice, dim2 neighbour coordinate

//global declaration of the rng using the mersenne twister
std::mt19937_64 eng(seed);
std::uniform_real_distribution<double> dist(0.0, 1.0);

int n_sites=0; //n_i,n_j,n_k

//description: this function reads the nearest neighbour list in the file and fills the nearest_neighbours vector
//params: filename of the nearest neighbour list
//returns: /
void load_nearest_neighbours(std::string filename)
{
        std::ifstream read_csv;
        read_csv.open(filename, std::ios::in);
        if (read_csv) //read entire file
        {
                std::string s="";
                std::string t="";
                std::stringstream strstream;
                int i=0;
                while(std::getline(read_csv,s))
                {
                        strstream.clear();
                        strstream << s;
			                  int j=0;
                        while (std::getline(strstream, t, ';'))
                        {
                                int tmp=std::stoi(t);
                                nearest_neighbours.at(i).push_back(tmp);
                        }
                        i++;
                }
        }
        read_csv.close();
        //filling the nearest_neighbours vector
	      for (int i=0; i<intn_sites; i++)
	      {
		        for (int k=0; k<nearest_neighbours.at(i).size(); k++)
		        {
			           std::cout << nearest_neighbours.at(i).at(k) << std::endl;
		        }
		        std::cout<< " " << std::endl;
	      }
}


//for using the nearest neighbour list, a one-dimensional array can be used instead of a nested array. The information of the lattice structure is provided by the nearest_neighbours list.
//description: Initialize the lattice with either with parallel spins
//params: / (lattice is a global vector)
//returns: /
void init_lattice()
{
	lattice.resize(n_sites);
	for (int i=0; i<n_sites; i++)
	{
		lattice.at(i)=1;
	}
}


//description:	calculate the internal energy
//params:	/ (lattice is a global vector)
//returns:	internal energy
double initial_energy()
{
  // initial temporary variable to store the two contribution to the energy as can be seen by the Hamiltonian
	double E_initial=0.0;
	double E_initial_B=0.0;

	//looping over all sites
	for (int i=0; i<n_sites; i++)
	{
    //looping over the nearest neighbours counting each pair of nearest neighbours twice
		for (int j=0; j<nearest_neighbours.at(i).size(); j++)
		    E_initial+=lattice.at(i)*lattice.at(nearest_neighbours.at(i).at(j));
		E_initial_B+=B*lattice.at(i);
	}
	//since each inteaction of two sites is counted twice, it must be divided by two
	E_initial=E_initial/2;

	return (-1*(E_initial+E_initial_B));
}

//description:	returns the magnetization of the current configuration
//params: / (lattice is a global vector)
//returns:	magnetization
int initial_magnetization()
{
	int M_initial=0;
  //looping over all lattice sites
	for (int i=0; i<n_sites; i++)
	{
		M_initial+=lattice.at(i);
	}
	return(M_initial);
}

//description:	Prints the lattice in the terminal
//params:	/ (lattice is a global vector)
//returns: /
void print()
{
	for (int i=0; i<n_sites; i++)
	{
		std::cout<<lattice.at(i) << ";";
	}
}


//description:	Randomly select a lattice site to considered for the spin flip
//params:	/ (lattice is a global vector)
//returns:	randomly selected lattice site as integer
int select_site()
{
	int selected_site=0;
	r=dist(eng);
	selected_site=floor(r*n_sites);
	return selected_site;
}



//description: writes data to a csv file
//params:	vector storing all calculated internal energies,  number of lattice sites, number of iterations, vector storing all calculated magnetizations
//returns:	true if data were successfully written to the csv file
bool publish_data(std::vector<int>& energy, int intn_sites, int n_iterations, std::vector<int>& magnetization)
{
	bool ret=true;
	std::string filename="";
  //write the data to a file in a "raw_data_csv" directory
	filename="raw_data_csv/lattice_"+std::to_string(n_sites)+"_iterations_"+std::to_string(n_iterations)+"_beta"+std::to_string(beta)+"_B"+std::to_string(B)+"_J"+std::to_string(J)+"_wolf_algorithm.csv";
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

//description: this function saves snapshots of the lattice spin configuration of the system to a csv file in the "snapshots" directory
//params: number of lattice sites, number of iterations, current index of iterations
//returns: /
void save_snapshots(int intn_sites,int n_iterations, int k)
{
	bool ret=true;
	std::cout<< k<< std::endl;
	std::string filename="";
	filename="snapshots/"+std::to_string(k)+"snap_lattice_"+std::to_string(beta)+"_"+std::to_string(n_sites)+"_iterations_"+std::to_string(n_iterations)+"_beta"+std::to_string(beta)+"_B"+std::to_string(B)+"_J"+std::to_string(J)+"_wolf_algorithm.csv";
	try
	{
		std::ofstream file(filename,std::ios::trunc);
		file << beta << std::endl;
		for (int i=0; i<n_sites; i++)
	        {
                        file<<lattice.at(i)<< ";";
	        }
		file.close();
	}
	catch (std::string filename)
	{
		ret=false;
	}

}

//description: this function checks whether a certain lattice site i is already in the cluster.
//params: lattice coordinate to be checked.
//returns: true if the site is already in the cluster.
bool is_site_in_cluster(int i)
{
	bool ret=false;
	for (int k=0; k<cluster.size(); k++)
	{
		if (cluster.at(k)==i)
		{
			ret=true;
		}
	}
	return ret;
}

//description: this function checks all the nearest neighbours of a given site i. If the neighbouring spins are parallel, it is checked whether they are already in the cluster. If not, they are added with a certain probabilty P_add.
//params: lattice coordinate of the site which neighbours must be checked.
//returns: /

void check_neighbours(int i)
{
  // looping over all nearest neighbours of lattice site i
	for (int k=0; k<nearest_neighbours.at(i).size(); k++)
	{
    // checking whether spins are parallel
		if (lattice.at(i)==lattice.at(nearest_neighbours.at(i).at(k)))
		{
      //checking whether neighbouring site is already in the lattice
			if (is_site_in_cluster(nearest_neighbours.at(i).at(k))==false)
			{
				int tmp_cluster_size = cluster.size();
				r=dist(eng);
        //add parallel spins on nearest neighbouring site with probabilty P_add(beta)
				double P_add=(1-exp(-2*beta*J));
				if (r<P_add)
				{
					cluster.resize(tmp_cluster_size+1);
					cluster.at(tmp_cluster_size)=nearest_neighbours.at(i).at(k);
				}
			}
		}
	}
}


//description: main function of the cluster generation logic.
//params: randomly selected site
//returns: /
void get_cluster(int tmp_selected_site)
{
	int i=tmp_selected_site;
	std::cout << i << std::endl;
	// add randomly selected site to cluster list
	cluster.resize(1);
	cluster.at(0)=i;

	int c=0;	// counter for the while loop and index of the cluster vector
	while (c < cluster.size())
	{
		check_neighbours(cluster.at(c));
		c++;
	}

}

//description: having finished the cluster creation this function is used to flip all the spins in the cluster
//params: /
//retuns: /
void flip_cluster()
{
	for (int i=0; i<cluster.size(); i++)
	{
		lattice.at(cluster.at(i))*=-1;
	}
	cluster.clear();
}

//description:	main function containing the the main logic of the Wolff algorithm as described in the header
//params: number of sites, number of iterations for the main loop of the Wolff algorithm, beta value, external magnetic field, filename of the file storing the nearest neighbours
//returns: /
int main(int argc, char *argv[])
{
	//variable declarations and initializing
	int tmp_selected_site;
	std::vector<int> energy;
	std::vector<int> magnetization;
	std::string filename=argv[5];
	int n_iterations=std::stoi(argv[2]);
	beta=std::stod(argv[3]);
	B=std::stoi(argv[4]);
	n_sites=std::stoi(argv[1]);
	intn_sites=n_sites;
	nearest_neighbours.resize(n_sites);

  //load list of nearest neighbours
	load_nearest_neighbours(filename);
  //initialize the lattice
	init_lattice();
	std::cout << "initializing done" << std::endl;
  //calculate the initial internal energy of the system in the current spin configuration
  double tmp_energy=initial_energy();
  //calculate the magnetization of the system in the current spin configuration
	double tmp_magnetization=initial_magnetization();
	std::cout << "energy and magnetization calculated" << std::endl;

  //******************************
  //    actual Wolf Algorithm   //
  //******************************

	int tmp_count=0;
  //major for loop doing a certain number of iterations as provided by the terminal launching the program
	for (int j=1;j<=n_iterations; j++)
	{
		std::cout << "iteration " << j << std::endl;
    //the following for loop is no real for loop. For the Metropolis algorithm e.g. the following for loop is necessary to perform sweeps. It is therefore left here, as the Wolff algorithm might also be run with sweeps
    //However, this would take more computational efforts reducing the speed of the algorithm. For most cases the Wolff algorithm without sweeps yields good results anyway.
    for (int k=1; k<=1; k++) //intn_sites
		{
			std::cout << "inner iteration " << k << std::endl;
      //randomly select lattice site
      tmp_selected_site=select_site();
			std::cout << "starting get cluster" << std::endl;
      //create the cluster of parallel aligned spins around the selected site
			get_cluster(tmp_selected_site);
			std::cout << "flippting spin" << std::endl;
      //having created the cluster, flip it
			flip_cluster();
		}
    //calculate the quantites of interest (internal energy u and magnetization m)
		std::cout << "calculating energy and magnetization" << std::endl;
		tmp_energy=initial_energy();
		tmp_magnetization=initial_magnetization();
		std::cout << "done calculating energy and magnetization" << std::endl;
    //store the calculated values for u and m in a vector to be written to a file later
		energy.push_back(tmp_energy);
		magnetization.push_back(tmp_magnetization);
		//The following if conditions eneables the algorithm to save snapshots of the lattice spin configuration. However, for the calculation of mean values, this is not necessary.
		//if (j>199 && j%100==0)
		//	save_snapshots(intn_sites, n_iterations,j);
	}
  //write the data to a csv file
	publish_data(energy, intn_sites, n_iterations, magnetization);
	return 0;
}
