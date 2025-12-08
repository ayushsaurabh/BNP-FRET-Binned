# BNP-FRET-Binned: Analyze binned smFRET data in a Bayesian nonparametrics (BNP) paradigm

BNP-FRET is a suite of software tools to analyze binned smFRET data collected using sCMOS and EMCCD detectors. It implements Markov chain Monte Carlo (MCMC) algorithms to learn distributions over parameters of interest: the number of states a biomolecule transitions through and associated transition rates. These tools can be used in a simple plug and play manner. Check the following set of papers to see details of all the mathematics involved in the development of BNP-FRET:

https://biorxiv.org/cgi/content/short/2022.07.20.500887v1

## Julia Installation

All the codes are written in Julia language for high performance/speed (similar to C and Fortran) and its open-source/free availability. Julia also allows easy parallelization of all the codes. To install julia, please download and install julia language installer from their official website (see below) for your operating system or use your package manager. The current version of the code has been successfully tested on linux (Ubuntu 24.04.1 LTS), macOS 12, and Windows.

https://julialang.org/

Like python, Julia also has an interactive environment (commonly known as REPL) which can be used to add packages and perform simple tests as shown in the picture below.


![Screenshot from 2025-06-13 04-52-30](https://github.com/user-attachments/assets/156c0e05-59a7-4f79-92ed-52cda9415747)


In Windows, this interactive environment can be started by clicking on the Julia icon on Desktop that is created upon installation or by going into the programs menu directly. On Linux or macOS machines, julia REPL can be accessed by simply typing julia in the terminal. We use this environment to install some essential julia packages that help simplify linear algebra and statistical calculations, and plotting. To add these packages via julia REPL, **first enter the julia package manager by executing `]` command in the REPL**. Then simply execute the following command to install all these packages at the same time.

```add Distributions SpecialFunctions LinearAlgebra Statistics StatsBase Plots HDF5```

Also, see the image below for an example of the package installation process in julia REPL.


![Screenshot from 2025-06-13 04-49-47](https://github.com/user-attachments/assets/fae7ecc4-9e54-4bf8-ad12-daaf1420822f)

**To get out of the package manager, simply hit the backspace key.**

### Environment Creation
**This is for advanced users who already have Julia installed.**
If you already have Julia and do not want to alter your default environment, you can go to the directory where this software is, then 
1. Run Julia then type `]` and `activate .`;
2. Or run Julia in terminal via `julia --project`.
   
These two ways are equivalent. Both of them create a new Julia environment the first time you run it, or otherwise switch to this environment.

## Code organization
BNP-FRET is organized in such a way that all the user input is accomplished via the "input_parameters.jl" file. It can be used to provide the file name for FRET data, camera sensitivity parameters, crosstalk probabilities, plotting options, and sampler options as shown below:

![Screenshot from 2025-06-13 05-06-22](https://github.com/user-attachments/assets/16e51112-acc2-49f2-a4a1-155cb246c622)

![Screenshot from 2025-06-13 05-06-30](https://github.com/user-attachments/assets/22dc8fc0-11d5-445b-aab4-fc40586c2c8d)

Furthermore, the functions used to perform all the computations are organized in a hierarchical structure. The file "main.jl" contains the main sampler. All the functions called in that file are written in the file "functions_layer_1.jl". Similarly, all the functions called in the file "functions_layer_1.jl" are in file "functions_layer_2.jl". Any new set of functions can be introduced by following the same heirarchical structure. A brief description of all the functions is given below in the sequence they are called:

1. get_FRET_data(): Used to obtain photon arrival data and corresponding detection channels from input files in HDF5 format. It can be easily modified if other file formats are desired.

2. sampler_HMM(): Used to generate samples for parameters of interest using Gibbs algorithm.

3. initialize_variables!(): Initializes all the parameters of interest to constitute the first set of MCMC samples.
   
4. get_log_full_posterior(): Called by sampler() to get full joint posterior. Sums logarithms of likelihood and all the priors.
   
5. get_FRET_efficiences!(): Called to obtain FRET efficiencies for the active states.
 
6. sample_transition_rates!(): Called to sample kinetic rates and FRET efficiencies.
    
7. sample_emission_parameters!(): Called to sample intermediate hidden variables like emitted photon counts.

8. sample_loads_state_trajectory!(): Called to jointly sample active states from a large collection of states and the state trajectory.
 
9. get_active_inactive_loads!(): Called to identify active and inactive states.
 
10. get_generator!(): Called to get generator matrix with kinetic rates on off-diagonal elements and negative row-sum on the diagonal.
 
11. plot_everything(): Called to plot current Monte Carlo sample.

12. save_mcmc_data(): Called by sampler() to save output to files.

13. get_reduced_propagator!(): Called to compute propagator/transition proability matrix from exponential of the generator matrix.

14. get_log_demarginalized_likelihood_rates(): Called to compute likelihood from sampled trajectory only as rates only affect trajectory.

15. get_log_demarginalized_likelihood_FRET(): Called to compute likelihood from measurements and hidden variables only.

16. get_log_likelihood_observations_only(): 

17. get_log_observation_prob(): Called to get observation probability for a give time bin.

18. sample_state_trajectory(): Called to sample state trajectory using Forward-filter-backward-sampling algorithm

19. get_log_demarginalized_likelihood(): Called to compute likelihood given a trajectory.

20. get_filter_terms!(): Called to get probability values for sampling state trajectory.


# Test Examples
We provide two examples: 1) a simulated FRET trace with 3 states and 2) experimental FRET trace for Holliday junction. As shown below, these traces are stored in HDF5 files and contain donor and acceptor channel digital counts, donor and acceptor channel background rates, and camera calibration parameters, readout offset and readout noise variance, generated for each individual trace from dark calibration images.

![Screenshot from 2025-06-13 05-16-28](https://github.com/user-attachments/assets/83d92ac9-04b1-45a8-82b2-b73f193c7fbc)


To run an example, we suggest putting BNP-FRET scripts and the input data files in the same folder/directory. Next, if running on a Windows machine, first confirm the current folder that julia is being run from by executing the following command in the REPL:

```pwd()```

**Please note here that Windows machines use backslashes "\\" to describe folder paths unlike Linux and macOS where forward slashes "/" are used.** Appropriate modifications therefore must be made to the folder paths. Now, if the output of the command above is different from the path containing the scripts and tiff files, the current path can be changed by executing the following command:

```cd("/home/username/BNP-FRET-Binned/")```

BNP-FRET code can now be executed by simply importing the "main.jl" in the REPL as shown in the picture below

```include("main.jl")```


![Screenshot from 2025-06-13 05-22-56](https://github.com/user-attachments/assets/411638ed-9649-4d85-ba5f-7ad936f684cf)


On a linux or macOS machine, the "main.jl" script can be run directly from the terminal after entering the B-SIM directory and executing the following command:

```julia main.jl```

**WARNING: Please note that when running the code through the REPL, restart the REPL if B-SIM throws an error. Every execution of B-SIM adds processors to the julia REPL and processor ID or label increases in value. To make sure that processor labels always start at 1, we suggest avoiding restarting B-SIM in the same REPL.**

Now, the BNP-FRET output below shows the MCMC (Markov Chain Monte Carlo) iteration number (number of samples generated), number of active system states, labels for all the active loads (states), the absorption rate (related to laser power), rate matrix for the biomolecule of interest with FRET efficiencies on the diagonal instead of zeros, logarithm of the full joint posterior, and acceptance rates.


![Screenshot from 2025-06-13 05-26-49](https://github.com/user-attachments/assets/87ab6ee8-cd8f-47de-9342-38eebfd2276e)


Depending on the chosen plotting frequency in the "input_parameters.jl" file, the code also generates a set of plots showing (from top to bottom) the input donor and channel counts, apparent FRET efficiency, estimated state trajectory for the current sample, donor and acceptor background photon counts for the current sample, donor and acceptor photon emissions for the current sample, and the logarithm of the posterior probability to observe convergence of the sampler, as shown below.


![Screenshot from 2025-06-13 05-38-25](https://github.com/user-attachments/assets/9652c745-1d0f-475b-acc3-ef3b31e8243d)

We note above that the logarithm of the posterior probability first increases ("burn in period") and then converges to its maximum value. Upon convergence, we store the sample onto hard drive at 5000 iteration ("save_burn_in_period" in the input parameters file). We then disturb the sample significantly by increasing "temperature" in order to eliminate correlations. We again wait for convergence and store the sample. We repeat this procedure, termed "simulated annealing", at the frequency set by "save_burn_in_period" parameter. Use of simulated annealing here helps uncorrelate the chain of samples by smoothing and widening the posterior at intermediate iterations by raising temperature, allowing the sampler to easily move far away from the current sample or a local maximum.

Finally, as samples are collected, BNP-FRET saves intermediate samples and analysis data onto the hard drive in the HDF5 format with file names that start with "mcmc_output" in the same directory as the data.


## A Brief Description of the Sampler

The samplers here execute a Markov Chain Monte Carlo (MCMC) algorithm (Gibbs) where samples for each parameter of interest are generated sequentially from their corresponding probability distributions (posterior). First, the sampler creates/initiates arrays to store all the samples, posterior values, and acceptance rates for proposed samples. Next, new samples are then iteratively proposed using proposal (normal) distributions for each parameter, to be accepted or rejected by the Metropolis-Hastings step if direct sampling is not available. If accepted, the proposed sample is stored in the arrays otherwise the previous sample is stored at the same MCMC iteraion. 

The variance of the proposal distribution typically decides how often proposals are accepted/rejected. A larger covariance or movement away from the previous sample would lead to a larger change in likelihood/posterior values. Since the sampler prefers movement towards high probability regions, a larger movement towards low probability regions would lead to likely rejection of the sample compared to smaller movement.

The collected samples can then be used to compute statistical quantities and plot probability distributions. The plotting function used by the sampler in this code allows monitoring of posterior values, most probable model for the molecule, and distributions over transition rates and FRET efficiencies.

As mentioned before, sampler prefers movement towards higher probability regions of the posterior distribution. This means that if parameters are initialized in low probability regions of the posterior, which is typically the case, the posterior would appear to increase initially for many iterations (hundreds to thousands depending on the complexity of the model). This initial period is called burn-in. After burn-in, convergence is achieved where the posterior would typically fluctuate around some mean/average value. The convergence typically indicates that the sampler has reached the maximum of the posterior distribution (the most probability region), that is, sampler generates most samples from higher probability region. In other words, given a large collection of samples, the probability density in a region of parameter space is proportional to the number of samples collected from that region. 
 
All the samples collected during burn-in are usually ignored when computing statistical properties and presenting the final posterior distribution. 
