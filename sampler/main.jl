# BNP-FRET-Binned

###########################################################
# Copyright (C) 2025 Presse Lab - All Rights Reserved
#
# Author: Ayush Saurabh
#
# You may use, distribute and modify this code under the
# terms of the MIT license.
################################################################################
#
# A brief description of the sampler:
#
# The sampler function below executes a Markov Chain Monte Carlo (MCMC)
# algorithm (Gibbs) where samples for each parameter of interest are generated
# sequentially from their corresponding probability distributions (posterior).
# First, the sampler creates/initiates arrays to store all the samples,
# posterior values, and acceptance rates for proposed samples. Next, new
# samples are then iteratively proposed using proposal (normal) distributions
# for each parameter, to be accepted or rejected by the Metropolis-Hastings
# step. If accepted, the proposed sample is stored in the arrays otherwise the
# previous sample is stored at the same MCMC iteration.
#
# The variance of the proposal distribution typically decides how often
# proposals are accepted/rejected. A larger covariance or movement away from
# the previous sample would lead to a larger change in likelihood/posterior
# values. Since the sampler prefers movement towards high probability regions,
# a larger movement towards low probability regions would lead to likely
# rejection of the sample compared to smaller movement.
#
# The collected samples can then be used to compute statistical quantities and
# plot probability distributions. The plotting function used by the sampler
# in this code allows monitoring of posterior values, most probable model for
# the molecule, and distributions over transition rates and FRET efficiencies.
#
# As mentioned before, sampler prefers movement towards higher probability
# regions of the posterior distribution. This means that if parameters are
# initialized in low probability regions of the posterior, which is typically
# the case, the posterior would appear to increase initially for many
# iterations (hundreds to thousands depending on the complexity of the model).
# This initial period is called burn-in. After burn-in, convergence is achieved
# where the posterior would typically fluctuate around some mean/average value.
# The convergence typically indicates that the sampler has reached the maximum
# of the posterior distribution (the most probability region), that is, sampler
# generates most samples from higher probability region. In other words, given
# a large collection of samples, the probability density in a region of
# parameter space is proportional to the number of samples collected from that
# region.
#
# All the samples collected during burn-in are usually ignored
# when computing statistical properties and presenting the final posterior
# distribution.
#
###############################################################################

using Random
rng = Xoshiro(123);

using Distributions, SpecialFunctions
using LinearAlgebra
using Statistics, StatsBase
using HDF5


include("input_parameters.jl")
include("functions_layer_1_nonparametrics_loads_data.jl")
include("functions_layer_2_nonparametrics_loads_data.jl")
include("functions_layer_3_nonparametrics_loads_data.jl")

# Input
const donor_channel_data::Vector{Float64}, # ADU
        acceptor_channel_data::Vector{Float64}, # ADU
        donor_channel_bg::Float64, # ADU
        acceptor_channel_bg::Float64, # ADU
		offset_donor::Float64, # ADU
		variance_donor::Float64, # ADU
		offset_acceptor::Float64, # ADU
		variance_acceptor::Float64, # ADU
			n_bins::Int64 = get_data()

const read_out_noise_donor::Float64 = sqrt(variance_donor)
const read_out_noise_acceptor::Float64 = sqrt(variance_acceptor)

# Background photons in each channel
if (donor_channel_bg > offset_donor)

        const mean_bg_photons_donor::Float64 =
                        abs(donor_channel_bg - offset_donor) *
                                        sensitivity_donor/detection_eff

else

        const mean_bg_photons_donor::Float64 = 0.0

end
if (acceptor_channel_bg > offset_acceptor)

        const mean_bg_photons_acceptor::Float64 =
                        abs(acceptor_channel_bg - offset_acceptor) *
                                        sensitivity_acceptor/detection_eff

else

        const mean_bg_photons_acceptor::Float64 = 0.0

end

# Parameter for Beta-Bernoulli Nonparametrics
const prior_success_probability::Float64 =
		1.0/(1.0 + ((max_n_system_states - 1)/
						expected_n_system_states))

# Parameters for Dirichlet Prior and Categorical Distributions
const ones_Dir::Vector{Float64} = [1.0, 1.0]
const sign_vec::Vector{Int64} = [-1, 1]
const vec_0pt5::Vector{Float64} = [0.5, 0.5]


function sampler_HMM()

	draw::Int64 = 1

	# Print some of the parameters
	println(" *********************************************************")
	@show file_prefix
	@show modeling_choice
	@show n_bins
	@show gamma_prior_shape, gamma_prior_scale
	@show mean_bg_photons_donor, mean_bg_photons_acceptor
	@show read_out_noise_donor, read_out_noise_acceptor
	@show offset_donor, offset_acceptor
	@show trajectory_collapse_required
	println(" *********************************************************")
	
	flush(stdout);


	# Allocate memory for each variable
	n_rates::Int64 = max_n_system_states^2

	mcmc_save_loads::Matrix{Int64} = zeros(n_samples_to_save, max_n_system_states)
	mcmc_save_rates::Matrix{Float64} = zeros(n_samples_to_save, n_rates)
	mcmc_save_absorption_rate::Vector{Float64} = zeros(n_samples_to_save)
	mcmc_save_linear_drift_rate::Vector{Float64} = zeros(n_samples_to_save)
  	mcmc_log_posterior::Vector{Float64} = zeros(total_draws)
	mcmc_save_trajectory = zeros(Int64, n_samples_to_save, n_bins)

	absorption_rate::Float64 = 0.0
	linear_drift_rate::Float64 = 0.0

 	bg_photons_donor = ones(Int64, n_bins)
 	bg_photons_acceptor = ones(Int64, n_bins)

	photons_absorbed::Vector{Int64} = zeros(n_bins)
	emitted_donor_photons::Vector{Int64} = zeros(n_bins)
	emitted_acceptor_photons::Vector{Int64} = zeros(n_bins)

	rates::Vector{Float64} = zeros(n_rates)
	state_trajectory::Vector{Int64} = zeros(n_bins)
	sorted_trajectory::Vector{Int64} = zeros(n_bins)


	n_system_states::Int64 = 1
	loads::Vector{Int64} = zeros(max_n_system_states)
	loads_active::Vector{Int64} = zeros(max_n_system_states)
	loads_inactive::Vector{Int64} = zeros(max_n_system_states)
	p_load::Vector{Float64} = zeros(2)

	FRET_efficiencies::Vector{Float64} = zeros(max_n_system_states)
	FRET_efficiencies_sorted::Vector{Float64} = zeros(max_n_system_states)

	generator::Matrix{Float64} = zeros(max_n_system_states, max_n_system_states)
	propagator::Matrix{Float64} = zeros(max_n_system_states, max_n_system_states)
	propagator_transpose::Matrix{Float64} = 
		zeros(max_n_system_states, max_n_system_states)
	reduced_propagator::Matrix{Float64} = 
		zeros(max_n_system_states, max_n_system_states)
	reduced_propagator_transpose::Matrix{Float64} = 
		zeros(max_n_system_states, max_n_system_states)

	rho::Vector{Float64} = zeros(max_n_system_states)

 	filter_terms::Matrix{Float64} = zeros(max_n_system_states, n_bins)
	log_observation_prob::Vector{Float64} = zeros(max_n_system_states)
	prob_vec::Vector{Float64} = zeros(max_n_system_states)
	intermediate_vec::Vector{Float64} = zeros(max_n_system_states)


	state_trajectory_active::Vector{Int64} = zeros(n_bins)
	state_trajectory_inactive::Vector{Int64} = zeros(n_bins)

	emissions::Vector{Int64} = zeros(2)
	old_emissions::Vector{Int64} = zeros(2)
	proposed_emissions::Vector{Int64} = zeros(2)
	proposed_prob_vec::Vector{Float64} = zeros(2)
	reversed_prob_vec::Vector{Float64} = zeros(2)

	intermediate_vec2::Vector{Float64} = zeros(2)
	intermediate_vec3::Vector{Float64} = zeros(2)



	#Variables for collapse
	
	if trajectory_collapse_required == true
		escape_rates::Vector{Float64} = zeros(max_n_system_states) 
		separation_matrix_FRET::Matrix{Float64} = zeros(max_n_system_states, max_n_system_states)
		collapsed_indices = Vector{CartesianIndex{2}}(undef, 
				Int64(max_n_system_states*(max_n_system_states -1)/2))
		collapsed_indices_sorted = Vector{CartesianIndex{2}}(undef, 
				Int64(max_n_system_states*(max_n_system_states -1)/2))
		separation_vector_collapsed_indices::Vector{Float64} = 
				zeros(Int64(max_n_system_states*(max_n_system_states -1)/2))
		separation_vector_collapsed_indices_sorted::Vector{Float64} =
				zeros(Int64(max_n_system_states*(max_n_system_states -1)/2))
		collapsed_labels::Vector{Int64} = 
			zeros(Int64(max_n_system_states*(max_n_system_states -1)))

	end


 	println(" Initializing all variables...")
 	flush(stdout);
 
 	loads, absorption_rate, linear_drift_rate,
		rates, state_trajectory, photons_absorbed,
 			emitted_donor_photons, emitted_acceptor_photons,
 				bg_photons_donor, bg_photons_acceptor,
				generator, propagator, rho,  
				loads_active, 
				loads_inactive,
				n_system_states = 
					initialize_variables!(loads,
						absorption_rate, 
						linear_drift_rate,
						rates, 
						state_trajectory, 
						photons_absorbed,
 						emitted_donor_photons, 
						emitted_acceptor_photons,
 						bg_photons_donor, 
						bg_photons_acceptor,
						generator,
						propagator,
						rho,
						loads_active,
						loads_inactive,
						n_system_states,
						emissions,
						intermediate_vec2)
 
    	mcmc_log_posterior[draw] = get_log_posterior(loads, 
					absorption_rate, 
					linear_drift_rate, 
					rates, 
					photons_absorbed,
    					emitted_donor_photons, 
					emitted_acceptor_photons,
  					bg_photons_donor, 
					bg_photons_acceptor,
					generator,
					propagator,
					rho,
					loads_active,
					n_system_states,
					reduced_propagator,
					reduced_propagator_transpose,
					prob_vec,
					intermediate_vec,
					intermediate_vec2)
  

	FRET_efficiencies = get_FRET_efficiencies(FRET_efficiencies, 
						  n_system_states, 
						  loads_active, 
						  rates)

	println("****************************************************************")
	@show draw
	@show loads, n_system_states
 	@show unique(state_trajectory), size(unique(state_trajectory))
 	@show absorption_rate
 	@show linear_drift_rate
 	println("generator matrix = ")
       	display(view(generator, 1:n_system_states, 1:n_system_states))
 	println("FRET efficiencies = ")
	display(view(FRET_efficiencies, 1:n_system_states))
 	@show mcmc_log_posterior[draw]
 	flush(stdout);


	println(" Starting Sampler...")
	flush(stdout);

	current_sample::Int64 = 0
 	for draw in 2:total_draws
 
 
 		temperature::Float64 = 1.0 + (starting_temperature - 1.0) * 
 				exp(-((draw - 1) % save_burn_in_period)/
 				    	annealing_constant)
 
 		# Get the new rates for conformation dynamics
    		rates = sample_transition_rates!(draw, 
 				absorption_rate,
 				linear_drift_rate,
 				rates,
    				state_trajectory,
    				photons_absorbed,
    				emitted_donor_photons,
    				emitted_acceptor_photons, 
 				bg_photons_donor,
 				bg_photons_acceptor,
 				temperature,
        			generator,
        			propagator,
        			rho,
        			loads_active,
        			loads_inactive,
        			n_system_states,
				intermediate_vec2,
				intermediate_vec3)
 
 		# Sample Parameters Governing Emissions
   		absorption_rate, linear_drift_rate, photons_absorbed, 
   			emitted_donor_photons,
   				emitted_acceptor_photons =
  					sample_emission_parameters!(draw, 
  						absorption_rate,
  						linear_drift_rate,
  						rates,
   						state_trajectory,
   						photons_absorbed,
   						emitted_donor_photons,
   						emitted_acceptor_photons, 
  						bg_photons_donor,
  						bg_photons_acceptor,
  						temperature,
						old_emissions,
						proposed_emissions,
						proposed_prob_vec,
						reversed_prob_vec,
						intermediate_vec2)
  
  
  		if (draw-1) % save_burn_in_period <= burn_in_period
  
          		loads, state_trajectory =
               			sample_loads_state_trajectory!(draw, 
  					loads, 
  					absorption_rate,
  					linear_drift_rate,
  					rates,
      					state_trajectory,
         				photons_absorbed,
      					emitted_donor_photons,
      					emitted_acceptor_photons, 
  					bg_photons_donor,
  					bg_photons_acceptor,
      					temperature,
 					generator,
 					propagator,
 					propagator_transpose,
 					rho,
 					loads_active,
 					loads_inactive,
					p_load,
 					n_system_states,
 					filter_terms,
 					log_observation_prob,
 					prob_vec,
					intermediate_vec,
 					state_trajectory_active,
 					state_trajectory_inactive)
  
  		end
  		loads_active, loads_inactive, n_system_states = 
 					get_active_inactive_loads!(loads,
 							loads_active,
 							loads_inactive,
 							n_system_states)
 		generator, propagator, rho = get_generator!(loads_active, 
 					n_system_states,
 					rates,
 					generator,
 					propagator,
 					rho)
 
 
     		mcmc_log_posterior[draw] = get_log_posterior(loads, 
 					absorption_rate, 
 					linear_drift_rate, 
 					rates, 
 					photons_absorbed,
     					emitted_donor_photons, 
 					emitted_acceptor_photons,
   					bg_photons_donor, 
 					bg_photons_acceptor,
 					generator,
 					propagator,
 					rho,
 					loads_active,
 					n_system_states,
 					reduced_propagator,
					reduced_propagator_transpose,
 					prob_vec,
					intermediate_vec,
					intermediate_vec2)

        	FRET_efficiencies = get_FRET_efficiencies(FRET_efficiencies, 
        					  n_system_states, 
        					  loads_active, 
        					  rates)


		# Print Data
   		if draw % output_frequency == 0
 
 			println("****************************************************************")
 			@show draw
 			@show loads, n_system_states
 			@show unique(state_trajectory), size(unique(state_trajectory))
 			@show temperature
 			@show absorption_rate
 			@show linear_drift_rate
 			println("generator matrix = ")
        		display(view(generator, 1:n_system_states, 1:n_system_states))
 			println("FRET efficiencies = ")
        		display(view(FRET_efficiencies, 1:n_system_states))
 			@show mcmc_log_posterior[draw]
 			flush(stdout);
		end

    		if plotting_on == true && draw % plotting_frequency == 0

 			# Rearrange so that the state labels increase with increasing FRET efficiency
    			FRET_efficiencies_sorted .= sort(FRET_efficiencies)
    			FRET_efficiencies_sorted .= circshift(FRET_efficiencies_sorted, n_system_states)
    			sorted_trajectory .= state_trajectory
    			for i in 1:n_bins
        			sorted_trajectory[i] = findall(x-> x == sorted_trajectory[i], loads_active)[1]
    				sorted_trajectory[i] = 
   					findall(x -> x == FRET_efficiencies[sorted_trajectory[i]], 
   								FRET_efficiencies_sorted)[1]
    			end
  
      			plot_everything(draw, current_sample+1, 
      				sorted_trajectory, n_system_states,
     					mcmc_log_posterior, 
     					photons_absorbed,
     					emitted_donor_photons,
         				emitted_acceptor_photons, 
    					bg_photons_donor,
    					bg_photons_acceptor)
   
   		end


 		# Collapse Trajectory	
 		# Collapse in order of increasing separation and only collapse 
 		# unique sets of labels. No overlapping labels.
 
		if trajectory_collapse_required == true
  		if (draw-1) % save_burn_in_period == burn_in_period || 
			(draw-1) % save_burn_in_period == burn_in_period + 200 
  
 			escape_rates .= 0.0
			escape_rates[1:n_system_states] .= 
				abs.(diag(view(generator, 1:n_system_states, 1:n_system_states)))
  			separation_matrix_FRET .= 0.0 
  			for i in 1:n_system_states
  				for j in i:n_system_states
  					separation_matrix_FRET[i, j] = 
  						abs(FRET_efficiencies[i]-FRET_efficiencies[j])
  				end
  			end
  			n_collapses::Int64 = count(x -> 0.0 < x < FRET_eff_min_separation, 
 								view(separation_matrix_FRET, 
								     1:n_system_states, 
								     1:n_system_states))
			collapsed_indices[1:n_collapses] .= findall(x -> 0.0 < x < FRET_eff_min_separation, 
 								view(separation_matrix_FRET, 
								     1:n_system_states, 
								     1:n_system_states))

			@show loads_active
			@show n_collapses
			println("Collapsed Indices:")
			@show display(view(collapsed_indices, 1:n_collapses))
			@show "Collapsed labels:"

			separation_vector_collapsed_indices[1:n_collapses] .= 
					separation_matrix_FRET[view(collapsed_indices, 1:n_collapses)]
			separation_vector_collapsed_indices_sorted[1:n_collapses] .= 
					sort(view(separation_vector_collapsed_indices, 1:n_collapses)) 

  			for i in 1:n_collapses
  				collapsed_indices_sorted[i] = 
  					collapsed_indices[findall(x-> x == separation_vector_collapsed_indices_sorted[i], 
								  view(separation_vector_collapsed_indices, 1:n_collapses))[1]]
  			end
  
  			
  			collapsed_labels .= 0
  			for i in 1:n_collapses

   				first_label::Int64 = loads_active[collapsed_indices_sorted[i][1]]
  				second_label::Int64 = loads_active[collapsed_indices_sorted[i][2]]
  			
  				if ((first_label in collapsed_labels) == false) && 
  						((second_label in collapsed_labels) ==false)
  					if (escape_rates[collapsed_indices_sorted[i][1]] > escape_rate_max) || 
  							(escape_rates[collapsed_indices_sorted[i][2]] > escape_rate_max)
  
  						state_trajectory[state_trajectory .== second_label] .= first_label
  						loads[second_label] = 0
  
        					collapsed_labels[2*i-1] = first_label 
        					collapsed_labels[2*i] = second_label
						@show first_label, second_label
  
  					end
  				end
  			end
  			flush(stdout);
  		end
		end
 
 
 
 		# Save Data 
      		if draw % save_frequency == 0
 
 			current_sample += 1
 
  			mcmc_save_loads[current_sample, 1:max_n_system_states] .= loads[:]
 			mcmc_save_rates[current_sample, 1:n_rates] .= rates[:]
 			mcmc_save_trajectory[current_sample, :] .= state_trajectory[:]
 
      			save_mcmc_data(draw, 
				current_sample, 
				mcmc_save_loads, 
				mcmc_save_rates,
      				mcmc_log_posterior, 
        			mcmc_save_trajectory,
      				photons_absorbed,
      				emitted_donor_photons,
      				emitted_acceptor_photons)
 
    		end
 	end

	return nothing
end

@time sampler_HMM()
