function get_data()

	file_name = string(working_directory, file_prefix,".h5")
	fid = h5open(file_name,"r")
	donor_channel_data = read(fid, "donor_channel")
	acceptor_channel_data = read(fid, "acceptor_channel")
	donor_channel_bg = read(fid, "donor_channel_bg")
	acceptor_channel_bg = read(fid, "acceptor_channel_bg")

	offset_donor = read(fid, "offset_donor")
	variance_donor = read(fid, "variance_donor")
	offset_acceptor = read(fid, "offset_acceptor")
	variance_acceptor = read(fid, "variance_acceptor")


	n_bins = size(donor_channel_data)[1]
	close(fid)

	return donor_channel_data, acceptor_channel_data,
		donor_channel_bg, acceptor_channel_bg, 
		offset_donor, variance_donor,
		offset_acceptor, variance_acceptor,
		n_bins
end

function initialize_variables!(loads::Vector{Int64},
			absorption_rate::Float64,
			linear_drift_rate::Float64,
			rates::Vector{Float64},
			state_trajectory::Vector{Int64},
			photons_absorbed::Vector{Int64},
			emitted_donor_photons::Vector{Int64},
 			emitted_acceptor_photons::Vector{Int64},
			bg_photons_donor::Vector{Int64},
			bg_photons_acceptor::Vector{Int64},
			generator::Matrix{Float64},
			propagator::Matrix{Float64},
			rho::Vector{Float64},
			loads_active::Vector{Int64},
			loads_inactive::Vector{Int64},
			n_system_states::Int64,
			emissions::Vector{Int64},
			intermediate_vec::Vector{Float64})

	absorption_rate = ((mean(donor_channel_data) -
			donor_channel_bg)*sensitivity_donor)/
			(quantum_yield_d) +  # s^-1
			((mean(acceptor_channel_data) -
					acceptor_channel_bg)*sensitivity_acceptor)/
			(quantum_yield_a)
	linear_drift_rate = 0.0

	for i in 1:max_n_system_states
		for j in 1:max_n_system_states
			ij = (i-1)*max_n_system_states + j
			if i == j # FRET Eff
   				rates[ij] = rand(rng)
			elseif i != j
 				rates[ij] = rand(rng, 
						Gamma(gamma_prior_shape, 
						gamma_prior_scale), 1)[1] # s^-1
			end
		end
	end

	if modeling_choice == "nonparametric"
		prior_success_probability::Float64 =
				1.0/(1.0 + ((max_n_system_states - 1)/
					expected_n_system_states))
		p_load::Vector{Float64} =
			[prior_success_probability, 1.0 - prior_success_probability]
		n_system_states::Int64 = 0
		for i in 1:max_n_system_states
			loads[i] = rand(rng, Categorical(p_load), 1)[1]
			if  loads[i]== 1 #Active
				loads[i] = i
				n_system_states = n_system_states + 1
			elseif loads[i] == 2 #Inactive
				loads[i] = 0
				if i == max_n_system_states && n_system_states == 0
					loads[i] = i
				end
			end
		end
	else
		for i in 1:expected_n_system_states
			loads[i] = i
		end
		n_system_states = expected_n_system_states
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

	previous_state::Int64 = 
		rand(rng, Categorical(rho[1:n_system_states]), 1)[1]
	FRET_eff::Float64 = 0.5
	intermediate_vec .= [(1.0-FRET_eff) * quantum_yield_d, FRET_eff * quantum_yield_a]

  	for bin in 1:n_bins

 		next_state::Int64 = 
			rand(rng, Categorical(propagator[previous_state, 
						1:n_system_states]))
    		state_trajectory[bin] = loads_active[next_state]
   		photons_absorbed[bin] = 
			rand(rng, Poisson(absorption_rate+bin*linear_drift_rate))
   		emissions .= rand(rng, Multinomial(photons_absorbed[bin], intermediate_vec))
   		emitted_donor_photons[bin] = emissions[1]
   		emitted_acceptor_photons[bin] = emissions[2]

   		bg_photons_donor[bin] = rand(rng, Poisson(mean_bg_photons_donor))
   		bg_photons_acceptor[bin] = rand(rng, Poisson(mean_bg_photons_acceptor))

   		previous_state = next_state

 	end

	return loads, absorption_rate, linear_drift_rate, 
		rates, state_trajectory, photons_absorbed,
		emitted_donor_photons, emitted_acceptor_photons,
		bg_photons_donor, bg_photons_acceptor,
		generator, propagator, rho, 
		loads_active, loads_inactive, n_system_states
end

function get_log_posterior(loads::Vector{Int64},
			absorption_rate::Float64,
			linear_drift_rate::Float64,
			rates::Vector{Float64},
			photons_absorbed::Vector{Int64},
			emitted_donor_photons::Vector{Int64},
			emitted_acceptor_photons::Vector{Int64},
			bg_photons_donor::Vector{Int64},
			bg_photons_acceptor::Vector{Int64},
			generator::Matrix{Float64}, 
			propagator::Matrix{Float64}, 
			rho::Vector{Float64},
			loads_active::Vector{Int64},
			n_system_states::Int64,
			reduced_propagator::Matrix{Float64},
			reduced_propagator_transpose::Matrix{Float64},
			prob_vector::Vector{Float64},
			intermediate_vec::Vector{Float64},
			intermediate_vec2::Vector{Float64})

	log_likelihood::Float64 = 0.0
	prob_vector .= rho
	
	for bin in 1:n_bins
		reduced_propagator = get_reduced_propagator!(bin, 
					loads, 
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
					reduced_propagator_transpose)
		mul!(intermediate_vec, reduced_propagator, prob_vector)
		prob_vector[1:n_system_states] .= view(intermediate_vec, 1:n_system_states)
		p::Float64 = sum(prob_vector)
		log_likelihood += log(p)
		prob_vector .= prob_vector ./ p

#		@show intermediate_vec
#		@show reduced_propagator
#		@show prob_vector
#		@show bin, log(p)

	end

	# Add all the priors
	log_prior::Float64 = 0.0

	# For absorption rate
	log_prior += logpdf(Gamma(1, 1.0e3), absorption_rate)

	# For gradient
	log_prior += logpdf(Normal(0.0, 1.0), linear_drift_rate)

	prior_success_probability::Float64 = 1.0/(1.0 + ((max_n_system_states - 1)/
						expected_n_system_states))
	for i in 1:max_n_system_states
		# For loads
		if loads[i] == i
			log_prior += log(prior_success_probability)
		else
			log_prior += log(1.0 - prior_success_probability)
		end

		for j in 1:max_n_system_states
			ij::Int64 = (i-1)*max_n_system_states + j
			if i != j # Confo Rates
				log_prior += logpdf(Gamma(gamma_prior_shape, 
						gamma_prior_scale), rates[ij])
			elseif i == j # FRET Eff
				intermediate_vec2[1] = rates[ij]+eps()
				intermediate_vec2[2] = 1.0-rates[ij]+eps()

				log_prior +=logpdf(Dirichlet(ones_Dir), 
						   	intermediate_vec2)
			end
		end
	end

	log_posterior::Float64 = log_likelihood + log_prior

	return log_posterior
end

function get_FRET_efficiencies(FRET_efficiencies, 
			n_system_states::Int64, 
			loads_active::Vector{Int64},
			rates::Vector{Float64})

	FRET_efficiencies .= 0.0
	for i in 1:n_system_states
		ii::Int64 = (loads_active[i]-1)*max_n_system_states + loads_active[i]
		FRET_efficiencies[i] = rates[ii]
	end

	return FRET_efficiencies
end

function sample_transition_rates!(draw::Int64,
			absorption_rate::Float64,
			linear_drift_rate::Float64,
			rates::Vector{Float64},
			state_trajectory::Vector{Int64},
			photons_absorbed::Vector{Int64},
			emitted_donor_photons::Vector{Int64},
			emitted_acceptor_photons::Vector{Int64},
			bg_photons_donor::Vector{Int64},
			bg_photons_acceptor::Vector{Int64},
			temperature::Float64,
			generator::Matrix{Float64},
			propagator::Matrix{Float64},
			rho::Vector{Float64},
			loads_active::Vector{Int64},
			loads_inactive::Vector{Int64},
			n_system_states::Int64,
			intermediate_vec2::Vector{Float64},
			intermediate_vec3::Vector{Float64})


	# For absorption rate
	old_log_conditional_likelihood_rates::Float64 = 
		get_log_demarginalized_likelihood_rates(
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
			n_system_states)

	old_log_conditional_likelihood_FRET::Float64 = 
		get_log_demarginalized_likelihood_FRET(
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
			n_system_states)

 
	old_log_prior::Float64 = 0.0
	old_log_conditional_posterior::Float64 = 0.0
	proposed_log_conditional_likelihood_rates::Float64 = 0.0
	proposed_log_conditional_likelihood_FRET::Float64 = 0.0
	proposed_log_prior::Float64 = 0.0
	proposed_log_conditional_posterior::Float64 = 0.0
	log_hastings::Float64 = 0.0
	proposed_rate::Float64 = 0.0
	old_rate::Float64 = 0.0

	# For active states
	for i in 1:n_system_states
		for j in 1:n_system_states

			ij::Int64 = (loads_active[i]-1)*max_n_system_states + loads_active[j]
			if i != j  # For Confo Rates

				old_rate = rates[ij]
				old_log_prior = logpdf(Gamma(gamma_prior_shape, 
							gamma_prior_scale), old_rate)
 				old_log_conditional_posterior =
 						old_log_conditional_likelihood_rates + 
						old_log_prior

  				proposed_rate = rand(rng, Normal(log(old_rate), 
								 step_size_rates))
  				proposed_rate = exp(proposed_rate)
				rates[ij] = proposed_rate

				generator, propagator, rho = get_generator!(loads_active, 
					n_system_states,
					rates,
					generator,
					propagator,
					rho)

				proposed_log_conditional_likelihood_rates =
					get_log_demarginalized_likelihood_rates(
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
							n_system_states)

 				proposed_log_prior = logpdf(Gamma(gamma_prior_shape, 
						gamma_prior_scale), proposed_rate)
 				proposed_log_conditional_posterior =
 						proposed_log_conditional_likelihood_rates + 
							proposed_log_prior

 				log_hastings = (proposed_log_conditional_posterior -
						old_log_conditional_posterior) +
 						log(proposed_rate) - log(old_rate)

 				if log_hastings >= log(rand(rng))
 					old_log_conditional_likelihood_rates =
						proposed_log_conditional_likelihood_rates
				else
					rates[ij] = old_rate
				end

				generator, propagator, rho = get_generator!(loads_active, 
					n_system_states,
					rates,
					generator,
					propagator,
					rho)

			elseif i == j # For FRET Eff

				old_rate = rates[ij]

				intermediate_vec2[1] = old_rate+eps()
				intermediate_vec2[2] = 1.0-old_rate+eps()

  				old_log_prior = logpdf(Dirichlet(ones_Dir), intermediate_vec2)
  				old_log_conditional_posterior =
  						old_log_conditional_likelihood_FRET + 
							old_log_prior

				intermediate_vec2[1] = conc_parameter_FRET*old_rate+eps()
				intermediate_vec2[2] = conc_parameter_FRET*(1.0 - old_rate)+eps()

				intermediate_vec2 .= rand(rng, Dirichlet(intermediate_vec2))
				proposed_rate = intermediate_vec2[1]
				rates[ij] = proposed_rate

				proposed_log_conditional_likelihood_FRET =
					get_log_demarginalized_likelihood_FRET(
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
							n_system_states)

				intermediate_vec2[1] = proposed_rate+eps()
				intermediate_vec2[2] = 1.0-proposed_rate+eps()
  				proposed_log_prior = logpdf(Dirichlet(ones_Dir), intermediate_vec2)

  				proposed_log_conditional_posterior =
  					proposed_log_conditional_likelihood_FRET + proposed_log_prior

				intermediate_vec2[1] = conc_parameter_FRET*proposed_rate+eps()
				intermediate_vec2[2] = conc_parameter_FRET*(1.0-proposed_rate)+eps()
				intermediate_vec3[1] = old_rate+eps()
				intermediate_vec3[2] = 1.0-old_rate+eps()

				log_backward_transition_prob::Float64 = logpdf(Dirichlet(intermediate_vec2),
  										intermediate_vec3)

				intermediate_vec2[1] = conc_parameter_FRET*old_rate+eps()
				intermediate_vec2[2] = conc_parameter_FRET*(1.0-old_rate)+eps()
				intermediate_vec3[1] = proposed_rate+eps()
				intermediate_vec3[2] = 1.0-proposed_rate+eps()

				log_forward_transition_prob::Float64 = logpdf(Dirichlet(intermediate_vec2),
  										intermediate_vec3)

  				log_hastings =(proposed_log_conditional_posterior -
  						old_log_conditional_posterior) +
						(log_backward_transition_prob -
						 log_forward_transition_prob)

 				if log_hastings >= log(rand(rng))
 					old_log_conditional_likelihood_FRET =
						proposed_log_conditional_likelihood_FRET
				else
					rates[ij] = old_rate
				end

			end
		end
	end


	#For inactive states
	for i in 1:max_n_system_states-n_system_states
		for j in 1:max_n_system_states-n_system_states

			ij::Int64 = (loads_inactive[i]-1)*max_n_system_states + loads_inactive[j]

			if i != j  # For Confo Rates

				old_rate = rates[ij]
				old_log_prior = logpdf(Gamma(gamma_prior_shape, 
							gamma_prior_scale), old_rate)
 				old_log_conditional_posterior = old_log_prior

  				proposed_rate = rand(rng, Normal(log(old_rate), 
							step_size_rates))
  				proposed_rate = exp(proposed_rate)
				rates[ij] = proposed_rate

 				proposed_log_prior = logpdf(Gamma(gamma_prior_shape, 
						gamma_prior_scale), proposed_rate)
 				proposed_log_conditional_posterior = proposed_log_prior

 				log_hastings = (proposed_log_conditional_posterior -
						old_log_conditional_posterior) +
 						log(proposed_rate) - log(old_rate)

 				if log_hastings >= log(rand(rng))
				else
					rates[ij] = old_rate
				end

			elseif i == j # For FRET Eff

				old_rate = rates[ij]

				intermediate_vec2[1] = old_rate+eps()
				intermediate_vec2[2] = 1.0-old_rate+eps()

  				old_log_prior = logpdf(Dirichlet(ones_Dir), intermediate_vec2)

  				old_log_conditional_posterior = old_log_prior

				intermediate_vec2[1] = conc_parameter_FRET*old_rate+eps()
				intermediate_vec2[2] = conc_parameter_FRET*(1.0 - old_rate)+eps()
				intermediate_vec2 .= rand(rng, Dirichlet(intermediate_vec2))
				proposed_rate = intermediate_vec2[1]
				rates[ij] = proposed_rate

				intermediate_vec2[1] = proposed_rate+eps()
				intermediate_vec2[2] = 1.0-proposed_rate+eps()
  				proposed_log_prior = logpdf(Dirichlet(ones_Dir), intermediate_vec2)

  				proposed_log_conditional_posterior = proposed_log_prior

				intermediate_vec2[1] = conc_parameter_FRET*proposed_rate+eps()
				intermediate_vec2[2] = conc_parameter_FRET*(1.0-proposed_rate)+eps()
				intermediate_vec3[1] = old_rate+eps()
				intermediate_vec3[2] = 1.0-old_rate+eps()

				log_backward_transition_prob::Float64 = logpdf(Dirichlet(intermediate_vec2),
  										intermediate_vec3)

				intermediate_vec2[1] = conc_parameter_FRET*old_rate+eps()
				intermediate_vec2[2] = conc_parameter_FRET*(1.0-old_rate)+eps()
				intermediate_vec3[1] = proposed_rate+eps()
				intermediate_vec3[2] = 1.0-proposed_rate+eps()

				log_forward_transition_prob::Float64 = logpdf(Dirichlet(intermediate_vec2),
  										intermediate_vec3)


  				log_hastings =(proposed_log_conditional_posterior -
  						old_log_conditional_posterior) +
						(log_backward_transition_prob -
						 log_forward_transition_prob)

 				if log_hastings >= log(rand(rng))
				else
					rates[ij] = old_rate
				end

			end
		end
	end

	return rates
end


function sample_emission_parameters!(draw::Int64,
			absorption_rate::Float64,
			linear_drift_rate::Float64,
			rates::Vector{Float64},
			state_trajectory::Vector{Int64},
			photons_absorbed::Vector{Int64},
			emitted_donor_photons::Vector{Int64},
			emitted_acceptor_photons::Vector{Int64},
			bg_photons_donor::Vector{Int64},
			bg_photons_acceptor::Vector{Int64},
			temperature::Float64,
			old_emissions::Vector{Int64},
			proposed_emissions::Vector{Int64},
			proposed_prob_vec::Vector{Float64},
			reversed_prob_vec::Vector{Float64},
			intermediate_vec2::Vector{Float64})


 	old_log_conditional_likelihood::Float64 =
				get_log_likelihood_observations_only(
							absorption_rate,
							linear_drift_rate,
							rates,
							state_trajectory,
							photons_absorbed,
							emitted_donor_photons,
							emitted_acceptor_photons,
							bg_photons_donor,
							bg_photons_acceptor)

	# For absorption rate
 	old_log_prior::Float64 = logpdf(Gamma(1, 1.0e3), absorption_rate)
 	old_log_conditional_posterior::Float64 =
 						old_log_conditional_likelihood + old_log_prior

 	proposed_absorption_rate::Float64 = rand(rng, Normal(log(absorption_rate), 1.0e-3))
 	proposed_absorption_rate = exp(proposed_absorption_rate)

 	proposed_log_conditional_likelihood::Float64 =
				get_log_likelihood_observations_only(
							proposed_absorption_rate,
							linear_drift_rate,
							rates,
							state_trajectory,
							photons_absorbed,
							emitted_donor_photons,
							emitted_acceptor_photons,
							bg_photons_donor,
							bg_photons_acceptor)

 	proposed_log_prior::Float64 = logpdf(Gamma(1, 1.0e3), proposed_absorption_rate)
 	proposed_log_conditional_posterior::Float64 =
 						proposed_log_conditional_likelihood + proposed_log_prior
 	log_hastings::Float64 =
		(proposed_log_conditional_posterior - old_log_conditional_posterior) +
 					log(proposed_absorption_rate) - log(absorption_rate)
 	if log_hastings >= log(rand(rng))
 		old_log_conditional_likelihood = proposed_log_conditional_likelihood
 		absorption_rate = proposed_absorption_rate
 	end

	# For linear drift rate
 	old_log_prior = logpdf(Normal(0.0, 1.0), linear_drift_rate)
 	old_log_conditional_posterior = old_log_conditional_likelihood + old_log_prior


 	proposed_linear_drift_rate::Float64 = rand(rng, Normal(linear_drift_rate, 1.0e-3))

 	proposed_log_conditional_likelihood =
				get_log_likelihood_observations_only(
							absorption_rate,
							proposed_linear_drift_rate,
							rates,
							state_trajectory,
							photons_absorbed,
							emitted_donor_photons,
							emitted_acceptor_photons,
							bg_photons_donor,
							bg_photons_acceptor)

 	proposed_log_prior = logpdf(Normal(0.0, 1.0), proposed_linear_drift_rate)
 	proposed_log_conditional_posterior = proposed_log_conditional_likelihood + proposed_log_prior
 	log_hastings =
		(proposed_log_conditional_posterior - old_log_conditional_posterior)
 	if log_hastings >= log(rand(rng))
 		old_log_conditional_likelihood = proposed_log_conditional_likelihood
 		linear_drift_rate = proposed_linear_drift_rate 
 	end

	# For other parameters

	accepted_emissions::Int64 = 0

	for bin in 1:n_bins

		bg_photons_donor[bin] = rand(rng, Poisson(mean_bg_photons_donor))
		bg_photons_acceptor[bin] = rand(rng, Poisson(mean_bg_photons_acceptor))


		FRET_eff::Float64 = rates[(state_trajectory[bin]-1)*
						max_n_system_states+state_trajectory[bin]]

		# For photons absorbed and emissions
 		old_log_observation_prob::Float64 = get_log_observation_prob(bin,
 						state_trajectory[bin],
 						photons_absorbed[bin],
 						emitted_donor_photons[bin],
 						emitted_acceptor_photons[bin],
        					bg_photons_donor[bin],
 						bg_photons_acceptor[bin],
        					absorption_rate,
        					linear_drift_rate,
 						rates)

        	old_emissions[1] = emitted_donor_photons[bin]
        	old_emissions[2] = emitted_acceptor_photons[bin]

        	intermediate_vec2[1] = (1.0-FRET_eff)*quantum_yield_d
        	intermediate_vec2[2] = FRET_eff*quantum_yield_a

 		old_log_prior =
 			logpdf(Multinomial(photons_absorbed[bin],
 				intermediate_vec2),
 				old_emissions)

 		old_log_conditional_posterior = old_log_observation_prob + old_log_prior


		chosen_sign::Int64 = sign_vec[rand(rng, Categorical(vec_0pt5))]
  		proposed_photons_absorbed::Int64 = photons_absorbed[bin] + chosen_sign *
 							rand(rng, Poisson(0.1), 1)[1]

        	proposed_prob_vec .= old_emissions ./ sum(old_emissions)
 		proposed_emissions .=
 				rand(rng, Multinomial(proposed_photons_absorbed,
 									proposed_prob_vec))
 		proposed_emitted_donor_photons::Int64 = proposed_emissions[1]
 		proposed_emitted_acceptor_photons::Int64 = proposed_emissions[2]

 		if proposed_emitted_donor_photons > 0 && proposed_emitted_acceptor_photons > 0
 			proposed_log_observation_prob::Float64 = get_log_observation_prob(bin,
 							state_trajectory[bin],
 							proposed_photons_absorbed,
 							proposed_emitted_donor_photons,
 							proposed_emitted_acceptor_photons,
        						bg_photons_donor[bin],
 							bg_photons_acceptor[bin],
        						absorption_rate,
        						linear_drift_rate,
 							rates)

 			intermediate_vec2[1] = (1.0-FRET_eff)*quantum_yield_d
        		intermediate_vec2[2] = FRET_eff*quantum_yield_a

 			proposed_log_prior =
 				logpdf(Multinomial(proposed_photons_absorbed,
 							intermediate_vec2),
 								proposed_emissions)

 			proposed_log_conditional_posterior = proposed_log_observation_prob + proposed_log_prior

        		reversed_prob_vec .= proposed_emissions ./ sum(proposed_emissions)

 			log_hastings = 	1/temperature * (proposed_log_conditional_posterior -
 							 old_log_conditional_posterior) +
 							logpdf(Multinomial(proposed_photons_absorbed,
 								reversed_prob_vec), old_emissions) -
 							logpdf(Multinomial(photons_absorbed[bin],
 								proposed_prob_vec), proposed_emissions)
 			if log_hastings >= log(rand(rng))
  				photons_absorbed[bin] = proposed_photons_absorbed
 	 			emitted_donor_photons[bin] = proposed_emitted_donor_photons
 	 			emitted_acceptor_photons[bin] = proposed_emitted_acceptor_photons
        			old_log_observation_prob = proposed_log_observation_prob
        			accepted_emissions += 1
 			end
 		end
	end

	return absorption_rate, linear_drift_rate,
			photons_absorbed,
			emitted_donor_photons,
			emitted_acceptor_photons,
			bg_photons_donor,
			bg_photons_acceptor
end #function




# State Trajectory
function sample_loads_state_trajectory!(draw::Int64,
			loads::Vector{Int64},
			absorption_rate::Float64,
			linear_drift_rate::Float64,
			rates::Vector{Float64},
			state_trajectory::Vector{Int64},
   			photons_absorbed::Vector{Int64},
			emitted_donor_photons::Vector{Int64},
			emitted_acceptor_photons::Vector{Int64},
			bg_photons_donor::Vector{Int64},
			bg_photons_acceptor::Vector{Int64},
			temperature::Float64,
			generator::Matrix{Float64},
			propagator::Matrix{Float64},
			propagator_transpose::Matrix{Float64},
			rho::Vector{Float64},
			loads_active::Vector{Int64},
			loads_inactive::Vector{Int64},
			p_load::Vector{Float64},
			n_system_states::Int64,
			filter_terms::Matrix{Float64},
			log_observation_prob::Vector{Float64},
			prob_vec::Vector{Float64},
			intermediate_vec::Vector{Float64},
			state_trajectory_active::Vector{Int64},
			state_trajectory_inactive::Vector{Int64})


	# Sample loads

	local accept_trajectory::Bool
	if modeling_choice == "parametric"

 		state_trajectory, accept_trajectory =
					sample_state_trajectory!( 
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
							n_system_states,
							filter_terms,
							log_observation_prob,
							prob_vec,
							intermediate_vec)

	else

 		log_likelihood_inactive::Float64 = 0.0
		log_likelihood_active::Float64 = 0.0
		log_p_inactive::Float64 = 0.0
		log_p_active::Float64 = 0.0
		old_log_demarginalized_likelihood::Float64 =
 				get_log_demarginalized_likelihood(
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
 						n_system_states)


 		for i in 1:max_n_system_states

 			if loads[i] == 0

				state_trajectory_inactive .= state_trajectory

 				log_likelihood_inactive = 
					old_log_demarginalized_likelihood
 				log_p_inactive = log_likelihood_inactive + log(1.0 -
 							prior_success_probability)

				# Proposed Load
 				loads[i] = i

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

 		    		state_trajectory_active, 
					accept_trajectory =
  						sample_state_trajectory!( 
							absorption_rate,
							linear_drift_rate,
							rates,
							state_trajectory_active,
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
							n_system_states,
							filter_terms,
							log_observation_prob,
							prob_vec,
							intermediate_vec)


 				if accept_trajectory == true

 					log_likelihood_active =
						get_log_demarginalized_likelihood(
							absorption_rate,
							linear_drift_rate,
							rates, 
							state_trajectory_active,
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
							n_system_states)

 					log_p_active = log_likelihood_active + 
							log(prior_success_probability)

 				elseif accept_trajectory == false

 					log_p_active = -Inf

 				end
				
 			elseif loads[i] == i

				state_trajectory_active .= state_trajectory

 				log_likelihood_active = old_log_demarginalized_likelihood
 				log_p_active = log_likelihood_active + 
						log(prior_success_probability)


				# Proposed Load
 				loads[i] = 0
				loads_active, loads_inactive, n_system_states = 
					get_active_inactive_loads!(loads,
							loads_active,
							loads_inactive,
							n_system_states)

 				if n_system_states > 0

					generator, propagator, rho = get_generator!(loads_active, 
							n_system_states,
							rates,
							generator,
							propagator,
							rho)

 		    			state_trajectory_inactive, 
						accept_trajectory = sample_state_trajectory!( 
								absorption_rate,
								linear_drift_rate,
								rates,
								state_trajectory_inactive,
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
								n_system_states,
								filter_terms,
								log_observation_prob,
								prob_vec,
								intermediate_vec)

 					if accept_trajectory == true
						
 						log_likelihood_inactive = get_log_demarginalized_likelihood(
								absorption_rate,
								linear_drift_rate,
								rates, 
								state_trajectory_inactive,
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
								n_system_states)

  						log_p_inactive = log_likelihood_inactive +
 								log(1.0 - prior_success_probability)

 					elseif accept_trajectory == false

 						log_p_inactive = -Inf

 					end

 				elseif n_system_states == 0

 					# Posterior is 0 for a zero-state model. So logarithm
 					# of posterior is -Inf
 				 
 					log_p_inactive = -Inf
 				end

 			end

 			# The following procedure helps avoid overflow issues
 			if log_p_inactive == -Inf && log_p_active == -Inf

 				println("Trajectories Problematic")

 			else
 				max_val = max(log_p_active, log_p_inactive)
 				p_active = exp(log_p_active-max_val)
 				p_inactive = exp(log_p_inactive-max_val)

 				# Probability vector to activate or deactivate a load
				p_load[1] = p_active
				p_load[2] = p_inactive

 				# Normalize this probability vector
 				p_load .= p_load ./ sum(p_load)
  				loads[i] =  rand(rng, Categorical(p_load), 1)[1]

 				if loads[i] == 1 #Active load

 					loads[i] = i
 					old_log_demarginalized_likelihood = log_likelihood_active
 					state_trajectory .= state_trajectory_active

 				elseif loads[i] == 2 #Inactive load

 					loads[i] = 0
 					old_log_demarginalized_likelihood = log_likelihood_inactive
					state_trajectory .= state_trajectory_inactive

 				end
 			end
 		end
	end

  	return loads, state_trajectory
end

function save_mcmc_data(current_draw::Int64, 
	current_sample::Int64,
	mcmc_save_loads::Matrix{Int64},
	mcmc_save_rates::Matrix{Float64},
	mcmc_log_posterior::Vector{Float64},
	mcmc_save_trajectory::Matrix{Int64},
	photons_absorbed::Vector{Int64},
	emitted_donor_photons::Vector{Int64},
	emitted_acceptor_photons::Vector{Int64})

	# Save the data in HDF5 format.
	file_name = string(working_directory, "mcmc_output_", file_prefix, ".h5")

	fid = h5open(file_name,"w")

	write_dataset(fid, "mcmc_loads",
			view(mcmc_save_loads, 1:current_sample, :))
	write_dataset(fid, "mcmc_rates",
			view(mcmc_save_rates, 1:current_sample, :))
	write_dataset(fid, "mcmc_log_posterior",
		      		view(mcmc_log_posterior, 1:current_draw))
	write_dataset(fid, "mcmc_trajectory",
			   		view(mcmc_save_trajectory, 1:current_sample, :))
	write_dataset(fid, "mcmc_photons_absorbed", photons_absorbed)
	write_dataset(fid, "mcmc_emitted_donor_photons",
			   			emitted_donor_photons)
	write_dataset(fid, "mcmc_emitted_acceptor_photons",
			   			emitted_acceptor_photons)
	close(fid)

	return nothing
end

if plotting_on == true

	using Plots, Plots.Measures
	function plot_everything(draw, 
			n_collected_samples, 
			step_found_trajectory, 
			n_system_states, mcmc_log_posterior,
			photons_absorbed,
			emitted_donor_photons,
	    		emitted_acceptor_photons,
			bg_photons_donor,
			bg_photons_acceptor)
	
	    	time_data = bin_width*collect(1:n_bins)
		plot(time_data, 
		     (donor_channel_data .- donor_channel_bg) ./1000.0, 
		     color = :green, 
		     label = ("Donor Data"),
		     legendfontsize=14)
		plot_data = plot!(time_data, 
			(acceptor_channel_data .- acceptor_channel_bg) ./1000.0, 
#			xlabel="Time (s)", 
			ylabel="Intensity (a.u.)",
	            	legend = :topright, linewidth= 1.5, label = ("Acceptor Data"),
	            	xtickfontsize=18, ytickfontsize=18,
	            	xguidefontsize = 20, yguidefontsize = 20,
	            	fontfamily = "Computer Modern",
			top_margin = 5mm,
	            	right_margin=5mm, bottom_margin = 5mm,
	            	left_margin = 5mm, color = :red,
	            	dpi = 300, format = :svg)
	    	plot_apparent = plot(time_data, 
			(acceptor_channel_data .- acceptor_channel_bg) ./ 
				(acceptor_channel_data + donor_channel_data .- 
				(acceptor_channel_bg + donor_channel_bg)),
#			xlabel="Time (s)", 
			ylabel="E\$_{FRET}\$",
	            	legend = false, linewidth= 1.5,
	            	ylims = (0.0, 1.0),
	            	xtickfontsize=18, ytickfontsize=18,
	            	xguidefontsize = 20, yguidefontsize = 20,
	            	fontfamily = "Computer Modern",
	            	right_margin=5mm, bottom_margin = 5mm,
	            	left_margin = 5mm, color = :blue,
	            	dpi = 300, format = :svg)
	    	plot_stepdata = plot(time_data, step_found_trajectory, 
			seriestype=:steppost,
#			xlabel="Time (s)", 
			ylabel="State",
	            	legend = false, linewidth= 1.5,
	        	ylims = (minimum(step_found_trajectory) - 0.5, 
				 maximum(step_found_trajectory) + 0.2),
	        	yticks = (collect(convert(Float64, 
				minimum(step_found_trajectory)):1.0:convert(Float64, 
				maximum(step_found_trajectory))),
				collect(minimum(step_found_trajectory):1:maximum(step_found_trajectory))),
	        	xtickfontsize=18, ytickfontsize=18,
	        	xguidefontsize = 20, yguidefontsize = 20,
	        	fontfamily = "Computer Modern",
	        	right_margin=5mm, bottom_margin = 5mm,
	        	left_margin = 5mm, color = :blue,
	        	dpi = 300, format = :svg)
	
	    	plot_posterior = plot(mcmc_log_posterior[1:draw],
	 		xlabel="Iterations", ylabel="log-posterior",
	             	legend = false, linewidth= 1.5,
	         	xtickfontsize=18, ytickfontsize=18,
	         	xguidefontsize = 20, yguidefontsize = 20,
	         	fontfamily = "Computer Modern",
	         	right_margin=5mm, bottom_margin = 20mm,
	         	left_margin = 20mm, color = :blue,
	         	dpi = 300, format = :svg)
	
	    	plot_photons_absorbed = plot(time_data, photons_absorbed,
			xlabel="Time (s)", ylabel="Counts",
	      		legend = false, linewidth= 1.5,
	        	xtickfontsize=18, ytickfontsize=18,
	        	xguidefontsize = 20, yguidefontsize = 20,
	        	fontfamily = "Computer Modern",
	        	right_margin=5mm, bottom_margin = 5mm,
	        	left_margin = 5mm, color = :blue,
	        	dpi = 300, format = :svg)
	
	    	plot_emitted = plot(time_data, emitted_donor_photons,
			xlabel="Time (s)", ylabel="Counts",
	      		legend = :topright, label = ("Donor Emissons"), linewidth= 1.5,
			legendfontsize=14,
	        	xtickfontsize=18, ytickfontsize=18,
	        	xguidefontsize = 20, yguidefontsize = 20,
	        	fontfamily = "Computer Modern",
	        	right_margin=5mm, bottom_margin = 5mm,
	        	left_margin = 5mm, color = :green,
	        	dpi = 300, format = :svg)
	
	    	plot_emitted = plot!(time_data, emitted_acceptor_photons,
			xlabel="Time (s)", ylabel="Counts",
	      		legend = :topright, label = ("Acceptor Emissions"), linewidth= 1.5,
			legendfontsize=14,
	        	xtickfontsize=18, ytickfontsize=18,
	        	xguidefontsize = 20, yguidefontsize = 20,
	        	fontfamily = "Computer Modern",
			top_margin = 10mm,
	        	right_margin=5mm, bottom_margin = 10mm,
	        	left_margin = 5mm, color = :red,
	        	dpi = 300, format = :svg)
	
	    	plot_bg = plot(time_data, bg_photons_donor,
#			xlabel="Time (s)", 
			ylabel="Counts",
	      		legend = :topright, label = "Donor Background",  linewidth= 1.5,
			legendfontsize=14,
	        	xtickfontsize=18, ytickfontsize=18,
	        	xguidefontsize = 20, yguidefontsize = 20,
	        	fontfamily = "Computer Modern",
	        	right_margin=5mm, bottom_margin = 5mm,
	        	left_margin = 5mm, color = :green,
	        	dpi = 300, format = :svg)
	
	    	plot_bg = plot!(time_data, bg_photons_acceptor,
#			xlabel="Time (s)", 
			ylabel="Counts",
	      		legend = :topright, label = "Acceptor Background", linewidth= 1.5,
			legendfontsize=14,
	        	xtickfontsize=18, ytickfontsize=18,
	        	xguidefontsize = 20, yguidefontsize = 20,
	        	fontfamily = "Computer Modern",
			top_margin = 10mm,
	        	right_margin=5mm, bottom_margin = 5mm,
	        	left_margin = 5mm, color = :red,
	        	dpi = 300, format = :svg)
	
		plot_full = plot(plot_data,
			plot_apparent,
			plot_stepdata,
			plot_bg,
			plot_emitted,
			plot_posterior,
			layout=(6, 1), size=(3000, 1670), format = :svg)
	    	display(plot_full)
	
	 	if draw % save_frequency == 0
	 		savefig(plot_full, 
				string(working_directory, "final_plot_", 
					file_prefix , "_", 
					n_collected_samples, ".png" ))
	 	end
	
	    return nothing
	end
end
