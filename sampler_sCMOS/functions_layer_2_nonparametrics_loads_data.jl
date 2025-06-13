#function choose_a_sign()
#	chosen_sign::Int64 = sign_vec[rand(rng, Categorical(vec_0pt5))]
#	return chosen_sign
#end


function get_active_inactive_loads!(loads::Vector{Int64}, 
			loads_active::Vector{Int64},
			loads_inactive::Vector{Int64},
			n_system_states::Int64,
			)

	n_system_states = count(x -> x != 0, loads) 
	n_inactive_states::Int64 = max_n_system_states - n_system_states

	loads_active .= 0
	loads_inactive .= 0

	loads_active[1:n_system_states] .= findall(x -> x != 0, loads)
	loads_inactive[1:n_inactive_states] .= findall(x -> x == 0, loads)

	return loads_active, loads_inactive, n_system_states
end


function get_generator!(loads_active::Vector{Int64}, 
			n_system_states::Int64,
			rates::Vector{Float64}, 
			generator::Matrix{Float64}, 
			propagator::Matrix{Float64}, 
			rho::Vector{Float64})

	generator .= 0.0 
	for i in 1:n_system_states
		for j in 1:n_system_states
			ij = (loads_active[i]-1)*max_n_system_states + loads_active[j]
			if i != j
				generator[i, j] = rates[ij]
			end
		end
	end
	for i in 1:n_system_states
		generator[i,i] = -sum(view(generator, i,:))
	end

	rho .= 0.0
	rho[1:n_system_states] .= abs.(vec(nullspace(Transpose(view(generator, 
						1:n_system_states, 
						1:n_system_states)), 
						rtol = 1.0e-14)))
	rho .= rho ./ sum(rho)

	if size(findall(x -> x != 0.0, rho))[1] != n_system_states 
		println("Numerical Instability: Please increase tolerance for generator 
			matrix rank/nullspace computation to estimate the steady-state 
			probability vector rho")
		println("generator =" )
		display(view(generator, 1:n_system_states, 1:n_system_states))
		println("rho =" )
		display(view(rho, 1:n_system_states))
	end

	propagator .= 0.0
	propagator[1:n_system_states, 1:n_system_states] .= 
					exp(bin_width .* 
					  view(generator, 1:n_system_states, 
						    1:n_system_states))

	return generator, propagator, rho
end

function get_log_demarginalized_likelihood_rates(
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
			n_system_states::Int64)

	log_demarginalized_likelihood::Float64 = 0.0

 	if n_system_states > 1
		for i in 1:n_system_states
			initial_state::Int64 = i
			final_state::Int64 = findall(x-> x == state_trajectory[1], loads_active)[1]
			log_demarginalized_likelihood += log(rho[initial_state]) +
								log(propagator[initial_state, final_state])
		end
		for bin in 2:n_bins
			initial_state::Int64 = findall(x-> x == state_trajectory[bin-1], loads_active)[1]
			final_state::Int64 = findall(x-> x == state_trajectory[bin], loads_active)[1]
			log_demarginalized_likelihood += log(propagator[initial_state,
								final_state])
		end
	end

	return log_demarginalized_likelihood
end

function get_log_demarginalized_likelihood_FRET(
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
			n_system_states::Int64)

	log_demarginalized_likelihood::Float64 = 0.0

 	for bin in 1:n_bins
 		log_demarginalized_likelihood +=
 			get_log_observation_prob(bin, 
					state_trajectory[bin],
 					photons_absorbed[bin],
					emitted_donor_photons[bin],
 					emitted_acceptor_photons[bin],
					bg_photons_donor[bin],
 					bg_photons_acceptor[bin],
					absorption_rate,
					linear_drift_rate,
					rates)
 	end

	return log_demarginalized_likelihood
end



function get_log_demarginalized_likelihood(
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
			n_system_states::Int64)

	log_demarginalized_likelihood::Float64 = 0.0

 	if n_system_states > 1
		for i in 1:n_system_states
			initial_state::Int64 = i
			final_state::Int64 = findall(x-> x == state_trajectory[1], loads_active)[1]
			log_demarginalized_likelihood += log(rho[initial_state]) +
								log(propagator[initial_state, final_state])
		end
		for bin in 2:n_bins
			initial_state::Int64 = findall(x-> x == state_trajectory[bin-1], loads_active)[1]
			final_state::Int64 = findall(x-> x == state_trajectory[bin], loads_active)[1]
			log_demarginalized_likelihood += log(propagator[initial_state,
								final_state])
		end
	end
 	for bin in 1:n_bins
 		log_demarginalized_likelihood +=
 			get_log_observation_prob(bin, 
					state_trajectory[bin],
 					photons_absorbed[bin],
					emitted_donor_photons[bin],
 					emitted_acceptor_photons[bin],
					bg_photons_donor[bin],
 					bg_photons_acceptor[bin],
					absorption_rate,
					linear_drift_rate,
					rates)
 	end

	return log_demarginalized_likelihood
end

function get_log_likelihood_observations_only(
			absorption_rate::Float64,
			linear_drift_rate::Float64,
			rates::Vector{Float64},
			state_trajectory::Vector{Int64},
			photons_absorbed::Vector{Int64},
			emitted_donor_photons::Vector{Int64},
			emitted_acceptor_photons::Vector{Int64},
			bg_photons_donor::Vector{Int64},
			bg_photons_acceptor::Vector{Int64})

 	log_likelihood_obs_only::Float64 = 0.0
 	for bin in 1:n_bins
 		log_likelihood_obs_only +=
 				get_log_observation_prob(bin, state_trajectory[bin],
 					photons_absorbed[bin],
					emitted_donor_photons[bin],
 					emitted_acceptor_photons[bin],
					bg_photons_donor[bin],
 					bg_photons_acceptor[bin],
					absorption_rate,
					linear_drift_rate,
					rates)
 	end
	return log_likelihood_obs_only
end




function get_reduced_propagator!(bin::Int64,
			loads::Vector{Int64},
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
			reduced_propagator_transpose::Matrix{Float64})

	reduced_propagator .= propagator

	for state in 1:n_system_states
		obs_prob::Float64 = exp(get_log_observation_prob(bin, loads_active[state],
 						photons_absorbed[bin],
						emitted_donor_photons[bin],
 						emitted_acceptor_photons[bin],
						bg_photons_donor[bin],
 						bg_photons_acceptor[bin],
						absorption_rate,
						linear_drift_rate,
						rates))
		reduced_propagator[state, :] .= obs_prob .* view(reduced_propagator, state, :)
	end

	transpose!(reduced_propagator_transpose, reduced_propagator)
	reduced_propagator .= reduced_propagator_transpose 

	return reduced_propagator
end

# State Trajectory
function sample_state_trajectory!(
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
			n_system_states::Int64,
			filter_terms::Matrix{Float64},
			log_observation_prob::Vector{Float64},
			prob_vec::Vector{Float64},
			intermediate_vec::Vector{Float64})

	# Get Filter Terms for State Trajectory
   	filter_terms, 
		accept_trajectory::Bool =
				get_filter_terms!( 
					absorption_rate,
					linear_drift_rate,
					rates, 
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

		#Backward Sampling
		state_trajectory[n_bins] = loads_active[rand(rng, Categorical(
						filter_terms[ 1:n_system_states, n_bins]), 1)[1]]
		prob_vec[n_system_states+1:end] .= 0.0

		for bin in n_bins-1:-1:1
			final_state::Int64 = findall(x-> x == state_trajectory[bin+1], 
					      loads_active)[1]
			prob_vec[1:n_system_states] .= view(filter_terms, 1:n_system_states, bin) .*
						view(propagator, 1:n_system_states, final_state)
			prob_vec .= prob_vec/sum(prob_vec)
			state_trajectory[bin] = loads_active[rand(rng, 
						Categorical(prob_vec[1:n_system_states]), 1)[1]]
		end


	else
 			println("TRAJECTORY REJECTED")
	end

	return state_trajectory, accept_trajectory
end
