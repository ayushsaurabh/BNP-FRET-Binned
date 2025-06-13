# Get observation probability
function get_log_observation_prob(bin::Int64,
			state_label::Int64,
			photons_absorbed_bin::Int64,
			emitted_donor_photons_bin::Int64,
			emitted_acceptor_photons_bin::Int64,
			bg_photons_donor_bin::Int64,
			bg_photons_acceptor_bin::Int64,
			absorption_rate::Float64,
			linear_drift_rate::Float64,
			rates::Vector{Float64})


	FRET_eff::Float64 = rates[(state_label - 1) * max_n_system_states +
							state_label]

	# Shot noise
 	log_observation_prob::Float64 = logpdf(Poisson(absorption_rate+
 				bin*linear_drift_rate), photons_absorbed_bin) +

	logpdf(Multinomial(photons_absorbed_bin,
		[(1.0-FRET_eff)*quantum_yield_d,
			FRET_eff*quantum_yield_a]),
		[emitted_donor_photons_bin,
		 	emitted_acceptor_photons_bin]) +

	# Background
 	logpdf(Poisson(mean_bg_photons_donor), bg_photons_donor_bin)+
 	logpdf(Poisson(mean_bg_photons_acceptor), bg_photons_acceptor_bin)+

	# Noise model after EM multiplication
 	logpdf(Normal(((1.0-da_crosstalk)*emitted_donor_photons_bin+
 		       	bg_photons_donor_bin)*detection_eff/sensitivity_donor+
 					offset_donor, read_out_noise_donor),
 					donor_channel_data[bin]) +
 	logpdf(Normal((da_crosstalk*emitted_donor_photons_bin+
 				emitted_acceptor_photons_bin+
 			bg_photons_acceptor_bin)*detection_eff/sensitivity_acceptor+
 					offset_acceptor, read_out_noise_acceptor),
 					acceptor_channel_data[bin])


 	return log_observation_prob
end




function get_filter_terms!(
		absorption_rate::Float64,
		linear_drift_rate::Float64,
		rates::Vector{Float64},
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
		intermediate_vec::Vector{Float64},
)

	filter_terms .= 0.0
	log_observation_prob .= 0.0
	prob_vec .= rho 

	bin::Int64 = 1
	for system_state in 1:n_system_states
			log_observation_prob[system_state] =
				get_log_observation_prob(bin, 
					loads_active[system_state],
					photons_absorbed[bin], 
					emitted_donor_photons[bin],
					emitted_acceptor_photons[bin],
					bg_photons_donor[bin],
 					bg_photons_acceptor[bin],
					absorption_rate,
					linear_drift_rate,
					rates)
	end

	log_prob_max::Float64 = maximum(view(log_observation_prob, 1:n_system_states))

	# The rescaling below helps avoid underflow/overflow issues.
        prob_vec .= prob_vec .* exp.(log_observation_prob .- log_prob_max)
        prob_vec[n_system_states+1:end] .= 0.0
        prob_vec .= prob_vec/sum(prob_vec)

	filter_terms[1:n_system_states, bin] .= view(prob_vec, 1:n_system_states)

	local accept_trajectory::Bool

        if size(findall(x-> isnan(x) == true, view(prob_vec, 1:n_system_states)))[1]> 0 ||
                        log_observation_prob == NaN

		accept_trajectory = false
 		println("TRAJECTORY REJECTED 1")

		return filter_terms, accept_trajectory

  	end
	for bin in 2:n_bins

		transpose!(propagator_transpose, propagator)
		mul!(intermediate_vec, propagator_transpose, prob_vec)
 		prob_vec .= intermediate_vec 

		for system_state in 1:n_system_states
			log_observation_prob[system_state] =
				get_log_observation_prob(bin, 
						loads_active[system_state],
						photons_absorbed[bin],
						emitted_donor_photons[bin],
						emitted_acceptor_photons[bin],
						bg_photons_donor[bin],
 						bg_photons_acceptor[bin],
						absorption_rate,
						linear_drift_rate,
						rates)
		end
		log_prob_max = maximum(log_observation_prob)
                prob_vec .= prob_vec .* exp.(log_observation_prob .- log_prob_max)
                prob_vec[n_system_states+1:end] .= 0.0
                prob_vec .= prob_vec/sum(prob_vec)

                filter_terms[1:n_system_states, bin] .= view(prob_vec, 1:n_system_states)

                if size(findall(x-> isnan(x) == true, view(prob_vec, 1:n_system_states)))[1]> 0 ||
                                log_observation_prob == NaN

			accept_trajectory = false
 			println("TRAJECTORY REJECTED 2")

			return filter_terms, accept_trajectory

  		end
	end
	accept_trajectory = true

	return filter_terms, accept_trajectory
end

