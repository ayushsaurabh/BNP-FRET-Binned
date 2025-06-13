working_directory = string(pwd(),"/")
file_prefix = string("synthetic_trace_sCMOS_1") 


# Detector Parameters
const detection_eff::Float64 = 1.0
const da_crosstalk::Float64 = 0.0

const quantum_yield_d::Float64 = 1.0 #0.15
const quantum_yield_a::Float64 = 1.0 #0.27
const bin_width::Float64 = 0.04 # in seconds
const sensitivity_donor::Float64 = 0.46 # electrons per ADU
const sensitivity_acceptor::Float64 = 0.46 # electrons per ADU

# Inference Parameters

# Choose "nonparametric" if number of states is unknown and kept variable
# Choose "parametric" if number of states is known
const modeling_choice = "nonparametric"

if modeling_choice == "nonparametric"
	const expected_n_system_states::Int64 = 2
	const max_n_system_states::Int64 = 10
elseif modeling_choice == "parametric"
	const expected_n_system_states::Int64 = 2
	const max_n_system_states::Int64 = expected_n_system_states
end

# Maximum conformational rate that can be effectively
# estimated based on Nyquist criterion.
const max_confo_rate::Float64 = 1.0/(2.0*bin_width)

# Parameters for the prior on the transition rates. Chosen
# in order to promote escape rates smaller than Nyquist
# criterion. If data (likelihood) dominates, these values
# have relatively small influence on the results.
const gamma_prior_shape::Float64 = 1.0
const gamma_prior_scale::Float64 = max_confo_rate/(max_n_system_states) 
# Scale above is equal to standard deviation if shape = 1.0

# Parameters for determining step size needed
# for efficient exploration of the parameter
# space for rates and FRET efficiencies. Rates
# are explored using Normal distribution. FRET
# efficiencies are explored using Dirichlet
# distribution since the values are limited
# to the [0, 1] interval.
const step_size_rates::Float64 = 0.2 # Larger value, larger step size
const conc_parameter_FRET::Float64 = 1.0e4 # Larger value, smaller step size

# Parameters for simulated annealing to more
# efficiently explore parameter space during
# Monte Carlo sampling. Simulated annealing smoothens
# and widens the probability distribution by
# increasing temperature in order to
# make it easy for the sampler to get out of local
# maxima and approach global maximum. Annealing is
# repeated with a frequency = save_burn_in_period
# parameter below. Temperature decays exponentially.
# Burn in period corresponds to the time during which
# sampler is approaching the maximum of the
# probability distribution and is not fully converged.
# Convegence is typically evident when the posterior
# probabilities fluctuate about a fixed value (ideally maximum).
const starting_temperature::Float64 = 100.0
const annealing_constant::Float64 = 500.0

# Furthermore trajectories can be forced to collapse and
# have a fewer number of system states if overfitting 
# is seen due to fluorescence fluctuations that cannot 
# be characterized with detector noise and 
# Poissonian background fluctuations.
const trajectory_collapse_required::Bool = true

if trajectory_collapse_required == true
	const burn_in_period::Int64 = 9.0*annealing_constant
	const collapsed_burn_in_period::Int64 = 
			10.0*annealing_constant - burn_in_period
	const save_burn_in_period::Int64 = 
			burn_in_period+collapsed_burn_in_period
else
	const burn_in_period::Int64 = 10.0*annealing_constant
	const save_burn_in_period::Int64 = burn_in_period
end

const n_samples_to_save::Int64 = 200
const total_draws::Int64 = n_samples_to_save * save_burn_in_period

# Parameters for trajectory collapse
const FRET_eff_min_separation = 0.15
const escape_rate_max = 0.80 * max_confo_rate

# Parameters for printing results, plotting, and
# saving results onto hard drive
const output_frequency::Int64 = 100
const plotting_on::Bool = true
const plotting_frequency::Int64 = output_frequency
const save_frequency::Int64 = save_burn_in_period 
