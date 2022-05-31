import pyomo.environ as pyo
from pyomo.dae import ContinuousSet
from pyomo.dae.flatten import flatten_dae_components

from nmpc_examples.pipeline.pipeline_model import make_model

from nmpc_examples.mhe.mhe_constructor import (
    construct_measurement_variables_constraints,
    construct_disturbed_model_constraints,
    get_error_disturbance_cost,
    activate_disturbed_constraints_based_on_original_constraints,
)

from nmpc_examples.nmpc.model_linker import DynamicVarLinker
from nmpc_examples.nmpc.model_helper import DynamicModelHelper
from nmpc_examples.nmpc.dynamic_data.series_data import TimeSeriesData

import json
import os
import matplotlib.pyplot as plt

"""
Script to run an MHE simulation with a single natural gas pipeline model.
"""


def run_mhe(
        simulation_horizon=20.0,
        estimator_horizon=20.0,
        sample_period=2.0,
        ):
    """
    Run a simple MHE problem on a pipeline model.
    """
    #
    # Make dynamic model
    #
    nxfe = 4
    ntfe_per_sample = 4
    n_cycles = round(simulation_horizon/sample_period)
    m_plant = make_model(
        horizon=sample_period,
        ntfe=ntfe_per_sample,
        nxfe=nxfe,
    )
    time = m_plant.fs.time
    t0 = time.first()
    tf = time.last()

    ipopt = pyo.SolverFactory("ipopt")

    # Fix "inputs" in dynamic model: inlet pressure and outlet flow rate
    space = m_plant.fs.pipeline.control_volume.length_domain
    x0 = space.first()
    xf = space.last()
    m_plant.fs.pipeline.control_volume.flow_mass[:, xf].fix()
    m_plant.fs.pipeline.control_volume.pressure[:, x0].fix()

    #
    # Make steady model for plant's and estimator's initializations
    #
    m_steady = make_model(dynamic=False, nxfe=nxfe)
    m_steady.fs.pipeline.control_volume.flow_mass[:, x0].fix(
        3.0e5*pyo.units.kg/pyo.units.hr
    )
    m_steady.fs.pipeline.control_volume.pressure[:, x0].fix(
        57.0*pyo.units.bar
    )

    ipopt.solve(m_steady, tee=True)

    #
    # Extract data from steady state model
    #
    m_steady_helper = DynamicModelHelper(m_steady, m_steady.fs.time)
    initial_data = m_steady_helper.get_data_at_time(t0)
    scalar_data = m_steady_helper.get_scalar_variable_data()

    #
    # Load data into dynamic model
    #
    use_linker = False
    # Doesn't work if I change use_linker to True.
    # TODO: fix this. If I construct the DynamicVarLinker,
    # the code runs but one of the solves doesn't converge...
    if use_linker:
        # If I want to use DynamicVarLinker:
        # (E.g. if we didn't know that names in the steady model would
        # be valid for the dynamic model.)
        steady_scalar_vars, steady_dae_vars = flatten_dae_components(
            m_steady, m_steady.fs.time, pyo.Var
        )
        steady_dae_vars_in_plant = [
            m_plant.find_component(var.referent) for var in steady_dae_vars
        ]
        plant_steady_linker = DynamicVarLinker(
            steady_dae_vars,
            steady_dae_vars_in_plant,
            m_steady.fs.time,
            m_plant.fs.time,
        )
        plant_steady_linker.transfer()

    else:
        # If I want to use DynamicModelHelper:
        m_plant_helper = DynamicModelHelper(m_plant, m_plant.fs.time)
        m_plant_helper.load_scalar_data(scalar_data)
        m_plant_helper.load_data_at_time(initial_data)

    # Solve as a sanity check -- should be square with zero infeasibility
    res = ipopt.solve(m_plant, tee=True)
    pyo.assert_optimal_termination(res)

    #
    # Initialize data structure for simulation data
    #
    # What is the simulation? The plant trajectories as a pre-determined
    # sequence of inputs is applied?
    # We also will store data from the estimator?
    # Do data from the simulation and estimator exist on the same "real"
    # time points?
    # The simulation always runs one sample ahead of the estimator.
    # So maybe we run the simulation, then apply the estimator so that they
    # determine/estimate states at the same time points.
    sim_data = m_plant_helper.get_data_at_time([t0])

    #
    # Construct dynamic model for estimator
    #
    samples_per_estimator = round(estimator_horizon/sample_period)
    ntfe_per_estimator = ntfe_per_sample * samples_per_estimator
    m_estimator = make_model(
        horizon=estimator_horizon,
        ntfe=ntfe_per_estimator,
        nxfe=nxfe,
    )
    # Fix inputs at all time points
    m_estimator.fs.pipeline.control_volume.pressure[:, x0].fix()
    m_estimator.fs.pipeline.control_volume.flow_mass[:, xf].fix()
    # Estimator should have control inputs fixed. These should be
    # supplied by some external source, and rotated every sample period.

    # Make helper object w.r.t. time
    m_estimator_time_helper = DynamicModelHelper(
        m_estimator, m_estimator.fs.time
    )
    # Estimator helper use this to load and extract data.
    # Data will be loaded from a previous simulation, plus noise.

    #
    # Make sample-point set for measurements and model disturbances
    #
    sample_points = [
        t0 + sample_period*i for i in range(samples_per_estimator+1)
    ]
    m_estimator.fs.sample_points = ContinuousSet(initialize=sample_points)
    # Does this need to be a ContinuousSet?
    # Note that the estimator time set is defined in the positive direction.
    # (Might be interesting to consider defining it in [-H, 0])

    #
    # Construct components for measurements and measurement errors
    #
    m_estimator.estimation_block = pyo.Block()
    esti_blo = m_estimator.estimation_block

    cv = m_estimator.fs.pipeline.control_volume
    # Flow rate at all spatial discretization points is chosen
    # as the measurements. Not a "sufficient subset" of "dynamic degrees
    # of freedom."
    # These are "the variables that get measured"
    measured_variables = [
        pyo.Reference(cv.flow_mass[:, x]) for x in list(space)[:-1]
    ]
    #
    measurement_info = construct_measurement_variables_constraints(
        m_estimator.fs.sample_points,
        measured_variables,
    )
    # These are a set of integer indices for measurement variables?
    esti_blo.measurement_set = measurement_info[0]
    # These are... the variables passed as arguments? Or "duplicated
    # variables" corresponding to them?
    esti_blo.measurement_variables = measurement_info[1]
    # Measurement variables should be fixed all the time.
    # These are fixed, rather than the "measured_variables" above.
    # Those above variables are the "estimates" (that happen to be measured)
    # rather than the "measurements" which are fixed.
    esti_blo.measurement_variables.fix()
    # Some delta for estimate-variable mismatch that we will minimize
    esti_blo.measurement_error_variables = measurement_info[2]
    # x + delta = err, I assume
    # No. delta ~is~ the error
    # est + delta = meas
    esti_blo.measurement_constraints = measurement_info[3]

    #
    # Construct disturbed model constraints
    #
    # I assume we are going to deactivate these constraints, then
    # construct new constraints with error variables added to the
    # model.
    flatten_momentum_balance = [
        # Why do we keep this equation at space.first()?
        pyo.Reference(cv.momentum_balance[:,x]) for x in space
    ]
    flatten_material_balance = [
        pyo.Reference(cv.material_balances[:,x,"Vap","natural_gas"])
        for x in list(space)[:-1]
    ]
    model_constraints_to_be_disturbed = \
        flatten_momentum_balance + flatten_material_balance

    # Returns new constraints, new variables, and a set that indexes
    # them. This seems reasonable. Disturbance constraints are
    # indexed by time, but variables are only indexed by sample points?
    # I think this makes sense. Is this the best way to parameterize
    # this function?
    # - construct_disturbance_constraints(time, constraints, disturbance_vars)
    #   where disturbance vars are indexed by time (somehow)?
    model_disturbance_info = construct_disturbed_model_constraints(
        m_estimator.fs.time,
        m_estimator.fs.sample_points,
        model_constraints_to_be_disturbed,
    )
    esti_blo.disturbance_set = model_disturbance_info[0]
    # Should this be an indexed variable, or a list of variables
    # indexed only by time? The latter makes it easier for these to have
    # human-readable names.
    esti_blo.disturbance_variables = model_disturbance_info[1]
    esti_blo.disturbed_constraints = model_disturbance_info[2]

    # What is going on here? Activates disturbance constraints...
    # only if the corresponding constraints are active in the model?
    activate_disturbed_constraints_based_on_original_constraints(
        m_estimator.fs.time,
        m_estimator.fs.sample_points,
        esti_blo.disturbance_variables,
        model_constraints_to_be_disturbed,
        esti_blo.disturbed_constraints,
    )

    # Make helper object w.r.t. sample points
    #
    # What are we doing here? This identifies variables indexed
    # by sample_points? Then we extract and pass around data?
    m_estimator_spt_helper = DynamicModelHelper(
        m_estimator, m_estimator.fs.sample_points
    )

    #
    # Construct least square objective to minimize measurement errors
    # and model disturbances
    #
    measurement_error_weights = {
        pyo.ComponentUID(var.referent): 1.0
        for var in measured_variables
    }
    # Here we are getting the tracking cost for measured variables.
    # Measurements are only defined at sample points, so we can only
    # penalize them at sample points. Why is time necessary?
    # Is the returned object indexed by sample_points? Yes.
    # We should be able to do this with existing get_tracking_cost function...
    m_estimator.measurement_error_cost = get_error_disturbance_cost(
        m_estimator.fs.time,
        m_estimator.fs.sample_points,
        measured_variables,
        esti_blo.measurement_error_variables,
        measurement_error_weights,
    )

    model_disturbance_weights = {
        # This is reminding me that I need a data structure for
        # constant data.
        **{pyo.ComponentUID(con.referent): 10.0
           for con in flatten_momentum_balance},
        **{pyo.ComponentUID(con.referent): 20.0
           for con in flatten_material_balance},
    }
    # Same comment as above, shouldn't we be able to do this with
    # existing functions?
    m_estimator.model_disturbance_cost = get_error_disturbance_cost(
        m_estimator.fs.time,
        m_estimator.fs.sample_points,
        model_constraints_to_be_disturbed,
        esti_blo.disturbance_variables,
        model_disturbance_weights,
    )

    m_estimator.squred_error_disturbance_objective = pyo.Objective(
        # Should we skip these cost functions at t=0?
        expr=(sum(m_estimator.measurement_error_cost.values()) +
              sum(m_estimator.model_disturbance_cost.values())
              )
    )

    #
    # Initialize dynamic model to initial steady state
    #
    # TODO: load a time-series measurements & controls as initialization
    # This looks fine.
    m_estimator_time_helper.load_scalar_data(scalar_data)
    m_estimator_time_helper.load_data_at_time(initial_data, m_estimator.fs.time)

    for index, var in enumerate(measured_variables):
        for spt in m_estimator.fs.sample_points:
            esti_blo.measurement_variables[index, spt].set_value(var[spt].value)

    #
    # Initialize data sturcture for estimates
    #
    estimate_data = m_estimator_time_helper.get_data_at_time([t0])

    #
    # Set up a model linker to send measurements to estimator to update
    # measurement variables
    #
    # I think a find_component to set up the lists of associated variables
    # is reasonable.
    measured_variables_in_plant = [m_plant.find_component(var.referent)
                                   for var in measured_variables
    ]
    # "measurement_variables" (as opposed to "measured_variables")
    # are fixed, right? Yes, they are.
    flatten_measurements = [
        pyo.Reference(esti_blo.measurement_variables[index, :])
        for index in esti_blo.measurement_set
    ]
    # This sends measurements from plant to estimator.
    measurement_linker = DynamicVarLinker(
        measured_variables_in_plant, # Indexed by plant's time
        flatten_measurements,        # Indexed by estimator sample points
    )

    # Set up a model linker to send measurements to estimator to initialize
    # measured variables
    #
    # Sends measurement values from plant to the actual variables in the
    # estimator. Just for initialization? Should we not already have a
    # way to associate the "measurement vars" and "measured vars"?
    # This DynamicVarLinker seems unnecessary to me. TODO: Try to refactor
    # it out.
    estimate_linker = DynamicVarLinker(
        measured_variables_in_plant, # Indexed by plant's time
        measured_variables,          # Indexed by controller's time
    )

    #
    # Load control input data for simulation
    #
    directory = os.path.abspath(os.path.dirname(__file__))
    # Where does this file come from?
    filepath = os.path.join(directory, "control_input_data.json")
    with open(filepath, "r") as fr:
        control_data = json.load(fr)

    # control_data seems to be a list?
    # TODO: Can we make it a serialized TimeSeriesData?
    # control_data is actually a {cuid: [value-list]} dict.

    # This seems incorrect. Why are we using len(control_data)?
    # This is the number of control inputs.
    control_input_time = [i*sample_period for i in range(len(control_data))]
    control_inputs = TimeSeriesData(
        control_data, control_input_time, time_set=None
    )

    for i in range(n_cycles):
        # time.first() in the model corresponds to sim_t0 in "simulation time"
        # time.last() in the model corresponds to sim_tf in "simulation time"
        sim_t0 = i*sample_period
        sim_tf = (i + 1)*sample_period

        #
        # Load inputs into plant
        #
        # Why do we get control inputs at sim_t0? Doesn't this correspond
        # to time.first() in the plant? Seems a little odd that we get
        # the input at sim_t0, then load it everywhere ~except~ the initial
        # point.
        current_control = control_inputs.get_data_at_time(time=sim_t0)
        non_initial_plant_time = list(m_plant.fs.time)[1:]
        m_plant_helper.load_data_at_time(
            current_control, non_initial_plant_time
        )

        res = ipopt.solve(m_plant, tee=True)
        pyo.assert_optimal_termination(res)

        #
        # Extract time series data from solved model
        #
        # Initial conditions have already been accounted for.
        # Note that this is only correct because we're using an implicit
        # time discretization.
        #
        # If we were using an explicit discretization, the initial conditions
        # would need to use the updated control inputs. Therefore, they would
        # be different after the solve from the final values of the previous
        # sample, and should be overridden.
        # For an explicit discretization, we should extend with
        # list(m_plant.fs.time)[:-1]
        #
        non_initial_time = list(m_plant.fs.time)[1:]
        model_data = m_plant_helper.get_data_at_time(non_initial_time)

        #
        # Apply offset to data from model
        #
        model_data.shift_time_points(sim_t0)

        #
        # Extend simulation data with result of new simulation
        #
        sim_data.concatenate(model_data)

        #
        # Load measurements from plant to estimator
        #
        plant_tf = m_plant.fs.time.last()
        estimator_tf = m_estimator.fs.time.last()
        # This loads measurements from the plant into what are essentially
        # mutable parameters in the controller.
        measurement_linker.transfer(plant_tf, estimator_tf)

        #
        # Initialize measured variables within the last sample period to
        # current measurements
        #
        # Here we are initializing the state variables that happen to be
        # measurable. And we are initializing them from the same state variables
        # in the plant.
        last_sample_period = list(m_estimator.fs.time)[-ntfe_per_sample:]
        # So yes, we could do this without the linker object.
        # And presumably we could update using the newly updated measurement
        # variables. The linker syntax is quite convenient, however.
        # for index, var in enumerate(measured_variables):
        #     for tp in last_sampel_period:
        #         var[tp].set_value(blo.measurement_variables[index, estimator_tf])
        estimate_linker.transfer(tf, last_sample_period)

        # Load inputs to estimator
        m_estimator_time_helper.load_data_at_time(
            current_control, last_sample_period
        )

        # Degrees of freedom in the estimator should be fine. Inputs fixed,
        # states unfixed everywhere.
        res = ipopt.solve(m_estimator, tee=True)
        pyo.assert_optimal_termination(res)

        #
        # Extract estimate data from estimator
        #
        # Extract the new estimates (at tf) from the estimator. Any reason we
        # don't extract all the estimates in the last sample? Because we only
        # penalize the estimate-measurement deviation at the sample points?
        #
        estimator_data = m_estimator_time_helper.get_data_at_time([estimator_tf])
        # Shift time points from "estimator time" to "simulation time"
        #
        # We want these estimates to exist at time sim_tf. It's a little
        # annoying that we have to shift by this difference. It would be nicer
        # if these values existed at t0... Then we would have to shift by t0,
        # but this is often zero, which is nice. NOTE: I think in my NMPC
        # simulation, I assume that t0 in the controller is zero. I probably
        # need to take a difference like this in the controller as well.
        #
        estimator_data.shift_time_points(sim_tf-estimator_tf)

        #
        # Extend data structure of estimates
        #
        # estimate_data vs. estimator_data might be a little confusing.
        # Maybe something like computed_estimates for the "real-time"
        # data structure and estimator_data for the single-model data
        # structure.
        #
        estimate_data.concatenate(estimator_data)

        #
        # Re-initialize estimator model
        #
        # Shift time and sample-point indexed variables by one sample
        # period.
        #
        m_estimator_time_helper.shift_values_by_time(sample_period)
        m_estimator_spt_helper.shift_values_by_time(sample_period)

        #
        # Re-initialize model to final values.
        # This sets new initial conditions, including inputs.
        #
        # We could also shift_values_by_time, but this is more explicit.
        #
        m_plant_helper.copy_values_at_time(source_time=plant_tf)

    return sim_data, estimate_data


def plot_states_estimates_from_data(
        state_data,
        estimate_data,
        names,
        show=False
        ):
    state_time = state_data.get_time_points()
    states = state_data.get_data()
    estimate_time = estimate_data.get_time_points()
    estimates = estimate_data.get_data()
    for i, name in enumerate(names):
        fig, ax = plt.subplots()
        state_values = states[name]
        estimate_values = estimates[name]
        ax.plot(state_time, state_values, label="Plant states")
        ax.plot(estimate_time, estimate_values, "o", label="Estimates")
        ax.set_title(name)
        ax.set_xlabel("Time (hr)")
        ax.legend()

        if show:
            fig.show()
        else:
            fname = "state_estimate%s.png" % i
            fig.savefig(fname, transparent=False)


if __name__ == "__main__":
    simulation_data, estimate_data = run_mhe(
        simulation_horizon=4.0,
    )
    plot_states_estimates_from_data(
        simulation_data,
        estimate_data,
        [
            pyo.ComponentUID("fs.pipeline.control_volume.flow_mass[*,0.0]"),
            pyo.ComponentUID("fs.pipeline.control_volume.pressure[*,1.0]"),
        ],
    )
