import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc

from nmpc_examples.pipeline.pipeline_model import (
    make_model,
    get_simulation_inputs,
)

"""
Script for running "rolling-horizon" type simulation with pipeline model.

"""

def run_simulation(
    simulation_horizon=20.0,
    t_ptb=4.0,
):
    """
    Runs a simulation where outlet (demand) flow rate is perturbed
    at a specified time.
    """
    #
    # Make steady model (for initial conditions) and dynamic model
    #
    m_steady = make_model(dynamic=False)
    model_horizon = 2.0
    ntfe = 4
    m = make_model(horizon=model_horizon, ntfe=ntfe)
    time = m.fs.time
    t0 = time.first()
    tf = time.last()
    t0_steady = m_steady.fs.time.first()

    # Make helper objects to extract data from flattened variables
    m_helper = mpc.DynamicModelInterface(m, m.fs.time)
    m_steady_helper = mpc.DynamicModelInterface(m_steady, m_steady.fs.time)

    # Fix "inputs" in dynamic model: inlet pressure and outlet flow rate
    space = m.fs.pipeline.control_volume.length_domain
    x0 = space.first()
    xf = space.last()
    m.fs.pipeline.control_volume.flow_mass[:, xf].fix()
    m.fs.pipeline.control_volume.pressure[:, x0].fix()

    #
    # Get sequence of input values. This is a TimeSeriesData object
    #
    input_sequence = get_simulation_inputs(
        simulation_horizon=simulation_horizon,
        t_ptb=t_ptb,
    )
    sim_sample_points = input_sequence.get_time_points()
    n_cycles = len(sim_sample_points) - 1
    simulation_horizon = sim_sample_points[-1]

    #
    # Load initial inputs into steady model
    #
    initial_inputs = input_sequence.get_data_at_time(t0_steady)
    m_steady_helper.load_data(initial_inputs)

    # Fix inputs in steady state model
    m_steady.fs.pipeline.control_volume.flow_mass[:, xf].fix()
    m_steady.fs.pipeline.control_volume.pressure[:, x0].fix()

    ipopt = pyo.SolverFactory("ipopt")
    ipopt.solve(m_steady, tee=True)

    #
    # Extract data from steady state model
    #
    scalar_data = m_steady_helper.get_scalar_variable_data()
    initial_data = m_steady_helper.get_data_at_time(time=t0_steady)

    #
    # Load data into dynamic model
    #
    m_helper.load_data(scalar_data)
    m_helper.load_data(initial_data, time_points=m.fs.time)
    # ^ time here is an optional argument. Default is to load at all time
    # Note here that we're loading "scalar data" into time-indexed variables
    # at all time. Is this clear from the code?

    # Solve as a sanity check -- should be square with zero infeasibility
    res = ipopt.solve(m, tee=True)
    pyo.assert_optimal_termination(res)

    #
    # Initialize data structure for simulation data
    #
    # The returned type is TimeSeriesData
    simulation_data = m_helper.get_data_at_time([t0])

    for i in range(n_cycles):
        # time.first() in the model corresponds to sim_t0 in "simulation time"
        # time.last() in the model corresponds to sim_tf in "simulation time"
        sim_t0 = i*model_horizon
        sim_tf = (i+1)*model_horizon

        #
        # Extract data from input sequence and load into the dynamic model
        #
        non_initial_time = list(time)[1:]
        inputs_to_apply = input_sequence.get_data_at_time(sim_tf)
        # Here we are assuming that we want to apply the same input
        # values at every point in time.
        # Becaucse we use an implicit discretization, we don't need
        # to apply inputs at t0.
        m_helper.load_data(inputs_to_apply, time_points=non_initial_time)

        ipopt.solve(m, tee=True)

        #
        # Extract time series data from solved model
        #
        # Initial conditions have already been accounted for.
        # Note that this is only correct because we're using an implicit
        # time discretization.
        model_data = m_helper.get_data_at_time(non_initial_time)

        #
        # Apply offset to data from model
        #
        model_data.shift_time_points(sim_t0)

        #
        # Extend simulation data with result of new simulation
        #
        simulation_data.concatenate(model_data)

        #
        # Re-initialize model to final values.
        # This sets new initial conditions, including inputs.
        #
        m_helper.copy_values_at_time(source_time=tf)

    return simulation_data


if __name__ == "__main__":
    run_simulation()
