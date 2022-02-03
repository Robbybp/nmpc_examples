import pyomo.environ as pyo
from pyomo.dae.flatten import flatten_dae_components

import idaes.core as idaes

from nmpc_examples.simple_pipeline.pipeline_model import (
    make_model,
    get_simulation_inputs,
)

from nmpc_examples.nmpc.dynamic_data import (
    find_nearest_index,
    interval_data_from_time_series,
    load_inputs_into_model,
)

from nmpc_examples.nmpc.model_helper import DynamicModelHelper

"""
Script for running "rolling-horizon" type simulatoin with pipeline model
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

    # Fix "inputs" in dynamic model: inlet pressure and outlet flow rate
    space = m.fs.pipeline.control_volume.length_domain
    x0 = space.first()
    xf = space.last()
    m.fs.pipeline.control_volume.flow_mass[:, xf].fix()
    m.fs.pipeline.control_volume.pressure[:, x0].fix()

    input_sequence = get_simulation_inputs(
        simulation_horizon=simulation_horizon,
        t_ptb=t_ptb,
    )
    sim_sample_points = input_sequence.get_time_points()
    n_cycles = len(sim_sample_points) - 1
    simulation_horizon = sim_sample_points[-1]
    #n_cycles = len(input_sequence[0])-1
    #simulation_horizon = input_sequence[0][-1]

    #
    # Load initial inputs into steady model
    #
    # With data series object:
    # - want to call helper.load_data_at_time
    # - this requires flattening the steady model, which I may not
    #   need to do to load inputs. Here it will be necessary to extract
    #   data later, so I won't worry about it for now.
    m_steady_helper = DynamicModelHelper(m_steady, m_steady.fs.time)
    initial_inputs = input_sequence.get_data_at_time(t0)
    #initial_inputs = {
    #    name: values[0] for name, values in input_sequence.get_data().items()
    #}
    m_steady_helper.load_data_at_time(initial_inputs)
    t0_steady = m_steady.fs.time.first()
    #for name, val in initial_inputs.items():
    #    m_steady.find_component(name)[t0_steady].fix(val)
    m_steady.fs.pipeline.control_volume.flow_mass[:, xf].fix()
    m_steady.fs.pipeline.control_volume.pressure[:, x0].fix()

    ipopt = pyo.SolverFactory("ipopt")
    ipopt.solve(m_steady, tee=True)

    #
    # Extract data from steady state model
    #
    #steady_scalar_vars, steady_dae_vars = flatten_dae_components(
    #    m_steady, m_steady.fs.time, pyo.Var
    #)
    #initial_data = {
    #    str(pyo.ComponentUID(var.referent)): var[t0_steady].value
    #    for var in steady_dae_vars
    #}
    #scalar_data = {
    #    str(pyo.ComponentUID(var)): var.value for var in steady_scalar_vars
    #}

    # With the helper class:
    scalar_data = m_steady_helper.get_scalar_data()
    initial_data = m_steady_helper.get_data_at_time(time=t0_steady)

    #
    # Load data into dynamic model
    #
    #for name, val in scalar_data.items():
    #    m.find_component(name).set_value(val)
    #for name, val in initial_data.items():
    #    var = m.find_component(name)
    #    for t in time:
    #        var[t].set_value(val)
    # With the model helper:
    # Again, we don't need to flatten to load data, but we will flatten
    # later anyway.
    m_helper = DynamicModelHelper(m, m.fs.time)
    m_helper.load_scalar_data(scalar_data)
    m_helper.load_data_at_time(initial_data, m.fs.time)

    # Solve as a sanity check -- should be square with zero infeasibility
    res = ipopt.solve(m, tee=True)
    pyo.assert_optimal_termination(res)

    #
    # Initialize data structure for simulation data
    #
    #scalar_vars, dae_vars = flatten_dae_components(m, time, pyo.Var)
    #_simulation_data = (
    #    [t0],
    #    {
    #        str(pyo.ComponentUID(var.referent)): [var[t0].value]
    #        for var in dae_vars
    #    },
    #)
    # With the model helper:
    simulation_data = m_helper.get_data_at_time([t0])

    simulation_time = simulation_data.get_time_points()
    for i in range(n_cycles):
        # time.first() in the model corresponds to sim_t0 in "simulation time"
        # time.last() in the model corresponds to sim_tf in "simulation time"
        sim_t0 = i*model_horizon
        sim_tf = (i+1)*model_horizon

        #
        # Extract inputs of sequence that are between sim_t0 and sim_tf
        #
        # ^ This is actually more general than we need
        #idx_t0 = find_nearest_index(simulation_time, sim_t0)
        #idx_tf = find_nearest_index(simulation_time, sim_tf)
        #extracted_inputs = (
        #    simulation_time[idx_t0:idx_tf + 1],
        #    {
        #        name: values[idx_t0:idx_tf + 1]
        #        for name, values in input_sequence[1].items()
        #    },
        #)
        #extracted_inputs = interval_data_from_time_series(extracted_inputs)

        #
        # Apply offset to time points so they are valid for model
        #
        #offset = sim_t0
        #inputs_for_model = {
        #    name: {
        #        (interval[0]-offset, interval[1]-offset): val
        #        for interval, val in inputs.items()
        #    } for name, inputs in extracted_inputs.items()
        #}

        #load_inputs_into_model(m, time, inputs_for_model)

        # With the model helper and data series object:
        # Here we are assuming that we want to apply the same input values
        # at every point in time. This is fine for now.
        # Note we don't need to apply an offset anywhere.
        inputs_to_apply = input_sequence.get_data_at_time(sim_tf)
        m_helper.load_data_at_time(inputs_to_apply)

        ipopt.solve(m, tee=True)

        #
        # Extract time series data from solved model
        #
        # Initial conditions have already been accounted for.
        # Note that this is only correct because we're using an implicit
        # time discretization.
        non_initial_time = list(time)[1:]
        #model_data = (
        #    non_initial_time,
        #    {
        #        str(pyo.ComponentUID(var.referent)): [
        #            var[t].value for t in non_initial_time
        #        ] for var in dae_vars
        #    },
        #)
        # With series data structure:
        model_data = m_helper.get_data_at_time(non_initial_time)

        #
        # Apply offset to data from model
        #
        #new_time_points = [t+offset for t in non_initial_time]
        #new_sim_data = (new_time_points, dict(model_data[1]))
        # With series data:
        model_data.shift_time_points(sim_t0)

        #
        # Extend simulation data with result of new simulation
        #
        #simulation_data[0].extend(new_time_points)
        #for name, values in simulation_data[1].items():
        #    values.extend(new_sim_data[1][name])
        # With series data:
        simulation_data.concatenate(model_data)

        #
        # Re-initialize model to final values.
        # This includes setting new initial conditions.
        #
        #for var in dae_vars:
        #    for t in time:
        #        var[t].set_value(var[tf].value)
        m_helper.propagate_values_at_time(tf)

    return simulation_data


if __name__ == "__main__":
    run_simulation()
