import pyomo.environ as pyo
from pyomo.dae.flatten import flatten_dae_components

from nmpc_examples.simple_pipeline.pipeline_model import make_model

from nmpc_examples.nmpc.dynamic_data import (
    find_nearest_index,
    interval_data_from_time_series,
    load_inputs_into_model,
)
from nmpc_examples.nmpc import (
    get_tracking_cost_from_constant_setpoint,
)

from nmpc_examples.nmpc.model_linker import DynamicVarLinker
from nmpc_examples.nmpc.model_helper import DynamicModelHelper
from nmpc_examples.nmpc.dynamic_data.series_data import TimeSeriesData

import matplotlib.pyplot as plt

"""
Script to run an NMPC simulation with a single natural gas pipeline model.
"""


def run_nmpc(
        simulation_horizon=20.0,
        controller_horizon=20.0,
        sample_period=2.0,
        ):
    """
    Runs a simulation where outlet (demand) flow rate is perturbed
    at a specified time.
    """
    #
    # Make steady model (for initial conditions) and dynamic model
    #
    nxfe = 4
    m_steady = make_model(dynamic=False, nxfe=nxfe)
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
    # Make steady model for setpoint
    #
    m_setpoint = make_model(dynamic=False, nxfe=nxfe)
    m_setpoint.fs.pipeline.control_volume.flow_mass[:, x0].fix(
        5.0e5*pyo.units.kg/pyo.units.hr
    )
    m_setpoint.fs.pipeline.control_volume.pressure[:, x0].fix(
        57.0*pyo.units.bar
    )
    ipopt.solve(m_setpoint, tee=True)

    #
    # Extract data from setpoint model
    #
    m_setpoint_helper = DynamicModelHelper(m_setpoint, m_setpoint.fs.time)
    setpoint_data = m_setpoint_helper.get_data_at_time()

    #
    # Load initial inputs into steady model for initial conditions
    #
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
    scalar_data = m_steady_helper.get_scalar_data()

    #
    # Load data into dynamic model
    #
    use_linker = False
    if use_linker:
        # If I want to use DynamicVarLinker:
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
    sim_data = m_plant_helper.get_data_at_time([t0])

    #
    # Construct dynamic model for controller
    #
    samples_per_controller = round(controller_horizon/sample_period)
    ntfe_per_controller = ntfe_per_sample * samples_per_controller
    m_controller = make_model(
        horizon=controller_horizon,
        ntfe=ntfe_per_controller,
        nxfe=nxfe,
    )
    # Fix inputs at initial condition
    m_controller.fs.pipeline.control_volume.pressure[t0, x0].fix()
    m_controller.fs.pipeline.control_volume.flow_mass[t0, xf].fix()

    # Make helper object
    m_controller_helper = DynamicModelHelper(m_controller, m_controller.fs.time)

    #
    # Construct tracking objective
    #
    cv = m_controller.fs.pipeline.control_volume
    tracking_variables = [
        pyo.Reference(cv.pressure[:, x0]),
        pyo.Reference(cv.pressure[:, xf]),
        pyo.Reference(cv.flow_mass[:, x0]),
        pyo.Reference(cv.flow_mass[:, xf]),
    ]
    weight_data = {
        "fs.pipeline.control_volume.flow_mass[*,%s]" % x0: 1e-10,
        "fs.pipeline.control_volume.flow_mass[*,%s]" % xf: 1e-10,
        "fs.pipeline.control_volume.pressure[*,%s]" % x0: 1e-2,
        "fs.pipeline.control_volume.pressure[*,%s]" % xf: 1e-2,
    }
    weight_data = {
        # get_tracking_cost_expression expects CUIDs as keys now
        pyo.ComponentUID(name): val for name, val in weight_data.items()
    }
    m_controller.tracking_cost = get_tracking_cost_from_constant_setpoint(
        tracking_variables,
        m_controller.fs.time,
        setpoint_data,
        weight_data=weight_data,
    )
    m_controller.tracking_objective = pyo.Objective(
        expr=sum(m_controller.tracking_cost.values())
    )

    #
    # Constrain inputs piecewise constant
    #
    # TODO: This should be handled by a function in another module.
    piecewise_constant_vars = [
        pyo.Reference(cv.pressure[:, x0]),
        pyo.Reference(cv.flow_mass[:, xf]),
    ]
    m_controller.piecewise_constant_vars_set = pyo.Set(
        initialize=list(range(len(piecewise_constant_vars)))
    )
    sample_points = [
        t0 + sample_period*i for i in range(samples_per_controller+1)
    ]
    sample_point_set = set(sample_points)
    def piecewise_constant_rule(m, i, t):
        var = piecewise_constant_vars[i]
        if t in sample_point_set:
            return pyo.Constraint.Skip
        else:
            t_next = m_controller.fs.time.next(t)
            return var[t] == var[t_next]
    m_controller.piecewise_constant_constraint = pyo.Constraint(
        m_controller.piecewise_constant_vars_set,
        m_controller.fs.time,
        rule=piecewise_constant_rule,
    )

    #
    # Initialize dynamic model to initial steady state
    #
    m_controller_helper.load_scalar_data(scalar_data)
    m_controller_helper.load_data_at_time(initial_data, time)

    #
    # Initialize data structure for controller inputs
    #
    input_names = [
        pyo.ComponentUID("fs.pipeline.control_volume.flow_mass[*,%s]" % xf),
        pyo.ComponentUID("fs.pipeline.control_volume.pressure[*,%s]" % x0),
    ]
    applied_inputs = m_controller_helper.get_data_at_time([t0])
    applied_inputs.project_onto_variables(input_names)

    #
    # Set up a "model linker" to transfer control inputs to plant
    #
    controller_input_vars = [
        m_controller.find_component(name) for name in input_names
    ]
    plant_input_vars = [
        m_plant.find_component(name) for name in input_names
    ]
    input_linker = DynamicVarLinker(
        controller_input_vars,
        plant_input_vars,
    )

    #
    # Set up linker to send initial conditions from plant to controller
    #
    # We will send values of all variables from plant to controller,
    # even though we only need to send those that are fixed as initial
    # conditions.
    plant_vars_in_controller = [
        m_controller.find_component(var.referent)
        for var in m_plant_helper.dae_vars
    ]
    init_cond_linker = DynamicVarLinker(
        m_plant_helper.dae_vars,
        plant_vars_in_controller,
    )

    for i in range(n_cycles):
        # time.first() in the model corresponds to sim_t0 in "simulation time"
        # time.last() in the model corresponds to sim_tf in "simulation time"
        sim_t0 = i*sample_period
        sim_tf = (i + 1)*sample_period

        #
        # Solve dynamic optimization problem to get inputs
        #
        ipopt.solve(m_controller, tee=True)

        ts = sample_points[1]
        #
        # Extract first inputs from controller
        #
        extracted_inputs = m_controller_helper.get_data_at_time([ts])
        # "Project" onto the subset of variables I want to store
        extracted_inputs.project_onto_variables(input_names)
        # Shift time points from "controller time" to "simulation time"
        extracted_inputs.shift_time_points(sim_t0)

        #
        # Extend data structure of applied inputs
        #
        applied_inputs.concatenate(extracted_inputs)

        #
        # Load inputs from controller into plant
        #
        non_initial_plant_time = list(m_plant.fs.time)[1:]
        input_linker.transfer(ts, non_initial_plant_time)

        res = ipopt.solve(m_plant, tee=True)
        pyo.assert_optimal_termination(res)

        #
        # Extract time series data from solved model
        #
        # Initial conditions have already been accounted for.
        # Note that this is only correct because we're using an implicit
        # time discretization.
        non_initial_time = list(time)[1:]
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
        # Re-initialize controller model
        #
        m_controller_helper.shift_values(sample_period)

        #
        # Re-initialize model to final values.
        # This sets new initial conditions, including inputs.
        #
        tf = m_plant.fs.time.last()
        m_plant_helper.propagate_values_at_time(tf)

        init_cond_linker.transfer(tf, t0)

    return sim_data, applied_inputs


def plot_states_from_data(data, names, show=False):
    time = data.get_time_points()
    state_data = data.get_data()
    for i, name in enumerate(names):
        values = state_data[name]
        fig, ax = plt.subplots()
        ax.plot(time, values)
        ax.set_title(name)
        ax.set_xlabel("Time (hr)")

        if show:
            fig.show()
        else:
            fname = "state%s.png" % i
            fig.savefig(fname, transparent=False)


def plot_inputs_from_data(data, names, show=False):
    time = data.get_time_points()
    input_data = data.get_data()
    for i, name in enumerate(names):
        values = input_data[name]
        fig, ax = plt.subplots()
        ax.step(time, values)
        ax.set_title(name)
        ax.set_xlabel("Time (hr)")

        if show:
            fig.show()
        else:
            fname = "input%s.png" % i
            fig.savefig(fname, transparent=False)
        

if __name__ == "__main__":
    simulation_data, applied_inputs = run_nmpc(
        simulation_horizon=4.0,
    )
    plot_states_from_data(
        simulation_data,
        [
            pyo.ComponentUID("fs.pipeline.control_volume.flow_mass[*,0.0]"),
            pyo.ComponentUID("fs.pipeline.control_volume.pressure[*,1.0]"),
        ],
    )
    plot_inputs_from_data(
        applied_inputs,
        [
            pyo.ComponentUID("fs.pipeline.control_volume.flow_mass[*,1.0]"),
            pyo.ComponentUID("fs.pipeline.control_volume.pressure[*,0.0]"),
        ],
    )
