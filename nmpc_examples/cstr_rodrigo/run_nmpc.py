import pyomo.environ as pyo
from pyomo.common.collections import ComponentMap
from pyomo.dae.flatten import flatten_dae_components
from pyomo.util.subsystems import TemporarySubsystemManager

from nmpc_examples.cstr_rodrigo.model import make_model
from nmpc_examples.nmpc import (
    get_tracking_cost_from_constant_setpoint,
    get_piecewise_constant_constraints,
)


def get_setpoint_data():
    return {
        pyo.ComponentUID("Ca[*]"): 0.018,
    }


def get_objective_weight_data():
    return {
        pyo.ComponentUID("Ca[*]"): 1.0,
    }


def set_up_controller(
        model,
        input_variables,
        tracking_variables,
        sample_points,
        ):
    # The controller model is also parameterized by the setpoint itself,
    # which we are not capturing here.
    time = model.t
    t0 = time.first()

    #
    # Construct steady model for setpoint, solve, and extract data
    #
    m_setpoint = make_model(steady=True, bounds=True)
    setpoint_data = get_setpoint_data()
    objective_weights = get_objective_weight_data()
    setpoint_variables = [pyo.Reference(m_setpoint.Ca[:])]
    m_setpoint.tracking_cost = get_tracking_cost_from_constant_setpoint(
        setpoint_variables,
        m_setpoint.t,
        setpoint_data,
        objective_weights,
    )
    t0_setpoint = m_setpoint.t.first()
    m_setpoint.obj = pyo.Objective(expr=sum(
        m_setpoint.tracking_cost[t] for t in m_setpoint.t
    ))
    m_setpoint.Tjinb.unfix()
    ipopt = pyo.SolverFactory("ipopt")
    ipopt.solve(m_setpoint, tee=True)
    setpoint_scalar_vars, setpoint_dae_vars = flatten_dae_components(
        m_setpoint, m_setpoint.t, pyo.Var
    )
    # Overriding setpoint_data here...
    setpoint_data = {
        pyo.ComponentUID(var.referent): var[t0_setpoint].value
        for var in setpoint_dae_vars
    }
    ###

    # Add objective function to controller
    model.tracking_cost = get_tracking_cost_from_constant_setpoint(
        tracking_variables, time, setpoint_data
    )
    model.obj = pyo.Objective(expr=sum(
        model.tracking_cost[t] for t in sample_points if t != t0
    ))

    #
    # Declare inputs and add piecewise constant constraints
    #
    input_set, pwc_con = get_piecewise_constant_constraints(
        input_variables, time, sample_points
    )
    model.input_set = input_set
    model.pwc_con = pwc_con
    ###

    for t in time:
        if t != t0:
            model.Tjinb[t].unfix()

    return model


def run_nmpc(
        n_cycles=2,
        horizon=10.0,
        sample_time=2.0,
        ntfe_plant=2,
        ntfe_controller=5,
        ntcp=2,
        ):
    #ntfe_plant = ntfe_per_sample
    # TODO: I should really specify horizon as an integer to be multiplied
    # by the sample time.
    samples_per_controller = round(horizon/sample_time)
    #ntfe_controller = ntfe_per_sample * samples_per_controller

    m_plant = make_model(
        horizon=sample_time,
        ntfe=ntfe_plant,
        ntcp=ntcp,
        steady=False,
        bounds=False,
    )
    time_plant = m_plant.t
    t0 = time_plant.first()
    tf = time_plant.last()

    m_controller = make_model(
        horizon=horizon,
        ntfe=ntfe_controller,
        ntcp=ntcp,
        bounds=True,
        steady=False,
    )
    time_controller = m_controller.t

    sample_points = [
        i*sample_time for i in range(samples_per_controller+1)
    ]
    input_variables = [
        pyo.Reference(m_controller.Tjinb[:]),
    ]
    tracking_variables = [
        pyo.Reference(m_controller.Ca[:]),
        pyo.Reference(m_controller.Tall[:, "T"]),
        pyo.Reference(m_controller.Tall[:, "Tj"]),
        pyo.Reference(m_controller.Tjinb[:]),
    ]
    set_up_controller(
        m_controller,
        input_variables,
        tracking_variables,
        sample_points,
    )

    # inputs at sample points
    dof_vars = [
        var[t] for var in input_variables for t in sample_points if t != t0
    ]

    # First point in "simulation time"
    t0_sim = 0.0

    ipopt = pyo.SolverFactory("ipopt")
    # Solver square plant and controller models. This is a lazy/inefficient
    # way to get consistent initial conditions.
    ipopt.solve(m_plant, tee=True)

    with TemporarySubsystemManager(to_fix=dof_vars):
        ipopt.solve(m_controller, tee=True)

    # These will be necessary when we initialize data structures for plant
    # simulation and controller prediction data.
    plant_scalar_vars, plant_dae_vars = flatten_dae_components(
        m_plant, time_plant, pyo.Var
    )
    controller_scalar_vars, controller_dae_vars = flatten_dae_components(
        m_controller, time_controller, pyo.Var
    )

    # Initialize data structures:
    plant_data = (
        [t0_sim],
        ComponentMap((var, [var[t0].value]) for var in plant_dae_vars),
    )
    controller_data = (
        [t0_sim],
        ComponentMap((var, [var[t0].value]) for var in controller_dae_vars),
    )

    # This holds the inputs we are currently planning to send to the plant.
    # We will update every time we solve a controller model.
    # Note that because it uses references as keys, planned_input_data
    # is not easily interpreted by the user.
    planned_input_data = ComponentMap(
        (var, [var[t].value for t in sample_points])
        for var in input_variables
    )

    applied_input_data = (
        [t0_sim],
        # Initialize inputs with initial value in the model. Because we use an
        # implicit discretization, this value does not affect the rest of the
        # model. We load inputs in input_sequence starting at first non-initial
        # time point.
        #
        # Open question whether we should initialize to the value in the model
        # or the first value in planned_input_data. The latter is probably
        # more consistent...
        ComponentMap(
            (var, [planned_input_data[var][0]]) for var in input_variables
        ),
    )

    #
    # Initialize controller to initial conditions
    #
    for var in controller_dae_vars:
        for t in time_controller:
            var[t].set_value(var[t0].value)

    sim_offset = t0_sim
    for i in range(n_cycles):
        t0_sim = sim_offset + sample_time * i
        tf_sim = sim_offset + sample_time * (i + 1)

        # Solve controller model to get control inputs
        # NOTE: Initial primal infeasibility is different between this solve
        # and KH's first NMPC solve. TODO: Figure out why this is...
        ipopt.solve(m_controller, tee=True)

        # Extract inputs from the controller
        planned_input_data = ComponentMap(
            (var, [var[t].value for t in sample_points])
            for var in input_variables
        )
        planned_input_data = {
            str(pyo.ComponentUID(var.referent)): [
                var[t].value for t in sample_points
            ] for var in input_variables
        }

        # Send inputs to model
        # We send the input at sample point index 1, that is, the
        # end of the first sample.
        for name, values in planned_input_data.items():
            var = m_plant.find_component(name)
            var[:].set_value(values[1])

        # Extend applied input data.
        input_time, var_map = applied_input_data
        input_time.append(tf_sim)
        for var, values in var_map.items():
            name = str(pyo.ComponentUID(var.referent))
            values.append(planned_input_data[name][1])

        # Simulate model
        ipopt.solve(m_plant, tee=True)

        # Extend state data with values from model
        #
        # If we used an explicit discretization, we should probably
        # override the values at t0.
        non_initial_time = list(time_plant)[1:]
        new_sim_time = [t + t0_sim for t in non_initial_time]
        sim_time, var_map = plant_data
        sim_time.extend(new_sim_time)
        for var, values in var_map.items():
            values.extend(var[t].value for t in non_initial_time)

        first_sample = [
            t for t in time_controller
            if t != t0 and t <= sample_points[1]
        ]
        new_controller_time = [t + t0_sim for t in first_sample]
        sim_time, var_map = controller_data
        sim_time.extend(new_controller_time)
        for var, values in var_map.items():
            values.extend(var[t].value for t in first_sample)

        #
        # Re-initialize controller model
        #
        seen = set()
        tf = time_controller.last()
        for var in controller_dae_vars:
            if id(var[t0]) in seen:
                continue
            else:
                seen.add(id(var[t0]))
            for t in time_controller:
                ts = t + sample_time
                idx = time_controller.find_nearest_index(ts)
                if idx is None:
                    # ts is outside the controller's horizon
                    var[t].set_value(var[tf].value)
                else:
                    ts = time_controller.at(idx)
                    var[t].set_value(var[ts].value)

        #
        # Re-initialize model to final values.
        # This includes setting new initial conditions.
        #
        tf = time_plant.last()
        for var in plant_dae_vars:
            final_value = var[tf].value
            for t in time_plant:
                var[t].set_value(final_value)
            controller_var = m_controller.find_component(var.referent)
            controller_var[t0].set_value(final_value)

    sim_time, var_map = plant_data
    plant_data = (
        sim_time,
        {
            str(pyo.ComponentUID(var.referent)): values
            for var, values in var_map.items()
        },
    )

    sim_time, var_map = controller_data
    controller_data = (
        sim_time,
        {
            str(pyo.ComponentUID(var.referent)): values
            for var, values in var_map.items()
        },
    )
        

    input_time, input_map = applied_input_data
    applied_input_data = (
        input_time,
        {
            str(pyo.ComponentUID(var)): values
            for var, values in input_map.items()
        },
    )

    return plant_data, applied_input_data


if __name__ == "__main__":
    run_nmpc(n_cycles=10)
