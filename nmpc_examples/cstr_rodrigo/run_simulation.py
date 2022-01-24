import pyomo.environ as pyo
from pyomo.common.collections import ComponentMap
from pyomo.dae.flatten import flatten_dae_components

from nmpc_examples.cstr_rodrigo.model import make_model


def get_input_sequence():
    # Note: no mention of the time points at which these inputs are
    # to be applied.
    return {
        "Tjinb": (
            #[250.0]
            #+ [(250.0 + 5*i) if i <= 5 else 260 - 5*i for i in range(11)]
            [(250.0 + 5*i) if i <= 5 else 260 - 5*i for i in range(11)]
        ),
    }


def run_simulation(
        n_cycles=2,
        horizon=10.0,
        ntfe=5,
        ntcp=3,
        ):
    horizon = horizon
    m = make_model(
        horizon=horizon,
        ntfe=ntfe,
        ntcp=ntcp,
        steady=False,
        bounds=False,
    )
    time = m.t
    t0 = time.first()
    tf = time.last()

    # input_sequence is model-instance agnostic, i.e. it uses strings
    # as keys
    input_sequence = get_input_sequence()

    # First point in "simulation time"
    t0_sim = 0.0

    ipopt = pyo.SolverFactory("ipopt")
    # Solve model before loading inputs to ensure consistent initial
    # conditions.
    ipopt.solve(m, tee=True)

    scalar_vars, dae_vars = flatten_dae_components(m, time, pyo.Var)
    
    # Initialize data structures:
    #recorded_states = [
    #    pyo.Reference(m.Tall[:, "T"]),
    #    pyo.Reference(m.Ca[:]),
    #]
    recorded_states = dae_vars
    sim_data = (
        [t0_sim],
        ComponentMap((var, [var[t0].value]) for var in recorded_states),
    )

    input_names = list(sorted(input_sequence.keys()))
    inputs = [m.find_component(name) for name in input_names]

    # Note that because it uses references as keys, planned_input_data
    # is not easily interpreted by the user.
    planned_input_data = ComponentMap(
        (var, input_sequence[name]) for var, name in zip(inputs, input_names)
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
        ComponentMap((var, [planned_input_data[var][0]]) for var in inputs),
    )

    sim_offset = t0_sim
    for i in range(n_cycles):
        t0_sim = sim_offset + horizon * i
        tf_sim = sim_offset + horizon * (i + 1)

        # Send inputs to model
        for var, values in planned_input_data.items():
            var[:].set_value(values[i])

        # Extend applied input data.
        input_time, var_map = applied_input_data
        input_time.append(tf_sim)
        for var, values in var_map.items():
            values.append(planned_input_data[var][i])

        # Simulate model
        ipopt.solve(m, tee=True)

        # Extend state data with values from model
        #
        # If we used an explicit discretization, we should probably
        # override the values at t0.
        non_initial_time = list(time)[1:]
        new_sim_time = [t + t0_sim for t in non_initial_time]
        sim_time, var_map = sim_data
        sim_time.extend(new_sim_time)
        for var, values in var_map.items():
            values.extend(var[t].value for t in non_initial_time)

        # TODO: update initial conditions in plant
        for var in dae_vars:
            for t in time:
                var[t].set_value(var[tf].value)

    sim_time, var_map = sim_data
    sim_data = (
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

    return sim_data, applied_input_data


if __name__ == "__main__":
    run_simulation()
