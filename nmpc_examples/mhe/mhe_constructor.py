#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2021
# by the software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia University
# Research Corporation, et al.  All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################
from pyomo.core.base.set import Set
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.expression import Expression

from nmpc_examples.nmpc.dynamic_data.series_data import get_time_indexed_cuid


def curr_sample_point(tp, sample_points):
    """
    This function returns the sample point to which the given time element
    belongs.

    Arguments
    ---------
    tp: float
        Time element
    sample_points: iterable
        Set of sample points

    Returns
    -------
    A sample point to which the given time element belongs.

    """
    for spt in sample_points:
        if spt >= tp:
            return spt


def construct_measurement_variables_constraints(
    sample_points,
    variables,
):
    """
    This function constructs components for measurments, including:
    1. Index set for measurements
    2. Variables for measurements and measurement errors
    2. Constraints for measured variables, measurements, and measurement
    errors.

    Arguments
    ---------
    sample_points: iterable
        Set of sample points
    variables: list
        List of time-indexed variables that are measured

    Returns
    -------
    meas_set: Pyomo Set
        Index set for measurements
    meas_var: Pyomo Var
        Measurement variable, indexed by meas_set and sample_points
    meas_error_var: Pyomo Var
        Measurement error variable, indexed by meas_set and sample_points
    meas_con: Pyomo Constraint
        measurement constraint, indexed by meas_set and sample_points,
            **measurement == measured_var + measurement_error**

    """
    meas_set = Set(initialize=range(len(variables)))
    meas_var = Var(meas_set, sample_points)
    meas_error_var = Var(meas_set, sample_points, initialize=0.0)

    def meas_con_rule(mod, index, sp):
        return meas_var[index, sp] == \
            variables[index][sp] + meas_error_var[index, sp]
    meas_con = Constraint(
        meas_set, sample_points, rule=meas_con_rule,
    )

    return meas_set, meas_var, meas_error_var, meas_con


def construct_disturbed_model_constraints(
    time,
    sample_points,
    mod_constraints,
):
    """
    This function constructs components for model disturbances, including:
    1. Index set for model disturbances
    2. Variables for model disturbances
    3. Disturbed model constraints, consisting of original model constraints
    and model disturbances.

    Arguments
    ----------
    time: iterable
        Set by which to index model constraints
    sample_points: iterable
        Set of sample points
    mod_constraints : list
        List of model constraints to add model disturbances

    Returns
    -------
    disturbance_set: Pyomo Set
        Index set of model disturbances
    disturbance_var: Pyomo Var
        Model disturbance variable, indexed by disturbance_set and time
    disturbed_con: Pyomo Constraint
        Model constraints with model disturbances, indexed by disturbance_set
        and time,
            ** original_constraint + disturbance == 0.0 **

    """
    disturbance_set = Set(initialize=range(len(mod_constraints)))
    disturbance_var = Var(disturbance_set, sample_points, initialize=0.0)

    def disturbed_con_rule(m, index, i):
        # try:
        con_ele = mod_constraints[index][i]
        # except KeyError:
        #     if i == 0:
        #         # Assuem we're using an implicit time discretization.
        #         return Constraint.Skip
        #     else:
        #         raise KeyError(i)

        if not con_ele.equality:
            raise RuntimeError(con_ele.name, " is not an equality constraint.")
        if con_ele.lower.value != con_ele.upper.value:
            raise RuntimeError(con_ele_name, " has different lower and upper "
                               "bounds.")
        if con_ele.upper.value != 0.0:
            raise RuntimeError(con_ele_name, " is an equality but its bound "
                               "is not zero.")

        spt = curr_sample_point(i, sample_points)

        return con_ele.body + disturbance_var[index, spt] == \
            con_ele.upper.value
    disturbed_con = Constraint(
        disturbance_set, time, rule=disturbed_con_rule,
    )

    return disturbance_set, disturbance_var, disturbed_con


def activate_disturbed_constraints_based_on_original_constraints(
    time,
    sample_points,
    disturbance_var,
    mod_constraints,
    disturbed_con,
):
    """
    This function activate the model constraint and deactivate the original
    constraint, if the original constarint is active. Also, if all model
    constraints within a specific sample period are not active, fix the
    disturbance variable at zero.

    Parameters
    ----------
    time: iterable
        Time set which indexes model constraints
    sample_points: iterable
        Set of sample points
    disturbance_var: Pyomo Var
        Model disturbances
    mod_constraints: list
        List of original model constraints
    disturbed_con: Pyomo Constraint
        Model constraints with model disturbances

    """
    # Deactivate original equalities and activate disturbed equalities
    for index, con in enumerate(mod_constraints):
        for tp in time:
            con_ele = con[tp]
            if con_ele.active:
                con_ele.deactivate()
                disturbed_con[index, tp].activate()
            else:
                # If the original equality is not active, deactivate the
                # disturbed one.
                disturbed_con[index, tp].deactivate()

    # If all constraints in a specific sample time are not active, fix that
    # model disturbance at zero.
    spt_saw_tp = {spt:[] for spt in sample_points}
    for tp in time:
        spt = curr_sample_point(tp, sample_points)
        spt_saw_tp[spt].append(tp)

    for index in range(len(mod_constraints)):
        for spt in sample_points:
            con_active_list = [
                disturbed_con[index, tp].active for tp in spt_saw_tp[spt]
            ]
            if not any(con_active_list):
                disturbance_var[index, spt].fix(0.0)


def get_error_disturbance_cost(
    time,
    sample_points,
    components,
    error_dist_var,
    weight_data=None,
):
    """
    #TODO: This function is similar to "get_tracking_cost_from_constant_setpoint",
    #Should I merge this one to that one?
    This function return a squared cost expression for measurement errors or
    model disturbances.

    Arguments
    ---------
    time: iterable
        Time set which indexes components (measured variables or
        model constraints)
    sample_points: iterable
        Set of sample points
    components: List
        List of components (measured variables or model constraints)
    error_dist_var: Pyomo Var
        Variable of measurement error
        from "construct_measurement_variables_constraints"
        or model disturbance
        from "construct_disturbed_model_constraints"
    weight_data: dict
        Optional. Maps variable names to squared cost weights. If not provided,
        weights of one are used.

    Returns
    -------
    Pyomo Expression, indexed by sample_points, containing the sum of squared
    errors or disturbances.

    """
    component_cuids = [
        get_time_indexed_cuid(comp, sets=(time,))
        for comp in components
    ]

    if weight_data is None:
        weight_data = {cuid: 1.0 for cuid in component_cuids}
    for i, cuid in enumerate(component_cuids):
        if cuid not in weight_data:
            raise KeyError(
                "Error/disturbance weight dictionary does not contain a key for "
                "variable\n%s with ComponentUID %s" % (components[i].name, cuid)
            )

    def error_disturbance_rule(m, spt):
        return sum(
            weight_data[cuid] * error_dist_var[index, spt]**2
            for index, cuid in enumerate(component_cuids)
        )
    error_disturbance_cost = Expression(
        sample_points, rule=error_disturbance_rule
    )
    return error_disturbance_cost
