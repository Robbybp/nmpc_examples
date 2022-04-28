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

from pyomo.common.collections import ComponentSet

from nmpc_examples.nmpc.dynamic_data.series_data import get_time_indexed_cuid


def curr_sample_point(tp, sample_points):
    for spt in sample_points:
        if spt >= tp:
            return spt


def construct_measurement_variables_constraints(
    # block,
    sample_points,
    variables,
):

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
            con_active_list = [disturbed_con[index, tp].active
                               for tp in spt_saw_tp[spt]
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

    component_cuids = [
        get_time_indexed_cuid(comp, sets=(time,))
        for comp in components
    ]

    def error_disturbance_rule(m, spt):
        return sum(
            weight_data[cuid] * error_dist_var[index, spt]**2
            for index, cuid in enumerate(component_cuids)
        )
    error_disturbance_cost = Expression(
        sample_points, rule=error_disturbance_rule
    )

    return error_disturbance_cost

# def get_error_disturbance_cost(
#     time,
#     sample_points,
#     measured_vars,
#     meas_error_var,
#     mod_cons,
#     mod_dist_var,
#     meas_error_weights,
#     mod_dist_weights,
# ):

#     measured_var_cuids = [
#         get_time_indexed_cuid(var, sets=(time,))
#         for var in measured_vars
#     ]

#     mod_con_cuids = [
#         get_time_indexed_cuid(con, sets=(time,))
#         for con in mod_cons
#     ]

#     def error_disturbance_rule(m, spt):
#         squared_error_sum = sum(
#             meas_error_weights[cuid] * meas_error_var[index, spt]**2
#             for index, cuid in enumerate(measured_var_cuids)
#         )

#         squared_dist_sum = sum(
#             mod_dist_weights[cuid] * mod_dist_var[index, spt]**2
#             for index, cuid in enumerate(mod_con_cuids)
#         )

#         return squared_error_sum + squared_dist_sum
#     error_disturbance_cost = Expression(sample_points,
#                                         rule=error_disturbance_rule)

#     return error_disturbance_cost


def get_squared_measurement_error(
    measurement_estimate_map,
    sample_points,
    weight_data=None,
):
    """
    This function returns a squared cost expression of measurement error for
    the given mapping of meausrements to estimates.

    Arguments
    ---------
    measurement_estimate_map: ComponentMap
        Mapping of samplepoint-indexed measurements to estimates
    sample_points: iterable
        Set by which to index the squared expression
    weight_data: dict
        Optional. Maps measurement names to squared cost weights. If not
        provided, weights of one are used.

    Returns
    -------
    Pyomo Expression, indexed by sample points, containing the sum of weighted
    squared difference between measurements and estimates, i.e., mearuerment
    errors.

    """
    cuids = [
        get_time_indexed_cuid(var, sets=(sample_points,))
        for var in measurement_estimate_map.keys()
    ]

    if weight_data is None:
        weight_data = {cuid: 1.0 for cuid in cuids}
    for i, cuid in enumerate(cuids):
        if cuid not in weight_data:
            raise KeyError(
                "Tracking weight dictionary does not contain a key for "
                "variable\n%s with ComponentUID %s" % (variables[i].name, cuid)
            )

    def sqr_meas_error_rule(m, t):
        return sum(
            weight_data[cuid] * (meas[t] - esti[t])**2
            for cuid, (meas, esti) in zip(cuids,
                                          measurement_estimate_map.items())
        )

    sqr_measurement_error = Expression(sample_points,
                                        rule=sqr_meas_error_rule)
    return sqr_measurement_error