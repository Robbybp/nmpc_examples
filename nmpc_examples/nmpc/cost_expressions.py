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
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.expression import Expression

from nmpc_examples.nmpc.dynamic_data.series_data import get_time_indexed_cuid


def get_tracking_cost_from_constant_setpoint(
        variables,
        time,
        setpoint_data,
        weight_data=None,
        ):
    """
    This function returns a tracking cost expression for the given time-indexed
    variables and associated setpoint data.

    Arguments
    ---------
    variables: list
        List of time-indexed variables to include in the tracking cost
        expression
    time: iterable
        Set by which to index the tracking expression
    setpoint_data: dict
        Maps variable names to setpoint values
    weight_data: dict
        Optional. Maps variable names to tracking cost weights. If not
        provided, weights of one are used.

    Returns
    -------
    Pyomo Expression, indexed by time, containing the sum of weighted
    squared difference between variables and setpoint values.

    """
    cuids = [
        get_time_indexed_cuid(var, sets=(time,))
        for var in variables
    ]
    # TODO: Weight data (and setpoint data) are user-provided and don't
    # necessarily have CUIDs as keys. Should I processes the keys here
    # with get_time_indexed_cuid?
    if weight_data is None:
        #weight_data = {name: 1.0 for name in variable_names}
        weight_data = {cuid: 1.0 for cuid in cuids}
    for i, cuid in enumerate(cuids):
        if cuid not in setpoint_data:
            raise KeyError(
                "Setpoint data dictionary does not contain a key for variable\n"
                "%s with ComponentUID %s" % (variables[i].name, cuid)
            )
        if cuid not in weight_data:
            raise KeyError(
                "Tracking weight dictionary does not contain a key for "
                "variable\n%s with ComponentUID %s" % (variables[i].name, cuid)
            )

    def tracking_rule(m, t):
        return sum(
            weight_data[cuid] * (var[t] - setpoint_data[cuid])**2
            for cuid, var in zip(cuids, variables)
        )
    tracking_expr = Expression(time, rule=tracking_rule)
    return tracking_expr
