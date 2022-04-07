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


def get_quadratic_penalty_at_time(var, t, target, weight=None):
    if weight is None:
        weight = 1.0
    return weight * (var[t] - target)**2


def get_penalty_expressions_at_time(
    variables,
    time,
    t,
    target_data,
    weight_data=None,
):
    """
    """
    cuids = [
        get_time_indexed_cuid(var, sets=(time,))
        for var in variables
    ]
    # TODO: Weight data (and setpoint data) are user-provided and don't
    # necessarily have CUIDs as keys. Should I processes the keys here
    # with get_time_indexed_cuid?
    if weight_data is None:
        weight_data = {cuid: 1.0 for cuid in cuids}
    for i, cuid in enumerate(cuids):
        if cuid not in target_data:
            raise KeyError(
                "Target data dictionary does not contain a key for variable\n"
                "%s with ComponentUID %s" % (variables[i].name, cuid)
            )
        if cuid not in weight_data:
            raise KeyError(
                "Terminal penalty weight dictionary does not contain a key for "
                "variable\n%s with ComponentUID %s" % (variables[i].name, cuid)
            )

    penalties = [
        get_quadratic_penalty_at_time(
            var, t, target_data[cuid], weight_data[cuid]
        ) for var, cuid in zip(variables, cuids)
    ]
    return penalties


def get_terminal_penalty(
    variables,
    time,
    target_data,
    weight_data=None,
):
    t = time.last()
    terminal_penalty = Expression(
        expr=sum(get_penalty_expressions_at_time(
            variables, time, t, target_data, weight_data
        ))
    )
    return terminal_penalty
