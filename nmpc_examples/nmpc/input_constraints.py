from pyomo.core.base.constraint import Constraint
from pyomo.core.base.set import Set


def get_piecewise_constant_constraint(inputs, time, sample_points, use_next=True):
    input_set = Set(initialize=range(len(inputs)))
    sample_point_set = set(sample_points)
    def piecewise_constant_rule(m, i, t):
        if t in sample_points:
            return Constraint.Skip
        else:
            var = inputs[i]
            # Should this use t_next or t_prev? I.e. should the first or
            # last endpoint of a control sample be used? I believe this
            # depends on our discretization
            if use_next:
                t1 = t
                t2 = time.next(t)
            else:
                t1 = time.prev(t)
                t2 = t
            return var[t1] == var[t2]
    pwc_con = Constraint(input_set, time, rule=piecewise_constant_rule)
    return input_set, pwc_con
