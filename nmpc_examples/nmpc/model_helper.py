from pyomo.dae.flatten import flatten_dae_components
from pyomo.core.base.var import Var
from pyomo.core.base.componentuid import ComponentUID

from nmpc_examples.nmpc.dynamic_data.series_data import TimeSeriesData

iterable_scalars = (str, bytes)


def _to_iterable(item):
    if hasattr(item, "__iter__"):
        if isinstance(item, iterable_scalars):
            yield item
        else:
            for obj in item:
                yield obj
    else:
        yield item


class DynamicModelHelper(object):
    """
    Somewhat like a serializer, with the ability to generate and load
    different types of data from a dynamic model. Also somewhat like a
    wrapper around the flattened indexed variables, with the ability
    to shift values between different points in time.

    """

    def __init__(self, model, time):
        scalar_vars, dae_vars = flatten_dae_components(model, time, Var)
        self.model = model
        self.time = time
        self.scalar_vars = scalar_vars
        self.dae_vars = dae_vars

        # Use buffer to reduce repeated work during name/cuid generation
        cuid_buffer = {}
        self.scalar_var_cuids = [
            ComponentUID(var, cuid_buffer=cuid_buffer)
            for var in self.scalar_vars
        ]
        self.dae_var_cuids = [
            ComponentUID(var.referent, cuid_buffer=cuid_buffer)
            for var in self.dae_vars
        ]

    def get_scalar_data(self):
        return {
            cuid: var.value
            for cuid, var in zip(self.scalar_var_cuids, self.scalar_vars)
        }

    def get_data_at_time(self, time=None):
        if time is None:
            time = self.time.first()
        try:
            time_list = list(time)
            data = {
                cuid: [var[t].value for t in time]
                for cuid, var in zip(self.dae_var_cuids, self.dae_vars)
            }
            # Here we're returning a data series object.
            # This makes the calling code simpler. Does it make sense for
            # this class? I.e. should we have some special class for
            # scalar data? Just the ability to process variable-like
            # keys makes this probably worthwhile.
            return TimeSeriesData(data, time_list, time_set=time)
            #return {
            #    cuid: [var[t].value for t in time]
            #    for cuid, var in zip(self.dae_var_cuids, self.dae_vars)
            #}
        except TypeError:
            # time is a scalar
            # Maybe checking if time is an instance of numeric_types would
            # be better.
            return {
                cuid: var[time].value
                for cuid, var in zip(self.dae_var_cuids, self.dae_vars)
            }

    def load_scalar_data(self, data):
        for cuid, val in data.items():
            var = self.model.find_component(cuid)
            var.set_value(val)

    def load_data_at_time(self, data, time_points=None):
        if time_points is None:
            time_points = self.time
        else:
            time_points = list(_to_iterable(time_points))
        for cuid, val in data.items():
            var = self.model.find_component(cuid)
            for t in time_points:
                var[t].set_value(val)

    def propagate_values_at_time(self, t, target_time=None):
        # TODO: name of this method?
        # Here I only implement a transfer from a single time point.
        # We could transfer to and from multiple time points, but
        # we could potentially override our desired values if a vardata
        # appears twice (e.g. because it is part of a reference)
        if target_time is None:
            target_time = self.time
        else:
            target_time = list(_to_iterable(target_time))
        for var in self.dae_vars:
            for t_targ in target_time:
                var[t_targ].set_value(var[t].value)

    def shift_values(self, dt):
        seen = set()
        t0 = self.time.first()
        tf = self.time.last()
        for var in self.dae_vars:
            if id(var[tf]) in seen:
                # Assume that if var[tf] has been encountered, this is a
                # reference to a "variable" we have already processed.
                continue
            else:
                seen.add(id(var[tf]))
            new_values = []
            for t in self.time:
                t_new = t + dt
                # TODO: What if t_shift is not a valid time point?
                # Right now we just proceed with the closest valid time point.
                # We're relying on the fact that indices of t0 or tf are
                # returned if t_new is outside the bounds of the time set.
                idx = self.time.find_nearest_index(t_new, tolerance=None)
                t_new = self.time.at(idx)
                new_values.append(var[t_new].value)
            for i, t in enumerate(self.time):
                var[t].set_value(new_values[i])
