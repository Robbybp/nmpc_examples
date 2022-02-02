from pyomo.dae.flatten import flatten_dae_components
from pyomo.core.base.var import Var
from pyomo.core.base.componentuid import ComponentUID


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

    def get_data_at_time(self, t0=None):
        if t0 is None:
            t0 = self.time.first()
        return {
            cuid: var[t0].value
            for cuid, var in zip(self.dae_var_cuids, self.dae_vars)
        }

    def load_scalar_data(self, data):
        # Unclear whether methods or loading data into a model should
        # go on this class.
        #
        # Also unclear whether mapping names to variables is actually
        # a good idea. If keys in data are just CUIDs, then computing
        # the hash (to look up a variable) is about the same amount
        # of work as calling find_component.
        for cuid, val in data.items():
            var = self.model.find_component(cuid)
            var.set_value(val)
