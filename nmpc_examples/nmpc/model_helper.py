from pyomo.dae.flatten import flatten_dae_components
from pyomo.core.base.var import Var
from pyomo.core.base.componentuid import ComponentUID

# TODO: Something like a "DynamicModelLinker"

class DynamicModelHelper(object):

    def __init__(self, model, time):
        scalar_vars, dae_vars = flatten_dae_components(model, time, Var)
        self.model = model
        self.time = time
        self.scalar_vars = scalar_vars
        self.dae_vars = dae_vars
        cuid_buffer = {}
        self.scalar_var_names = [
            str(ComponentUID(var, cuid_buffer=cuid_buffer))
            for var in self.scalar_vars
        ]
        self.dae_var_names = [
            str(ComponentUID(var.referent, cuid_buffer=cuid_buffer))
            for var in self.dae_vars
        ]
        self.scalar_name_var_map = dict(
            zip(self.scalar_var_names, self.scalar_vars)
        )
        self.dae_name_var_map = dict(
            zip(self.dae_var_names, self.dae_vars)
        )

    def get_scalar_data(self):
        return {
            name: var.value
            for name, var in zip(self.scalar_var_names, self.scalar_vars)
        }

    def get_data_at_time(self, t0=None):
        if t0 is None:
            t0 = self.time.first()
        return {
            name: var[t0].value
            for name, var in zip(self.dae_var_names, self.dae_vars)
        }

    def load_scalar_data(self, data):
        for name, val in data.items():
            self.scalar_name_var_map[name].set_value(val)
