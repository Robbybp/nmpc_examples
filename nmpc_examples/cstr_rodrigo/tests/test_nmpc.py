import pyomo.common.unittest as unittest
import pyomo.environ as pyo

from idaes.core.util.model_statistics import degrees_of_freedom

from nmpc_examples.cstr_rodrigo import make_model, set_up_controller, run_nmpc


class TestNMPC(unittest.TestCase):

    def test_set_up_controller(self):
        samples_per_controller = 5
        sample_time = 2.0
        horizon = sample_time * samples_per_controller
        ntfe_per_sample = 5
        ntfe = samples_per_controller * ntfe_per_sample
        m = make_model(
            horizon=horizon,
            ntfe=ntfe,
            ntcp=3,
            steady=False,
            bounds=True,
        )
        sample_points = [
            i * sample_time for i in range(samples_per_controller+1)
        ]
        input_variables = [
            pyo.Reference(m.Tjinb[:]),
        ]
        tracking_variables = [
            pyo.Reference(m.Ca[:]),
            pyo.Reference(m.Tall[:, "T"]),
            pyo.Reference(m.Tall[:, "Tj"]),
        ]
        set_up_controller(m, input_variables, tracking_variables, sample_points)

        self.assertTrue(isinstance(m.obj, pyo.Objective))
        self.assertTrue(isinstance(m.pwc_con, pyo.Constraint))

        self.assertEqual(degrees_of_freedom(m), samples_per_controller)


if __name__ == "__main__":
    unittest.main()
