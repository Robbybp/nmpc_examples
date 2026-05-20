import pyomo.common.unittest as unittest
import pyomo.environ as pyo

from nmpc_examples.pipeline.run_pipeline_mhe import run_mhe

class TestPipelineMHE(unittest.TestCase):

    def test_small_mhe_simulation(self):
        simulation_data, estimate_data = run_mhe(
            simulation_horizon=4.0,
            sample_period=2.0,
            # NOTE: When I reduce the estimator horizon to 4.0, the test
            # runs much faster, but the results here don't change.
            # Is this what I expect?
            # It seems plausible, as the estimated values are the same
            # as those simulated...
            estimator_horizon=4.0,
        )

        # These values were not reproduced with another model, just taken
        # from this MHE simulation. They are here to make sure the result
        # doesn't change.
        pred_inlet_flow = [
            3.000e5, 4.927e5, 4.446e5, 4.460e5, 4.557e5, 4.653e5,
            4.722e5, 4.785e5, 4.842e5,
        ]
        pred_outlet_pressure = [
            50.91, 47.82, 46.19, 45.06, 43.58, 43.08, 42.62, 42.21, 41.83,
        ]

        pred_estimate_inlet_flow = [3.00000e5, 4.557e5, 4.842e5]
        pred_estimate_outlet_pressure = [50.91, 43.58, 41.83]

        actual_inlet_flow = simulation_data.get_data()[
            pyo.ComponentUID("fs.pipeline.control_volume.flow_mass[*,0.0]")
        ]
        actual_outlet_pressure = simulation_data.get_data()[
            pyo.ComponentUID("fs.pipeline.control_volume.pressure[*,1.0]")
        ]

        estimate_inlet_flow = estimate_data.get_data()[
            pyo.ComponentUID("fs.pipeline.control_volume.flow_mass[*,0.0]")
        ]
        estimate_outlet_pressure = estimate_data.get_data()[
            pyo.ComponentUID("fs.pipeline.control_volume.pressure[*,1.0]")
        ]

        self.assertStructuredAlmostEqual(
            pred_inlet_flow, actual_inlet_flow, reltol=1e-3
        )
        self.assertStructuredAlmostEqual(
            pred_outlet_pressure, actual_outlet_pressure, reltol=1e-3
        )
        self.assertStructuredAlmostEqual(
            pred_estimate_inlet_flow, estimate_inlet_flow, reltol=1e-3
        )
        self.assertStructuredAlmostEqual(
            pred_estimate_outlet_pressure, estimate_outlet_pressure, reltol=1e-3
        )


if __name__ == "__main__":
    unittest.main()
