import pyomo.common.unittest as unittest

from nmpc_examples.simple_pipeline.run_pipeline_simulation import (
    run_simulation,
)

class TestPipelineSimulation(unittest.TestCase):

    def test_small_simulation(self):
        simulation_data = run_simulation(
            simulation_horizon=6.0,
            t_ptb=2.0,
        )

        # From a simulation with a pipeline model with a 6 hr horizon
        # and the same perturbation.
        pred_inlet_flow = [
            3.000e5, 3.000e5, 3.000e5, 3.000e5, 3.000e5, 3.126e5, 3.225e5,
            3.302e5, 3.362e5, 3.409e5, 3.447e5, 3.477e5, 3.500e5,
        ]
        pred_outlet_pressure = [
            51.00, 51.00, 51.00, 51.00, 51.00, 50.45, 50.02,
            49.68, 49.40, 49.18, 49.00, 48.86, 48.74,
        ]

        actual_inlet_flow = simulation_data[1][
            "fs.pipeline.control_volume.flow_mass[*,0.0]"
        ]
        actual_outlet_pressure = simulation_data[1][
            "fs.pipeline.control_volume.pressure[*,1.0]"
        ]

        self.assertStructuredAlmostEqual(
            pred_inlet_flow, actual_inlet_flow, reltol=1e-3
        )
        self.assertStructuredAlmostEqual(
            pred_outlet_pressure, actual_outlet_pressure, reltol=1e-3
        )


if __name__ == "__main__":
    unittest.main()
