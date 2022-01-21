import pyomo.common.unittest as unittest

from nmpc_examples.cstr_rodrigo import run_simulation


class TestSimulation(unittest.TestCase):

    def test_initial_conditions(self):
        sim_data, input_data = run_simulation(n_cycles=0)

        # From Kuan-Han's tests
        pred_state_map = {
            "Ca[*]": [0.0192],
            "k[*]": [1596.67471],
            "Tjinb[*]": [250.0],
            "Tall[*,T]": [384.0],
            "Tall[*,Tj]": [371.0],
        }

        sim_time, sim_state_map = sim_data
        sim_state_map = {name: sim_state_map[name] for name in pred_state_map}
        self.assertStructuredAlmostEqual(pred_state_map, sim_state_map)

        pred_input_map = {
            # TODO: Should this key have a "[*]" in it?
            "Tjinb": [250.0],
        }
        input_time, input_map = input_data
        input_map = {name: input_map[name] for name in pred_input_map}
        self.assertEqual(pred_input_map, input_map)

    def test_one_cycle(self):
        sim_data, input_data = run_simulation(
            n_cycles=1,
            ntfe=4,
            ntcp=2,
            horizon=2.0,
        )

        # From Kuan-Han's tests
        pred_state_map = {
            "Ca[*]": 0.01919,
            "Tall[*,T]": 384.00519,
            "Tall[*,Tj]": 371.27157,
            "k[*]": 1597.19180,
            "Tjinb[*]": 250.0,
        }

        sim_time, sim_state_map = sim_data
        sim_state_map = {
            name: sim_state_map[name][-1]
            for name in pred_state_map
        }
        self.assertStructuredAlmostEqual(
            pred_state_map, sim_state_map, reltol=1e-3
        )

    def test_two_cycles(self):
        sim_data, input_data = run_simulation(
            n_cycles=2,
            ntfe=4,
            ntcp=2,
            horizon=2.0,
        )

        # From Kuan-Han's tests
        pred_state_map = {
            "Ca[*]": 0.018655,
            "Tall[*,T]": 384.930409,
            "Tall[*,Tj]": 372.581354,
            "k[*]": 1691.919933,
            "Tjinb[*]": 255.0,
        }

        sim_time, sim_state_map = sim_data
        sim_state_map = {
            name: sim_state_map[name][-1]
            for name in pred_state_map
        }
        self.assertStructuredAlmostEqual(
            pred_state_map, sim_state_map, reltol=1e-3
        )


    def test_three_cycles(self):
        sim_data, input_data = run_simulation(
            n_cycles=3,
            ntfe=4,
            ntcp=2,
            horizon=2.0,
        )

        # From Kuan-Han's tests
        pred_state_map = {
            "Ca[*]": 0.018101,
            "Tall[*,T]": 385.913247,
            "Tall[*,Tj]": 373.945749,
            "k[*]": 1798.164102,
            "Tjinb[*]": 260.0,
        }

        sim_time, sim_state_map = sim_data
        sim_state_map = {
            name: sim_state_map[name][-1]
            for name in pred_state_map
        }
        self.assertStructuredAlmostEqual(
            pred_state_map, sim_state_map, reltol=1e-3
        )


if __name__ == "__main__":
    unittest.main()
