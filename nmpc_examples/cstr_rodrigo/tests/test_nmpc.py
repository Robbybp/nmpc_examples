import pyomo.common.unittest as unittest

from nmpc_examples.cstr_rodrigo import run_nmpc


class TestNMPC(unittest.TestCase):

    def test_initial_conditions(self):
        run_nmpc(n_cycles=0)


if __name__ == "__main__":
    unittest.main()
