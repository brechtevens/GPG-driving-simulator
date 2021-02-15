import unittest
import casadi.casadi as cs
import customopengen as og


class ProblemBuilderTestCase(unittest.TestCase):

    def test_add_sx_decision_variables(self):
        u = cs.SX.sym("u", 10)
        x = cs.SX.sym("x", 4)
        pb = og.builder.ProblemBuilder()
        pb.add_decision_variable(u, x)
        self.assertEqual(14, len(pb.decision_variables))

    def test_add_mx_decision_variables(self):
        u = cs.MX.sym("u", 10)
        x = cs.MX.sym("x", 4)
        pb = og.builder.ProblemBuilder()
        pb.add_decision_variable(u, x)
        self.assertEqual(2, len(pb.decision_variables))


if __name__ == '__main__':
    unittest.main()
