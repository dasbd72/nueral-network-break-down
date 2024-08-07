import unittest

from .activation_test import TestReLU  # noqa: F401
from .linear_test import TestLinear  # noqa: F401
from .optimizer_test import TestOptimizers  # noqa: F401
from .parameter_test import TestParameter  # noqa: F401

if __name__ == "__main__":
    unittest.main()
