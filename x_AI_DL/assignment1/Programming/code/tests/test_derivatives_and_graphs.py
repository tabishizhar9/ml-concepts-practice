#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
This script was written as a test module for the 'derivatives_and_graphs.ipynb'-jupyter notebook.
That notebook tests your knowledge on how to compute derivatives and gradients for Pytorch tensors.
Students may use the functions below to validate their solutions of the proposed tasks.

@author: Sebastian Doerrich
@copyright: Copyright (c) 2022-2026 xAILab Bamberg, Chair of Explainable Machine Learning (xAI), Otto-Friedrich University of Bamberg
@credits: [Christian Ledig, Sebastian Doerrich]
@license: CC BY-SA
@version: see changelog.md
@python: Python 3
@maintainer: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
@status: Production
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
# Import packages
import pathlib

import numpy as np
import torch
import unittest
import argparse

import sys
import os
import traceback

# Import own files
# Adds the parent folder relative to this to the path to use the extracted logic from notebook.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
# Extracts and imports the module.
from extract_logic_module import extract_module
derivatives_and_graphs = extract_module("derivatives_and_graphs")

# Add the code directory to sys.path
root_dir = os.path.join(os.path.dirname(__file__), '..', '..')

# Define the base data directory path
data_dir = os.path.abspath(os.path.join(root_dir, 'data', 'derivatives_and_graphs'))


class TestManager:
    """
    Class to manage the registration of test scores.
    """
    def __init__(self, test_scores):
        """
        Initialize with a reference to the dictionary where scores will be stored.
        """
        self.test_scores = test_scores

    def register_test(func):
        """
        Decorator to register a test as executed.
        """
        def wrapper(self, *args, **kwargs):
            try:
                result = func(self, *args, **kwargs)
                self.test_scores[func.__name__] = result
                return result
            except (AssertionError, Exception) as e:
                error_type = "AssertionError" if isinstance(e, AssertionError) else "Exception"
                print(f"Test {func.__name__} failed with {error_type}...")
                print(traceback.format_exc())
                print(50 * "-")
                self.test_scores[func.__name__] = 0
                return 0
        return wrapper


class TestDerivativeSimpleFunctionAtPosition(unittest.TestCase):
    """
    The class contains all test cases for task "2.1 - Derivative of a Simple Function at a Specific Position".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 2.1 - Derivative of a Simple Function at a Specific Position...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

    @TestManager.register_test
    def _test_x(self):
        """ Test for x. """
        student_version, _ = derivatives_and_graphs.derivative_of_simple_function_at_position()
        expected_version = torch.load(os.path.join(data_dir, 'references.pt'), weights_only=True)["t1"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_gradient_fx_at_x(self):
        """ Test for f'(x) at x. """
        student_version, _ = derivatives_and_graphs.derivative_of_simple_function_at_position()
        student_version = student_version.grad.detach()
        expected_version = torch.load(os.path.join(data_dir, 'references.pt'), weights_only=True)["t2"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_fx(self):
        """ Test for f(x). """
        _, student_version = derivatives_and_graphs.derivative_of_simple_function_at_position()
        expected_version = torch.load(os.path.join(data_dir, 'references.pt'), weights_only=True)["t3"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

        # Return score if all assertions pass
        return 0.25

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_x()
        self._test_gradient_fx_at_x()
        self._test_fx()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 1.0]")


class TestDerivativeComplicatedFunctionAtPosition(unittest.TestCase):
    """
    The class contains all test cases for task "2.2 - Derivative of a More Complicated Function at a Specific Position".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 2.2 - Derivative of a More Complicated Function at a Specific Position...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the test parameters
        pass

    @TestManager.register_test
    def _test_x(self):
        """ Test for x. """
        student_version, _ = derivatives_and_graphs.derivative_of_more_complicated_function_at_position()
        expected_version = torch.load(os.path.join(data_dir, 'references.pt'), weights_only=True)["t4"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_gradient_gx_at_x(self):
        """ Test for g'(x) at x. """
        student_version, _ = derivatives_and_graphs.derivative_of_more_complicated_function_at_position()
        student_version = student_version.grad.detach()
        expected_version = torch.load(os.path.join(data_dir, 'references.pt'), weights_only=True)["t5"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_gx(self):
        """ Test for g(x). """
        _, student_version = derivatives_and_graphs.derivative_of_more_complicated_function_at_position()
        expected_version = torch.load(os.path.join(data_dir, 'references.pt'), weights_only=True)["t6"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

        # Return score if all assertions pass
        return 0.25

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_x()
        self._test_gradient_gx_at_x()
        self._test_gx()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 1.0]")


class TestDerivativeEntireFunction(unittest.TestCase):
    """
    The class contains all test cases for task "2.3 - Derivative of an Entire Function".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 2.3 - Derivative of an Entire Function...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the test parameters
        pass

    @TestManager.register_test
    def _test_x(self):
        """ Test for x. """
        student_version, _ = derivatives_and_graphs.derivative_of_relu()
        expected_version = torch.load(os.path.join(data_dir, 'references.pt'), weights_only=True)["t7"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_gradient_fx_at_x(self):
        """ Test for f'(x) at x. """
        student_version, _ = derivatives_and_graphs.derivative_of_relu()
        student_version = student_version.grad.detach()
        expected_version = torch.load(os.path.join(data_dir, 'references.pt'), weights_only=True)["t8"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_fx(self):
        """ Test for f(x). """
        _, student_version = derivatives_and_graphs.derivative_of_relu()
        expected_version = torch.load(os.path.join(data_dir, 'references.pt'), weights_only=True)["t9"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

        # Return score if all assertions pass
        return 0.5

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_x()
        self._test_gradient_fx_at_x()
        self._test_fx()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 1.5]")


class TestPartialDerivatives(unittest.TestCase):
    """
    The class contains all test cases for task "3 - Partial Derivatives".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 3 - Partial Derivatives...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the test parameters
        pass

    @TestManager.register_test
    def _test_u(self):
        """ Test for u. """
        student_version, _, _ = derivatives_and_graphs.partial_derivatives()
        expected_version = torch.load(os.path.join(data_dir, 'references.pt'), weights_only=True)["t10"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_v(self):
        """ Test for v. """
        _, student_version, _ = derivatives_and_graphs.partial_derivatives()
        expected_version = torch.load(os.path.join(data_dir, 'references.pt'), weights_only=True)["t11"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_gradient_fu(self):
        """ Test for f_u. """
        student_version, _, _ = derivatives_and_graphs.partial_derivatives()
        student_version = student_version.grad.detach()
        expected_version = torch.load(os.path.join(data_dir, 'references.pt'), weights_only=True)["t12"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_gradient_fv(self):
        """ Test for f_v. """
        _, student_version, _ = derivatives_and_graphs.partial_derivatives()
        student_version = student_version.grad.detach()
        expected_version = torch.load(os.path.join(data_dir, 'references.pt'), weights_only=True)["t13"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_fuv(self):
        """ Test for f(uv). """
        _, _, student_version = derivatives_and_graphs.partial_derivatives()
        expected_version = torch.load(os.path.join(data_dir, 'references.pt'), weights_only=True)["t14"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

        # Return score if all assertions pass
        return 0.5

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_u()
        self._test_v()
        self._test_gradient_fu()
        self._test_gradient_fv()
        self._test_fuv()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 2.0]")


if __name__ == '__main__':
    # Instantiate the command line parser
    parser = argparse.ArgumentParser()

    # Add the option to run only a specific test case
    parser.add_argument('--test_case', help='Name of the test case you want to run')
    parser.add_argument('--test_function', help='Name of the single test function you want to run')

    # Read the command line parameters
    args = parser.parse_args()

    # Run only a single test class
    if args.test_case:
        test_class = eval(args.test_case)

        if args.test_function:
            suite = unittest.TestSuite()
            suite.addTest(test_class(args.test_function))

        else:
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)

        unittest.TextTestRunner().run(suite)

    # Run all test classes
    else:
        unittest.main(argv=[''], verbosity=1, exit=False)
