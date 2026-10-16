#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
This script was written as a test module for the 'linear_regression.ipynb'-jupyter notebook.
That notebook lets you implement and apply a linear regression on some sample data.
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
import unittest
import argparse

import sys
import os
# Add the code directory to sys.path
root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.abspath(root_dir))

# Import own files
# Adds the parent folder relative to this to the path to use the extracted logic from notebook.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
# Extracts and imports the module.
from extract_logic_module import extract_module
linear_regression = extract_module("linear_regression")

from code.tests.utils import TestManager

class TestDataLoader(unittest.TestCase):
    """
    The class contains all test cases for task "2 - Dataset Loading".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 2 - Dataset Loading...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the test parameters
        self.file_path = os.path.abspath(os.path.join(root_dir, 'data', 'linear_regression_dataset.npz'))

        # Load the data
        self.data = np.load(self.file_path)

        # Create the students data loader
        self.students_data_loader = linear_regression.DataLoader(self.file_path)

    @TestManager.register_test
    def _test_x(self):
        """ Test loading the x-values of all samples. """
        student_version, _ = self.students_data_loader.get_samples()
        expected_version = self.data['x']

        self.assertEqual(type(student_version), type(expected_version), "Type of x is not correct!")
        self.assertEqual(len(student_version), len(expected_version), "Length of x is not correct!")
        self.assertTrue(np.allclose(student_version, expected_version, atol=0.0001), "x is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_y(self):
        """ Test loading the y-values of all samples. """
        _, student_version = self.students_data_loader.get_samples()
        expected_version = self.data['y']

        self.assertEqual(type(student_version), type(expected_version), "Type of y is not correct!")
        self.assertEqual(len(student_version), len(expected_version), "Length of y is not correct!")
        self.assertTrue(np.allclose(student_version, expected_version, atol=0.0001), "y is not correct!")

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
        self._test_y()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 1.0]")


class TestLossFunction(unittest.TestCase):
    """
    The class contains all test cases for task "3.1 - Loss Function of Linear Regression".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 3.1 - Loss Function of Linear Regression...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the test parameters
        self.file_path = os.path.abspath(os.path.join(root_dir, 'data', 'linear_regression_dataset.npz'))

        # Load the expected data
        self.data = np.load(self.file_path)

        # Create the students data loader
        self.students_data_loader = linear_regression.DataLoader(self.file_path)

    @TestManager.register_test
    def _test_loss_function(self):
        """ Test the loss function. """
        student_version = linear_regression.loss_linear_regression(self.data['x'], self.data['y'], self.data['init_w0'],
                                                                   self.data['init_w1'])
        reference_path = os.path.abspath(os.path.join(root_dir, 'code', 'tests', 'references', 'linear_regression.npz'))
        expected_version = np.load(reference_path)['test_loss_function']

        self.assertAlmostEqual(student_version, expected_version, delta=0.0001, msg="Loss is not correct!")

        # Return score if all assertions pass
        return 0.5

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_loss_function()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 0.5]")



class TestLossDerivatives(unittest.TestCase):
    """
    The class contains all test cases for task "3.2 - Derivatives of Loss Function of Linear Regression".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 3.2 - Derivatives of Loss Function of Linear Regression...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the test parameters
        self.file_path = os.path.abspath(os.path.join(root_dir, 'data', 'linear_regression_dataset.npz'))

        # Load the expected data
        self.data = np.load(self.file_path)

        # Create the students data loader
        self.students_data_loader = linear_regression.DataLoader(self.file_path)

    @TestManager.register_test
    def _test_w0_derivative(self):
        """ Test the partial derivative of the loss function w.r.t w0. """
        student_version, _ = linear_regression.compute_derivatives_of_loss(self.data['x'], self.data['y'],
                                                                           self.data['init_w0'], self.data['init_w1'])
        reference_path = os.path.abspath(os.path.join(root_dir, 'code', 'tests', 'references', 'linear_regression.npz'))
        expected_version = np.load(reference_path)['test_loss_derivatives_w0']

        self.assertAlmostEqual(student_version, expected_version, delta=0.0001, msg="Partial derivative w.r.t w0 is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_w1_derivative(self):
        """ Test the partial derivative of the loss function w.r.t w1. """
        _, student_version = linear_regression.compute_derivatives_of_loss(self.data['x'], self.data['y'],
                                                                           self.data['init_w0'], self.data['init_w1'])
        reference_path = os.path.abspath(os.path.join(root_dir, 'code', 'tests', 'references', 'linear_regression.npz'))
        expected_version = np.load(reference_path)['test_loss_derivatives_w1']

        self.assertAlmostEqual(student_version, expected_version, delta=0.0001, msg="Partial derivative w.r.t w1 is not correct!")

        # Return score if all assertions pass
        return 0.5

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_w0_derivative()
        self._test_w1_derivative()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 1.0]")


class TestGradientDescent(unittest.TestCase):
    """
    The class contains all test cases for task "3.3 - Optimize the Model Parameters".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 3.3 - Optimize the Model Parameters...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the test parameters
        self.file_path = os.path.abspath(os.path.join(root_dir, 'data', 'linear_regression_dataset.npz'))

        # Load the expected data
        self.data = np.load(self.file_path)

        # Create the students data loader
        self.students_data_loader = linear_regression.DataLoader(self.file_path)

    @TestManager.register_test
    def _test_gradient_descent_step(self):
        """ Test the gradient descent algorithm. """
        reference_path = os.path.abspath(os.path.join(root_dir, 'code', 'tests', 'references', 'linear_regression.npz'))
        theta_derivative = np.load(reference_path)['test_gradient_descent_theta_dev']

        student_version = linear_regression.gradient_descent_update_step(self.data['theta'], theta_derivative,
                                                                         self.data['lr'])
        expected_version = np.load(reference_path)['test_gradient_descent_new_theta']

        self.assertAlmostEqual(student_version, expected_version, delta=0.0001, msg="Gradient descent step is not correct!")

        # Return score if all assertions pass
        return 0.5

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_gradient_descent_step()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 0.5]")


class TestTraining(unittest.TestCase):
    """
    The class contains all test cases for task "3.4 - Let’s Put Everything Together and Train Our Model".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 3.4 - Let’s Put Everything Together and Train Our Model...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the test parameters
        self.file_path = os.path.abspath(os.path.join(root_dir, 'data', 'linear_regression_dataset.npz'))

        # Load the expected data
        self.data = np.load(self.file_path)

    @TestManager.register_test
    def _test_optimal_w0(self):
        """ Test the optimization of w0. """
        student_version, _, _ = linear_regression.train(self.data['x'], self.data['y'], self.data['init_w0'],
                                                        self.data['init_w1'], self.data['epochs'], self.data['lr'],
                                                        verbose=False)
        reference_path = os.path.abspath(os.path.join(root_dir, 'code', 'tests', 'references', 'linear_regression.npz'))
        expected_version = np.load(reference_path)['test_training_w0']

        self.assertAlmostEqual(student_version, expected_version, delta=0.0001, msg="Training for w0 is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_optimal_w1(self):
        """ Test the optimization of w1. """
        _, student_version, _ = linear_regression.train(self.data['x'], self.data['y'], self.data['init_w0'],
                                                        self.data['init_w1'], self.data['epochs'], self.data['lr'],
                                                        verbose=False)

        reference_path = os.path.abspath(os.path.join(root_dir, 'code', 'tests', 'references', 'linear_regression.npz'))
        expected_version = np.load(reference_path)['test_training_w1']

        self.assertAlmostEqual(student_version, expected_version, delta=0.0001, msg="Training for w1 is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_losses(self):
        """ Test the losses. """
        _, _, student_version = linear_regression.train(self.data['x'], self.data['y'], self.data['init_w0'],
                                                        self.data['init_w1'], self.data['epochs'], self.data['lr'],
                                                        verbose=False)

        reference_path = os.path.abspath(os.path.join(root_dir, 'code', 'tests', 'references', 'linear_regression.npz'))
        expected_version = np.load(reference_path)['test_training_all_losses']

        self.assertEqual(len(student_version), len(expected_version), "Length of all_losses is not correct!")
        self.assertTrue(np.allclose(np.array(student_version), np.array(expected_version), atol=0.0001),
                        "Some of the calculated losses are not correct!")

        # Return score if all assertions pass
        return 0.5

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_optimal_w0()
        self._test_optimal_w1()
        self._test_losses()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 1.0]")


class TestRealWorldDataLoader(unittest.TestCase):
    """
    The class contains all test cases for task "4.1 - Load the data".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 4.1 - Load the data...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the test parameters
        self.file_path = os.path.abspath(os.path.join(root_dir, 'data', 'linear_regression_real_world_dataset.npz'))

        # Load the data
        self.data = np.load(self.file_path)

    @TestManager.register_test
    def _test_x(self):
        """ Test loading the years. """
        student_version, _ = linear_regression.load_real_world_dataset(self.file_path)
        reference_path = os.path.abspath(os.path.join(root_dir, 'code', 'tests', 'references', 'linear_regression.npz'))
        expected_version = np.load(reference_path)['test_real_world_x']

        self.assertEqual(type(student_version), type(expected_version), "Type of years is not correct!")
        self.assertEqual(len(student_version), len(expected_version), "Length of years is not correct!")
        self.assertTrue(np.allclose(student_version, expected_version, atol=0.0001), "years is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_y(self):
        """ Test loading the median sales prices. """
        _, student_version = linear_regression.load_real_world_dataset(self.file_path)
        reference_path = os.path.abspath(os.path.join(root_dir, 'code', 'tests', 'references', 'linear_regression.npz'))
        expected_version = np.load(reference_path)['test_real_world_y']

        self.assertEqual(type(student_version), type(expected_version), "Type of prices is not correct!")
        self.assertEqual(len(student_version), len(expected_version), "Length of prices is not correct!")
        self.assertTrue(np.allclose(student_version, expected_version, atol=0.0001), "prices is not correct!")

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
        self._test_y()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 1.0]")


class TestRealWorldTraining(unittest.TestCase):
    """
    The class contains all test cases for task "4.2 - Run the Linear Regression Algorithm".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 4.2 - Run the Linear Regression Algorithm...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the test parameters
        self.file_path = os.path.abspath(os.path.join(root_dir, 'data', 'linear_regression_real_world_dataset.npz'))

        # Load the expected data
        self.data = np.load(self.file_path)

    @TestManager.register_test
    def _test_optimal_w0(self):
        """ Test the optimization of w0. """
        student_version, _, _ = linear_regression.train_for_real_world_dataset(self.file_path, self.data['init_w0'],
                                                                               self.data['init_w1'],
                                                                               self.data['epochs'],
                                                                               self.data['lr'],
                                                                               verbose=False)

        reference_path = os.path.abspath(os.path.join(root_dir, 'code', 'tests', 'references', 'linear_regression.npz'))
        expected_version = np.load(reference_path)['test_training_real_world_w0']

        self.assertAlmostEqual(student_version, expected_version, delta=0.0001, msg="Training for w0 is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_optimal_w1(self):
        """ Test the optimization of w1. """
        _, student_version, _ = linear_regression.train_for_real_world_dataset(self.file_path, self.data['init_w0'],
                                                                               self.data['init_w1'],
                                                                               self.data['epochs'],
                                                                               self.data['lr'],
                                                                               verbose=False)

        reference_path = os.path.abspath(os.path.join(root_dir, 'code', 'tests', 'references', 'linear_regression.npz'))
        expected_version = np.load(reference_path)['test_training_real_world_w1']

        self.assertAlmostEqual(student_version, expected_version, delta=0.0001, msg="Training for w1 is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_losses(self):
        """ Test the losses. """
        _, _, student_version = linear_regression.train_for_real_world_dataset(self.file_path, self.data['init_w0'],
                                                                               self.data['init_w1'],
                                                                               self.data['epochs'],
                                                                               self.data['lr'],
                                                                               verbose=False)

        reference_path = os.path.abspath(os.path.join(root_dir, 'code', 'tests', 'references', 'linear_regression.npz'))
        expected_version = np.load(reference_path)['test_training_real_world_all_losses']

        self.assertEqual(len(student_version), len(expected_version), "Length of all_losses is not correct!")
        self.assertTrue(np.allclose(np.array(student_version), np.array(expected_version), atol=0.0001),
                        "Some of the calculated losses are not correct!")

        # Return score if all assertions pass
        return 0.5

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_optimal_w0()
        self._test_optimal_w1()
        self._test_losses()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 1.0]")


class TestRealWorldPrediction(unittest.TestCase):
    """
    The class contains all test cases for task "4.3 - Calculate the Expected Median Sales Price in 2021".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 4.3 - Calculate the Expected Median Sales Price in 2021...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the test parameters
        self.file_path = os.path.abspath(os.path.join(root_dir, 'data', 'linear_regression_real_world_dataset.npz'))

        # Load the expected data
        self.data = np.load(self.file_path)

    @TestManager.register_test
    def _test_expected_median_sale_price(self):
        """ Test the prediction of the median sales prices. """
        reference_path = os.path.abspath(os.path.join(root_dir, 'code', 'tests', 'references', 'linear_regression.npz'))
        trained_w0 = np.load(reference_path)['test_training_real_world_w0']
        trained_w1 = np.load(reference_path)['test_training_real_world_w1']

        student_version = linear_regression.predict(trained_w0, trained_w1, self.data['predicted_year'])
        expected_version = np.load(reference_path)['test_prediction']

        self.assertAlmostEqual(student_version, expected_version, delta=0.0001, msg="Expected Median Sale Price is not correct!")

        # Return score if all assertions pass
        return 1.0

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_expected_median_sale_price()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 1.0]")


if __name__ == '__main__':
    # Instantiate the command line parser
    parser = argparse.ArgumentParser()

    # Add the option to run only a specific test case
    parser.add_argument('--test_case', help='Name of the test case you want to run')

    # Read the command line parameters
    args = parser.parse_args()

    # Run only a single test class
    if args.test_case:
        test_class = eval(args.test_case)
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        unittest.TextTestRunner().run(suite)

    # Run all test classes
    else:
        unittest.main(argv=[''], verbosity=1, exit=False)
