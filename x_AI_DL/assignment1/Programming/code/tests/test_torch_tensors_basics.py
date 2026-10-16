#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
This script was written as a test module for the 'torch_tensors_basics.ipynb'-jupyter notebook.
That notebook tests your knowledge about the basic torch.Tensor-functions.
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
torch_tensors_basics = extract_module("torch_tensors_basics")

# Add the code directory to sys.path
root_dir = os.path.join(os.path.dirname(__file__), '..', '..')

# Define the base data directory path
data_dir = os.path.abspath(os.path.join(root_dir, 'data', 'torch_tensors_basics'))


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


class TestTensorCreation(unittest.TestCase):
    """
    The class contains all test cases for task "2.1 - Create a Tensor".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 2.1 - Create a Tensor...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the test parameters
        self.shape = (2, 3, 4)
        self.lst = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
        self.np_array = np.array([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
        self.tensor = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

    @TestManager.register_test
    def _test_creation_of_tensor_with_specified_shape(self):
        """ Test creation of tensor with specified shape. """
        student_version = torch_tensors_basics.create_tensor_with_specified_shape(self.shape)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_basics.pt'), weights_only=True)["t1"]

        self.assertEqual(student_version.size(), expected_version.size(), "Shape of tensor is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_creation_of_tensor_from_list(self):
        """ Test creation of tensor from a list. """
        student_version = torch_tensors_basics.create_tensor_from_list(self.lst)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_basics.pt'), weights_only=True)["t2"]

        self.assertEqual(student_version.size(), expected_version.size(), "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_creation_of_tensor_from_numpy_array(self):
        """ Test creation of tensor from a numpy array. """
        student_version = torch_tensors_basics.create_tensor_from_numpy_array(self.np_array)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_basics.pt'), weights_only=True)["t3"]

        self.assertEqual(student_version.size(), expected_version.size(), "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version.to(expected_version.dtype), expected_version, atol=0.0001), "Tensor is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_creation_of_random_tensor(self):
        """ Test creation of a random tensor with a specified shape. """
        student_version = torch_tensors_basics.create_random_tensor(self.shape)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_basics.pt'), weights_only=True)["t4"]

        self.assertEqual(student_version.size(), expected_version.size(), "Shape of tensor is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_creation_of_zero_tensor(self):
        """ Test creation of a tensor with a specified shape filled with zeros. """
        student_version = torch_tensors_basics.create_tensor_filled_with_zeros(self.shape)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_basics.pt'), weights_only=True)["t5"]

        self.assertEqual(student_version.size(), expected_version.size(), "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_creation_of_one_tensor(self):
        """ Test creation of a tensor with a specified shape filled with ones. """
        student_version = torch_tensors_basics.create_tensor_filled_with_ones(self.shape)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_basics.pt'), weights_only=True)["t6"]

        self.assertEqual(student_version.size(), expected_version.size(), "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_creation_of_one_diagonal_tensor(self):
        """ Test creation of 2D-tensor with a specified number of rows and columns for which its diagonal is filled
        with ones. """
        student_version = torch_tensors_basics.create_tensor_with_the_diagonal_filled_with_ones((self.shape[0],
                                                                                                 self.shape[1]))
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_basics.pt'), weights_only=True)["t7"]

        self.assertEqual(student_version.size(), expected_version.size(), "Shape of tensor is not correct!")
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
        self._test_creation_of_tensor_with_specified_shape()
        self._test_creation_of_tensor_from_list()
        self._test_creation_of_tensor_from_numpy_array()
        self._test_creation_of_random_tensor()
        self._test_creation_of_zero_tensor()
        self._test_creation_of_one_tensor()
        self._test_creation_of_one_diagonal_tensor()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 1.75]")


class TestTensorInformationExtraction(unittest.TestCase):
    """
    The class contains all test cases for task "2.2 - Extract Tensor Information".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 2.2 - Extract Tensor Information...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the test parameters
        self.tensor = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

    @TestManager.register_test
    def _test_extraction_of_tensor_type(self):
        """ Test extracting the type of a tensor. """
        student_version = torch_tensors_basics.extract_the_type_of_a_tensor(self.tensor)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_basics.pt'), weights_only=True)["t8"]

        self.assertEqual(student_version, expected_version, "Type of tensor is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_extraction_of_tensor_data_type(self):
        """ Test extracting the data type of a tensor. """
        student_version = torch_tensors_basics.extract_the_data_type_of_a_tensor(self.tensor)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_basics.pt'), weights_only=True)["t9"]

        self.assertEqual(student_version, expected_version, "Data type of tensor is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_extraction_of_tensor_dimension(self):
        """ Test extracting the dimension of a tensor. """
        student_version = torch_tensors_basics.extract_the_dimensions_of_a_tensor(self.tensor)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_basics.pt'), weights_only=True)["t10"]

        self.assertEqual(student_version, expected_version, "Dimension of tensor is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_extraction_of_tensor_shape(self):
        """ Test extracting the shape of a tensor. """
        student_version = torch_tensors_basics.extract_the_shape_of_a_tensor(self.tensor)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_basics.pt'), weights_only=True)["t11"]

        self.assertEqual(student_version, expected_version, "Shape of tensor is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_extraction_of_tensor_number_elements(self):
        """ Test extracting the number of elements of a tensor. """
        student_version = torch_tensors_basics.extract_the_number_of_elements_of_a_tensor(self.tensor)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_basics.pt'), weights_only=True)["t12"]

        self.assertEqual(student_version, expected_version, "Number of elements of tensor is not correct!")

        # Return score if all assertions pass
        return 0.25

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_extraction_of_tensor_type()
        self._test_extraction_of_tensor_data_type()
        self._test_extraction_of_tensor_dimension()
        self._test_extraction_of_tensor_shape()
        self._test_extraction_of_tensor_number_elements()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 1.25]")


class TestTensorSlicing(unittest.TestCase):
    """
    The class contains all test cases for task "2.3 - Indexing and Slicing".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 2.3 - Indexing and Slicing...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the test parameters
        self.tensor_slicing = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])

    @TestManager.register_test
    def _test_slicing_01(self):
        """ Test slicing of a tensor. """
        student_version = torch_tensors_basics.slicing_01(self.tensor_slicing)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_basics.pt'), weights_only=True)["t13"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_slicing_02(self):
        """ Test different slicing of a tensor. """
        student_version = torch_tensors_basics.slicing_02(self.tensor_slicing)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_basics.pt'), weights_only=True)["t14"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_slicing_03(self):
        """ Test different slicing of a tensor. """
        student_version = torch_tensors_basics.slicing_03(self.tensor_slicing)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_basics.pt'), weights_only=True)["t15"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_slicing_04(self):
        """ Test different slicing of a tensor. """
        student_version = torch_tensors_basics.slicing_04(self.tensor_slicing)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_basics.pt'), weights_only=True)["t16"]

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
        self._test_slicing_01()
        self._test_slicing_02()
        self._test_slicing_03()
        self._test_slicing_04()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 1.0]")


class TestTensorCalculations(unittest.TestCase):
    """
    The class contains all test cases for task "3 - Tensor Calculations".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 3 - Tensor Calculations...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the test parameters
        self.t1 = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
        self.t2 = torch.tensor([[41, 42, 43], [51, 52, 53], [61, 62, 63]])
        self.scalar = 2.0
        self.new_shape = (9,)

    @TestManager.register_test
    def _test_summation_of_tensors(self):
        """ Test summation of two tensors. """
        student_version = torch_tensors_basics.tensor_summation(self.t1, self.t2)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_operations_t1.pt'), weights_only=True)

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_scalar_multiplication(self):
        """ Test multiplying a scalar with a tensor. """
        student_version = torch_tensors_basics.scalar_multiplication(self.t1, self.scalar)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_operations_t2.pt'), weights_only=True)

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_hadamard_product(self):
        """ Test calculating the hadamard product of two tensors. """
        student_version = torch_tensors_basics.hadamard_product(self.t1, self.t2)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_operations_t3.pt'), weights_only=True)

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_matrix_product(self):
        """ Test calculating the matrix product of two tensors. """
        student_version = torch_tensors_basics.matrix_product(self.t1, self.t2)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_operations_t4.pt'), weights_only=True)

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_dot_product(self):
        """ Test calculating the dot product of two tensors. """
        student_version = torch_tensors_basics.dot_product(self.t1.flatten(), self.t2.flatten())
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_operations_t5.pt'), weights_only=True)

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_square_tensor(self):
        """ Test squaring all elements of a tensor. """
        student_version = torch_tensors_basics.square_tensor(self.t1)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_operations_t6.pt'), weights_only=True)

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_tensor_broadcasting(self):
        """ Test filling a tensor with a value using broadcasting. """
        t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
        t2 = torch.tensor([[[1., 2.5], [3., 4.]], [[5.5, 6.5], [7.5, 8.]]])
        t3 = torch.tensor([[1e4, 2e3], [3e5, 4e6]])
        value1, value2, value3 = 7, -5, 1e9

        student_version1 = torch_tensors_basics.fill_tensor_with_value(t1, value1)
        student_version2 = torch_tensors_basics.fill_tensor_with_value(t2, value2)
        student_version3 = torch_tensors_basics.fill_tensor_with_value(t3, value3)

        # Test 1
        self.assertTrue(torch.is_floating_point(student_version1), "Type of tensor is not correct!")
        self.assertEqual(student_version1.shape, t1.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.all(student_version1 == value1), "Tensor is not correct!")

        # Test 2
        self.assertTrue(torch.is_floating_point(student_version2), "Type of tensor is not correct!")
        self.assertEqual(student_version2.shape, t2.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.all(student_version2 == value2), "Tensor is not correct!")

        # Test 3
        self.assertTrue(torch.is_floating_point(student_version3), "Type of tensor is not correct!")
        self.assertEqual(student_version3.shape, t3.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.all(student_version3 == value3), "Tensor is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_matrix_multiplication_broadcasting(self):
        student_answers1 = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 0, 0]])
        student_answers2 = torch.tensor([[1, 1, 1, 0], [0, 1, 0, 1], [1, 0, 1, 1]])
        question_points1 = torch.tensor([2, 3, 1])
        question_points2 = torch.tensor([4, 2, 1, 3])

        student_version1 = torch_tensors_basics.find_top_scoring_student(student_answers1, question_points1)
        student_version2 = torch_tensors_basics.find_top_scoring_student(student_answers2, question_points2)

        # Test 1
        self.assertIsInstance(student_version1, torch.Tensor, "Result should be a tensor.")
        self.assertEqual(student_version1.shape, torch.tensor(100).shape, "Shape of tensor is not correct!")

        self.assertIsInstance(student_version1.item(), int, "Index is not an integer!")
        self.assertTrue(0 <= student_version1 < student_answers1.size(0), "Index is not within the correct range!")

        reference_index1 = torch.load(os.path.join(data_dir, 'reference_top_students_t1.pt'), weights_only=True)
        self.assertEqual(student_version1.item(), reference_index1.item(), "Index is not correct!")

        # Test 2
        self.assertIsInstance(student_version2, torch.Tensor, "Result should be a tensor.")
        self.assertEqual(student_version2.shape, torch.tensor(100).shape, "Shape of tensor is not correct!")

        self.assertIsInstance(student_version2.item(), int, "Index is not an integer!")
        self.assertTrue(0 <= student_version2 < student_answers2.size(0), "Index is not within the correct range!")

        reference_index2 = torch.load(os.path.join(data_dir, 'reference_top_students_t2.pt'), weights_only=True)
        self.assertEqual(student_version2.item(), reference_index2.item(), "Index is not correct!")

        # Return score if all assertions pass
        return 0.5

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_summation_of_tensors()
        self._test_scalar_multiplication()
        self._test_hadamard_product()
        self._test_matrix_product()
        self._test_dot_product()
        self._test_square_tensor()
        self._test_tensor_broadcasting()
        self._test_matrix_multiplication_broadcasting()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 2.5]")


class TestTensorStatistics(unittest.TestCase):
    """
    The class contains all test cases for task "4 - Tensor Statistics".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 4 - Tensor Statistics...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the test parameters
        self.t1 = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
        self.t2 = torch.tensor([[41, 42, 43], [51, 52, 53], [61, 62, 63]])

    @TestManager.register_test
    def _test_minimum(self):
        """ Test finding the minimum of a tensor. """
        student_version = torch_tensors_basics.minimum(self.t1)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_operations_t7.pt'), weights_only=True)

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_maximum(self):
        """ Test finding the maximum of a tensor. """
        student_version = torch_tensors_basics.maximum(self.t1)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_operations_t8.pt'), weights_only=True)

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_mean(self):
        """ Test finding the mean of a tensor. """
        student_version = torch_tensors_basics.mean(self.t1)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_operations_t9.pt'), weights_only=True)

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_standard_deviation(self):
        """ Test finding the standard deviation of a tensor. """
        student_version = torch_tensors_basics.standard_deviation(self.t1)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_operations_t10.pt'), weights_only=True)

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_statistics_applied(self):
        """ Test finding the row with highest mean of a tensor. """
        t1 = torch.tensor([[1, 2], [4, 5], [7, 8]])
        t2 = torch.tensor([[1, 1, 1], [2, 0, 2], [1, 2, 1]])

        student_version1 = torch_tensors_basics.find_row_with_highest_mean(t1)
        student_version2 = torch_tensors_basics.find_row_with_highest_mean(t2)

        # Test 1
        self.assertIsInstance(student_version1, int, "Result should be an integer.")
        self.assertTrue(0 <= student_version1 < t1.size(0), "Index is not within the correct range!")

        reference_index1 = torch.load(os.path.join(data_dir, 'reference_top_means_i1.pt'), weights_only=True)
        self.assertEqual(student_version1, reference_index1, "Index is not correct!")

        # Test 2
        self.assertIsInstance(student_version2, int, "Result should be an integer.")
        self.assertTrue(0 <= student_version2 < t2.size(0), "Index is not within the correct range!")

        reference_index2 = torch.load(os.path.join(data_dir, 'reference_top_means_i2.pt'), weights_only=True)
        self.assertEqual(student_version2, reference_index2, "Index is not correct!")

        # Return score if all assertions pass
        return 0.5

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_minimum()
        self._test_maximum()
        self._test_mean()
        self._test_standard_deviation()
        self._test_statistics_applied()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 1.5]")


class TestTensorCasting(unittest.TestCase):
    """
    The class contains all test cases for task "5.1 - Tensor Operations (Casting)".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 5.1 - Tensor Operations (Casting)...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the test parameters
        self.t1 = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
        self.t2 = torch.tensor([[41, 42, 43], [51, 52, 53], [61, 62, 63]])

    @TestManager.register_test
    def _test_cast_to_list(self):
        """ Test casting a tensor to a Python list. """
        student_version = torch_tensors_basics.cast_to_list(self.t1)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_operations_t11.pt'), weights_only=True)

        self.assertEqual(type(student_version), type(expected_version), "Type of solution is not correct!")
        self.assertEqual(len(student_version), len(expected_version), "Shape of solution is not correct!")
        self.assertTrue(student_version == expected_version, "Solution is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_cast_to_numpy(self):
        """ Test casting a tensor to a Numpy array. """
        student_version = torch_tensors_basics.cast_to_numpy(self.t1)
        expected_version = np.load(os.path.join(data_dir, 'reference_tensor_operations_t12.npz'))["data"]

        self.assertEqual(type(student_version), type(expected_version), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(np.array_equal(student_version, expected_version), "Solution is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_cast_to_double(self):
        """ Test casting an integer tensor to a double tensor. """
        student_version = torch_tensors_basics.cast_to_double(self.t1)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_operations_t13.pt'), weights_only=True)

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

        # Return score if all assertions pass
        return 0.25

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_cast_to_list()
        self._test_cast_to_numpy()
        self._test_cast_to_double()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 0.75]")


class TestTensorTransforming(unittest.TestCase):
    """
    The class contains all test cases for task "5.2 - Tensor Operations (Transform a Tensor)".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 5.2 - Tensor Operations (Transform a Tensor)...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the test parameters
        self.t1 = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])

    @TestManager.register_test
    def _test_reshape_contiguous(self):
        """ Test reshaping a contiguous tensor to a new shape. """
        t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
        new_shape = (3, 2)

        student_version = torch_tensors_basics.reshape_contiguous_tensor(t1, new_shape)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_reshaping_t1.pt'), weights_only=True)

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

        random_number = torch.randint(low=100, high=1000, size=(1,)).item()
        student_version[0, 0] = random_number
        self.assertEqual(t1[0, 0], random_number, "Output tensor should be a view of the input tensor!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_reshape_non_contiguous(self):
        """ Test reshaping a non-contiguous tensor to a new shape. """
        t1 = torch.tensor([[1, 2], [3, 4]])
        new_shape1 = (4, 1)

        t2 = torch.arange(1, 13).view(6, 2)
        new_shape2 = (4, 3)

        student_version1 = torch_tensors_basics.reshape_non_contiguous_tensor(t1, new_shape1)
        expected_version1 = torch.load(os.path.join(data_dir, 'reference_tensor_reshaping_t2.pt'), weights_only=True)

        student_version2 = torch_tensors_basics.reshape_non_contiguous_tensor(t2, new_shape2)
        expected_version2 = torch.load(os.path.join(data_dir, 'reference_tensor_reshaping_t3.pt'), weights_only=True)

        random_number = torch.randint(low=100, high=1000, size=(1,)).item()

        # Test 1
        self.assertEqual(student_version1.type(), expected_version1.type(), "Type of solution is not correct!")
        self.assertEqual(student_version1.shape, expected_version1.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version1, expected_version1, atol=0.0001), "Solution is not correct!")

        self.assertTrue(student_version1.is_contiguous(), "Output tensor should be contiguous!")

        student_version1[0, 0] = random_number
        self.assertNotEqual(t1[0, 0], random_number, "Output tensor should be a view of the input tensor!")

        # Test 2
        self.assertEqual(student_version2.type(), expected_version2.type(), "Type of solution is not correct!")
        self.assertEqual(student_version2.shape, expected_version2.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version2, expected_version2, atol=0.0001), "Solution is not correct!")

        self.assertTrue(student_version2.is_contiguous(), "Output tensor should be contiguous!")

        student_version2[0, 0] = random_number
        self.assertNotEqual(t2[0, 0], random_number, "Output tensor should be a view of the input tensor!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_swap(self):
        """ Test swapping the first axis of a 3D-tensor with its last one. """
        student_version = torch_tensors_basics.swap(torch.unsqueeze(self.t1, dim=0))
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_operations_t15.pt'), weights_only=True)

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_trans(self):
        """ Test transposing a tensor. """
        student_version = torch_tensors_basics.trans(self.t1)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_operations_t16.pt'), weights_only=True)

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_pave(self):
        """ Test paving an nD-tensor to 1D. """
        student_version = torch_tensors_basics.pave(self.t1)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_operations_t17.pt'), weights_only=True)

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

        # Return score if all assertions pass
        return 0.25

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_reshape_contiguous()
        self._test_reshape_non_contiguous()
        self._test_swap()
        self._test_trans()
        self._test_pave()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 1.75]")


class TestTensorConcatenation(unittest.TestCase):
    """
    The class contains all test cases for task "5.3 - Tensor Operations (Concatenate Tensors)".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 5.3 - Tensor Operations (Concatenate Tensors)...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the test parameters
        self.axis = 0

    @TestManager.register_test
    def _test_concat1(self):
        """ Test concatenating two tensors along a specified axis. """
        t1 = torch.tensor([[11, 12], [21, 22], [31, 32]])
        t2 = torch.tensor([[41, 42], [51, 52], [61, 62]])

        student_version = torch_tensors_basics.concat(t1, t2, self.axis)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_operations_t18.pt'), weights_only=True)

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_concat2(self):
        """ Test concatenating another two tensors along a specified axis. """
        t1 = torch.tensor([[11, 12], [21, 22], [31, 32]])
        t2 = torch.tensor([[41, 42, 43], [51, 52, 53]])

        student_version = torch_tensors_basics.concat(t1, t2, self.axis)
        expected_version = torch.load(os.path.join(data_dir, 'reference_tensor_operations_t19.pt'), weights_only=True)

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

        # Return score if all assertions pass
        return 0.5

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_concat1()
        self._test_concat2()

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