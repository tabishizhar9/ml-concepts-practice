#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
This script was written as a test module for the 'numpy_basics.ipynb'-jupyter notebook.
That notebook introduces the basic functions and processes of the free numpy package.
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
import traceback

# Add the code directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import own files
# Adds the parent folder relative to this to the path to use the extracted logic from notebook.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
# Extracts and imports the module.
from extract_logic_module import extract_module
numpy_basics = extract_module("numpy_basics")

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


class TestArrayProperties(unittest.TestCase):
    """
    The class contains all test cases for task 1.1 - General Array Properties.
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 1.1 - General Array Properties...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the tests
        self.np_a1 = np.array([0, 1, 2, 3, 4, 5])
        self.np_a2 = np.array([[0, 1], [1, 0]])
        self.np_a3 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    @TestManager.register_test
    def _test_shape(self):
        """ Test shape of arrays. """
        student_shape1 = numpy_basics.extract_array_dimensions_01(self.np_a1)
        student_shape2 = numpy_basics.extract_array_dimensions_01(self.np_a2)
        student_shape3 = numpy_basics.extract_array_dimensions_01(self.np_a3)
        expected_shape1 = (6, )
        expected_shape2 = (2, 2)
        expected_shape3 = (3, 3)

        self.assertEqual(expected_shape1, student_shape1)
        self.assertEqual(expected_shape2, student_shape2)
        self.assertEqual(expected_shape3, student_shape3)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_size(self):
        """ Test size of arrays. """
        student_size1 = numpy_basics.extract_array_dimensions_02(self.np_a1)
        student_size2 = numpy_basics.extract_array_dimensions_02(self.np_a2)
        student_size3 = numpy_basics.extract_array_dimensions_02(self.np_a3)
        expected_size1 = 6
        expected_size2 = 4
        expected_size3 = 9

        self.assertEqual(expected_size1, student_size1)
        self.assertEqual(expected_size2, student_size2)
        self.assertEqual(expected_size3, student_size3)

        # Return score if all assertions pass
        return 0.25

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_shape()
        self._test_size()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 0.5]")


class TestArrayCreation(unittest.TestCase):
    """
    The class contains all test cases for task 1.2 - Creating Arrays.
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 1.2 - Creating Arrays...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the tests
        self.zeros_shape1 = (1, )
        self.zeros_shape2 = (2, 2)
        self.zeros_shape3 = (3, 5, 4, 1)

        self.ones_shape1 = (1, )
        self.ones_shape2 = (2, 2)
        self.ones_shape3 = (3, 5, 4, 1)

        self.array_like1 = np.array([0, 1, 2, 3, 4, 5])
        self.array_like2 = np.array([[0, 1], [1, 0]])
        self.array_like3 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

        self.range1 = [0, 11, 1]
        self.range2 = [-5, 5, 2]
        self.range3 = [100, 1000, 10]

        self.elements1 = [0, 11, 11]
        self.elements2 = [-5, 5, 100]
        self.elements3 = [0, 1, 1]

    @TestManager.register_test
    def _test_zeros(self):
        """ Test creation of array filled only with zeros. """
        student_version1 = numpy_basics.create_numpy_arrays_01(self.zeros_shape1)
        student_version2 = numpy_basics.create_numpy_arrays_01(self.zeros_shape2)
        student_version3 = numpy_basics.create_numpy_arrays_01(self.zeros_shape3)

        # Test 1
        self.assertEqual(student_version1.shape, self.zeros_shape1)
        self.assertEqual(np.sum(student_version1), 0)

        # Test 2
        self.assertEqual(student_version2.shape, self.zeros_shape2)
        self.assertEqual(np.sum(student_version2), 0)

        # Test 3
        self.assertEqual(student_version3.shape, self.zeros_shape3)
        self.assertEqual(np.sum(student_version3), 0)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_ones(self):
        """ Test creation of array filled only with ones. """
        student_version1 = numpy_basics.create_numpy_arrays_02(self.ones_shape1)
        student_version2 = numpy_basics.create_numpy_arrays_02(self.ones_shape2)
        student_version3 = numpy_basics.create_numpy_arrays_02(self.ones_shape3)

        # Test 1
        self.assertEqual(student_version1.shape, self.ones_shape1)
        self.assertEqual(np.max(student_version1), 1)
        self.assertEqual(np.min(student_version1), 1)

        # Test 2
        self.assertEqual(student_version2.shape, self.ones_shape2)
        self.assertEqual(np.max(student_version2), 1)
        self.assertEqual(np.min(student_version2), 1)

        # Test 3
        self.assertEqual(student_version3.shape, self.ones_shape3)
        self.assertEqual(np.max(student_version3), 1)
        self.assertEqual(np.min(student_version3), 1)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_array_like(self):
        """ Test creation of array which entries shall be replaced with zeros. """
        student_version1 = numpy_basics.create_numpy_arrays_03(self.array_like1)
        student_version2 = numpy_basics.create_numpy_arrays_03(self.array_like2)
        student_version3 = numpy_basics.create_numpy_arrays_03(self.array_like3)

        # Test 1
        self.assertEqual(student_version1.shape, self.array_like1.shape)
        self.assertEqual(np.sum(student_version1), 0)

        # Test 2
        self.assertEqual(student_version2.shape, self.array_like2.shape)
        self.assertEqual(np.sum(student_version2), 0)

        # Test 3
        self.assertEqual(student_version3.shape, self.array_like3.shape)
        self.assertEqual(np.sum(student_version3), 0)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_range(self):
        """ Test creation of ranged array. """
        student_version1 = numpy_basics.create_numpy_arrays_04(self.range1)
        student_version2 = numpy_basics.create_numpy_arrays_04(self.range2)
        student_version3 = numpy_basics.create_numpy_arrays_04(self.range3)

        # Test 1
        self.assertEqual(student_version1.shape, (11, ))
        self.assertEqual(np.sum(student_version1), 55)

        # Test 2
        self.assertEqual(student_version2.shape, (5, ))
        self.assertEqual(np.sum(student_version2), -5)

        # Test 3
        self.assertEqual(student_version3.shape, (90, ))
        self.assertEqual(np.sum(student_version3), 49050)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_elements(self):
        """ Test creation of array with specified amount of elements. """
        student_version1 = numpy_basics.create_numpy_arrays_05(self.elements1)
        student_version2 = numpy_basics.create_numpy_arrays_05(self.elements2)
        student_version3 = numpy_basics.create_numpy_arrays_05(self.elements3)

        # Test 1
        self.assertEqual(student_version1.shape, (11, ))
        self.assertAlmostEqual(float(np.sum(student_version1)), 60.5, None, "Values are not almost equal!", 0.0001)

        # Test 2
        self.assertEqual(student_version2.shape, (100, ))
        self.assertAlmostEqual(float(np.sum(student_version2)), 0.0, None, "Values are not almost equal!", 0.0001)

        # Test 3
        self.assertEqual(student_version3.shape, (1, ))
        self.assertAlmostEqual(float(np.sum(student_version3)), 0.0, None, "Values are not almost equal!", 0.0001)

        # Return score if all assertions pass
        return 0.5

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_zeros()
        self._test_ones()
        self._test_array_like()
        self._test_range()
        self._test_elements()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 1.5]")


class TestArrayHandling(unittest.TestCase):
    """
    The class contains all test cases for task 1.3 - Array Handling.
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 1.3 - Array Handling...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the tests
        # Reshape array
        self.reshape_array1 = np.array([0, 1, 2, 3, 4, 5])
        self.reshape_array2 = np.array([[0, 1], [1, 0]])
        self.reshape_array3 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

        self.reshaped_shape1 = (2, 3)
        self.reshaped_shape2 = (1, 4)
        self.reshaped_shape3 = (1, 1, 9)

        # Vertical and horizontal stack
        self.stack1 = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
        self.stack2 = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9], [10, 11, 12]])]
        self.stack3 = [np.array([[1], [2], [3]]), np.array([[4], [5], [6]]), np.array([[7], [8], [9]])]

        # Concatenating
        self.concat_arrays1 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        self.concat_arrays2 = [np.array([[1, 2, 3], [3, 4, 5]]), np.array([[6, 7, 8, 9], [10, 11, 12, 13]])]
        self.concat_arrays3 = [np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]], [[5, 5], [6, 6]]]),
                               np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]])]

        self.concat_axis1 = 0
        self.concat_axis2 = 1
        self.concat_axis3 = 2

        # Repeating
        self.repeat_array1 = np.array([0, 1, 2, 3, 4, 5])
        self.repeat_array2 = np.array([[0, 1], [1, 0]])
        self.repeat_array3 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

        self.repeat_times1 = (2, )
        self.repeat_times2 = (1, 2)
        self.repeat_times3 = (2, 2)

        # Transposing
        self.transpose_array1 = np.array([0, 1, 2, 3, 4, 5])
        self.transpose_array2 = np.array([[0, 1], [1, 0]])
        self.transpose_array3 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    @TestManager.register_test
    def _test_reshape(self):
        """ Test reshaping arrays. """
        student_version1 = numpy_basics.handle_numpy_arrays_01([self.reshape_array1, self.reshaped_shape1])
        student_version2 = numpy_basics.handle_numpy_arrays_01([self.reshape_array2, self.reshaped_shape2])
        student_version3 = numpy_basics.handle_numpy_arrays_01([self.reshape_array3, self.reshaped_shape3])

        # Test 1
        self.assertEqual(student_version1.shape, self.reshaped_shape1)
        self.assertEqual(np.min(student_version1), 0)
        self.assertEqual(np.max(student_version1), 5)
        self.assertEqual(np.sum(student_version1), 15)
        self.assertTrue(np.allclose(student_version1[0], np.array([0, 1, 2])))

        # Test 2
        self.assertEqual(student_version2.shape, self.reshaped_shape2)
        self.assertEqual(np.min(student_version2), 0)
        self.assertEqual(np.max(student_version2), 1)
        self.assertEqual(np.sum(student_version2), 2)
        self.assertTrue(np.allclose(student_version2[0], np.array([0, 1, 1, 0])))

        # Test 3
        self.assertEqual(student_version3.shape, self.reshaped_shape3)
        self.assertEqual(np.min(student_version3), -1)
        self.assertEqual(np.max(student_version3), 1)
        self.assertEqual(np.sum(student_version3), 0)
        self.assertTrue(np.allclose(student_version3[0], np.array([[1,  1,  1,  0,  0,  0, -1, -1, -1]])))

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_vstack(self):
        """ Test vertically stacking arrays. """
        student_version1 = numpy_basics.handle_numpy_arrays_02(self.stack1)
        student_version2 = numpy_basics.handle_numpy_arrays_02(self.stack2)
        student_version3 = numpy_basics.handle_numpy_arrays_02(self.stack3)

        # Test 1
        self.assertEqual(student_version1.shape, (3, 3))
        self.assertEqual(np.min(student_version1), 1)
        self.assertEqual(np.max(student_version1), 9)
        self.assertEqual(np.sum(student_version1), 45)
        self.assertTrue(np.allclose(student_version1[0], np.array([1, 2, 3])))

        # Test 2
        self.assertEqual(student_version2.shape, (4, 3))
        self.assertEqual(np.min(student_version2), 1)
        self.assertEqual(np.max(student_version2), 12)
        self.assertEqual(np.sum(student_version2), 78)
        self.assertTrue(np.allclose(student_version2[0], np.array([1, 2, 3])))
        self.assertTrue(np.allclose(student_version2[2], np.array([7, 8, 9])))

        # Test 3
        self.assertEqual(student_version3.shape, (9, 1))
        self.assertEqual(np.min(student_version3), 1)
        self.assertEqual(np.max(student_version3), 9)
        self.assertEqual(np.sum(student_version3), 45)
        self.assertTrue(np.allclose(student_version3[0], np.array([1])))
        self.assertTrue(np.allclose(student_version3[4], np.array([5])))

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_hstack(self):
        """ Test horizontally stacking arrays. """
        student_version1 = numpy_basics.handle_numpy_arrays_03(self.stack1)
        student_version2 = numpy_basics.handle_numpy_arrays_03(self.stack2)
        student_version3 = numpy_basics.handle_numpy_arrays_03(self.stack3)

        # Test 1
        self.assertEqual(student_version1.shape, (9,))
        self.assertEqual(np.min(student_version1), 1)
        self.assertEqual(np.max(student_version1), 9)
        self.assertEqual(np.sum(student_version1), 45)
        self.assertTrue(np.allclose(student_version1[0], np.array([1])))

        # Test 2
        self.assertEqual(student_version2.shape, (2, 6))
        self.assertEqual(np.min(student_version2), 1)
        self.assertEqual(np.max(student_version2), 12)
        self.assertEqual(np.sum(student_version2), 78)
        self.assertTrue(np.allclose(student_version2[0], np.array([1, 2, 3, 7, 8, 9])))

        # Test 3
        self.assertEqual(student_version3.shape, (3, 3))
        self.assertEqual(np.min(student_version3), 1)
        self.assertEqual(np.max(student_version3), 9)
        self.assertEqual(np.sum(student_version3), 45)
        self.assertTrue(np.allclose(student_version3[0], np.array([1, 4, 7])))

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_concatenation(self):
        """ Test concatenating of arrays. """
        student_version1 = numpy_basics.handle_numpy_arrays_04([*self.concat_arrays1, self.concat_axis1])
        student_version2 = numpy_basics.handle_numpy_arrays_04([*self.concat_arrays2, self.concat_axis2])
        student_version3 = numpy_basics.handle_numpy_arrays_04([*self.concat_arrays3, self.concat_axis3])

        # Test 1
        self.assertEqual(student_version1.shape, (6,))
        self.assertEqual(np.min(student_version1), 1)
        self.assertEqual(np.max(student_version1), 6)
        self.assertEqual(np.sum(student_version1), 21)
        self.assertTrue(np.allclose(student_version1[0], np.array([1])))

        # Test 2
        self.assertEqual(student_version2.shape, (2, 7))
        self.assertEqual(np.min(student_version2), 1)
        self.assertEqual(np.max(student_version2), 13)
        self.assertEqual(np.sum(student_version2), 94)
        self.assertTrue(np.allclose(student_version2[0], np.array([1, 2, 3, 6, 7, 8, 9])))

        # Test 3
        self.assertEqual(student_version3.shape, (3, 2, 5))
        self.assertEqual(np.min(student_version3), 1)
        self.assertEqual(np.max(student_version3), 6)
        self.assertEqual(np.sum(student_version3), 105)
        self.assertTrue(np.allclose(student_version3[0], np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])))

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_repeating(self):
        """ Test repeating arrays. """
        student_version1 = numpy_basics.handle_numpy_arrays_05([self.repeat_array1, self.repeat_times1])
        student_version2 = numpy_basics.handle_numpy_arrays_05([self.repeat_array2, self.repeat_times2])
        student_version3 = numpy_basics.handle_numpy_arrays_05([self.repeat_array3, self.repeat_times3])

        # Test 1
        self.assertEqual(student_version1.shape, (12,))
        self.assertEqual(np.min(student_version1), 0)
        self.assertEqual(np.max(student_version1), 5)
        self.assertEqual(np.sum(student_version1), 30)
        self.assertTrue(np.allclose(student_version1[0], 0))

        # Test 2
        self.assertEqual(student_version2.shape, (2, 4))
        self.assertEqual(np.min(student_version2), 0)
        self.assertEqual(np.max(student_version2), 1)
        self.assertEqual(np.sum(student_version2), 4)
        self.assertTrue(np.allclose(student_version2[0], np.array([0, 1, 0, 1])))

        # Test 3
        self.assertEqual(student_version3.shape, (6, 6))
        self.assertEqual(np.min(student_version3), -1)
        self.assertEqual(np.max(student_version3), 1)
        self.assertEqual(np.sum(student_version3), 0)
        self.assertTrue(np.allclose(student_version3[0], np.array([1, 1, 1, 1, 1, 1])))

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_transposing(self):
        """ Test transposing arrays. """
        student_version1 = numpy_basics.handle_numpy_arrays_06(self.transpose_array1)
        student_version2 = numpy_basics.handle_numpy_arrays_06(self.transpose_array2)
        student_version3 = numpy_basics.handle_numpy_arrays_06(self.transpose_array3)

        # Test 1
        self.assertEqual(student_version1.shape, (6,))
        self.assertEqual(np.min(student_version1), 0)
        self.assertEqual(np.max(student_version1), 5)
        self.assertEqual(np.sum(student_version1), 15)
        self.assertTrue(np.allclose(student_version1[0], 0))

        # Test 2
        self.assertEqual(student_version2.shape, (2, 2))
        self.assertEqual(np.min(student_version2), 0)
        self.assertEqual(np.max(student_version2), 1)
        self.assertEqual(np.sum(student_version2), 2)
        self.assertTrue(np.allclose(student_version2[0], np.array([0, 1])))

        # Test 3
        self.assertEqual(student_version3.shape, (3, 3))
        self.assertEqual(np.min(student_version3), -1)
        self.assertEqual(np.max(student_version3), 1)
        self.assertEqual(np.sum(student_version3), 0)
        self.assertTrue(np.allclose(student_version3[0], np.array([1, 0, -1])))

        # Return score if all assertions pass
        return 0.25

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_reshape()
        self._test_vstack()
        self._test_hstack()
        self._test_concatenation()
        self._test_repeating()
        self._test_transposing()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 1.5]")


class TestArrayCalculations(unittest.TestCase):
    """
    The class contains all test cases for task 1.4 - Calculating with Arrays.
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 1.4 - Calculating with Arrays...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the tests
        self.array11 = np.array([1, 2, 3])
        self.array12 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        self.array13 = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]], [[5, 5], [6, 6]]])

        self.array21 = np.array([6, 7, 8])
        self.array22 = np.array([[5, 5, 5], [1, 2, 3], [18, 19, 20]])
        self.array23 = np.array([[[12, 12], [11, 11]], [[10, 10], [9, 9]], [[8, 8], [7, 7]]])

        # NumPy 2.0 removed the 2-dimensional case of np.cross (deprecated since 1.25),
        # so the cross-product test cannot use array13/array23, whose innermost vectors
        # have two components. These are the same arrays with three components each.
        # The construction is unchanged: both vectors of a pair have equal components,
        # so every cross product is the zero vector and the total is still 0.
        self.array14 = np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]],
                                 [[5, 5, 5], [6, 6, 6]]])
        self.array24 = np.array([[[12, 12, 12], [11, 11, 11]], [[10, 10, 10], [9, 9, 9]],
                                 [[8, 8, 8], [7, 7, 7]]])

        self.sum_axis1 = 0
        self.sum_axis2 = 1
        self.sum_axis3 = 2

        self.threshold1 = 3
        self.threshold2 = 0
        self.threshold3 = 4

    @TestManager.register_test
    def _test_adding(self):
        """ Test adding arrays element-wise. """
        student_version1 = numpy_basics.calculations_with_numpy_arrays_01([self.array11, self.array21])
        student_version2 = numpy_basics.calculations_with_numpy_arrays_01([self.array12, self.array22])
        student_version3 = numpy_basics.calculations_with_numpy_arrays_01([self.array13, self.array23])

        # Test 1
        self.assertEqual(student_version1.shape, (3,))
        self.assertEqual(np.sum(student_version1), 27)

        # Test 2
        self.assertEqual(student_version2.shape, (3, 3))
        self.assertEqual(np.sum(student_version2), 78)

        # Test 3
        self.assertEqual(student_version3.shape, (3, 2, 2))
        self.assertEqual(np.sum(student_version3), 156)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_dot_product(self):
        """ Test dot product of two arrays. """
        student_version1 = numpy_basics.calculations_with_numpy_arrays_02([self.array11, self.array21])
        student_version2 = numpy_basics.calculations_with_numpy_arrays_02([self.array12, self.array22])
        student_version3 = numpy_basics.calculations_with_numpy_arrays_02([self.array13, self.array23])

        # Test 1
        self.assertEqual(student_version1.shape, ())
        self.assertEqual(np.sum(student_version1), 44)

        # Test 2
        self.assertEqual(student_version2.shape, (3, 3))
        self.assertEqual(np.sum(student_version2), 0)

        # Test 3
        self.assertEqual(student_version3.shape, (3, 2, 3, 2))
        self.assertEqual(np.sum(student_version3), 2394)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_multiply(self):
        """ Test multiplying arrays elment-wise. """
        student_version1 = numpy_basics.calculations_with_numpy_arrays_03([self.array11, self.array21])
        student_version2 = numpy_basics.calculations_with_numpy_arrays_03([self.array12, self.array22])
        student_version3 = numpy_basics.calculations_with_numpy_arrays_03([self.array13, self.array23])

        # Test 1
        self.assertEqual(student_version1.shape, (3, ))
        self.assertEqual(np.sum(student_version1), 44)

        # Test 2
        self.assertEqual(student_version2.shape, (3, 3))
        self.assertEqual(np.sum(student_version2), -42)

        # Test 3
        self.assertEqual(student_version3.shape, (3, 2, 2))
        self.assertEqual(np.sum(student_version3), 364)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_sum(self):
        """ Test summing arrays along a specified axis. """
        student_version1 = numpy_basics.calculations_with_numpy_arrays_04([self.array11, self.sum_axis1])
        student_version2 = numpy_basics.calculations_with_numpy_arrays_04([self.array12, self.sum_axis2])
        student_version3 = numpy_basics.calculations_with_numpy_arrays_04([self.array13, self.sum_axis3])

        # Test 1
        self.assertEqual(student_version1.shape, ())
        self.assertEqual(np.sum(student_version1), 6)

        # Test 2
        self.assertEqual(student_version2.shape, (3, ))
        self.assertEqual(np.sum(student_version2), 0)

        # Test 3
        self.assertEqual(student_version3.shape, (3, 2))
        self.assertEqual(np.sum(student_version3), 42)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_grad(self):
        """ Test gradient of array. """
        student_version1 = numpy_basics.calculations_with_numpy_arrays_05(self.array11)
        student_version2 = numpy_basics.calculations_with_numpy_arrays_05(self.array12)
        student_version3 = numpy_basics.calculations_with_numpy_arrays_05(self.array13)

        # Test 1
        self.assertEqual(student_version1.shape, (3,))
        self.assertEqual(np.sum(student_version1), 3.0)

        # Test 2
        self.assertEqual(student_version2[0].shape, (3, 3))
        self.assertEqual(np.sum(student_version2), -9.0)

        # Test 3
        self.assertEqual(student_version3[0].shape, (3, 2, 2))
        self.assertEqual(np.sum(student_version3), 36.0)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_cross(self):
        """ Test cross product of arrays. """
        student_version1 = numpy_basics.calculations_with_numpy_arrays_06([self.array11, self.array21])
        student_version2 = numpy_basics.calculations_with_numpy_arrays_06([self.array12, self.array22])
        student_version3 = numpy_basics.calculations_with_numpy_arrays_06([self.array14, self.array24])

        # Test 1
        self.assertEqual(student_version1.shape, (3,))
        self.assertEqual(np.sum(student_version1), 0)

        # Test 2
        self.assertEqual(student_version2.shape, (3, 3))
        self.assertEqual(np.sum(student_version2), 0)

        # Test 3
        self.assertEqual(student_version3.shape, (3, 2, 3))
        self.assertEqual(np.sum(student_version3), 0)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_square_root(self):
        """ Test element-wise square root of array. """
        student_version1 = numpy_basics.calculations_with_numpy_arrays_07(self.array11)
        student_version2 = numpy_basics.calculations_with_numpy_arrays_07(self.array22)
        student_version3 = numpy_basics.calculations_with_numpy_arrays_07(self.array13)

        # Test 1
        self.assertEqual(student_version1.shape, (3,))
        self.assertAlmostEqual(float(np.sum(student_version1)), 4.146264369941973, None, "Values are not almost equal!",
                               0.0001)

        # Test 2
        self.assertEqual(student_version2.shape, (3, 3))
        self.assertAlmostEqual(float(np.sum(student_version2)), 23.92814388810088, None, "Values are not almost equal!",
                               0.0001)

        # Test 3
        self.assertEqual(student_version3.shape, (3, 2, 2))
        self.assertAlmostEqual(float(np.sum(student_version3)), 21.663644180449882, None, "Values are not almost equal!",
                               0.0001)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_masking(self):
        """ Test masking of array. """
        student_version1 = numpy_basics.calculations_with_numpy_arrays_08([self.array11, self.threshold1])
        student_version2 = numpy_basics.calculations_with_numpy_arrays_08([self.array12, self.threshold2])
        student_version3 = numpy_basics.calculations_with_numpy_arrays_08([self.array13, self.threshold3])

        # Test 1
        self.assertEqual(student_version1.shape, (2,))
        self.assertEqual(np.sum(student_version1), 3)

        # Test 2
        self.assertEqual(student_version2.shape, (3,))
        self.assertEqual(np.sum(student_version2), -3)

        # Test 3
        self.assertEqual(student_version3.shape, (6, ))
        self.assertEqual(np.sum(student_version3), 12)

        # Return score if all assertions pass
        return 0.25

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_adding()
        self._test_dot_product()
        self._test_multiply()
        self._test_sum()
        self._test_grad()
        self._test_cross()
        self._test_square_root()
        self._test_masking()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 2.0]")


class TestArrayStatistics(unittest.TestCase):
    """
    The class contains all test cases for task 1.5 - Statistics with Arrays.
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 1.5 - Statistics with Arrays...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the tests
        self.array1 = np.array([1, 2, 3])
        self.array2 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        self.array3 = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]], [[5, 5], [6, 6]]])

        self.axis1 = None
        self.axis2 = 1
        self.axis3 = 2

    @TestManager.register_test
    def _test_min_value(self):
        """ Test extracting the minimum value of an array. """
        student_version1 = numpy_basics.statistics_with_numpy_arrays_01(self.array1, self.axis1)
        student_version2 = numpy_basics.statistics_with_numpy_arrays_01(self.array2, self.axis2)
        student_version3 = numpy_basics.statistics_with_numpy_arrays_01(self.array3, self.axis3)

        # Test 1
        self.assertEqual(student_version1.shape, ())
        self.assertEqual(np.sum(student_version1), 1)

        # Test 2
        self.assertEqual(student_version2.shape, (3,))
        self.assertEqual(np.sum(student_version2), 0)

        # Test 3
        self.assertEqual(student_version3.shape, (3, 2))
        self.assertEqual(np.sum(student_version3), 21)

        # Return score if all assertions pass
        return 0.175

    @TestManager.register_test
    def _test_max_value(self):
        """ Test extracting the maximum value of an array. """
        student_version1 = numpy_basics.statistics_with_numpy_arrays_02(self.array1, self.axis1)
        student_version2 = numpy_basics.statistics_with_numpy_arrays_02(self.array2, self.axis2)
        student_version3 = numpy_basics.statistics_with_numpy_arrays_02(self.array3, self.axis3)

        # Test 1
        self.assertEqual(student_version1.shape, ())
        self.assertEqual(np.sum(student_version1), 3)

        # Test 2
        self.assertEqual(student_version2.shape, (3,))
        self.assertEqual(np.sum(student_version2), 0)

        # Test 3
        self.assertEqual(student_version3.shape, (3, 2))
        self.assertEqual(np.sum(student_version3), 21)

        # Return score if all assertions pass
        return 0.175

    @TestManager.register_test
    def _test_min_index(self):
        """ Test extracting the index of the minimum value of an array. """
        student_version1 = numpy_basics.statistics_with_numpy_arrays_03(self.array1, self.axis1)
        student_version2 = numpy_basics.statistics_with_numpy_arrays_03(self.array2, self.axis2)
        student_version3 = numpy_basics.statistics_with_numpy_arrays_03(self.array3, self.axis3)

        # Test 1
        self.assertEqual(student_version1.shape, ())
        self.assertEqual(np.sum(student_version1), 0)

        # Test 2
        self.assertEqual(student_version2.shape, (3,))
        self.assertEqual(np.sum(student_version2), 0)

        # Test 3
        self.assertEqual(student_version3.shape, (3, 2))
        self.assertEqual(np.sum(student_version3), 0)

        # Return score if all assertions pass
        return 0.175

    @TestManager.register_test
    def _test_max_index(self):
        """ Test extracting the index of the maximum value of an array. """
        student_version1 = numpy_basics.statistics_with_numpy_arrays_04(self.array1, self.axis1)
        student_version2 = numpy_basics.statistics_with_numpy_arrays_04(self.array2, self.axis2)
        student_version3 = numpy_basics.statistics_with_numpy_arrays_04(self.array3, self.axis3)

        # Test 1
        self.assertEqual(student_version1.shape, ())
        self.assertEqual(np.sum(student_version1), 2)

        # Test 2
        self.assertEqual(student_version2.shape, (3,))
        self.assertEqual(np.sum(student_version2), 0)

        # Test 3
        self.assertEqual(student_version3.shape, (3, 2))
        self.assertEqual(np.sum(student_version3), 0)

        # Return score if all assertions pass
        return 0.175

    @TestManager.register_test
    def _test_mean(self):
        """ Test calculating the mean of an array along a given axis. """
        student_version1 = numpy_basics.statistics_with_numpy_arrays_05(self.array1, self.axis1)
        student_version2 = numpy_basics.statistics_with_numpy_arrays_05(self.array2, self.axis2)
        student_version3 = numpy_basics.statistics_with_numpy_arrays_05(self.array3, self.axis3)

        # Test 1
        self.assertEqual(student_version1.shape, ())
        self.assertEqual(np.sum(student_version1), 2.0)

        # Test 2
        self.assertEqual(student_version2.shape, (3,))
        self.assertEqual(np.sum(student_version2), 0)

        # Test 3
        self.assertEqual(student_version3.shape, (3, 2))
        self.assertEqual(np.sum(student_version3), 21)

        # Return score if all assertions pass
        return 0.15

    @TestManager.register_test
    def _test_std(self):
        """ Test calculating the standard deviation of an array along a given axis. """
        student_version1 = numpy_basics.statistics_with_numpy_arrays_06(self.array1, self.axis1)
        student_version2 = numpy_basics.statistics_with_numpy_arrays_06(self.array2, self.axis2)
        student_version3 = numpy_basics.statistics_with_numpy_arrays_06(self.array3, self.axis3)

        # Test 1
        self.assertEqual(student_version1.shape, ())
        self.assertAlmostEqual(float(np.sum(student_version1)), 0.8164, None, "Values are not almost equal!", 0.0001)

        # Test 2
        self.assertEqual(student_version2.shape, (3,))
        self.assertEqual(np.sum(student_version2), 0)

        # Test 3
        self.assertEqual(student_version3.shape, (3, 2))
        self.assertEqual(np.sum(student_version3), 0)

        # Return score if all assertions pass
        return 0.15

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_min_value()
        self._test_max_value()
        self._test_min_index()
        self._test_max_index()
        self._test_mean()
        self._test_std()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 1.0]")


class TestArraySlicing1D(unittest.TestCase):
    """
    The class contains the first test cases for task 1.6 - Array Slicing 1D.
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 1.6 - Array Slicing 1D...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the tests
        self.array1 = None
        self.array2 = np.arange(1, 11)
        self.array3 = np.array([1, 5, 6, 9, 13, 4, -5, 17, -3, 2, 3, 7, -5])

    @TestManager.register_test
    def _test_first_element(self):
        """ Test extracting the first element of an array. """
        student_version1 = numpy_basics.slicing_of_numpy_arrays_1d_01(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_1d_01(self.array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_1d_01(self.array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertEqual(student_version2.shape, ())
        self.assertEqual(np.sum(student_version2), 1)

        # Test 3
        self.assertEqual(student_version3.shape, ())
        self.assertEqual(np.sum(student_version3), 1)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_last_element(self):
        """ Test extracting the last element of an array. """
        student_version1 = numpy_basics.slicing_of_numpy_arrays_1d_02(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_1d_02(self.array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_1d_02(self.array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertEqual(student_version2.shape, ())
        self.assertEqual(np.sum(student_version2), 10)

        # Test 3
        self.assertEqual(student_version3.shape, ())
        self.assertEqual(np.sum(student_version3), -5)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_first_half(self):
        """ Test extracting the first half of an array. """
        student_version1 = numpy_basics.slicing_of_numpy_arrays_1d_03(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_1d_03(self.array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_1d_03(self.array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertIsInstance(student_version2, np.ndarray)
        self.assertEqual(student_version2.shape, (5,))
        self.assertEqual(np.sum(student_version2), 15)

        # Test 3
        self.assertIsInstance(student_version3, np.ndarray)
        self.assertEqual(student_version3.shape, (6,))
        self.assertEqual(np.sum(student_version3), 38)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_second_half(self):
        """ Test extracting the second half of an array. """
        student_version1 = numpy_basics.slicing_of_numpy_arrays_1d_04(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_1d_04(self.array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_1d_04(self.array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertIsInstance(student_version2, np.ndarray)
        self.assertEqual(student_version2.shape, (5,))
        self.assertEqual(np.sum(student_version2), 40)

        # Test 3
        self.assertIsInstance(student_version3, np.ndarray)
        self.assertEqual(student_version3.shape, (7,))
        self.assertEqual(np.sum(student_version3), 16)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_elements_at_odd_positions(self):
        """ Test extracting all elements at an odd position within an array. """
        student_version1 = numpy_basics.slicing_of_numpy_arrays_1d_05(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_1d_05(self.array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_1d_05(self.array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertIsInstance(student_version2, np.ndarray)
        self.assertEqual(student_version2.shape, (5,))
        self.assertEqual(np.sum(student_version2), 25)

        # Test 3
        self.assertIsInstance(student_version3, np.ndarray)
        self.assertEqual(student_version3.shape, (7,))
        self.assertEqual(np.sum(student_version3), 10)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_elements_at_even_positions(self):
        """ Test extracting all elements at an even position within an array. """
        student_version1 = numpy_basics.slicing_of_numpy_arrays_1d_06(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_1d_06(self.array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_1d_06(self.array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertIsInstance(student_version2, np.ndarray)
        self.assertEqual(student_version2.shape, (5,))
        self.assertEqual(np.sum(student_version2), 30)

        # Test 3
        self.assertIsInstance(student_version3, np.ndarray)
        self.assertEqual(student_version3.shape, (6,))
        self.assertEqual(np.sum(student_version3), 44)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_special_pattern(self):
        """ Test extracting all elements as presented in task 7. """
        student_version1 = numpy_basics.slicing_of_numpy_arrays_1d_07(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_1d_07(self.array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_1d_07(self.array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertIsInstance(student_version2, np.ndarray)
        self.assertEqual(student_version2.shape, (4,))
        self.assertEqual(np.sum(student_version2), 22)

        # Test 3
        self.assertIsInstance(student_version3, np.ndarray)
        self.assertEqual(student_version3.shape, (5,))
        self.assertEqual(np.sum(student_version3), 2)

        # Return score if all assertions pass
        return 0.5

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_first_element()
        self._test_last_element()
        self._test_first_half()
        self._test_second_half()
        self._test_elements_at_odd_positions()
        self._test_elements_at_even_positions()
        self._test_special_pattern()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 2.0]")


class TestArraySlicing2D(unittest.TestCase):
    """
    The class contains the remainder of test cases for task 1.7 - Array Slicing 2D.
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 1.7 - Array Slicing 2D...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the tests
        self.array1 = None
        self.array2 = np.arange(1, 17).reshape((4, 4))
        self.array3 = np.array([[-1, 15, 4, 6], [5, -4, 17, 0], [3, 3, -8, 6], [7, -1, -10, 2]])

    @TestManager.register_test
    def _test_first_element(self):
        """ Test extracting the first element of an array. """
        student_version1 = numpy_basics.slicing_of_numpy_arrays_2d_01(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_2d_01(self.array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_2d_01(self.array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertEqual(student_version2.shape, ())
        self.assertEqual(np.sum(student_version2), 1)

        # Test 3
        self.assertEqual(student_version3.shape, ())
        self.assertEqual(np.sum(student_version3), -1)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_last_element(self):
        """ Test extracting the last element of an array. """
        student_version1 = numpy_basics.slicing_of_numpy_arrays_2d_02(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_2d_02(self.array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_2d_02(self.array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertEqual(student_version2.shape, ())
        self.assertEqual(np.sum(student_version2), 16)

        # Test 3
        self.assertEqual(student_version3.shape, ())
        self.assertEqual(np.sum(student_version3), 2)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_first_half_row_wise(self):
        """ Test extracting the row-wise first half of elements of an array. """
        array2 = np.arange(1, 26).reshape((5, 5))
        array3 = np.array([[-1, 15, 4, 6, 1], [5, -4, 17, 0, 1], [3, 3, -8, 6, 1], [7, -1, -10, 2, 1], [1, 1, 1, 1, 1]])

        student_version1 = numpy_basics.slicing_of_numpy_arrays_2d_03(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_2d_03(array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_2d_03(array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertIsInstance(student_version2, np.ndarray)
        self.assertEqual(student_version2.shape, (2, 5))
        self.assertEqual(np.sum(student_version2), 55)

        # Test 3
        self.assertIsInstance(student_version3, np.ndarray)
        self.assertEqual(student_version3.shape, (2, 5))
        self.assertEqual(np.sum(student_version3), 44)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_second_half_row_wise(self):
        """ Test extracting the row-wise second half of elements of an array. """
        array2 = np.arange(1, 26).reshape((5, 5))
        array3 = np.array([[-1, 15, 4, 6, 1], [5, -4, 17, 0, 1], [3, 3, -8, 6, 1], [7, -1, -10, 2, 1], [1, 1, 1, 1, 1]])

        student_version1 = numpy_basics.slicing_of_numpy_arrays_2d_04(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_2d_04(array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_2d_04(array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertIsInstance(student_version2, np.ndarray)
        self.assertEqual(student_version2.shape, (3, 5))
        self.assertEqual(np.sum(student_version2), 270)

        # Test 3
        self.assertIsInstance(student_version3, np.ndarray)
        self.assertEqual(student_version3.shape, (3, 5))
        self.assertEqual(np.sum(student_version3), 9)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_first_half_column_wise(self):
        """ Test extracting the column-wise first half of elements of an array. """
        array2 = np.arange(1, 26).reshape((5, 5))
        array3 = np.array([[-1, 15, 4, 6, 1], [5, -4, 17, 0, 1], [3, 3, -8, 6, 1], [7, -1, -10, 2, 1], [1, 1, 1, 1, 1]])

        student_version1 = numpy_basics.slicing_of_numpy_arrays_2d_05(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_2d_05(array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_2d_05(array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertIsInstance(student_version2, np.ndarray)
        self.assertEqual(student_version2.shape, (5, 2))
        self.assertEqual(np.sum(student_version2), 115)

        # Test 3
        self.assertIsInstance(student_version3, np.ndarray)
        self.assertEqual(student_version3.shape, (5, 2))
        self.assertEqual(np.sum(student_version3), 29)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_second_half_column_wise(self):
        """ Test extracting the column-wise second half of elements of an array. """
        array2 = np.arange(1, 26).reshape((5, 5))
        array3 = np.array([[-1, 15, 4, 6, 1], [5, -4, 17, 0, 1], [3, 3, -8, 6, 1], [7, -1, -10, 2, 1], [1, 1, 1, 1, 1]])

        student_version1 = numpy_basics.slicing_of_numpy_arrays_2d_06(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_2d_06(array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_2d_06(array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertIsInstance(student_version2, np.ndarray)
        self.assertEqual(student_version2.shape, (5, 3))
        self.assertEqual(np.sum(student_version2), 210)

        # Test 3
        self.assertIsInstance(student_version3, np.ndarray)
        self.assertEqual(student_version3.shape, (5, 3))
        self.assertEqual(np.sum(student_version3), 24)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_odd_columns(self):
        """ Test extracting every element at an odd column in row-wise order of an array. """
        student_version1 = numpy_basics.slicing_of_numpy_arrays_2d_07(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_2d_07(self.array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_2d_07(self.array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertIsInstance(student_version2, np.ndarray)
        self.assertEqual(student_version2.shape, (4, 2))
        self.assertEqual(np.sum(student_version2), 64)

        # Test 3
        self.assertIsInstance(student_version3, np.ndarray)
        self.assertEqual(student_version3.shape, (4, 2))
        self.assertEqual(np.sum(student_version3), 17)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_even_columns(self):
        """ Test extracting every element at an even column in row-wise order of an array. """
        student_version1 = numpy_basics.slicing_of_numpy_arrays_2d_08(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_2d_08(self.array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_2d_08(self.array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertIsInstance(student_version2, np.ndarray)
        self.assertEqual(student_version2.shape, (4, 2))
        self.assertEqual(np.sum(student_version2), 72)

        # Test 3
        self.assertIsInstance(student_version3, np.ndarray)
        self.assertEqual(student_version3.shape, (4, 2))
        self.assertEqual(np.sum(student_version3), 27)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_diagonal(self):
        """ Test extracting every element on the diagonal of an array. """
        student_version1 = numpy_basics.slicing_of_numpy_arrays_2d_09(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_2d_09(self.array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_2d_09(self.array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertIsInstance(student_version2, np.ndarray)
        self.assertEqual(student_version2.shape, (4, ))
        self.assertEqual(np.sum(student_version2), 34)

        # Test 3
        self.assertIsInstance(student_version3, np.ndarray)
        self.assertEqual(student_version3.shape, (4, ))
        self.assertEqual(np.sum(student_version3), -11)

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_diagonal_reverse(self):
        """ Test extracting every element on the reverse diagonal of an array. """
        student_version1 = numpy_basics.slicing_of_numpy_arrays_2d_10(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_2d_10(self.array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_2d_10(self.array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertIsInstance(student_version2, np.ndarray)
        self.assertEqual(student_version2.shape, (4, ))
        self.assertEqual(np.sum(student_version2), 34)

        # Test 3
        self.assertIsInstance(student_version3, np.ndarray)
        self.assertEqual(student_version3.shape, (4, ))
        self.assertEqual(np.sum(student_version3), 33)

        # Return score if all assertions pass
        return 0.25

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_first_element()
        self._test_last_element()
        self._test_first_half_row_wise()
        self._test_second_half_row_wise()
        self._test_first_half_column_wise()
        self._test_second_half_column_wise()
        self._test_odd_columns()
        self._test_even_columns()
        self._test_diagonal()
        self._test_diagonal_reverse()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 2.5]")


class TestArrayAdvancedOperations(unittest.TestCase):
    """
    The class contains all test cases for task 1.8 - More Advanced Operations with Arrays.
    """

    def setUp(self):
        """ Initialize the tests. """
        # Create the tests
        print("Running tests for Task 1.8 - More Advanced Operations with Arrays...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

    @TestManager.register_test
    def _test_sort(self):
        """ Test sorting arrays by a given column. """
        array1 = np.array([[4, 3, 1], [9, 2, 5], [0, 7, 8]])
        array2 = np.array([[4, 7, 1], [5, 2, 9], [3, 8, 6], [0, 5, 3]])
        array3 = np.array([[7, 4, 1, 5], [3, 9, 5, 0], [2, 6, 8, 4], [0, 3, 5, 1]])

        student_version1 = numpy_basics.sort_array(array1, 0)
        student_version2 = numpy_basics.sort_array(array2, 1)
        student_version3 = numpy_basics.sort_array(array3, 2)

        expected_version1 = np.array([[0, 7, 8], [4, 3, 1], [9, 2, 5]])
        expected_version2 = np.array([[5, 2, 9],[0, 5, 3],[4, 7, 1],[3, 8, 6]])
        expected_version3 = np.array([[7, 4, 1, 5],[3, 9, 5, 0],[0, 3, 5, 1],[2, 6, 8, 4]])

        # Test 1 - sorting of arrays by column 0
        self.assertTrue(np.allclose(expected_version1, student_version1))

        # Test 2 - sorting of arrays by column 1
        self.assertTrue(np.allclose(expected_version2, student_version2))

        # Test 3 - sorting of arrays by column 2
        self.assertTrue(np.allclose(expected_version3, student_version3))

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_broadcasting(self):
        """ Test multiplying arrays of different shapes using broadcasting. """
        array_1D_1 = np.array([1, 2, 3])
        array_1D_2 = np.array([4, 5])
        array_nD_1 = np.random.rand(2, 3, 4)
        array_nD_2 = np.random.rand(2, 3)

        student_version_1D, _ = numpy_basics.multiply_arrays_with_broadcasting(array_1D_1, array_1D_2, array_nD_1, array_nD_2)
        _, student_version_nD = numpy_basics.multiply_arrays_with_broadcasting(array_1D_1, array_1D_2, array_nD_1, array_nD_2)

        # Test 1
        self.assertEqual(student_version_1D.shape, (3, 2))
        self.assertEqual(student_version_1D[0, 0], array_1D_1[0] * array_1D_2[0])
        self.assertEqual(student_version_1D[2, 1], array_1D_1[2] * array_1D_2[1])

        # Test 2
        self.assertEqual(student_version_nD.shape, array_nD_1.shape)
        self.assertEqual(student_version_nD[0, 0, 0], array_nD_1[0, 0, 0] * array_nD_2[0, 0])
        self.assertEqual(student_version_nD[1, 2, 1], array_nD_1[1, 2, 1] * array_nD_2[1, 2])

        # Return score if all assertions pass
        return 0.5

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_sort()
        self._test_broadcasting()

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



