#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
This script was written as a test module for the 'numpy_applied.ipynb'-jupyter notebook.
That notebook tests your knowledge about the basic numpy functions by letting you apply them to real world tasks.
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
import matplotlib
matplotlib.use('Agg') #Set matplotlib to use a non-GUI backend to avoid errors when running tests in a non-interactive terminal

import unittest
from unittest import mock
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
numpy_applied = extract_module("numpy_applied")

from code.tests.utils import TestManager

# Define the base data directory path
data_dir = os.path.abspath(os.path.join(root_dir, 'data', 'numpy_applied'))


class TestArrayBorder(unittest.TestCase):
    """
    The class contains all test cases for task "1 - Simple Array Border Pattern".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 1 - Simple Array Border Pattern...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        self.shape1 = (3, 4)
        self.shape2 = (5, 5)

    @TestManager.register_test
    def _test_pattern_1(self):
        # Create the student version of the pattern
        student_version1 = numpy_applied.create_array_border(self.shape1)

        # Load the reference image
        expected_shape1 = (5, 6)
        expected_version1 = np.load(os.path.join(data_dir, 'reference_bordered_array1.npz'))["data"]

        # Compare the student version to the expected output
        self.assertEqual(student_version1.shape, expected_shape1)
        self.assertEqual(student_version1[0, 0], 0, "Pattern is not correct!")
        self.assertEqual(student_version1[1, 1], 1, "Pattern is not correct!")
        self.assertEqual(np.min(student_version1), 0, "Pattern is not correct!")
        self.assertEqual(np.max(student_version1), 1, "Pattern is not correct!")
        self.assertEqual(np.sum(student_version1), 10.0, "Pattern is not correct!")
        self.assertTrue(np.allclose(student_version1, expected_version1), "Pattern is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_pattern_2(self):
        # Create the student version of the pattern
        student_version2 = numpy_applied.create_array_border(self.shape2)

        # Load the reference image
        expected_shape2 = (7, 7)
        expected_version2 = np.load(os.path.join(data_dir, 'reference_bordered_array2.npz'))["data"]

        # Compare the student version to the expected output
        self.assertEqual(student_version2.shape, expected_shape2)
        self.assertEqual(student_version2[0, 0], 0, "Pattern is not correct!")
        self.assertEqual(student_version2[1, 1], 1, "Pattern is not correct!")
        self.assertEqual(np.min(student_version2), 0, "Pattern is not correct!")
        self.assertEqual(np.max(student_version2), 1, "Pattern is not correct!")
        self.assertEqual(np.sum(student_version2), 16.0, "Pattern is not correct!")
        self.assertTrue(np.allclose(student_version2, expected_version2), "Pattern is not correct!")

        # Return score if all assertions pass
        return 0.5

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_pattern_1()
        self._test_pattern_2()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 1.0]")


class TestCheckerBoard(unittest.TestCase):
    """
    The class contains all test cases for task "2 - Checkerboard Pattern".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 2 - Checkerboard Pattern...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Load the reference images
        self.checker_expected1 = np.load(os.path.join(data_dir, 'reference_checker1.npz'))["data"]
        self.checker_expected2 = np.load(os.path.join(data_dir, 'reference_checker2.npz'))["data"]
        self.checker_expected3 = np.load(os.path.join(data_dir, 'reference_checker3.npz'))["data"]

        # Initialize the used sizes and tile sizes for the reference images
        self.size1, self.tile_size1 = 10, 1
        self.size2, self.tile_size2 = 100, 25
        self.size3, self.tile_size3 = 200, 25

    @TestManager.register_test
    def _test_attributes(self):
        """
        Test whether the students implemented an attribute 'pattern'.
        """
        # Create the student version of the pattern
        student_checker = numpy_applied.CheckerBoard(self.size1, self.tile_size1)

        # Check if the 'pattern' attribute exists
        self.assertTrue(
            hasattr(student_checker, 'pattern'),
            "Class CheckerBoard does not contain an attribute 'pattern'!"
        )

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_relation_size_to_tilesize(self):
        """
        Test whether the students pattern omit non-matching size and tile size parameter.
        """
        # create the student version of the pattern with non-matching size and tile size parameter
        student_checker = numpy_applied.CheckerBoard(10, 3)

        try:
            student_checker.draw()
            student_version = student_checker.pattern

            # Check whether the pattern does contain values
            self.assertIsNone(student_version, "Pattern was created despite having a non-suitable size and tile size "
                                               "relation!")

        except:
            # Extract the pattern
            student_version = student_checker.pattern

            # Check whether the pattern does contain values
            self.assertIsNone(student_version, "Pattern was created despite having a non-suitable size and tile size "
                                               "relation!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_pattern1(self):
        """
        Test whether the students pattern matches the expected one.
        """
        # Create the student version of the pattern
        student_checker = numpy_applied.CheckerBoard(self.size1, self.tile_size1)
        student_checker.draw()
        student_version = student_checker.pattern

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.checker_expected1.shape,
                         "Shape of pattern is not correct!")
        self.assertEqual(student_version[0, 0], 0, "Pattern is not correct!")
        self.assertEqual(np.min(student_version), 0, "Pattern is not correct!")
        self.assertEqual(np.max(student_version), 1, "Pattern is not correct!")
        self.assertEqual(float(np.sum(student_version)), float(np.sum(self.checker_expected1)),
                         "Pattern is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_pattern2(self):
        """
        Test whether the students pattern matches the expected one with different parameters.
        """
        # create the student version of the pattern
        student_checker = numpy_applied.CheckerBoard(self.size2, self.tile_size2)
        student_checker.draw()
        student_version = student_checker.pattern

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.checker_expected2.shape,
                         "Shape of pattern is not correct!")
        self.assertEqual(student_version[0, 0], 0, "Pattern is not correct!")
        self.assertEqual(np.min(student_version), 0, "Pattern is not correct!")
        self.assertEqual(np.max(student_version), 1, "Pattern is not correct!")
        self.assertEqual(float(np.sum(student_version)), float(np.sum(self.checker_expected2)),
                         "Pattern is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_pattern3(self):
        """
        Test whether the students pattern matches the expected one with different parameters.
        """
        # create the student version of the pattern
        student_checker = numpy_applied.CheckerBoard(self.size3, self.tile_size3)
        student_checker.draw()
        student_version = student_checker.pattern

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.checker_expected3.shape,
                         "Shape of pattern is not correct!")
        self.assertEqual(student_version[0, 0], 0, "Pattern is not correct!")
        self.assertEqual(np.min(student_version), 0, "Pattern is not correct!")
        self.assertEqual(np.max(student_version), 1, "Pattern is not correct!")
        self.assertEqual(float(np.sum(student_version)), float(np.sum(self.checker_expected3)),
                         "Pattern is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_visualize_pattern(self):
        """
        Test whether the student correctly implemented the `show` method.
        """
        checker = numpy_applied.CheckerBoard(size=self.size1, tile_size=self.tile_size1)

        # Directly assign a reference pattern for testing purposes without calling draw()
        checker.pattern = self.checker_expected1

        with mock.patch("matplotlib.pyplot.imshow") as mock_imshow, \
                mock.patch("matplotlib.pyplot.show") as mock_show:
            checker.show()

            mock_imshow.assert_called_once_with(checker.pattern, cmap='gray')
            mock_show.assert_called_once()

        # Return score if all assertions pass
        return 0.5

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_attributes()
        self._test_relation_size_to_tilesize()
        self._test_pattern1()
        self._test_pattern2()
        self._test_pattern3()
        self._test_visualize_pattern()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 3.0]")


class TestRectangle(unittest.TestCase):
    """
    The class contains all test cases for task "3.1 - Create the Rectangle Shape".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 3.1 - Create the Rectangle Shape...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Load the reference images
        self.rectangle_expected1 = np.load(os.path.join(data_dir, 'reference_rectangle1.npz'))["data"]
        self.rectangle_expected2 = np.load(os.path.join(data_dir, 'reference_rectangle2.npz'))["data"]
        self.rectangle_expected3 = np.load(os.path.join(data_dir, 'reference_rectangle3.npz'))["data"]

        # Initialize the used sizes and coordinates for the reference images
        self.size1, self.x1y1_1, self.x2y2_1 = 10, (0, 0), (10, 10)
        self.size2, self.x1y1_2, self.x2y2_2 = 100, (75, 40), (25, 60)
        self.size3, self.x1y1_3, self.x2y2_3 = 200, (20, 180), (180, 20)

    def _IoU(self, a1, a2):
        # Utility function returning the intersection over union value
        intersection = np.sum(a1 * a2)
        union = a1 + a2
        union = np.sum(union.astype(np.bool_))
        iou = intersection/union

        return iou

    @TestManager.register_test
    def _test_attributes(self):
        """
        Test whether the students implemented an attribute 'pattern'.
        """
        # Create the student version of the pattern
        student_rectangle = numpy_applied.Rectangle(self.size1, self.x1y1_1, self.x2y2_1)

        # Check if the 'pattern' attribute exists
        self.assertTrue(
            hasattr(student_rectangle, 'pattern'),
            "Class Rectangle does not contain an attribute 'pattern'!"
        )

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_pattern1(self):
        """
        Test whether the students pattern matches the expected one.
        """
        # Create the student version of the pattern
        student_rectangle = numpy_applied.Rectangle(self.size1, self.x1y1_1, self.x2y2_1)
        student_rectangle.draw()
        student_version = student_rectangle.pattern

        # Calculate the IOU
        iou = self._IoU(student_version, self.rectangle_expected1)

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.rectangle_expected1.shape,
                         "Shape of pattern is not correct!")
        self.assertEqual(student_version[0, 0], 1, "Pattern is not correct!")
        self.assertEqual(np.min(student_version), 1, "Pattern is not correct!")
        self.assertEqual(np.max(student_version), 1, "Pattern is not correct!")
        self.assertEqual(float(np.sum(student_version)), float(np.sum(self.rectangle_expected1)),
                         "Pattern is not correct!")
        self.assertAlmostEqual(iou, 1.0, 1, "Pattern is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_pattern2(self):
        """
        Test whether the students pattern matches the expected one with different parameters.
        """
        # Create the student version of the pattern
        student_rectangle = numpy_applied.Rectangle(self.size2, self.x1y1_2, self.x2y2_2)
        student_rectangle.draw()
        student_version = student_rectangle.pattern

        # Calculate the IOU
        iou = self._IoU(student_version, self.rectangle_expected2)

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.rectangle_expected2.shape,
                         "Shape of pattern is not correct!")
        self.assertEqual(student_version[0, 0], 0, "Pattern is not correct!")
        self.assertEqual(np.min(student_version), 0, "Pattern is not correct!")
        self.assertEqual(np.max(student_version), 1, "Pattern is not correct!")
        self.assertEqual(float(np.sum(student_version)), float(np.sum(self.rectangle_expected2)),
                         "Pattern is not correct!")
        self.assertAlmostEqual(iou, 1.0, 1, "Pattern is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_pattern3(self):
        """
        Test whether the students pattern matches the expected one with different parameters.
        """
        # Create the student version of the pattern
        student_rectangle = numpy_applied.Rectangle(self.size3, self.x1y1_3, self.x2y2_3)
        student_rectangle.draw()
        student_version = student_rectangle.pattern

        # Calculate the IOU
        iou = self._IoU(student_version, self.rectangle_expected3)

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.rectangle_expected3.shape,
                         "Shape of pattern is not correct!")
        self.assertEqual(student_version[0, 0], 0, "Pattern is not correct!")
        self.assertEqual(np.min(student_version), 0, "Pattern is not correct!")
        self.assertEqual(np.max(student_version), 1, "Pattern is not correct!")
        self.assertEqual(float(np.sum(student_version)), float(np.sum(self.rectangle_expected3)),
                         "Pattern is not correct!")
        self.assertAlmostEqual(iou, 1.0, 1, "Pattern is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_visualize_pattern(self):
        """
        Test whether the student correctly implemented the `show` method.
        """
        rectangle = numpy_applied.Rectangle(self.size1, self.x1y1_1, self.x2y2_1)

        # Directly assign a reference pattern for testing purposes without calling draw()
        rectangle.pattern = self.rectangle_expected1

        with mock.patch("matplotlib.pyplot.imshow") as mock_imshow, \
                mock.patch("matplotlib.pyplot.show") as mock_show:
            rectangle.show()

            mock_imshow.assert_called_once_with(rectangle.pattern, cmap='gray')
            mock_show.assert_called_once()

        # Return score if all assertions pass
        return 0.25

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_attributes()
        self._test_pattern1()
        self._test_pattern2()
        self._test_pattern3()
        self._test_visualize_pattern()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 2.0]")


class TestCircle(unittest.TestCase):
    """
    The class contains all test cases for task "3.2 - Create the Circle Shape".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 3.2 - Create the Circle Shape...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Load the reference images
        self.circle_expected1 = np.load(os.path.join(data_dir, 'reference_circle1.npz'))["data"]
        self.circle_expected2 = np.load(os.path.join(data_dir, 'reference_circle2.npz'))["data"]
        self.circle_expected3 = np.load(os.path.join(data_dir, 'reference_circle3.npz'))["data"]

        # Initialize the used sizes, radia and center for the reference images
        self.size1, self.radius1, self.center1 = 10, 0, (5, 2)
        self.size2, self.radius2, self.center2 = 100, 20, (0, 0)
        self.size3, self.radius3, self.center3 = 200, 50, (70, 120)

    def _IoU(self, a1, a2):
        # Utility function returning the intersection over union value
        intersection = np.sum(a1 * a2)
        union = a1 + a2
        union = np.sum(union.astype(np.bool_))
        iou = intersection/union

        return iou

    @TestManager.register_test
    def _test_attributes(self):
        """
        Test whether the students implemented an attribute 'pattern'.
        """
        # Create the student version of the pattern
        student_circle = numpy_applied.Circle(self.size1, self.radius1, self.center1)

        # Check if the 'pattern' attribute exists
        self.assertTrue(
            hasattr(student_circle, 'pattern'),
            "Class Circle does not contain an attribute 'pattern'!"
        )

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_pattern1(self):
        """
        Test whether the students pattern matches the expected one.
        """
        # Create the student version of the pattern
        student_circle = numpy_applied.Circle(self.size1, self.radius1, self.center1)
        student_circle.draw()
        student_version = student_circle.pattern

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.circle_expected1.shape,
                         "Shape of pattern is not correct!")
        self.assertEqual(student_version[0, 0], 0, "Pattern is not correct!")
        self.assertEqual(np.min(student_version), 0, "Pattern is not correct!")
        self.assertEqual(np.max(student_version), 0, "Pattern is not correct!")
        self.assertEqual(float(np.sum(student_version)), float(np.sum(self.circle_expected1)),
                         "Pattern is not correct!")

        # Return score if all assertions pass
        return 0.75

    @TestManager.register_test
    def _test_pattern2(self):
        """
        Test whether the students pattern matches the expected one with different parameters.
        """
        # Create the student version of the pattern
        student_circle = numpy_applied.Circle(self.size2, self.radius2, self.center2)
        student_circle.draw()
        student_version = student_circle.pattern

        # Calculate the IOU
        iou = self._IoU(student_version, self.circle_expected2)

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.circle_expected2.shape,
                         "Shape of pattern is not correct!")
        self.assertEqual(student_version[0, 0], 1, "Pattern is not correct!")
        self.assertEqual(np.min(student_version), 0, "Pattern is not correct!")
        self.assertEqual(np.max(student_version), 1, "Pattern is not correct!")
        self.assertEqual(float(np.sum(student_version)), float(np.sum(self.circle_expected2)),
                         "Pattern is not correct!")
        self.assertAlmostEqual(iou, 1.0, 1, "Pattern is not correct!")

        # Return score if all assertions pass
        return 0.75

    @TestManager.register_test
    def _test_pattern3(self):
        """
        Test whether the students pattern matches the expected one with different parameters.
        """
        # Create the student version of the pattern
        student_circle = numpy_applied.Circle(self.size3, self.radius3, self.center3)
        student_circle.draw()
        student_version = student_circle.pattern

        # Calculate the IOU
        iou = self._IoU(student_version, self.circle_expected3)

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.circle_expected3.shape,
                         "Shape of pattern is not correct!")
        self.assertEqual(student_version[0, 0], 0, "Pattern is not correct!")
        self.assertEqual(np.min(student_version), 0, "Pattern is not correct!")
        self.assertEqual(np.max(student_version), 1, "Pattern is not correct!")
        self.assertEqual(float(np.sum(student_version)), float(np.sum(self.circle_expected3)),
                         "Pattern is not correct!")
        self.assertAlmostEqual(iou, 1.0, 1, "Pattern is not correct!")

        # Return score if all assertions pass
        return 0.75

    @TestManager.register_test
    def _test_visualize_pattern(self):
        """
        Test whether the student correctly implemented the `show` method.
        """
        circle = numpy_applied.Circle(self.size1, self.radius1, self.center1)

        # Directly assign a reference pattern for testing purposes without calling draw()
        circle.pattern = self.circle_expected1

        with mock.patch("matplotlib.pyplot.imshow") as mock_imshow, \
                mock.patch("matplotlib.pyplot.show") as mock_show:
            circle.show()

            mock_imshow.assert_called_once_with(circle.pattern, cmap='gray')
            mock_show.assert_called_once()

        # Return score if all assertions pass
        return 0.5

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_attributes()
        self._test_pattern1()
        self._test_pattern2()
        self._test_pattern3()
        self._test_visualize_pattern()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 3.0]")


class TestShapes(unittest.TestCase):
    """
    The class contains all test cases for task "3.3 - Combine and Plot the Shapes".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 3.3 - Combine and Plot the Shapes...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Load the reference images
        self.shapes_expected1 = np.load(os.path.join(data_dir, 'reference_shapes1.npz'))["data"]
        self.shapes_expected2 = np.load(os.path.join(data_dir, 'reference_shapes2.npz'))["data"]
        self.shapes_expected3 = np.load(os.path.join(data_dir, 'reference_shapes3.npz'))["data"]

        # Initialize the used sizes, coordinates, radia and center for the reference images
        self.size1, self.x1y1_1, self.x2y2_1, self.radius1, self.center1 = 10, (0, 0), (10, 10), 0, (5, 2)
        self.size2, self.x1y1_2, self.x2y2_2, self.radius2, self.center2 = 100, (75, 40), (25, 60), 20, (0, 0)
        self.size3, self.x1y1_3, self.x2y2_3, self.radius3, self.center3 = 200, (20, 180), (180, 20), 50, (70, 120)

    def _IoU(self, a1, a2):
        # Utility function returning the intersection over union value
        intersection = np.sum(a1 * a2)
        union = a1 + a2
        union = np.sum(union.astype(np.bool_))
        iou = intersection/union

        return iou

    @TestManager.register_test
    def _test_pattern1(self):
        """
        Test whether the students pattern matches the expected one.
        """
        # Create the student version of the pattern
        student_version = numpy_applied.combine_and_plot_shapes(self.size1, self.x1y1_1, self.x2y2_1, self.radius1,
                                                                self.center1)

        # Calculate the IOU
        iou = self._IoU(student_version, self.shapes_expected1)

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.shapes_expected1.shape,
                         "Shape of pattern is not correct!")
        self.assertEqual(student_version[0, 0], 1, "Pattern is not correct!")
        self.assertEqual(np.min(student_version), 1, "Pattern is not correct!")
        self.assertEqual(np.max(student_version), 1, "Pattern is not correct!")
        self.assertEqual(float(np.sum(student_version)), float(np.sum(self.shapes_expected1)),
                         "Pattern is not correct!")
        self.assertAlmostEqual(iou, 1.0, 1, "Pattern is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_pattern2(self):
        """
        Test whether the students pattern matches the expected one with different parameters.
        """
        # Create the student version of the pattern
        student_version = numpy_applied.combine_and_plot_shapes(self.size2, self.x1y1_2, self.x2y2_2, self.radius2,
                                                                self.center2)

        # Calculate the IOU
        iou = self._IoU(student_version, self.shapes_expected2)

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.shapes_expected2.shape,
                         "Shape of pattern is not correct!")
        self.assertEqual(student_version[0, 0], 1, "Pattern is not correct!")
        self.assertEqual(np.min(student_version), 0, "Pattern is not correct!")
        self.assertEqual(np.max(student_version), 1, "Pattern is not correct!")
        self.assertEqual(float(np.sum(student_version)), float(np.sum(self.shapes_expected2)),
                         "Pattern is not correct!")
        self.assertAlmostEqual(iou, 1.0, 1, "Pattern is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_pattern3(self):
        """
        Test whether the students pattern matches the expected one with different parameters.
        """
        # Create the student version of the pattern
        student_version = numpy_applied.combine_and_plot_shapes(self.size3, self.x1y1_3, self.x2y2_3, self.radius3,
                                                                self.center3)

        # Calculate the IOU
        iou = self._IoU(student_version, self.shapes_expected3)

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.shapes_expected3.shape,
                         "Shape of pattern is not correct!")
        self.assertEqual(student_version[0, 0], 0, "Pattern is not correct!")
        self.assertEqual(np.min(student_version), 0, "Pattern is not correct!")
        self.assertEqual(np.max(student_version), 1, "Pattern is not correct!")
        self.assertEqual(float(np.sum(student_version)), float(np.sum(self.shapes_expected3)),
                         "Pattern is not correct!")
        self.assertAlmostEqual(iou, 1.0, 1, "Pattern is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_visualize_pattern(self):
        """
        Test whether the student correctly implemented the plotting.
        """
        with mock.patch("matplotlib.pyplot.imshow") as mock_imshow, \
                mock.patch("matplotlib.pyplot.show") as mock_show:

            numpy_applied.combine_and_plot_shapes(self.size1, self.x1y1_1, self.x2y2_1, self.radius1,
                                              self.center1)
            # Check if imshow was called with any array and the correct color map
            mock_imshow.assert_called_once()
            args, kwargs = mock_imshow.call_args
            # Check if 'cmap' keyword argument is set to 'gray'
            self.assertIn('cmap', kwargs, "cmap keyword argument is missing.")
            self.assertEqual(kwargs['cmap'], 'gray', "cmap is not set to 'gray'.")

            # Check if show() was called once to display the plot
            mock_show.assert_called_once()

        # Return score if all assertions pass
        return 0.5

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_pattern1()
        self._test_pattern2()
        self._test_pattern3()
        self._test_visualize_pattern()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 2.0]")


class TestSpectrum(unittest.TestCase):
    """
    The class contains all test cases for task "4 - Color Spectrum (Optional)".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 4 - Color Spectrum (Optional)...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Load the reference images
        self.spectrum_expected2 = np.load(os.path.join(data_dir, 'reference_spectrum2.npz'))["data"]
        self.spectrum_expected3 = np.load(os.path.join(data_dir, 'reference_spectrum3.npz'))["data"]

        # Initialize the used sizes
        #self.size1 = 10
        self.size2 = 100
        self.size3 = 255

    @TestManager.register_test
    def _test_attributes(self):
        """
        Test whether the students implemented an attribute 'pattern'.
        """
        # Create the student version of the pattern
        student_spectrum = numpy_applied.Spectrum(self.size2)

        # Check if the 'pattern' attribute exists
        self.assertTrue(
            hasattr(student_spectrum, 'pattern'),
            "Class Spectrum does not contain an attribute 'pattern'!"
        )

        # Return score if all assertions pass
        return 1.0

    #def test_pattern1(self):
    #    """
    #    Test whether the students pattern matches the expected one.
    #    """
    #
    #    # Create the student version of the pattern
    #    student_spectrum = numpy_applied.Spectrum(self.size1)
    #    student_spectrum.draw()
    #    student_version = student_spectrum.pattern
    #
    #    # Compare the student version to the expected output
    #    self.assertEqual(student_version.shape, self.spectrum_expected1.shape,
    #                     "Shape of pattern is not correct!")
    #    np.testing.assert_almost_equal(student_version, self.spectrum_expected1, decimal=2)

    @TestManager.register_test
    def _test_pattern2(self):
        """
        Test whether the students pattern matches the expected one with different parameters.
        """
        # Create the student version of the pattern
        student_spectrum = numpy_applied.Spectrum(self.size2)
        student_spectrum.draw()
        student_version = student_spectrum.pattern

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.spectrum_expected2.shape,
                         "Shape of pattern is not correct!")
        np.testing.assert_almost_equal(student_version, self.spectrum_expected2, decimal=2)

        # Return score if all assertions pass
        return 2.0


    @TestManager.register_test
    def _test_pattern3(self):
        """
        Test whether the students pattern matches the expected one with different parameters.
        """
        # Create the student version of the pattern
        student_spectrum = numpy_applied.Spectrum(self.size3)
        student_spectrum.draw()
        student_version = student_spectrum.pattern

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.spectrum_expected3.shape,
                         "Shape of pattern is not correct!")
        np.testing.assert_almost_equal(student_version, self.spectrum_expected3, decimal=2)

        # Return score if all assertions pass
        return 2.0

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_attributes()
        self._test_pattern2()
        self._test_pattern3()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Bonus score achieved: [{total_score_achieved} / 5.0]")


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
#%%
