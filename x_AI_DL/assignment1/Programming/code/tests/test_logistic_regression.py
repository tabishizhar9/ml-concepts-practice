#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
This script was written as a test module for the 'logistic_regression.ipynb'-jupyter notebook.
That notebook lets you implement and apply a logistic regression to classify cats and dog images.
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
import pickle

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
logistic_regression = extract_module("logistic_regression")

from code.tests.utils import TestManager

import copy
import functools
from PIL import Image


def load_reference():
    """ A fresh copy of the reference, so a test that modifies its inputs cannot affect the next one. """
    return copy.deepcopy(_build_reference())


@functools.lru_cache(maxsize=1)
def _build_reference():
    """
    Load the reference values the tests compare against.

    The reference file stores only what is needed to rebuild them: which image files, in
    which order, the train/test split as indices, and the resized and flattened images as
    float32. This keeps it at a few MB instead of 200 MB of images stored at every stage.
    """
    file_path_references = os.path.abspath(os.path.join(root_dir, 'code', 'tests', 'references', 'logistic_regression.pkl'))
    with open(file_path_references, 'rb') as f:
        ref = pickle.load(f)

    data_dir = os.path.abspath(os.path.join(root_dir, 'data', 'catsAndDogs'))
    ref['load_images_01'] = [np.array(Image.open(os.path.join(data_dir, 'Cat', name))) for name in ref['image_files_01']]
    ref['load_images_02'] = [np.array(Image.open(os.path.join(data_dir, 'Dog', name))) for name in ref['image_files_02']]

    images = ref['load_images_01'] + ref['load_images_02']
    labels = ref['initialize_labels_01'] + ref['initialize_labels_02']
    train, test = ref['split_indices'][:ref['split_index']], ref['split_indices'][ref['split_index']:]
    ref['train_test_split_01'] = [images[i] for i in train]
    ref['train_test_split_02'] = [labels[i] for i in train]
    ref['train_test_split_03'] = [images[i] for i in test]
    ref['train_test_split_04'] = [labels[i] for i in test]

    for key in ('reshape_01', 'reshape_02', 'flatten_01', 'flatten_02'):
        ref[key] = ref[key].astype(np.float64)
    ref['resize_images_01'] = list(ref['reshape_01'])
    ref['resize_images_02'] = list(ref['reshape_02'])
    ref['normalized_01'] = ref['flatten_01'] / 255.
    ref['normalized_02'] = ref['flatten_02'] / 255.
    return ref


class TestDataLoader(unittest.TestCase):
    """
    The class contains all test cases for task "1 - Dataset Loading".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 1 - Dataset Loading...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Load the parameters
        file_path_parameters = os.path.abspath(os.path.join(root_dir, 'data', 'logistic_regression.pkl'))

        with open(file_path_parameters, 'rb') as f:
            self.p = pickle.load(f)

        # Load the references
        self.ref = load_reference()

        file_path_cat_img = os.path.abspath(os.path.join(root_dir, 'data', 'catsAndDogs', 'Cat'))
        file_path_dog_img = os.path.abspath(os.path.join(root_dir, 'data', 'catsAndDogs', 'Dog'))

        # Create the students data loader
        self.students_data_loader = logistic_regression.DataLoader(file_path_cat_img,
                                                                   file_path_dog_img,
                                                                   self.p['cat_label'], self.p['dog_label'],
                                                                   self.p['train_split_ratio'])

    @TestManager.register_test
    def _test_image_loading(self):
        """ Test loading the images as a list of np.arrays. """
        # Debug
        file_path_cat_img = os.path.abspath(os.path.join(root_dir, 'data', 'catsAndDogs', 'Cat'))
        file_path_dog_img = os.path.abspath(os.path.join(root_dir, 'data', 'catsAndDogs', 'Dog'))

        stud_vers_01, stud_vers_02 = logistic_regression.DataLoader.load_images(file_path_cat_img,
                                                                                file_path_dog_img)

        exp_vers_01 = self.ref['load_images_01']
        exp_vers_02 = self.ref['load_images_02']

        self.assertEqual(type(stud_vers_01), type(exp_vers_01), "Type of cat_images is not correct!")
        self.assertEqual(len(stud_vers_01), len(exp_vers_01), "Length of cat_images is not correct!")

        self.assertEqual(type(stud_vers_02), type(exp_vers_02), "Type of dog_images is not correct!")
        self.assertEqual(len(stud_vers_02), len(exp_vers_02), "Length of dog_images is not correct!")

        # The images must come in file-name order (sorted()), or the train/test split differs from computer to computer
        self.assertEqual([im.shape for im in stud_vers_01[:10]], [im.shape for im in exp_vers_01[:10]],
                         "cat_images are not in file-name order. Did you sort the file paths?")
        self.assertEqual([im.shape for im in stud_vers_02[:10]], [im.shape for im in exp_vers_02[:10]],
                         "dog_images are not in file-name order. Did you sort the file paths?")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_initialize_class_dictionary(self):
        """ Test initializing the class dictionary. """
        stud_vers = logistic_regression.DataLoader.initialize_classes(self.p['cat_label'], self.p['dog_label'])

        exp_vers = self.ref['initialize_classes']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of the return value is not a correct!")
        self.assertTrue((stud_vers == exp_vers), "Return value is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_initialize_labels(self):
        """ Test initializing the labels for cats and dogs. """
        stud_vers_01, stud_vers_02 = logistic_regression.DataLoader.initialize_labels(self.ref['load_images_01'],
                                                                                      self.ref['load_images_02'],
                                                                                      self.p['cat_label'],
                                                                                      self.p['dog_label'])

        exp_vers_01 = self.ref['initialize_labels_01']
        exp_vers_02 = self.ref['initialize_labels_02']

        self.assertEqual(type(stud_vers_01), type(exp_vers_01), "Type of cat_labels is not correct!")
        self.assertEqual(len(stud_vers_01), len(exp_vers_01), "Length of cat_labels is not correct!")
        self.assertTrue(np.allclose(np.array(stud_vers_01), np.array(exp_vers_01), atol=0.0001), "cat_labels is not correct!")

        self.assertEqual(type(stud_vers_02), type(exp_vers_02), "Type of dog_labels is not correct!")
        self.assertEqual(len(stud_vers_02), len(exp_vers_02), "Length of dog_labels is not correct!")
        self.assertTrue(np.allclose(np.array(stud_vers_02), np.array(exp_vers_02), atol=0.0001), "dog_labels is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_split_images_and_labels_into_train_and_test_sets(self):
        """ Test splitting the images and labels into train and test sets. """
        stud_vers_01, stud_vers_02, stud_vers_03, stud_vers_04 =\
            logistic_regression.DataLoader.split_images_and_labels_into_train_and_test_sets(self.ref['load_images_01'],
                                                                                            self.ref['load_images_02'],
                                                                                            self.ref['initialize_labels_01'],
                                                                                            self.ref['initialize_labels_02'],
                                                                                            self.p['train_split_ratio'])

        exp_vers_01 = self.ref['train_test_split_01']
        exp_vers_02 = self.ref['train_test_split_02']
        exp_vers_03 = self.ref['train_test_split_03']
        exp_vers_04 = self.ref['train_test_split_04']

        self.assertEqual(type(stud_vers_01), type(exp_vers_01), "Type of train_images is not correct!")
        self.assertEqual(len(stud_vers_01), len(exp_vers_01), "Length of train_images is not correct!")
        for sv_01, exp_01 in zip(stud_vers_01[:10], exp_vers_01[:10]):
            self.assertTrue(np.allclose(sv_01, exp_01, atol=0.0001), "train_images is not correct!")

        self.assertEqual(type(stud_vers_02), type(exp_vers_02), "Type of train_labels is not correct!")
        self.assertEqual(len(stud_vers_02), len(exp_vers_02), "Length of train_labels is not correct!")
        for sv_02, exp_02 in zip(stud_vers_02[:10], exp_vers_02[:10]):
            self.assertTrue(np.allclose(sv_02, exp_02, atol=0.0001), "train_labels is not correct!")

        self.assertEqual(type(stud_vers_03), type(exp_vers_03), "Type of test_images is not correct!")
        self.assertEqual(len(stud_vers_03), len(exp_vers_03), "Length of test_images is not correct!")
        for sv_03, exp_03 in zip(stud_vers_03[:10], exp_vers_03[:10]):
            self.assertTrue(np.allclose(sv_03, exp_03, atol=0.0001), "test_images is not correct!")

        self.assertEqual(type(stud_vers_04), type(exp_vers_04), "Type of test_labels is not correct!")
        self.assertEqual(len(stud_vers_04), len(exp_vers_04), "Length of test_labels is not correct!")
        for sv_04, exp_04 in zip(stud_vers_04[:10], exp_vers_04[:10]):
            self.assertTrue(np.allclose(sv_04, exp_04, atol=0.0001), "test_labels is not correct!")

        # Return score if all assertions pass
        return 0.75

    @TestManager.register_test
    def _test_load_train_and_test_sets(self):
        """ Test loading the train and test sets. """
        self.students_data_loader.load_train_and_test_sets()
        stud_vers_01 = self.students_data_loader.train_images
        stud_vers_02 = self.students_data_loader.train_labels
        stud_vers_03 = self.students_data_loader.test_images
        stud_vers_04 = self.students_data_loader.test_labels

        exp_vers_01 = self.ref['train_test_split_01']
        exp_vers_02 = self.ref['train_test_split_02']
        exp_vers_03 = self.ref['train_test_split_03']
        exp_vers_04 = self.ref['train_test_split_04']

        self.assertEqual(type(stud_vers_01), type(exp_vers_01), "Type of self.train_images is not correct!")
        self.assertEqual(len(stud_vers_01), len(exp_vers_01), "Length of self.train_images is not correct!")

        self.assertEqual(type(stud_vers_02), type(exp_vers_02), "Type of self.train_labels is not correct!")
        self.assertEqual(len(stud_vers_02), len(exp_vers_02), "Length of self.train_labels is not correct!")

        self.assertEqual(type(stud_vers_03), type(exp_vers_03), "Type of self.test_images is not correct!")
        self.assertEqual(len(stud_vers_03), len(exp_vers_03), "Length of self.test_images is not correct!")

        self.assertEqual(type(stud_vers_04), type(exp_vers_04), "Type of self.test_labels is not correct!")
        self.assertEqual(len(stud_vers_04), len(exp_vers_04), "Length of self.test_labels is not correct!")

        # Return score if all assertions pass
        return 0.75

    @TestManager.register_test
    def _test_get_train_and_test_sets(self):
        """ Test loading the train and test sets. """
        self.students_data_loader.load_train_and_test_sets()

        stud_vers_01, stud_vers_02, stud_vers_03, stud_vers_04 = self.students_data_loader.get_train_and_test_sets()

        exp_vers_01 = self.ref['train_test_split_01']
        exp_vers_02 = self.ref['train_test_split_02']
        exp_vers_03 = self.ref['train_test_split_03']
        exp_vers_04 = self.ref['train_test_split_04']

        self.assertEqual(type(stud_vers_01), type(exp_vers_01), "Type of self.train_images is not correct!")
        self.assertEqual(len(stud_vers_01), len(exp_vers_01), "Length of self.train_images is not correct!")

        self.assertEqual(type(stud_vers_02), type(exp_vers_02), "Type of self.train_labels is not correct!")
        self.assertEqual(len(stud_vers_02), len(exp_vers_02), "Length of self.train_labels is not correct!")

        self.assertEqual(type(stud_vers_03), type(exp_vers_03), "Type of self.test_images is not correct!")
        self.assertEqual(len(stud_vers_03), len(exp_vers_03), "Length of self.test_images is not correct!")

        self.assertEqual(type(stud_vers_04), type(exp_vers_04), "Type of self.test_labels is not correct!")
        self.assertEqual(len(stud_vers_04), len(exp_vers_04), "Length of self.test_labels is not correct!")

        # Return score if all assertions pass
        return 0.5

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_image_loading()
        self._test_initialize_class_dictionary()
        self._test_initialize_labels()
        self._test_split_images_and_labels_into_train_and_test_sets()
        self._test_load_train_and_test_sets()
        self._test_get_train_and_test_sets()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 3.0]")


class TestPreprocessor(unittest.TestCase):
    """
    The class contains all test cases for task "2 - Preprocessing".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 2 - Preprocessing...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Load the parameters
        file_path_parameters = os.path.abspath(os.path.join(root_dir, 'data', 'logistic_regression.pkl'))

        with open(file_path_parameters, 'rb') as f:
            self.p = pickle.load(f)

        # Load the references
        self.ref = load_reference()

    @TestManager.register_test
    def _test_resize_images(self):
        """ Test resizing all images from their original shape to a new shape. """
        stud_vers_01, stud_vers_02 = logistic_regression.Preprocessor.resize_images(self.ref['train_test_split_01'],
                                                                                    self.ref['train_test_split_03'],
                                                                                    self.p['new_shape'])

        exp_vers_01 = self.ref['resize_images_01']
        exp_vers_02 = self.ref['resize_images_02']

        self.assertEqual(type(stud_vers_01), type(exp_vers_01), "Type of 'resized_train_images' is not correct!")
        self.assertEqual(len(stud_vers_01), len(exp_vers_01), "Length of 'resized_train_images' is not correct!")
        for sv_01, exp_01 in zip(stud_vers_01[:10], exp_vers_01[:10]):
            self.assertTrue(np.allclose(sv_01, exp_01, atol=0.0001), "'resized_train_images' is not correct!")

        self.assertEqual(type(stud_vers_02), type(exp_vers_02), "Type of 'resized_test_images' is not correct!")
        self.assertEqual(len(stud_vers_02), len(exp_vers_02), "Length of 'resized_test_images' is not correct!")
        for sv_02, exp_02 in zip(stud_vers_02[:10], exp_vers_02[:10]):
            self.assertTrue(np.allclose(sv_02, exp_02, atol=0.0001), "'resized_test_images' is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_reshape_sets(self):
        """ Test reshaping the sets from lists to np.arrays. """
        stud_vers_01, stud_vers_02, stud_vers_03, stud_vers_04 =\
            logistic_regression.Preprocessor.reshape_sets(self.ref['resize_images_01'], self.ref['resize_images_02'],
                                                          self.ref['train_test_split_02'],
                                                          self.ref['train_test_split_04'])

        exp_vers_01 = self.ref['reshape_01']
        exp_vers_02 = self.ref['reshape_02']
        exp_vers_03 = self.ref['reshape_03']
        exp_vers_04 = self.ref['reshape_04']

        self.assertEqual(type(stud_vers_01), type(exp_vers_01), "Type of 'reshaped_train_images' is not correct!")
        self.assertEqual(stud_vers_01.shape, exp_vers_01.shape, "Shape of 'reshaped_train_images' is not correct!")
        for sv_01, exp_01 in zip(stud_vers_01[:10], exp_vers_01[:10]):
            self.assertTrue(np.allclose(sv_01, exp_01, atol=0.0001), "'reshaped_train_images' is not correct!")

        self.assertEqual(type(stud_vers_02), type(exp_vers_02), "Type of 'reshaped_test_images' is not correct!")
        self.assertEqual(stud_vers_02.shape, exp_vers_02.shape, "Shape of 'reshaped_test_images' is not correct!")
        for sv_02, exp_02 in zip(stud_vers_02[:10], exp_vers_02[:10]):
            self.assertTrue(np.allclose(sv_02, exp_02, atol=0.0001), "'reshaped_test_images' is not correct!")

        self.assertEqual(type(stud_vers_03), type(exp_vers_03), "Type of 'reshaped_train_labels' is not correct!")
        self.assertEqual(stud_vers_03.shape, exp_vers_03.shape, "Shape of 'reshaped_train_labels' is not correct!")
        self.assertTrue(np.allclose(stud_vers_03, exp_vers_03, atol=0.0001), "'reshaped_train_labels' is not correct!")

        self.assertEqual(type(stud_vers_04), type(exp_vers_04), "Type of 'reshaped_test_labels' is not correct!")
        self.assertEqual(stud_vers_04.shape, exp_vers_04.shape, "Shape of 'reshaped_test_labels' is not correct!")
        self.assertTrue(np.allclose(stud_vers_04, exp_vers_04, atol=0.0001), "'reshaped_test_labels' is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_flatten_images(self):
        """ Test flattening the train and test images over the spatial dimension. """
        stud_vers_01, stud_vers_02 = logistic_regression.Preprocessor.flatten_images(self.ref['reshape_01'],
                                                                                     self.ref['reshape_02'])

        exp_vers_01 = self.ref['flatten_01']
        exp_vers_02 = self.ref['flatten_02']

        self.assertEqual(type(stud_vers_01), type(exp_vers_01), "Type of 'flattened_train_images' is not correct!")
        self.assertEqual(stud_vers_01.shape, exp_vers_01.shape, "Shape of 'flattened_train_images' is not correct!")
        for sv_01, exp_01 in zip(stud_vers_01[:10], exp_vers_01[:10]):
            self.assertTrue(np.allclose(sv_01, exp_01, atol=0.0001), "'flattened_train_images' is not correct!")

        self.assertEqual(type(stud_vers_02), type(exp_vers_02), "Type of 'flattened_test_images' is not correct!")
        self.assertEqual(stud_vers_02.shape, exp_vers_02.shape, "Shape of 'flattened_test_images' is not correct!")
        for sv_02, exp_02 in zip(stud_vers_02[:10], exp_vers_02[:10]):
            self.assertTrue(np.allclose(sv_02, exp_02, atol=0.0001), "'flattened_test_images' is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_normalize_images(self):
        """ Test which normalizing the train and test image sets. """
        stud_vers_01, stud_vers_02 = logistic_regression.Preprocessor.normalize_images(self.ref['flatten_01'],
                                                                                       self.ref['flatten_02'])

        exp_vers_01 = self.ref['normalized_01']
        exp_vers_02 = self.ref['normalized_02']

        self.assertEqual(type(stud_vers_01), type(exp_vers_01), "Type of 'normalized_train_images' is not correct!")
        self.assertEqual(stud_vers_01.shape, exp_vers_01.shape, "Shape of 'normalized_train_images' is not correct!")
        for sv_01, exp_01 in zip(stud_vers_01[:10], exp_vers_01[:10]):
            self.assertTrue(np.allclose(sv_01, exp_01, atol=0.0001), "'normalized_train_images' is not correct!")

        self.assertEqual(type(stud_vers_02), type(exp_vers_02), "Type of 'normalized_test_images' is not correct!")
        self.assertEqual(stud_vers_02.shape, exp_vers_02.shape, "Shape of 'normalized_test_images' is not correct!")
        for sv_02, exp_02 in zip(stud_vers_02[:10], exp_vers_02[:10]):
            self.assertTrue(np.allclose(sv_02, exp_02, atol=0.0001), "'normalized_test_images' is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_preprocess(self):
        """ Test preprocessing the training and test sets """
        stud_vers_01, stud_vers_02, stud_vers_03, stud_vers_04 = \
            logistic_regression.Preprocessor.preprocess(self.ref['train_test_split_01'],
                                                        self.ref['train_test_split_03'],
                                                        self.ref['train_test_split_02'],
                                                        self.ref['train_test_split_04'], self.p['new_shape'])

        exp_vers_01 = self.ref['normalized_01']
        exp_vers_02 = self.ref['normalized_02']
        exp_vers_03 = self.ref['reshape_03']
        exp_vers_04 = self.ref['reshape_04']

        self.assertEqual(type(stud_vers_01), type(exp_vers_01), "Type of 'preprocessed_train_images' is not correct!")
        self.assertEqual(stud_vers_01.shape, exp_vers_01.shape, "Shape of 'preprocessed_train_images' is not correct!")
        for sv_01, exp_01 in zip(stud_vers_01[:10], exp_vers_01[:10]):
            self.assertTrue(np.allclose(sv_01, exp_01, atol=0.0001), "'preprocessed_train_images' is not correct!")

        self.assertEqual(type(stud_vers_02), type(exp_vers_02), "Type of 'preprocessed_test_images' is not correct!")
        self.assertEqual(stud_vers_02.shape, exp_vers_02.shape, "Shape of 'preprocessed_test_images' is not correct!")
        for sv_02, exp_02 in zip(stud_vers_02[:10], exp_vers_02[:10]):
            self.assertTrue(np.allclose(sv_02, exp_02, atol=0.0001), "'preprocessed_test_images' is not correct!")

        self.assertEqual(type(stud_vers_03), type(exp_vers_03), "Type of 'preprocessed_train_labels' is not correct!")
        self.assertEqual(stud_vers_03.shape, exp_vers_03.shape, "Shape of 'preprocessed_train_labels' is not correct!")
        self.assertTrue(np.allclose(stud_vers_03, exp_vers_03, atol=0.0001), "'preprocessed_train_labels' is not correct!")

        self.assertEqual(type(stud_vers_04), type(exp_vers_04), "Type of 'preprocessed_test_labels' is not correct!")
        self.assertEqual(stud_vers_04.shape, exp_vers_04.shape, "Shape of 'preprocessed_test_labels' is not correct!")
        self.assertTrue(np.allclose(stud_vers_04, exp_vers_04, atol=0.0001), "'preprocessed_test_labels' is not correct!")

        # Return score if all assertions pass
        return 0.5

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_resize_images()
        self._test_reshape_sets()
        self._test_flatten_images()
        self._test_normalize_images()
        self._test_preprocess()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 2.0]")


class TestActivation(unittest.TestCase):
    """
    The class contains all test cases for task "3.1 - Activation Function".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 3.1 - Activation Function...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the tests
        self.test_float_01 = 5.0
        self.test_array_01 = np.array([0, 1, 2, 3, 4, 5])
        self.test_array_02 = np.array([[0, 1], [1, 0]])
        self.test_array_03 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

        # Load the references
        self.ref = load_reference()

    @TestManager.register_test
    def _test_activation_01(self):
        """ Test the activation function (1). """
        stud_ver = logistic_regression.sigmoid(self.test_float_01)
        exp_vers = self.ref['sigmoid_01']

        self.assertEqual(type(stud_ver), type(exp_vers), "Type of 's' is not correct!")
        self.assertTrue(np.allclose(stud_ver, exp_vers, atol=0.0001), "'s' is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_activation_02(self):
        """ Test the activation function (2). """
        stud_ver = logistic_regression.sigmoid(self.test_array_01)
        exp_vers = self.ref['sigmoid_02']

        self.assertEqual(type(stud_ver), type(exp_vers), "Type of 's' is not correct!")
        self.assertEqual(stud_ver.shape, exp_vers.shape, "Shape of 's' is not correct!")
        self.assertTrue(np.allclose(stud_ver, exp_vers, atol=0.0001), "'s' is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_activation_03(self):
        """ Test the activation function (3). """
        stud_ver = logistic_regression.sigmoid(self.test_array_02)
        exp_vers = self.ref['sigmoid_03']

        self.assertEqual(type(stud_ver), type(exp_vers), "Type of 's' is not correct!")
        self.assertEqual(stud_ver.shape, exp_vers.shape, "Shape of 's' is not correct!")
        self.assertTrue(np.allclose(stud_ver, exp_vers, atol=0.0001), "'s' is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_activation_04(self):
        """ Test the activation function (4). """
        stud_ver = logistic_regression.sigmoid(self.test_array_03)
        exp_vers = self.ref['sigmoid_04']

        self.assertEqual(type(stud_ver), type(exp_vers), "Type of 's' is not correct!")
        self.assertEqual(stud_ver.shape, exp_vers.shape, "Shape of 's' is not correct!")
        self.assertTrue(np.allclose(stud_ver, exp_vers, atol=0.0001), "'s' is not correct!")

        # Return score if all assertions pass
        return 0.25

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_activation_01()
        self._test_activation_02()
        self._test_activation_03()
        self._test_activation_04()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 1.0]")


class TestInitialization(unittest.TestCase):
    """
    The class contains all test cases for task "3.2 - Parameter Initialization".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 3.2 - Parameter Initialization...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the tests
        self.dim_01 = 1
        self.dim_02 = 5
        self.dim_03 = 100

        # Load the references
        self.ref = load_reference()

    @TestManager.register_test
    def _test_initialization_01(self):
        """ Test the initialization (1). """
        stud_ver_01, stud_ver_02 = logistic_regression.initialize_with_zeros(self.dim_01)
        exp_vers_01, exp_vers_02 = self.ref['init_01']

        self.assertEqual(type(stud_ver_01), type(exp_vers_01), "Type of 'w' is not correct!")
        self.assertEqual(stud_ver_01.shape, exp_vers_01.shape, "Shape of 'w' is not correct!")
        self.assertTrue(np.allclose(stud_ver_01, exp_vers_01, atol=0.0001), "'w' is not correct!")

        self.assertTrue(np.allclose(stud_ver_02, exp_vers_02, atol=0.0001), "'b' is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_initialization_02(self):
        """ Test the initialization (2). """
        stud_ver_01, stud_ver_02 = logistic_regression.initialize_with_zeros(self.dim_02)
        exp_vers_01, exp_vers_02 = self.ref['init_02']

        self.assertEqual(type(stud_ver_01), type(exp_vers_01), "Type of 'w' is not correct!")
        self.assertEqual(stud_ver_01.shape, exp_vers_01.shape, "Shape of 'w' is not correct!")
        self.assertTrue(np.allclose(stud_ver_01, exp_vers_01, atol=0.0001), "'w' is not correct!")

        self.assertTrue(np.allclose(stud_ver_02, exp_vers_02, atol=0.0001), "'b' is not correct!")

        # Return score if all assertions pass
        return 0.25

    @TestManager.register_test
    def _test_initialization_03(self):
        """ Test the initialization (3). """
        stud_ver_01, stud_ver_02 = logistic_regression.initialize_with_zeros(self.dim_03)
        exp_vers_01, exp_vers_02 = self.ref['init_03']

        self.assertEqual(type(stud_ver_01), type(exp_vers_01), "Type of 'w' is not correct!")
        self.assertEqual(stud_ver_01.shape, exp_vers_01.shape, "Shape of 'w' is not correct!")
        self.assertTrue(np.allclose(stud_ver_01, exp_vers_01, atol=0.0001), "'w' is not correct!")

        self.assertTrue(np.allclose(stud_ver_02, exp_vers_02, atol=0.0001), "'b' is not correct!")

        # Return score if all assertions pass
        return 0.25

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_initialization_01()
        self._test_initialization_02()
        self._test_initialization_03()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 0.75]")


class TestForwardAndBackward(unittest.TestCase):
    """
    The class contains all test cases for task "3.3 - Forward and Backward Propagation".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 3.3 - Forward and Backward Propagation...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the tests
        self.X = np.array([[1., 2., -1., 0], [3., 4., -3.2, 1], [3., 4., -3.2, -3.5]])
        self.Y = np.array([[1, 1, 0, 0]])
        self.w, self.b = np.array([[1.], [2.], [-1]]), 2.5

        # Load the references
        self.ref = load_reference()

    @TestManager.register_test
    def _test_forward(self):
        """ Test the forward propagation. """
        stud_ver_01, stud_ver_02 = logistic_regression.forward(self.X, self.Y, self.w, self.b)
        exp_vers_01, exp_vers_02 = self.ref['forward']

        self.assertEqual(type(stud_ver_01), type(exp_vers_01), "Type of 'A' is not correct!")
        self.assertEqual(stud_ver_01.shape, exp_vers_01.shape, "Shape of 'A' is not correct!")
        self.assertTrue(np.allclose(stud_ver_01, exp_vers_01, atol=0.0001), "'A' is not correct!")

        self.assertEqual(type(stud_ver_02), type(exp_vers_02), "Type of 'cost' is not correct!")
        self.assertTrue(np.allclose(stud_ver_02, exp_vers_02, atol=0.0001), "'cost' is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_backward(self):
        """ Test the backward propagation. """
        stud_ver = logistic_regression.backward(self.X, self.Y, self.ref['forward'][0])
        exp_vers = self.ref['backward']

        self.assertEqual(type(stud_ver), type(exp_vers), "Type of 'grads' is not correct!")
        self.assertEqual(len(stud_ver.keys()), len(exp_vers.keys()), "Number of entries of 'grads' is not correct!")

        self.assertEqual(type(stud_ver["dw"]), type(exp_vers["dw"]), "Type of 'dw' is not correct!")
        self.assertEqual(stud_ver["dw"].shape, exp_vers["dw"].shape, "Shape of 'dw' is not correct!")
        self.assertTrue(np.allclose(stud_ver["dw"], exp_vers["dw"], atol=0.0001), "'dw' is not correct!")

        self.assertEqual(type(stud_ver["db"]), type(exp_vers["db"]), "Type of 'db' is not correct!")
        self.assertEqual(stud_ver["db"].shape, exp_vers["db"].shape, "Shape of 'db' is not correct!")
        self.assertTrue(np.allclose(stud_ver["db"], exp_vers["db"], atol=0.0001), "'db' is not correct!")

        # Return score if all assertions pass
        return 0.5

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_forward()
        self._test_backward()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 1.0]")


class TestOptimization(unittest.TestCase):
    """
    The class contains all test cases for task "3.4 - Optimization".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 3.4 - Optimization...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the tests
        self.X = np.array([[1., 2., -1.], [3., 4., -3.2]])
        self.Y = np.array([[1, 0, 1]])
        self.w, self.b = np.array([[1.], [2.]]), 2.
        self.grads = {"dw": np.array([[0.06188603], [-0.01407361]]), "db": np.float64(-0.04709353)}
        self.learning_rate = 0.1

        # Load the references
        self.ref = load_reference()

    @TestManager.register_test
    def _test_optimize(self):
        """ Test the optimization. """
        stud_ver = logistic_regression.optimize(self.w, self.b, self.grads, self.learning_rate)
        exp_vers = self.ref['optimize']

        self.assertEqual(type(stud_ver), type(exp_vers), "Type of 'params' is not correct!")
        self.assertEqual(len(stud_ver.keys()), len(exp_vers.keys()), "Number of entries of 'params' is not correct!")

        self.assertEqual(type(stud_ver["w"]), type(exp_vers["w"]), "Type of 'w' is not correct!")
        self.assertEqual(stud_ver["w"].shape, exp_vers["w"].shape, "Shape of 'w' is not correct!")
        self.assertTrue(np.allclose(stud_ver["w"], exp_vers["w"], atol=0.0001), "'w' is not correct!")

        self.assertTrue(np.allclose(stud_ver["b"], exp_vers["b"], atol=0.0001), "'b' is not correct!")

        # Return score if all assertions pass
        return 0.5

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_optimize()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 0.5]")


class TestPrediction(unittest.TestCase):
    """
    The class contains all test cases for task "3.5 - Prediction".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 3.5 - Prediction...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the tests
        self.X = np.array([[1., -0.3, 1.5],[2, 0, 1], [0, -1.5, 2]])
        self.w = np.array([[0.3], [0.5], [-0.2]])
        self.b = -0.33333

        # Load the references
        self.exp_vers = np.array([[1., 0., 1]])

    @TestManager.register_test
    def _test_predict(self):
        """ Test the prediction. """
        stud_ver = logistic_regression.predict(self.X, self.w, self.b)

        self.assertEqual(type(stud_ver), type(self.exp_vers), "Type of 'Y_pred' is not correct!")
        self.assertEqual(stud_ver.shape, self.exp_vers.shape, "Shape of 'Y_pred' is not correct!")
        self.assertTrue(np.allclose(stud_ver, self.exp_vers, atol=0.0001), "'Y_pred' is not correct!")

        # Return score if all assertions pass
        return 0.75

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_predict()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 0.75]")


class TestModel(unittest.TestCase):
    """
    The class contains all test cases for task "3.6 - Merge all Functions into a Model".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 3.6 - Merge all Functions into a Model...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

        # Create the tests
        np.random.seed(0)

        # Three samples for training
        self.X = np.random.randn(4, 7)
        self.Y = np.array([1, 0, 0, 1, 0, 0, 1]).reshape(1, 7)
        self.b = 1.5

        # 6 samples for testing
        self.X_test = np.random.randn(4, 3)
        self.Y_test = np.array([0, 1, 0])

        # Hyperparameters
        self.num_iterations = 50
        self.learning_rate = 0.01

        # Create the student version
        self.stud_ver = logistic_regression.model(self.X, self.Y, self.X_test, self.Y_test, self.num_iterations,
                                             self.learning_rate)

        # Load the references
        self.exp_vers = {'costs': [np.array(0.69314718)],
                         'Y_pred_test': np.array([[1., 1., 0.]]),
                         'Y_pred_train': np.array([[1., 1., 0., 1., 0., 0., 1.]]),
                         'w': np.array([[ 0.08639757],
                                        [-0.08231268],
                                        [-0.11798927],
                                        [ 0.12866053]]),
                         'b': -0.03983236094816321,
                         "learning_rate": self.learning_rate,
                         "num_iterations": self.num_iterations}

    @TestManager.register_test
    def _test_output(self):
        """ Test the model output. """
        self.assertEqual(type(self.stud_ver), type(self.exp_vers), "Type of 'model_output' is not correct!")
        self.assertEqual(len(self.stud_ver.keys()), len(self.exp_vers.keys()), "Number of entries of 'model_output' is not correct!")
        # The skeleton already returns this dictionary with every value set to None, so the
        # checks above alone gave points for an untouched notebook.
        self.assertEqual(set(self.stud_ver.keys()), set(self.exp_vers.keys()), "Keys of 'model_output' are not correct!")
        for key, value in self.stud_ver.items():
            self.assertIsNotNone(value, f"model_output['{key}'] is not filled in!")
        self.assertEqual(self.stud_ver['learning_rate'], self.exp_vers['learning_rate'], "'learning_rate' is not correct!")
        self.assertEqual(self.stud_ver['num_iterations'], self.exp_vers['num_iterations'], "'num_iterations' is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_costs(self):
        """ Test the costs. """
        self.assertEqual(type(self.stud_ver['costs']), type(self.exp_vers['costs']), "Type of 'costs' is not correct!")
        self.assertEqual(len(self.stud_ver['costs']), len(self.exp_vers['costs']), "Length of 'costs' is not correct!")
        self.assertTrue(np.allclose(self.stud_ver['costs'], self.exp_vers['costs'], atol=0.0001), "'costs' is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_weights(self):
        """ Test the weights. """
        self.assertEqual(type(self.stud_ver['w']), type(self.exp_vers['w']), "Type of 'w' is not correct!")
        self.assertEqual(self.stud_ver['w'].shape, self.exp_vers['w'].shape, "Shape of 'w' is not correct!")
        self.assertTrue(np.allclose(self.stud_ver['w'], self.exp_vers['w'], atol=0.0001), "'w' is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_bias(self):
        """ Test the bias. """
        self.assertTrue(np.allclose(self.stud_ver['b'], self.exp_vers['b'], atol=0.0001), "'b' is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_prediction_test_samples(self):
        """ Test the prediction for the test samples. """
        self.assertEqual(type(self.stud_ver['Y_pred_test']), type(self.exp_vers['Y_pred_test']), "Type of 'Y_prediction_test' is not correct!")
        self.assertEqual(self.stud_ver['Y_pred_test'].shape, self.exp_vers['Y_pred_test'].shape, "Shape of 'Y_prediction_test' is not correct!")
        self.assertTrue(np.allclose(self.stud_ver['Y_pred_test'], self.exp_vers['Y_pred_test'], atol=0.0001), "'Y_prediction_test' is not correct!")

        # Return score if all assertions pass
        return 0.5

    @TestManager.register_test
    def _test_prediction_train_samples(self):
        """ Test the prediction for the train samples. """
        self.assertEqual(type(self.stud_ver['Y_pred_train']), type(self.exp_vers['Y_pred_train']), "Type of 'Y_prediction_test' is not correct!")
        self.assertEqual(self.stud_ver['Y_pred_train'].shape, self.exp_vers['Y_pred_train'].shape, "Shape of 'Y_prediction_test' is not correct!")
        self.assertTrue(np.allclose(self.stud_ver['Y_pred_train'], self.exp_vers['Y_pred_train'], atol=0.0001), "'Y_prediction_test' is not correct!")

        # Return score if all assertions pass
        return 0.5

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_output()
        self._test_costs()
        self._test_weights()
        self._test_bias()
        self._test_prediction_test_samples()
        self._test_prediction_train_samples()

        # Compute total achieved score
        total_score_achieved = sum(self.test_scores.values())
        num_passed_tests = sum(1 for score in self.test_scores.values() if score > 0)

        print(f"[{num_passed_tests} / {len(self.test_scores)}] test(s) passed.")
        print(f"Score achieved: [{total_score_achieved} / 3.0]")


class TestEvaluation(unittest.TestCase):
    """
    The class contains all test cases for task "4 - Viewing and Evaluate the Results".
    """

    def setUp(self):
        """ Initialize the tests. """
        print("Running tests for Task 4 - Viewing and Evaluate the Results...")
        print(50 * "-")

        self.test_scores = {}  # Store the scores for individual tests
        self.test_manager = TestManager(self.test_scores)

    @TestManager.register_test
    def _test_evaluate(self):
        checkpoint_path = os.path.abspath(os.path.join(root_dir, 'checkpoints', 'logistic_regression.pkl'))
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        train_accuracy = 100 - np.mean(np.abs(checkpoint['model']['Y_pred_train'] - checkpoint['Y_train'])) * 100
        test_accuracy = 100 - np.mean(np.abs(checkpoint['model']['Y_pred_test'] - checkpoint['Y_test'])) * 100

        self.assertGreater(train_accuracy, 50.0)
        self.assertGreater(test_accuracy, 40.0)

        # Return score if all assertions pass
        return 1.0

    def test_all(self):
        """
        Validate that all tests have run properly and compute the total score.
        """
        # Reset test_scores to avoid duplication
        self.test_scores = {}

        # Run individual tests
        self._test_evaluate()

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
#%%
