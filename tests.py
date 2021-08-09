import unittest

import numpy as np

from num_seq_generator import generate_numbers_sequence


class TestGenerator(unittest.TestCase):
    def test_shape(self):
        data = np.zeros((10, 64, 48))
        labels = np.array([i for i in range(10)])
        seq = generate_numbers_sequence([1, 6, 4], (0, 1), 100, data, labels)
        expected_shape = (28, 100)
        self.assertEqual(seq.shape, expected_shape, f"Should be {expected_shape}")

    def test_sequence(self):
        # create matrices of 28x20 with the main diagonal filled
        data = np.array([np.eye(28, 20) for i in range(10)])
        labels = np.array([i for i in range(10)])
        seq = generate_numbers_sequence([1, 2], (0, 1), 40, data, labels)
        expected_array = np.zeros_like(seq)
        # manually fill consecutive diagonals
        for i in range(20):
            expected_array[i, i] = 1 / 255
            expected_array[i, i + 20] = 1 / 255
        self.assertIsNone(
            np.testing.assert_array_equal(seq, expected_array),
            f"Failed to match diagonal sequence",
        )

    def test_zero_width(self):
        with self.assertRaises(ValueError):
            data = np.zeros((10, 28, 28))
            labels = np.array([i for i in range(10)])
            generate_numbers_sequence([1, 6, 4], (0, 1), 0, data, labels)

    def test_no_num_label(self):
        with self.assertRaises(ValueError):
            data = np.zeros((5, 28, 28))
            labels = np.array([i for i in range(5)])
            generate_numbers_sequence([1, 8], (0, 1), 0, data, labels)

    def test_empty_digits(self):
        with self.assertRaises(ValueError):
            data = np.zeros((5, 28, 28))
            labels = np.array([i for i in range(5)])
            generate_numbers_sequence([], (0, 1), 0, data, labels)

    def test_diff_num_samples(self):
        with self.assertRaises(ValueError):
            data = np.zeros((5, 28, 28))
            labels = np.array([i for i in range(8)])
            generate_numbers_sequence([], (0, 1), 0, data, labels)


if __name__ == "__main__":
    unittest.main()
