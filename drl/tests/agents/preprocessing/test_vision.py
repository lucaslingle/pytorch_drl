import unittest

import torch as tc

from drl.agents.preprocessing.vision import ToChannelMajor


class TestToChannelMajor(unittest.TestCase):
    def test_permute_batchonly(self):
        input_ = tc.tensor(tc.arange(16).reshape(2, 2, 2, 2))
        target = tc.tensor([[[[0, 2], [4, 6]], [[1, 3], [5, 7]]], [[[8, 10], [12, 14]], [[9, 11], [13, 15]]]])
        preprocessing = ToChannelMajor()
        self.assertEqual(
            (1 - tc.eq(preprocessing(input_), target).int()).sum().item(),
            0
        )

if __name__ == '__main__':
    unittest.main()