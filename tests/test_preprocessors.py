import unittest
import torch

from ndsl.module.preprocessor import IdentityPreprocessor, CLSPreprocessor

class TestPreprocessors(unittest.TestCase):

    def setUp(self):
        self.sequence = torch.tensor([
            [0, 1, 2],
            [3, 4, 5]
        ])
        
        
    def test_identity_preprocessor(self):
        pre = IdentityPreprocessor()
        result = pre(self.sequence)
        
        self.assertTrue(
            torch.equal(self.sequence, result),
            "The identity preprocessor output is not the expected"
        )


    def test_cls_preprocessor(self):
        pre = CLSPreprocessor()
        result = pre(self.sequence)
        
        self.assertEqual(
            result.size(), 
            torch.Size([2, 4]), 
            "CLS preprocessor output dimension is not correct"
        )

        self.assertTrue(
            torch.equal(self.sequence, result[:, 1:]), 
            "The sequence was modified in CLS preprocessor"
        )


if __name__ == '__main__':
    unittest.main()