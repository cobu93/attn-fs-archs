import unittest
import torch

from ndsl.architecture.attention import TTransformerEncoderLayer

class TestEncoderLayer(unittest.TestCase):

    def setUp(self):
        self.features = torch.tensor([
            [-1],
            [0],
            [1],
            [2],
            [3],
            [4]
        ])
        

    def test_categorical_oh_encoder(self):
        enc = CategoricalOneHotEncoder(10, 3)
        result = enc(self.features)

        self.assertEqual(
            result.size(), 
            torch.Size([6, 10]), 
            "The output don't have the expected size"
        )
        
        self.assertTrue(
            torch.equal(result[-2], result[-1]) \
            and torch.equal(result[0], result[-1]) \
            and not torch.equal(result[1], result[2]) \
            and not torch.equal(result[1], result[3]) \
            and not torch.equal(result[1], result[0]) \
            and not torch.equal(result[2], result[3]) \
            and not torch.equal(result[2], result[0]) \
            and not torch.equal(result[3], result[0]), 
            "The nan encoding is not valid"
        )


    def test_numerical_encoder(self):
        enc = NumericalEncoder(10)
        # Parameter in format (B, S, E)
        result = enc(self.features.float())

        self.assertEqual(
            result.size(), 
            torch.Size([6, 10]), 
            "Output shape should be (2, 10)"
        )

        self.assertTrue(
            torch.allclose(result[2] - result[1], (result[0] - result[1]) / self.features[0]) \
            and torch.allclose(result[2] - result[1], (result[3] - result[1]) / self.features[3]) \
            and torch.allclose(result[2] - result[1], (result[4] - result[1]) / self.features[4]) \
            and torch.allclose(result[2] - result[1], (result[5] - result[1]) / self.features[5]), 
            "The numerical encoding is not valid"
        )


if __name__ == '__main__':
    unittest.main()