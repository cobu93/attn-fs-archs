import unittest
import torch
from ndsl.module.aggregator import ConcatenateAggregator, SumAggregator, MeanAggregator, MaxAggregator, CLSAggregator, RNNAggregator

class TestAggregators(unittest.TestCase):

    def setUp(self):
        self.sequence = torch.tensor([
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ],
            [
                [-1, -2, -3],
                [-4, -5, -6],
                [-7, -8, -9]
            ]
        ]).float()
        

    def test_concatenate_aggregator(self):
        agg = ConcatenateAggregator(9)
        # Parameter in format (B, S, E)
        
        expected_result = torch.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [-1, -2, -3, -4, -5, -6, -7, -8, -9]
            
        ]).float()

        result = agg(self.sequence)
        self.assertTrue(
            torch.equal(result, expected_result), 
            "Result is not as expected {}".format(result)
        )

    def test_sum_aggregator(self):

        agg = SumAggregator(3)

        expected_result = torch.tensor([
            [12, 15, 18],
            [-12, -15, -18]
            
        ]).float()

        result = agg(self.sequence)

        result = agg(self.sequence)
        self.assertTrue(
            torch.equal(result, expected_result), 
            "Result is not as expected {}".format(result)
        )

    def test_mean_aggregator(self):

        agg = MeanAggregator(3)

        expected_result = torch.tensor([
            [4, 5, 6],
            [-4, -5, -6]
            
        ]).float()

        result = agg(self.sequence)

        self.assertTrue(
            torch.equal(result, expected_result), 
            "Result is not as expected {}".format(result)
        )

    def test_max_aggregator(self):

        agg = MaxAggregator(3)

        expected_result = torch.tensor([
            [7, 8, 9],
            [-1, -2, -3]
            
        ]).float()

        result = agg(self.sequence)

        self.assertTrue(
            torch.equal(result, expected_result), 
            "Result is not as expected {}".format(result)
        )

    def test_cls_aggregator(self):

        agg = CLSAggregator(3)

        expected_result = torch.tensor([
            [1, 2, 3],
            [-1, -2, -3]
            
        ]).float()

        result = agg(self.sequence)

        self.assertTrue(
            torch.equal(result, expected_result), 
            "Result is not as expected {}".format(result)
        )

    def test_rnn_aggregator(self):

        agg = RNNAggregator(output_size=8, cell="LSTM", input_size=3, num_layers=1, dropout=0)

        result = agg(self.sequence)

        self.assertEqual(
            result.size(),
            torch.Size([2, 8]), 
            "Result is not as expected {}".format(result)
        )



if __name__ == '__main__':
    unittest.main()