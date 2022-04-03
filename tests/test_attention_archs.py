import unittest
import torch
import torch.nn.functional as F

from ndsl.module.aggregator import ConcatenateAggregator, SumAggregator
from ndsl.module.encoder import CategoricalOneHotEncoder, NumericalEncoder
from ndsl.module.preprocessor import IdentityPreprocessor

from ndsl.architecture.attention import TabularTransformer, MixtureModels

from ndsl.loss.entropy import CELWithWeightsEntropyMinimization


class TestAggregator(unittest.TestCase):

    def test_concatenate_aggregator(self):
        agg = ConcatenateAggregator(50)
        # Parameter in format (B, S, E)
        result = agg(torch.rand((2, 10, 5)))
        self.assertEqual(
            result.size(), 
            torch.Size([2, 50]), 
            "Output shape should be (2, 50)"
        )

    def test_sum_aggregator(self):

        agg = SumAggregator(5)
        # Parameter in format (B, S, E)
        result = agg(torch.rand((2, 10, 5)))

        self.assertEqual(
            result.size(), 
            torch.Size([2, 5]), 
            "Output shape should be (2, 5)"
        )


class TestEncoder(unittest.TestCase):

    def test_oh_encoder(self):
        enc = CategoricalOneHotEncoder(10, 5)
        # Parameter in format (B, S, E)
        result = enc(torch.randint(0, 5, (2, 1)))
        self.assertEqual(
            result.size(), 
            torch.Size([2, 10]), 
            "Output shape should be (2, 10)"
        )

    def test_numerical_encoder(self):
        enc = NumericalEncoder(10)
        # Parameter in format (B, S, E)
        result = enc(torch.rand((2, 1)))

        self.assertEqual(
            result.size(), 
            torch.Size([2, 10]), 
            "Output shape should be (2, 10)"
        )

class TestPreprocessor(unittest.TestCase):

    def test_identity_preprocessor(self):
        preprocessor = IdentityPreprocessor()
        input = torch.randint(0, 5, (2, 10))
        result = preprocessor(input)
        self.assertTrue(
            torch.equal(input, result),
            "Output must be equal"
        )


class TestTransformer(unittest.TestCase):

    def test_transformer(self):
        trans = TabularTransformer(
            n_head=2, # Number of heads per layer
            n_hid=128, # Size of the MLP inside each transformer encoder layer
            n_layers=2, # Number of transformer encoder layers    
            n_output=5, # The number of output neurons
            encoders=torch.nn.ModuleList([
                NumericalEncoder(10),
                CategoricalOneHotEncoder(10, 4),
                NumericalEncoder(10),
            ]), # List of features encoders
            dropout=0.1, # Used dropout
            aggregator=None,
            preprocessor=IdentityPreprocessor()
        )

        input = torch.tensor([
            [0.5, 1, 0.7],
            [0.5, 3, 0.8]
        ])

        result = trans(input)

        self.assertEqual(
            result.size(), 
            torch.Size([2, 5]), 
            "Output shape should be (2, 10)"
        )

    def test_encoders_size(self):
        error = False

        trans = TabularTransformer(
            n_head=2, # Number of heads per layer
            n_hid=128, # Size of the MLP inside each transformer encoder layer
            n_layers=2, # Number of transformer encoder layers    
            n_output=5, # The number of output neurons
            encoders=torch.nn.ModuleList([
                NumericalEncoder(10),
                CategoricalOneHotEncoder(10, 4),
            ]), # List of features encoders
            dropout=0.1, # Used dropout
            aggregator=None,
            preprocessor=IdentityPreprocessor()
        )

        input = torch.tensor([
            [0.5, 1, 0.7],
            [0.5, 3, 0.8]
        ])

        try:
            result = trans(input)
        except ValueError as e:
            if "The number of features must be the same as the number of encoders." in str(e):
                error = True

        self.assertTrue(
            error,
            "Error should be raised because difference between features and encoders"
        )

    def test_attention_single_layer(self):
        error = False

        trans = TabularTransformer(
            n_head=5, # Number of heads per layer
            n_hid=128, # Size of the MLP inside each transformer encoder layer
            n_layers=1, # Number of transformer encoder layers    
            n_output=5, # The number of output neurons
            encoders=torch.nn.ModuleList([
                NumericalEncoder(10),
                CategoricalOneHotEncoder(10, 4),
                NumericalEncoder(10)
            ]), # List of features encoders
            dropout=0.1, # Used dropout
            aggregator=None,
            preprocessor=IdentityPreprocessor(),
            need_weights=True
        )

        input = torch.tensor([
            [0.5, 1, 0.7],
            [0.5, 3, 0.8],
            [0.8, 3, 0.6]
        ])

        result, attn = trans(input)
        self.assertEqual(
            attn.size(),
            # [num_layers, batch, number of heads, number of features, number of features]
            torch.Size([1, 3, 5, 3, 3]),
            f"Attention should be of size [2, 1, 3, 3]. Got {attn.size()} instead"
        )

    def test_attention_multi_layer(self):
        error = False

        trans = TabularTransformer(
            n_head=2, # Number of heads per layer
            n_hid=128, # Size of the MLP inside each transformer encoder layer
            n_layers=3, # Number of transformer encoder layers    
            n_output=5, # The number of output neurons
            encoders=torch.nn.ModuleList([
                NumericalEncoder(10),
                CategoricalOneHotEncoder(10, 4),
                NumericalEncoder(10)
            ]), # List of features encoders
            dropout=0.1, # Used dropout
            aggregator=None,
            preprocessor=IdentityPreprocessor(),
            need_weights=True
        )

        input = torch.tensor([
            [0.5, 1, 0.7],
            [0.5, 3, 0.8]
        ])

        result, attn = trans(input)
        
        self.assertEqual(
            attn.size(),
            # [num_layers, batch, number of heads, number of features, number of features]
            torch.Size([3, 2, 2, 3, 3]),
            f"Attention shoul be of size [2, 2, 3, 3]. Got {attn.size()} instead"
        )

    def test_switch_return_attention(self):
        has_attention = False

        trans = TabularTransformer(
            n_head=2, # Number of heads per layer
            n_hid=128, # Size of the MLP inside each transformer encoder layer
            n_layers=2, # Number of transformer encoder layers    
            n_output=5, # The number of output neurons
            encoders=torch.nn.ModuleList([
                NumericalEncoder(10),
                CategoricalOneHotEncoder(10, 4),
                NumericalEncoder(10)
            ]), # List of features encoders
            dropout=0.1, # Used dropout
            aggregator=None,
            preprocessor=IdentityPreprocessor(),
            need_weights=False
        )

        input = torch.tensor([
            [0.5, 1, 0.7],
            [0.5, 3, 0.8]
        ])

        trans.eval()

        result = trans(input)
        trans.need_weights = True
        result_attn, attn = trans(input)


        # [num_layers, batch, number of heads, number of features, number of features]

        if attn.size() == torch.Size([2, 2, 2, 3, 3]) \
            and torch.equal(result, result_attn):
            has_attention = True

        self.assertTrue(
            has_attention,
            "Attention not switched"
        )

    def test_numerical_passthrough(self):
        error = False

        trans = TabularTransformer(
            n_head=2, # Number of heads per layer
            n_hid=128, # Size of the MLP inside each transformer encoder layer
            n_layers=3, # Number of transformer encoder layers    
            n_output=5, # The number of output neurons
            encoders=torch.nn.ModuleList([
                NumericalEncoder(10),
                CategoricalOneHotEncoder(10, 4),
                NumericalEncoder(10)
            ]), # List of features encoders
            dropout=0.1, # Used dropout
            aggregator=None,
            preprocessor=IdentityPreprocessor(),
            need_weights=True,
            numerical_passthrough=True
        )

        input = torch.tensor([
            [0.5, 1, 0.7],
            [0.5, 3, 0.8]
        ])

        result, attn = trans(input)
        
        self.assertEqual(
            attn.size(),
            # [num_layers, batch, number of heads, number of features, number of features]
            torch.Size([3, 2, 2, 1, 1]),
            f"Attention should be of size [2, 2, 3, 3]. Got {attn.size()} instead"
        )

        self.assertEqual(
            result.size(),
            # [num_layers, batch, number of heads, number of features, number of features]
            torch.Size([2, 5]),
            f"Output should be of size [2, 5]. Got {result.size()} instead"
        )



class TestMixtureModels(unittest.TestCase):

    def test_mixture_models(self):
        trans = MixtureModels(
            n_head=2, # Number of heads per layer
            n_hid=128, # Size of the MLP inside each transformer encoder layer
            n_output=5, # The number of output neurons
            encoders=torch.nn.ModuleList([
                NumericalEncoder(10),
                CategoricalOneHotEncoder(10, 4),
                NumericalEncoder(10),
            ]), # List of features encoders
            n_models=3,
            dropout=0.1, # Used dropout
        )

        input = torch.tensor([
            [0.5, 1, 0.7],
            [0.5, 3, 0.8],
            [0.5, 1, 0.7],
            [0.5, 3, 0.8],
            [0.5, 1, 0.7],
            [0.5, 3, 0.8]
        ])

        result = trans(input)

        self.assertEqual(
            result.size(), 
            torch.Size([6, 5]), 
            "Output shape should be (2, 10)"
        )

class TestLoss(unittest.TestCase):

    def test_cel_weights_entropy_minimization(self):
        criterion = CELWithWeightsEntropyMinimization()

        input = torch.rand((5, 4))
        target = torch.LongTensor([1, 0, 1, 2, 3])
         # [num_layers, batch, number of heads, number of features, number of features]
        weights = torch.rand((3, 5, 8, 4, 4))
        weights = F.softmax(weights, dim=-2)

        loss = criterion(input, target, weights)

        self.assertIsNotNone(loss, "Loss function shouldn't return none")




if __name__ == '__main__':
    unittest.main()