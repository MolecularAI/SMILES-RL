import numpy as np
import torch
import torch.nn as tnn

from .vocabulary import Vocabulary
from .smiles_tokenizer import SMILESTokenizer


from .rnn import RNN


class CriticModel:
    """
    Implements an RNN model using SMILES for Actor.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        tokenizer,
        network_params=None,
        max_sequence_length=256,
        no_cuda=False,
    ):
        """
        Implements an RNN.
        :param vocabulary: Vocabulary to use.
        :param tokenizer: Tokenizer to use.
        :param network_params: Dictionary with all parameters required to correctly initialize the RNN class.
        :param max_sequence_length: The max size of SMILES sequence that can be generated.
        """
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

        if not isinstance(network_params, dict):
            network_params = {}

        self.network = RNN(len(self.vocabulary), **network_params)
        if torch.cuda.is_available() and not no_cuda:
            self.network.cuda()

        self._nll_loss = tnn.NLLLoss(reduction="none")

    def set_mode(self, mode: str):
        if mode == "training":
            self.network.train()
        elif mode == "inference":
            self.network.eval()
        else:
            raise ValueError(f"Invalid model mode '{mode}")

    @classmethod
    def load_from_file(cls, file_path: str, sampling_mode: bool = False):
        """
        Loads a model from a single file
        :param file_path: input file path
        :return: new instance of the RNN or an exception if it was not possible to load it.
        """
        if torch.cuda.is_available():
            save_dict = torch.load(file_path)
        else:
            save_dict = torch.load(file_path, map_location=lambda storage, loc: storage)

        network_params = save_dict.get("network_params", {})
        model = CriticModel(
            vocabulary=save_dict["vocabulary"],
            tokenizer=save_dict.get("tokenizer", SMILESTokenizer()),
            network_params=network_params,
            max_sequence_length=save_dict["max_sequence_length"],
        )
        model.network.load_state_dict(save_dict["network"])

        # Change final layer since output of value function should just a single value
        model.network._linear = tnn.Linear(model.network._layer_size, 1)

        if sampling_mode:
            model.set_mode("inference")
        else:
            model.set_mode("training")

        return model

    def save(self, file: str):
        """
        Saves the model into a file
        :param file: it's actually a path
        """
        save_dict = {
            "vocabulary": self.vocabulary,
            "tokenizer": self.tokenizer,
            "max_sequence_length": self.max_sequence_length,
            "network": self.network.state_dict(),
            "network_params": self.network.get_params(),
        }
        torch.save(save_dict, file)

    def values(self, sequences: torch.Tensor):
        """
        Retrieves the action-values of a given sequence.

        :param sequences: (batch_size, sequence_length) A batch of sequences
        :return:  (batch_size, sequence_length-1) action-value for each batch.
        """

        # All steps done at once
        # Excluding last token for consistency with likelihoods
        value, _ = self.network(sequences[:, :-1])

        return value.squeeze(-1)

    def get_network_parameters(self):
        return self.network.parameters()

    def save_to_file(self, path: str):
        self.save(path)

    def get_vocabulary(self):
        return self.vocabulary

    def load_state_dict(self, state_dict: dict):
        self.network.load_state_dict(state_dict)

    def state_dict(
        self,
    ):
        return self.network.state_dict()
