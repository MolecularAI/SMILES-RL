from typing import List, Tuple, Iterable
import numpy as np
import torch
import torch.nn as tnn

from .vocabulary import Vocabulary
from .smiles_tokenizer import SMILESTokenizer


from .rnn import RNN


class DefaultModel:
    """
    Implements an RNN model using SMILES.
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
            print("Network in CUDA", flush=True)
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
    def load_from_file(cls, file_path: str, sampling_mode=False):
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
        model = DefaultModel(
            vocabulary=save_dict["vocabulary"],
            tokenizer=save_dict.get("tokenizer", SMILESTokenizer()),
            network_params=network_params,
            max_sequence_length=save_dict["max_sequence_length"],
        )
        model.network.load_state_dict(save_dict["network"])

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

    def smiles_to_sequences(self, smiles):
        tokens = [self.tokenizer.tokenize(smile) for smile in smiles]
        encoded = [self.vocabulary.encode(token) for token in tokens]
        sequences = [torch.tensor(encode, dtype=torch.long) for encode in encoded]

        def collate_fn(encoded_seqs):
            """Function to take a list of encoded sequences and turn them into a batch"""
            max_length = max([seq.size(0) for seq in encoded_seqs])
            collated_arr = torch.zeros(
                len(encoded_seqs), max_length, dtype=torch.long
            )  # padded with zeroes
            for i, seq in enumerate(encoded_seqs):
                collated_arr[i, : seq.size(0)] = seq
            return collated_arr

        padded_sequences = collate_fn(sequences)

        return padded_sequences.to("cuda")

    def likelihood_smiles(self, smiles: Iterable) -> torch.Tensor:
        """
        Retrieves the (negative log) likelihood of a given Iterable of SMILES.

        :param sequences: (batch_size) A batch of SMILES
        :return:  (batch_size) negative Log likelihood for each example.
        """

        tokens = [self.tokenizer.tokenize(smile) for smile in smiles]
        encoded = [self.vocabulary.encode(token) for token in tokens]
        sequences = [torch.tensor(encode, dtype=torch.long) for encode in encoded]

        def collate_fn(encoded_seqs):
            """Function to take a list of encoded sequences and turn them into a batch"""
            max_length = max([seq.size(0) for seq in encoded_seqs])
            collated_arr = torch.zeros(
                len(encoded_seqs), max_length, dtype=torch.long
            )  # padded with zeroes
            for i, seq in enumerate(encoded_seqs):
                collated_arr[i, : seq.size(0)] = seq
            return collated_arr

        padded_sequences = collate_fn(sequences)
        return self.likelihood(padded_sequences)

    def likelihood(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Retrieves the likelihood of a given sequence. Used in training.

        :param sequences: (batch_size, sequence_length) A batch of sequences
        :return:  (batch_size) Log likelihood for each example.
        """
        logits, _ = self.network(sequences[:, :-1])  # all steps done at once
        log_probs = logits.log_softmax(dim=2)
        return self._nll_loss(log_probs.transpose(1, 2), sequences[:, 1:]).sum(dim=1)

    def sample_smiles(self, num=128, batch_size=128) -> Tuple[List, np.ndarray]:
        """
        Samples n SMILES from the model.
        :param num: Number of SMILES to sample.
        :param batch_size: Number of sequences to sample at the same time.
        :return:
            :smiles: (n) A list with SMILES.
            :likelihoods: (n) A list of likelihoods.
        """
        batch_sizes = [batch_size for _ in range(num // batch_size)] + [
            num % batch_size
        ]
        smiles_sampled = []
        likelihoods_sampled = []

        for size in batch_sizes:
            if not size:
                break
            seqs, likelihoods = self._sample(batch_size=size)
            smiles = [
                self.tokenizer.untokenize(self.vocabulary.decode(seq))
                for seq in seqs.cpu().numpy()
            ]

            smiles_sampled.extend(smiles)
            likelihoods_sampled.append(likelihoods.data.cpu().numpy())

            del seqs, likelihoods
        return smiles_sampled, np.concatenate(likelihoods_sampled)

    def sample_sequences_and_smiles(
        self, batch_size=128
    ) -> Tuple[torch.Tensor, List, torch.Tensor]:
        seqs, likelihoods = self._sample(batch_size=batch_size)
        smiles = [
            self.tokenizer.untokenize(self.vocabulary.decode(seq))
            for seq in seqs.cpu().numpy()
        ]
        return seqs, smiles, likelihoods

    # @torch.no_grad()
    def _sample(self, batch_size=128) -> Tuple[torch.Tensor, torch.Tensor]:
        start_token = torch.zeros(batch_size, dtype=torch.long)
        start_token[:] = self.vocabulary["^"]
        input_vector = start_token
        sequences = [
            self.vocabulary["^"] * torch.ones([batch_size, 1], dtype=torch.long)
        ]
        # NOTE: The first token never gets added in the loop so the sequences are initialized with a start token
        hidden_state = None
        nlls = torch.zeros(batch_size)
        for _ in range(self.max_sequence_length - 1):
            logits, hidden_state = self.network(input_vector.unsqueeze(1), hidden_state)
            logits = logits.squeeze(1)
            probabilities = logits.softmax(dim=1)
            log_probs = logits.log_softmax(dim=1)
            input_vector = torch.multinomial(probabilities, 1).view(-1)
            sequences.append(input_vector.view(-1, 1))
            nlls += self._nll_loss(log_probs, input_vector)
            if input_vector.sum() == 0:
                break

        sequences = torch.cat(sequences, 1)
        return sequences.data, nlls

    def log_and_probabilities(self, sequences: torch.Tensor):
        """
        Retrieves the log probabilities and probabilities of all actions given sequence.

        :param sequences: (batch_size, sequence_length) A batch of sequences
        :return:  (batch_size, sequence_length-1, num_actions) Log probabilities for action in sequence.
                (batch_size, sequence_length-1, num_actions) Probabilities for action in sequence.
        """

        # Excluding last token for consistency with likelihoods
        seqs = sequences[:, :-1]

        logits, _ = self.network(seqs, None)  # all steps done at once

        log_probs = logits.log_softmax(dim=-1)
        probs = logits.softmax(dim=-1)

        assert log_probs.size() == (
            sequences.size(0),
            sequences.size(1) - 1,
            len(self.vocabulary),
        ), f"log probs {log_probs.size()}, correct {(sequences.size(0),sequences.size(1),len(self.vocabulary))}"

        return log_probs, probs

    def get_network_parameters(self):
        return self.network.parameters()

    def save_to_file(self, path: str):
        self.save(path)

    def sample(self, batch_size: int):
        return self.sample_sequences_and_smiles(batch_size)

    def get_vocabulary(self):
        return self.vocabulary
