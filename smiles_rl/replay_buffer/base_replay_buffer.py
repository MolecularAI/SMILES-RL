from abc import ABC, abstractmethod


class BaseReplayBuffer(ABC):
    @abstractmethod
    def __call__(self):
        """Sample, and optionally put, elements from replay buffer

        Raises:
            NotImplementedError: method is required in replay buffer
        """
        raise NotImplementedError("__call__ method is not implemented")

    @abstractmethod
    def __len__(self):
        """Number of elements currently stored in replay buffer

        Raises:
            NotImplementedError: method is required in replay buffer
        """
        raise NotImplementedError("__len__ method is not implemented")
