from abc import ABC, abstractmethod
from typing import Dict, List

import torch


class AbstractModelWrapper(ABC):
    @abstractmethod
    def encode_queries(self, queries: List[str], *args, **kwargs) -> torch.Tensor:
        """
        Encode a list of queries into embeddings.
        """
        pass

    @abstractmethod
    def encode_corpus(
        self, corpus: List[Dict[str, str]], *args, **kwargs
    ) -> torch.Tensor:
        """
        Encode a list of documents into embeddings.
        """
        pass
