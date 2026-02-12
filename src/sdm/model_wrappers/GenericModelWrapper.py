from functools import partial
from typing import Dict, List

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding
from transformers.modeling_outputs import BaseModelOutput

from sdm.model_wrappers import AbstractModelWrapper
from sdm.model_wrappers.utils import *


class GenericModelWrapper(AbstractModelWrapper):
    def __init__(
        self,
        pretrained_model_name: str,
        batch_size: int = 64,
        pool_type: str = "cls",
        query_instruct: str = "",
    ):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.pool_type = pool_type
        self.batch_size = batch_size
        self.query_instruct = query_instruct

        self.encoder = None
        self.tokenizer = None

    def _lazy_loading(self):
        self.encoder = AutoModel.from_pretrained(
            self.pretrained_model_name, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)

        self.encoder.cuda()
        self.encoder.eval()

    def encode_queries(self, queries: List[str], *args, **kwargs) -> torch.Tensor:
        if self.encoder is None or self.tokenizer is None:
            self._lazy_loading()

        input_texts = ["{}{}".format(self.query_instruct, q) for q in queries]
        return self._do_encode(input_texts)

    def encode_corpus(
        self, corpus: List[Dict[str, str]], *args, **kwargs
    ) -> torch.Tensor:
        input_texts = [construct_document(doc) for doc in corpus]
        return self._do_encode(input_texts)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str]) -> torch.Tensor:
        if self.encoder is None or self.tokenizer is None:
            self._lazy_loading()

        dataset: Dataset = Dataset.from_dict({"contents": input_texts})
        dataset.set_transform(partial(transform_func, self.tokenizer))

        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            collate_fn=data_collator,
            pin_memory=True,
        )

        encoded_embeds = []
        for batch_dict in tqdm(data_loader, desc="Encoding", mininterval=10):
            batch_dict = move_to_cuda(batch_dict)

            with torch.amp.autocast("cuda"):
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                embeds = pool(
                    outputs.last_hidden_state,
                    batch_dict["attention_mask"],
                    self.pool_type,
                )
                encoded_embeds.append(embeds.cpu())

        embeds = torch.vstack(encoded_embeds)
        return torch.nn.functional.normalize(embeds, p=2, dim=1)
