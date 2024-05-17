from typing import Dict, Iterable

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from dalaran.datastructures import SupervisedFineTuningModelInput


def padded_collate(
    batch: Iterable[SupervisedFineTuningModelInput],
    padding_idx: int = 0,
    ignore_idx: int = -100,
) -> Dict[str, torch.Tensor]:
    input_ids = pad_sequence(
        [torch.tensor(model_input.tokens) for model_input in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    labels = pad_sequence(
        [torch.tensor(model_input.labels) for model_input in batch],
        batch_first=True,
        padding_value=ignore_idx,
    )

    input_ids_seq_len = input_ids.shape[-1]
    labels_seq_len = labels.shape[-1]

    # Hack to pad correctly and not use max_seq_len, which is costly
    if input_ids_seq_len > labels_seq_len:
        labels = F.pad(
            labels, (0, input_ids_seq_len - labels_seq_len), value=ignore_idx
        )
    elif labels_seq_len > input_ids_seq_len:
        input_ids = F.pad(
            input_ids,
            (0, labels_seq_len - input_ids_seq_len),
            value=padding_idx,
        )
    return {"tokens": input_ids, "labels": labels}
