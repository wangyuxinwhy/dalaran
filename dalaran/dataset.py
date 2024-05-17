from typing import Callable, Iterable, Sequence

from torch.utils.data import Dataset

from dalaran.datastructures import SupervisedFineTuningModelInput
from dalaran.message import Message
from dalaran.template import Template

EncodeFunction = Callable[[str], list[int]]


class InstructDataset(Dataset):
    def __init__(
        self,
        encode_function: EncodeFunction,
        records: Iterable[Sequence[Message]],
        template: Template,
        max_length: int | None = None,
    ) -> None:
        self.encode_function = encode_function
        self.records = list(records)
        self.template = template
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> SupervisedFineTuningModelInput:
        messages = self.records[idx]

        tokens: list[int] = []
        target_masks: list[bool] = []

        for message in messages:
            text = self.template.format_message(message)
            message_tokens = self.encode_function(text)
            tokens.extend(message_tokens)
            match message.role:
                case "system" | "user":
                    target_masks.extend([False] * len(message_tokens))
                case "assistant":
                    target_masks.extend([True] * len(message_tokens))
        labels = [
            token if target_mask else -100
            for token, target_mask in zip(tokens, target_masks)
        ]
        if self.max_length is not None:
            tokens = tokens[: self.max_length]
            labels = labels[: self.max_length]
        return SupervisedFineTuningModelInput(tokens=tokens, labels=labels)
