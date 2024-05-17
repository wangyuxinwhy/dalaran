from typing import Protocol, Sequence

from dalaran.datastructures import SupervisedFineTuningModelInput
from dalaran.message import Message
from dalaran.template import Template


class Tokenizer(Protocol):
    pad_token_id: int

    def encode(self, text: str) -> list[int]:
        """
        Given a string, return the a list of token ids.
        """
        ...


def tokenize_messages(
    tokenizer: Tokenizer,
    messages: Sequence[Message],
    template: Template,
    max_length: int | None = None,
) -> SupervisedFineTuningModelInput:
    tokens: list[int] = []
    target_masks: list[bool] = []

    for message in messages:
        text = template.format_message(message)
        message_tokens = tokenizer.encode(text)
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
    if max_length is not None:
        tokens = tokens[:max_length]
        labels = labels[:max_length]
    return SupervisedFineTuningModelInput(tokens=tokens, labels=labels)
