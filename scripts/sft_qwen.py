from functools import partial
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen2ForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from dalaran.collator import padded_collate
from dalaran.dataset import InstructDataset
from dalaran.files import JsonDataFile, UnionDataFile
from dalaran.single_device_train import LossTracker, set_seed
from dalaran.template import FstringTemplate
from dalaran.utils import clear_memory, use_dtype

EndOfText = "<|endoftext|>"
QwenTemplate = FstringTemplate(
    system_format_string="<|im_start|>system\n{content}<|im_end|>\n",
    user_format_string="<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
    assistant_format_string="{content}<|im_end|>\n",
)


def singel_device_train(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    gradient_accumulation_steps: int = 1,
) -> None:
    clear_memory()
    model.train()

    loss_tracker = LossTracker()
    for current_epoch in range(1, epochs + 1):
        progress_bar = tqdm(total=len(dataloader), unit="batch")
        for batch_index, batch in enumerate(dataloader, start=1):
            batch_output = model(**batch)
            loss = batch_output["loss"]
            loss_tracker.update(loss)
            loss = loss / gradient_accumulation_steps
            loss.backward()

            progress_bar.set_description(
                f"Epoch {current_epoch}/{epochs} - loss: {loss_tracker.smoothed_loss:.4f}"
            )
            if batch_index % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        loss_tracker.on_epoch_end()


def main(
    model_dir: Path,
    data_file: UnionDataFile,
    shuffle: bool = True,
    save_dir: Path | None = None,
    batch_size: int = 8,
    use_bfloat16: bool = False,
    lr: float = 5e-5,
    epochs: int = 3,
    gradient_accumulation_steps: int = 1,
    seed: int | None = None,
):
    if seed is not None:
        set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, use_fast=True, trust_remote_code=True
    )
    tokenizer = cast(PreTrainedTokenizerFast, tokenizer)
    dataset = InstructDataset(
        encode_function=partial(tokenizer.encode, add_special_tokens=False),
        records=data_file.iter_messages(),
        template=QwenTemplate,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=shuffle,
        seed=0,
    )
    pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if not isinstance(pad_token_id, int):
        raise ValueError("pad_token_id must be an integer")
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=partial(padded_collate, padding_idx=pad_token_id),
    )

    if use_bfloat16:
        with use_dtype(torch.bfloat16):
            model: Qwen2ForCausalLM = Qwen2ForCausalLM.from_pretrained(model_dir)  # type: ignore
    else:
        model = Qwen2ForCausalLM.from_pretrained(model_dir)  # type: ignore

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    singel_device_train(
        model,
        dataloader,
        optimizer,
        epochs=epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    save_dir = save_dir or (model_dir / "fine_tuned")
    save_dir.mkdir(exist_ok=True, parents=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main(
        model_dir=Path("/Users/wangyuxin/.cache/modelscope/hub/qwen/Qwen-1_8B-Chat"),
        data_file=JsonDataFile(path=Path("../data/resume_records.dalaran.cn.v1.json")),
        save_dir=None,
        batch_size=2,
        use_bfloat16=False,
        lr=2e-5,
        epochs=3,
        gradient_accumulation_steps=4,
        seed=None,
    )
