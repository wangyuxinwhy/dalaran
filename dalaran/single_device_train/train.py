import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dalaran.single_device_train.loss_tracker import LossTracker
from dalaran.utils import clear_memory


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
