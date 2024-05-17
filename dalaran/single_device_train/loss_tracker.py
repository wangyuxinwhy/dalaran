import torch
from pydantic import BaseModel


class LossRecord(BaseModel):
    average_loss: float
    smoothed_loss: float


class LossTracker:
    """
    A class for tracking and calculating loss values.

    Attributes:
        momentum (float): The momentum value used for smoothing the loss.
        warmup_steps (int): The number of warmup steps before applying momentum.
        history (list[LossRecord]): A list of loss records.

    Args:
        momentum (float): The momentum value used for smoothing the loss (default: 0.9).
        warmup_steps (int): The number of warmup steps before applying momentum (default: 100).
    """

    def __init__(
        self,
        momentum: float = 0.9,
        warmup_steps: int = 100,
    ) -> None:
        self.momentum = momentum
        self.warmup_steps = warmup_steps

        self._loss_sum: float = 0.0
        self._count: int = 0
        self._smoothed_loss: float | None = None
        self.history: list[LossRecord] = []

    def update(self, loss: torch.Tensor | float) -> None:
        """
        Update the loss tracker with a new loss value.

        Args:
            loss (torch.Tensor | float): The loss value to be added to the tracker.
        """
        loss = loss.item() if isinstance(loss, torch.Tensor) else loss
        self._loss_sum += loss
        self._count += 1

        if self._smoothed_loss is None:
            self._smoothed_loss = loss
        else:
            if self._count > self.warmup_steps:
                self._smoothed_loss = (
                    self.momentum * self._smoothed_loss + (1 - self.momentum) * loss
                )
            else:
                weight = 1 - self._count / self.warmup_steps
                weight = min(weight, self.momentum)
                self._smoothed_loss = weight * self._smoothed_loss + (1 - weight) * loss

    def reset(self) -> None:
        """
        Reset the loss tracker to its initial state.
        """
        self._loss_sum = 0
        self._count = 0
        self._smoothed_loss = None

    def on_epoch_end(self, reset: bool = True) -> None:
        """
        Perform actions at the end of an epoch.

        Args:
            reset (bool): Whether to reset the loss tracker after recording the loss history (default: True).
        """
        self.history.append(
            LossRecord(average_loss=self.loss, smoothed_loss=self.smoothed_loss)
        )
        if reset:
            self.reset()

    @property
    def loss(self) -> float:
        """
        Get the average loss value.

        Returns:
            float: The average loss value.

        Raises:
            ValueError: If no loss has been recorded yet.
        """
        if self._count == 0:
            raise ValueError("No loss has been recorded yet")
        return self._loss_sum / self._count

    @property
    def smoothed_loss(self) -> float:
        """
        Get the smoothed loss value.

        Returns:
            float: The smoothed loss value.

        Raises:
            ValueError: If no loss has been recorded yet.
        """
        if self._smoothed_loss is None:
            raise ValueError("No loss has been recorded yet")
        return self._smoothed_loss
