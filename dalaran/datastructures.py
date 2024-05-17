from pydantic import BaseModel


class SupervisedFineTuningModelInput(BaseModel):
    tokens: list[int]
    labels: list[int]
