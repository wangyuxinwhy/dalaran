import json
from pathlib import Path
from typing import Protocol, TypeAlias

from pydantic import BaseModel

from dalaran.message import Message


class DataFile(Protocol):
    type: str

    def iter_messages(self) -> list[list[Message]]: ...


class JsonDataFile(BaseModel):
    type: str = "json"
    path: Path

    def iter_messages(self) -> list[list[Message]]:
        return [
            [Message.model_validate(message) for message in messages]
            for messages in json.loads(self.path.read_text())
        ]


class JsonLinesDataFile(BaseModel):
    type: str = "jsonl"
    path: Path

    def iter_messages(self) -> list[list[Message]]:
        return [
            [
                Message.model_validate(json.loads(line))
                for line in self.path.read_text().splitlines()
            ]
        ]


UnionDataFile: TypeAlias = JsonDataFile | JsonLinesDataFile
