from typing import Protocol

from dalaran.message import Message


class Template(Protocol):
    def format_message(self, message: Message) -> str: ...


class FstringTemplate(Template):
    def __init__(
        self,
        system_format_string: str,
        user_format_string: str,
        assistant_format_string: str,
    ) -> None:
        self.system_format_string = system_format_string
        self.user_format_string = user_format_string
        self.assistant_format_string = assistant_format_string

    def format_message(self, message: Message) -> str:
        match message.role:
            case "system":
                return self.system_format_string.format(content=message.content)
            case "user":
                return self.user_format_string.format(content=message.content)
            case "assistant":
                return self.assistant_format_string.format(content=message.content)
