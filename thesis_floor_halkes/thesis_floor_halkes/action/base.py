from abc import ABC, abstractmethod
from typing import Any


class Action(ABC):
    """
    Abstract base class for actions.
    """

    def __init__(self, name: str):
        self.name = name

    def __call__(self) -> Any:
        """
        Call the action.
        """
        return self._get_action()

    @abstractmethod
    def _get_action(self) -> Any:
        """
        Abstract method to get the action.
        """
        pass
