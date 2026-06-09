from __future__ import annotations

from typing import Annotated, Any
import operator
from typing_extensions import TypedDict


class ShoppingState(TypedDict, total=False):
    question: str
    route: dict[str, Any]
    policy_result: dict[str, Any]
    data_result: dict[str, Any]
    final_answer: str
    trace: Annotated[list[dict[str, Any]], operator.add]
