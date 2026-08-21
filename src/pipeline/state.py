"""LangGraph shared state for the Spiritual AI pipeline."""

from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class SpiritualState(TypedDict):
    messages:    Annotated[Sequence[BaseMessage], add_messages]
    context:     str          # retrieved passages for current question
    sources:     list[str]    # citation strings e.g. "Bhagavad Gita Ch.2 V.47"
    is_relevant: bool         # set by guard node
    intent:      str          # "greeting" | "spiritual" | "irrelevant"
    query:       str          # rewritten standalone search query
