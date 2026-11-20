"""Test doubles shared by several test files: import them with `from fakes import ...`.

This works because tests/ has no __init__.py, so pytest (default "prepend" import mode)
puts the tests/ folder on sys.path. Fixtures belong in conftest.py; plain classes here.
"""

from dataclasses import dataclass, field
from itertools import count

from langchain_core.messages import AIMessage

from jobpilot.ai_service.schemas import SkillResources

# Call.schema for a call made through bind_tools() instead of with_structured_output()
TOOLS = "tools"


@dataclass
class Call:
    """One call made to a FakeModel."""

    schema: type | str  # the output schema, or TOOLS for a tool-calling turn
    kwargs: dict  # e.g. method/strict, or tool_choice
    messages: list
    tools: list = field(default_factory=list)  # the tools bound for this call, if any


class FakeModel:
    """Stands in for the chat model: one fixed answer (or error) per output schema.

        FakeModel(SkillExtraction(...), GapReview(...))           # answers, picked by type
        FakeModel(extraction, errors={GapReview: RuntimeError()})  # the review fails
        FakeModel(turns=[search_call("..."), submit_call(...)])    # tool-calling turns

    For tool calling, `turns` are the AIMessages returned one after the other; an
    exception in the list is raised when its turn comes.

    A stub (fixed answers) and a spy (records every call in .calls). Duck typing: it does
    not subclass BaseChatModel, it just has the methods our code calls.
    """

    def __init__(self, *responses, errors=None, turns=()):
        self.responses = {type(response): response for response in responses}
        self.errors = dict(errors or {})
        self.turns = list(turns)
        self.calls: list[Call] = []

    def with_structured_output(self, schema, **kwargs):
        # Like a real model: returns a new runnable bound to this schema
        return _FakeStructuredModel(self, schema, kwargs)

    def bind_tools(self, tools, **kwargs):
        # Like a real model: returns a new runnable that knows the tools
        return _FakeToolModel(self, tools, kwargs)

    def call_for(self, schema) -> Call:
        """The first call made with this output schema (or TOOLS)."""
        return next(call for call in self.calls if call.schema is schema)

    def tool_calls_made(self) -> list[Call]:
        """Every tool-calling turn, in order."""
        return [call for call in self.calls if call.schema is TOOLS]


class _FakeStructuredModel:
    """What FakeModel.with_structured_output returns: answers for one schema."""

    def __init__(self, fake: FakeModel, schema: type, kwargs: dict):
        self.fake = fake
        self.schema = schema
        self.kwargs = kwargs

    async def ainvoke(self, messages):
        self.fake.calls.append(Call(self.schema, self.kwargs, messages))
        if self.schema in self.fake.errors:
            raise self.fake.errors[self.schema]
        if self.schema not in self.fake.responses:
            # Fail loudly: a test that forgot to set an answer must not pass by accident
            raise LookupError(f"FakeModel has no answer for {self.schema.__name__}")
        return self.fake.responses[self.schema]


class _FakeToolModel:
    """What FakeModel.bind_tools returns: the scripted turns, one per call."""

    def __init__(self, fake: FakeModel, tools: list, kwargs: dict):
        self.fake = fake
        self.tools = tools
        self.kwargs = kwargs

    async def ainvoke(self, messages):
        # list(messages): a copy, because the graph's message list keeps growing
        self.fake.calls.append(Call(TOOLS, self.kwargs, list(messages), self.tools))
        if not self.fake.turns:
            raise LookupError("FakeModel has no more tool-calling turns")
        turn = self.fake.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn


_call_ids = count(1)


def tool_call_message(name: str, args: dict) -> AIMessage:
    """An AIMessage asking to call one tool, as a real model would return it."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": f"call_{next(_call_ids)}"}],
    )


def search_call(query: str) -> AIMessage:
    return tool_call_message("SearchLearningResources", {"query": query})


def submit_call(**urls_per_skill: list[str]) -> AIMessage:
    """submit_call(Docker=[url, ...], GitHub_Actions=[...]): underscores become spaces."""
    selections = [
        {"skill": skill.replace("_", " "), "urls": urls} for skill, urls in urls_per_skill.items()
    ]
    return tool_call_message("SubmitLearningPlan", {"selections": selections})


class FakeResourceFinder:
    """Stands in for ResourceFinder (pool + embedding model): no database needed.

    Two ways to answer:
    - `results`: the same fixed list for every call,
    - `catalog`: {skill or query: [ResourceMatch, ...]}, one SkillResources per skill
      (what the real finder returns), cut to `limit`.
    Records every call (skills, limit) in .calls; .skills is the last call's skills.
    """

    def __init__(self, results=None, error=None, catalog=None):
        self.results = results or []
        self.error = error
        self.catalog = catalog
        self.calls: list[tuple[list[str], int]] = []
        self.skills = None

    def __call__(self, skills, limit=3):
        self.calls.append((skills, limit))
        self.skills = skills
        if self.error:
            raise self.error
        if self.catalog is not None:
            return [
                SkillResources(skill=s, resources=self.catalog.get(s, [])[:limit]) for s in skills
            ]
        return self.results
