"""Test doubles shared by several test files: import them with `from fakes import ...`.

This works because tests/ has no __init__.py, so pytest (default "prepend" import mode)
puts the tests/ folder on sys.path. Fixtures belong in conftest.py; plain classes here.
"""

from dataclasses import dataclass


@dataclass
class Call:
    """One structured call made to a FakeModel."""

    schema: type
    kwargs: dict  # e.g. method/strict
    messages: list


class FakeModel:
    """Stands in for the chat model: one fixed answer (or error) per output schema.

        FakeModel(SkillExtraction(...), GapReview(...))           # answers, picked by type
        FakeModel(extraction, errors={GapReview: RuntimeError()})  # the review fails

    A stub (fixed answers) and a spy (records every call in .calls). Duck typing: it does
    not subclass BaseChatModel, it just has the methods our code calls.
    """

    def __init__(self, *responses, errors=None):
        self.responses = {type(response): response for response in responses}
        self.errors = dict(errors or {})
        self.calls: list[Call] = []

    def with_structured_output(self, schema, **kwargs):
        # Like a real model: returns a new runnable bound to this schema
        return _FakeStructuredModel(self, schema, kwargs)

    def call_for(self, schema) -> Call:
        """The first call made with this output schema."""
        return next(call for call in self.calls if call.schema is schema)


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


class FakeResourceFinder:
    """Stands in for ResourceFinder (pool + embedding model): no database needed.

    Records the skills it was called with; returns fixed results, or raises an error.
    """

    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.skills = None

    def __call__(self, skills):
        self.skills = skills
        if self.error:
            raise self.error
        return self.results
