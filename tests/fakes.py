"""Test doubles shared by several test files: import them with `from fakes import ...`.

This works because tests/ has no __init__.py, so pytest (default "prepend" import mode)
puts the tests/ folder on sys.path. Fixtures belong in conftest.py; plain classes here.
"""


class FakeModel:
    """Stands in for the chat model: returns a fixed extraction, or raises an error.

    A stub (fixed answer) and a spy (records how it was called). Duck typing: it does
    not subclass BaseChatModel, it just has the methods our code calls.
    """

    def __init__(self, extraction=None, error=None):
        self.extraction = extraction
        self.error = error
        self.structured_output_kwargs = None
        self.messages = None

    def with_structured_output(self, schema, **kwargs):
        self.structured_output_kwargs = kwargs  # lets tests check method/strict
        return self

    async def ainvoke(self, messages):
        self.messages = messages  # lets tests check what was sent
        if self.error:
            raise self.error
        return self.extraction


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
