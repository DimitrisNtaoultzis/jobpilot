from langchain_core.utils.function_calling import convert_to_openai_tool

from jobpilot.ai_service.resource_agent import TOOLS


def test_the_tools_are_described_to_the_model():
    # What bind_tools() sends to the model: a name, a description and a JSON Schema
    search, submit = (convert_to_openai_tool(tool)["function"] for tool in TOOLS)

    assert search["name"] == "SearchLearningResources"
    assert search["parameters"]["required"] == ["query"]
    assert "poor or off-topic" in search["description"]

    assert submit["name"] == "SubmitLearningPlan"
    assert submit["parameters"]["required"] == ["selections"]
