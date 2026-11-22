import asyncio
import uuid

import pytest
from fakes import TEXT, TOOLS, FakeModel, FakeResourceFinder, search_call, submit_call
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END
from langgraph.runtime import Runtime
from langgraph.types import Command

from jobpilot.ai_service.extraction import SkillExtraction
from jobpilot.ai_service.gap_analysis import CoveredSkill, GapReview, ReviewedGap
from jobpilot.ai_service.graph import (
    AnalysisContext,
    analysis_graph,
    apply_resource_plan,
    compare,
    extract,
    gap_analysis,
    prefetch_resources,
    resource_agent,
    route_after_agent,
    route_after_review,
    run_tools,
    write_letter,
)
from jobpilot.ai_service.resource_agent import (
    LAST_TURN_REMINDER,
    MAX_AGENT_TURNS,
    SEARCH_LIMIT,
)
from jobpilot.ai_service.schemas import AnalyzeResponse, ResourceMatch, SkillGap, SkillResources

STATE = {
    "cv_text": "Python developer",
    "job_title": "Backend Engineer",
    "job_description": "We need Python and Docker",
}


def make_runtime(model=None, finder=None):
    """A Runtime like the one LangGraph passes to the nodes, built by hand."""
    context = AnalysisContext(
        model=model or FakeModel(),
        find_resources=finder or FakeResourceFinder(),
    )
    return Runtime(context=context)


def test_extract_returns_only_the_skill_keys():
    model = FakeModel(SkillExtraction(job_skills=["Python", "Docker"], cv_skills=["Python"]))

    # Nodes are plain async functions: call one directly, no graph needed.
    update = asyncio.run(extract(STATE, make_runtime(model=model)))

    assert update == {"job_skills": ["Python", "Docker"], "cv_skills": ["Python"]}


def test_extract_sends_the_state_texts_to_the_model():
    model = FakeModel(SkillExtraction(job_skills=[], cv_skills=[]))

    asyncio.run(extract(STATE, make_runtime(model=model)))

    user_message = model.call_for(SkillExtraction).messages[-1].content
    assert "Python developer" in user_message
    assert "Backend Engineer" in user_message
    assert "We need Python and Docker" in user_message


def test_compare_builds_the_result_from_the_skill_lists():
    state = {**STATE, "job_skills": ["Python", "Docker"], "cv_skills": ["python"]}

    update = compare(state)

    result = update["result"]
    assert result.matched_skills == ["Python"]
    assert result.missing_skills == ["Docker"]
    assert result.match_score == 0.5


def resource(slug):
    return ResourceMatch(
        title=f"Title {slug}",
        url=f"https://example.com/{slug}",
        description=f"About {slug}.",
        similarity=0.7,
    )


def url(slug):
    return f"https://example.com/{slug}"


# What the fake finder knows, by skill or query
CATALOG = {
    "Docker": [resource("docker"), resource("k8s"), resource("linux")],
    "Docker containers tutorial": [resource("docker-book")],
}
DOCKER_RESOURCES = SkillResources(skill="Docker", resources=[resource("docker")])


def make_result():
    return AnalyzeResponse(
        required_skills=["Python", "Docker"],
        matched_skills=["Python"],
        missing_skills=["Docker"],
        match_score=0.5,
    )


def agent_state(*messages, turns=0):
    """The state inside the agent loop, after the prefetch of the Docker gap."""
    found = [SkillResources(skill="Docker", resources=CATALOG["Docker"])]
    return {
        **STATE,
        "result": make_result(),
        "found_resources": found,
        "messages": [HumanMessage("gaps and results"), *messages],
        "agent_turns": turns,
    }


def test_prefetch_searches_every_gap_and_starts_the_conversation():
    finder = FakeResourceFinder(catalog=CATALOG)

    update = asyncio.run(
        prefetch_resources({**STATE, "result": make_result()}, make_runtime(finder=finder))
    )

    assert finder.calls == [(["Docker"], SEARCH_LIMIT)]
    assert update["found_resources"][0].resources == CATALOG["Docker"]
    system, user = update["messages"]
    assert "https://example.com/k8s" in user.content


def test_prefetch_returns_no_update_when_the_search_fails(caplog):
    finder = FakeResourceFinder(error=RuntimeError("database is down"))

    update = asyncio.run(
        prefetch_resources({**STATE, "result": make_result()}, make_runtime(finder=finder))
    )

    assert update == {}  # no found_resources: the graph skips the agent
    assert "Learning resource search failed" in caplog.text


def test_resource_agent_requires_a_tool_call_and_counts_the_turn():
    answer = search_call("Docker containers tutorial")
    model = FakeModel(turns=[answer])

    update = asyncio.run(resource_agent(agent_state(), make_runtime(model=model)))

    assert update == {"messages": [answer], "agent_turns": 1}
    call = model.call_for(TOOLS)
    assert call.kwargs == {"tool_choice": "required"}
    assert [tool.__name__ for tool in call.tools] == [
        "SearchLearningResources",
        "SubmitLearningPlan",
    ]


def test_resource_agent_must_submit_on_its_last_turn():
    model = FakeModel(turns=[submit_call(Docker=[url("docker")])])

    asyncio.run(resource_agent(agent_state(turns=MAX_AGENT_TURNS - 1), make_runtime(model=model)))

    call = model.call_for(TOOLS)
    # Only the submit tool is offered: the model cannot search any more
    assert [tool.__name__ for tool in call.tools] == ["SubmitLearningPlan"]
    assert call.kwargs == {"tool_choice": "required"}
    assert call.messages[-1].content == LAST_TURN_REMINDER


def test_resource_agent_adds_no_message_when_the_llm_fails(caplog):
    model = FakeModel(turns=[RuntimeError("Groq is down")])

    update = asyncio.run(resource_agent(agent_state(), make_runtime(model=model)))

    assert update == {"agent_turns": 1}
    assert "Resource agent failed" in caplog.text


def test_run_tools_answers_each_search_with_its_results():
    request = search_call("Docker containers tutorial")
    finder = FakeResourceFinder(catalog=CATALOG)

    update = asyncio.run(run_tools(agent_state(request), make_runtime(finder=finder)))

    [message] = update["messages"]
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == request.tool_calls[0]["id"]  # answers this exact call
    assert "https://example.com/docker-book" in message.content
    assert update["found_resources"][0].skill == "Docker containers tutorial"
    assert finder.calls == [(["Docker containers tutorial"], SEARCH_LIMIT)]


def test_run_tools_still_answers_when_the_search_fails():
    finder = FakeResourceFinder(error=RuntimeError("database is down"))

    update = asyncio.run(run_tools(agent_state(search_call("x")), make_runtime(finder=finder)))

    [message] = update["messages"]
    assert message.content.startswith("Search failed")
    assert update["found_resources"] == []


def test_run_tools_answers_a_malformed_call_with_an_error():
    bad_call = AIMessage(
        content="", tool_calls=[{"name": "SearchLearningResources", "args": {}, "id": "c1"}]
    )

    update = asyncio.run(run_tools(agent_state(bad_call), make_runtime()))

    [message] = update["messages"]
    assert message.tool_call_id == "c1"
    assert message.content.startswith("Error")


def test_route_after_agent():
    search = search_call("Docker containers tutorial")
    submit = submit_call(Docker=[url("docker")])

    assert route_after_agent(agent_state(search, turns=1)) == "run_tools"
    assert route_after_agent(agent_state(submit, turns=1)) == "apply_resource_plan"
    # Out of turns, or the LLM failed (no AI message last): finish with what we have
    assert route_after_agent(agent_state(search, turns=MAX_AGENT_TURNS)) == "apply_resource_plan"
    assert route_after_agent(agent_state(turns=1)) == "apply_resource_plan"


def test_apply_resource_plan_puts_the_checked_choice_in_the_result():
    state = agent_state(submit_call(Docker=[url("linux"), url("made-up")]), turns=1)

    update = apply_resource_plan(state)

    [entry] = update["result"].learning_resources
    assert entry.resources == [resource("linux")]  # the invented URL is dropped


def test_apply_resource_plan_falls_back_without_a_plan():
    update = apply_resource_plan(agent_state(turns=1))  # the agent failed

    [entry] = update["result"].learning_resources
    assert entry.resources == [resource("docker")]  # the top prefetch result


DOCKER_IS_A_GAP = GapReview(covered=[], gaps=[ReviewedGap(skill="Docker", priority="must_have")])


def gap_state():
    """The state after compare: the job asks for SQL and Docker, the CV has PostgreSQL."""
    result = AnalyzeResponse(
        required_skills=["Python", "SQL", "Docker"],
        matched_skills=["Python"],
        missing_skills=["SQL", "Docker"],
        match_score=0.33,
    )
    return {**STATE, "cv_skills": ["Python", "PostgreSQL"], "result": result}


def test_gap_analysis_applies_the_review():
    model = FakeModel(
        GapReview(
            covered=[CoveredSkill(skill="SQL", cv_skill="PostgreSQL")],
            gaps=[ReviewedGap(skill="Docker", priority="nice_to_have")],
        )
    )

    update = asyncio.run(gap_analysis(gap_state(), make_runtime(model=model)))

    result = update["result"]
    assert result.matched_skills == ["Python", "SQL"]
    assert result.skill_gaps == [SkillGap(skill="Docker", priority="nice_to_have")]


def test_gap_analysis_keeps_every_gap_as_must_have_when_the_llm_fails(caplog):
    model = FakeModel(errors={GapReview: RuntimeError("Groq is down")})

    update = asyncio.run(gap_analysis(gap_state(), make_runtime(model=model)))

    result = update["result"]
    assert result.missing_skills == ["SQL", "Docker"]  # plain matching is kept
    assert result.skill_gaps == [
        SkillGap(skill="SQL", priority="must_have"),
        SkillGap(skill="Docker", priority="must_have"),
    ]
    assert "Gap review failed" in caplog.text


def new_config():
    """A new thread for every run, so tests never share a paused state."""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def run_graph(model, finder=None, config=None):
    """Run the whole compiled graph, the way /analyze does: it pauses at human_review."""
    context = AnalysisContext(model=model, find_resources=finder or FakeResourceFinder())
    return asyncio.run(analysis_graph.ainvoke(STATE, config or new_config(), context=context))


def resume_graph(config, decision, model, finder=None):
    """Resume a paused run with the user's answer (same thread, a new context)."""
    context = AnalysisContext(model=model, find_resources=finder or FakeResourceFinder())
    return asyncio.run(analysis_graph.ainvoke(Command(resume=decision), config, context=context))


def docker_gap_model(*turns, text=None):
    """A model for a run where Docker is the only gap; `turns` are the agent's answers,
    `text` the cover letter (if the run gets that far).
    """
    return FakeModel(
        SkillExtraction(job_skills=["Python", "Docker"], cv_skills=["python"]),
        DOCKER_IS_A_GAP,
        turns=turns,
        text=text,
    )


def test_graph_runs_all_nodes_when_skills_are_missing():
    model = docker_gap_model(submit_call(Docker=[url("docker")]))
    finder = FakeResourceFinder(catalog=CATALOG)

    final_state = run_graph(model, finder)

    result = final_state["result"]
    assert result.missing_skills == ["Docker"]
    assert result.learning_resources == [DOCKER_RESOURCES]
    assert finder.calls == [(["Docker"], SEARCH_LIMIT)]  # only the prefetch


def test_graph_skips_the_search_when_nothing_is_missing():
    model = FakeModel(SkillExtraction(job_skills=["Python"], cv_skills=["Python"]))
    finder = FakeResourceFinder()

    final_state = run_graph(model, finder)

    assert final_state["result"].match_score == 1.0
    assert finder.skills is None  # the conditional edge went straight to END


def test_graph_keeps_the_result_when_the_search_fails():
    model = docker_gap_model()  # no agent turns: the agent must be skipped
    finder = FakeResourceFinder(error=RuntimeError("database is down"))

    final_state = run_graph(model, finder)

    assert final_state["result"].missing_skills == ["Docker"]
    assert final_state["result"].learning_resources == []


def test_graph_lets_extraction_errors_propagate():
    # The graph does not hide LLM failures: /analyze turns them into a 502
    model = FakeModel(errors={SkillExtraction: RuntimeError("Groq is down")})

    with pytest.raises(RuntimeError, match="Groq is down"):
        run_graph(model)


def test_graph_searches_only_the_gaps_left_after_the_review():
    model = FakeModel(
        SkillExtraction(job_skills=["SQL", "Docker"], cv_skills=["PostgreSQL"]),
        GapReview(
            covered=[CoveredSkill(skill="SQL", cv_skill="PostgreSQL")],
            gaps=[ReviewedGap(skill="Docker", priority="must_have")],
        ),
    )
    finder = FakeResourceFinder()

    final_state = run_graph(model, finder)

    assert final_state["result"].matched_skills == ["SQL"]
    assert finder.skills == ["Docker"]  # no search for the covered SQL


def test_graph_skips_the_search_when_the_review_covers_every_gap():
    model = FakeModel(
        SkillExtraction(job_skills=["SQL"], cv_skills=["PostgreSQL"]),
        GapReview(covered=[CoveredSkill(skill="SQL", cv_skill="PostgreSQL")], gaps=[]),
    )
    finder = FakeResourceFinder()

    final_state = run_graph(model, finder)

    assert final_state["result"].match_score == 1.0
    assert finder.skills is None  # the second conditional edge went straight to END


def test_graph_lets_the_agent_search_again_before_it_submits():
    model = docker_gap_model(
        search_call("Docker containers tutorial"),
        submit_call(Docker=[url("docker-book"), url("docker")]),
    )
    finder = FakeResourceFinder(catalog=CATALOG)

    final_state = run_graph(model, finder)

    assert finder.calls[1] == (["Docker containers tutorial"], SEARCH_LIMIT)
    [entry] = final_state["result"].learning_resources
    # A URL from the agent's own search is accepted
    assert entry.resources == [resource("docker-book"), resource("docker")]
    # The 2nd LLM call saw its search answered: AI tool call, then the ToolMessage
    second_call = model.tool_calls_made()[1]
    assert isinstance(second_call.messages[-1], ToolMessage)


def test_graph_forces_a_submit_when_the_agent_runs_out_of_turns():
    model = docker_gap_model(
        search_call("Docker containers tutorial"),
        search_call("Docker for beginners"),
        submit_call(Docker=[url("docker")]),
    )

    final_state = run_graph(model, FakeResourceFinder(catalog=CATALOG))

    calls = model.tool_calls_made()
    assert len(calls) == MAX_AGENT_TURNS
    assert [len(call.tools) for call in calls] == [2, 2, 1]  # the last turn: submit only
    assert final_state["result"].learning_resources == [DOCKER_RESOURCES]


def test_graph_falls_back_to_the_top_result_when_the_agent_fails():
    model = docker_gap_model(RuntimeError("Groq is down"))

    final_state = run_graph(model, FakeResourceFinder(catalog=CATALOG))

    assert final_state["result"].learning_resources == [DOCKER_RESOURCES]


LETTER = "Dear hiring team,\n\nI build Python services."


def test_graph_pauses_for_review_after_the_analysis():
    model = docker_gap_model(submit_call(Docker=[url("docker")]), text=LETTER)
    config = new_config()

    paused_state = run_graph(model, FakeResourceFinder(catalog=CATALOG), config)

    [pause] = paused_state["__interrupt__"]
    assert pause.value == paused_state["result"]  # the analysis is what the user reviews
    assert analysis_graph.get_state(config).next == ("human_review",)
    assert model.text_calls_made() == []  # no letter before the user's answer


def test_graph_pauses_for_review_when_nothing_is_missing():
    model = FakeModel(SkillExtraction(job_skills=["Python"], cv_skills=["Python"]))
    config = new_config()

    paused_state = run_graph(model, config=config)

    assert "__interrupt__" in paused_state
    assert analysis_graph.get_state(config).next == ("human_review",)


def test_graph_writes_the_letter_when_the_user_asks_for_it():
    model = docker_gap_model(submit_call(Docker=[url("docker")]), text=LETTER)
    finder = FakeResourceFinder(catalog=CATALOG)
    config = new_config()
    run_graph(model, finder, config)
    calls_before_resume = len(model.calls)

    final_state = resume_graph(config, {"write_letter": True}, model, finder)

    assert final_state["cover_letter"] == LETTER
    # The analysis came from the checkpoint: the letter was the only new LLM call
    assert len(model.calls) == calls_before_resume + 1
    assert model.calls[-1].schema is TEXT
    assert finder.calls == [(["Docker"], SEARCH_LIMIT)]  # no search again either
    assert final_state["result"].learning_resources == [DOCKER_RESOURCES]
    assert analysis_graph.get_state(config).next == ()  # the run is finished


def test_graph_ends_without_a_letter_when_the_user_declines():
    model = docker_gap_model(submit_call(Docker=[url("docker")]), text=LETTER)
    config = new_config()
    run_graph(model, FakeResourceFinder(catalog=CATALOG), config)

    final_state = resume_graph(config, {"write_letter": False}, model)

    assert "cover_letter" not in final_state
    assert model.text_calls_made() == []
    assert analysis_graph.get_state(config).next == ()


def test_route_after_review():
    assert route_after_review({**STATE, "letter_requested": True}) == "write_letter"
    assert route_after_review({**STATE, "letter_requested": False}) == END


def letter_state(priority):
    """The state after the analysis: Python matched, Docker a gap of this priority."""
    result = make_result().model_copy(
        update={"skill_gaps": [SkillGap(skill="Docker", priority=priority)]}
    )
    return {**STATE, "result": result}


def test_write_letter_returns_the_checked_letter():
    model = FakeModel(text="Dear hiring team,\n\nI build Python services.")

    update = asyncio.run(write_letter(letter_state("must_have"), make_runtime(model=model)))

    assert update == {"cover_letter": "Dear hiring team,\n\nI build Python services."}
    user_message = model.call_for(TEXT).messages[-1].content
    analysis = user_message.split("<analysis>")[1]
    assert "Matched skills: Python" in analysis
    assert "Docker" not in analysis  # the gaps are never sent


def test_write_letter_gives_none_when_the_letter_fails_its_checks(caplog):
    # A single text is the answer to every call: the rewrite fails the same way
    model = FakeModel(text="Dear [Hiring Manager],\n\nI build Python services.")

    update = asyncio.run(write_letter(letter_state("must_have"), make_runtime(model=model)))

    assert update == {"cover_letter": None}
    assert "Cover letter failed" in caplog.text
    assert "I build Python services" not in caplog.text  # the letter never reaches the logs


def test_write_letter_gives_none_when_the_llm_fails():
    model = FakeModel(text=RuntimeError("Groq is down"))

    update = asyncio.run(write_letter(letter_state("must_have"), make_runtime(model=model)))

    assert update == {"cover_letter": None}


def test_write_letter_keeps_a_rewrite_that_passes_the_checks():
    model = FakeModel(text=["I have not used Docker yet.", "I build Python services."])

    update = asyncio.run(write_letter(letter_state("must_have"), make_runtime(model=model)))

    assert update == {"cover_letter": "I build Python services."}


def test_write_letter_gives_none_when_the_rewrite_also_names_a_missing_skill(caplog):
    model = FakeModel(text=["I have not used Docker yet.", "Docker is new to me."])

    update = asyncio.run(write_letter(letter_state("must_have"), make_runtime(model=model)))

    assert update == {"cover_letter": None}
    assert "does not show: Docker" in caplog.text
