from jobpilot.ai_service.matching import compare_skills


def test_skills_match_after_normalization():
    result = compare_skills(job_skills=["Python", "Kubernetes"], cv_skills=["python", "K8s"])

    assert result.matched_skills == ["Python", "Kubernetes"]
    assert result.missing_skills == []
    assert result.match_score == 1.0


def test_missing_skills_and_partial_score():
    result = compare_skills(job_skills=["Python", "Docker"], cv_skills=["Python"])

    assert result.matched_skills == ["Python"]
    assert result.missing_skills == ["Docker"]
    assert result.match_score == 0.5


def test_output_keeps_the_job_postings_names_and_order():
    result = compare_skills(
        job_skills=["Docker", "PostgreSQL", "Python"], cv_skills=["python", "postgres"]
    )

    assert result.required_skills == ["Docker", "PostgreSQL", "Python"]
    assert result.matched_skills == ["PostgreSQL", "Python"]


def test_duplicate_job_skills_count_once():
    result = compare_skills(job_skills=["Python", " python ", "Docker"], cv_skills=["Python"])

    assert result.required_skills == ["Python", "Docker"]
    assert result.match_score == 0.5


def test_blank_skills_are_ignored():
    result = compare_skills(job_skills=["Python", "", "   "], cv_skills=["", "Python"])

    assert result.required_skills == ["Python"]
    assert result.match_score == 1.0


def test_extra_cv_skills_are_not_reported():
    result = compare_skills(job_skills=["Python"], cv_skills=["Python", "Rust"])

    assert result.required_skills == ["Python"]
    assert result.missing_skills == []


def test_score_is_rounded_to_two_decimals():
    result = compare_skills(job_skills=["A", "B", "C"], cv_skills=["A"])

    assert result.match_score == 0.33


def test_no_job_skills_gives_a_zero_score():
    # Decision: nothing to compare is not a match, so 0.0 rather than a misleading 1.0.
    result = compare_skills(job_skills=[], cv_skills=["Python"])

    assert result.required_skills == []
    assert result.match_score == 0.0
