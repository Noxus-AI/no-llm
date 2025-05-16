from __future__ import annotations

from no_llm.providers import EnvVar, VertexProvider


def test_vertex_provider_iter(monkeypatch):
    provider = VertexProvider(
        project_id=EnvVar[str]("$VERTEX_PROJECT_ID"),
        locations=["location1", "location2"],
        model_family="gemini",
    )
    monkeypatch.setenv("VERTEX_PROJECT_ID", "test-project")

    # Test iteration
    providers = list(provider.iter())
    assert len(providers) == 2

    # Check each provider variant
    vertex_providers = [p for p in providers if isinstance(p, VertexProvider)]
    assert vertex_providers[0]._value == "location1"
    assert vertex_providers[1]._value == "location2"

    # Check they're different instances
    assert vertex_providers[0] is not vertex_providers[1]

    # Check other attributes are copied
    for p in vertex_providers:
        project_id = p.project_id
        if isinstance(project_id, EnvVar):
            assert project_id.var_name == "$VERTEX_PROJECT_ID"
        else:
            assert project_id == "$VERTEX_PROJECT_ID"


def test_vertex_provider_current():
    provider = VertexProvider(
        project_id=EnvVar[str]("$VERTEX_PROJECT_ID"),
        locations=["location1", "location2"],
        model_family="gemini",
    )

    # Test default current (first location)
    assert provider.current == "location1"

    # Test after setting _value
    provider._value = "location2"
    assert provider.current == "location2"


def test_vertex_provider_reset_variants():
    provider = VertexProvider(
        project_id=EnvVar[str]("$VERTEX_PROJECT_ID"),
        locations=["location1", "location2"],
        model_family="gemini",
    )

    provider._value = "location2"
    provider.reset_variants()
    assert provider._value is None
    assert provider.current == "location1"  # Should return first location after reset
