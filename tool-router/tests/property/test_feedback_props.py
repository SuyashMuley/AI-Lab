"""The feedback helpers run while reporting errors, so they must never raise or blow up in size."""

from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from tests.fakes.sample_tools import ALL, blob
from tests.fuzz import fuzz
from toolrouter.feedback import _params, _type_text, did_you_mean, signature

keywords = st.sampled_from(
    [
        "type", "properties", "required", "items", "prefixItems", "anyOf", "oneOf", "enum",
        "const", "default", "minimum", "maximum", "exclusiveMinimum", "minLength", "maxLength",
        "format", "additionalProperties",
    ]
)  # fmt: skip
scalars = (
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(max_size=8)
)
type_names = st.sampled_from(
    ["string", "integer", "number", "boolean", "null", "array", "object", "weird", 5, None]
)

junk_schemas = st.recursive(
    scalars | type_names,
    lambda children: (
        st.lists(children, max_size=4)
        | st.dictionaries(keywords | st.text(max_size=6), children, max_size=5)
    ),
    max_leaves=40,
)


@fuzz(500)
@given(junk_schemas)
def test_type_text_survives_arbitrary_schemas(schema: Any) -> None:
    text = _type_text(schema, 0)
    assert isinstance(text, str)
    assert len(text) < 20_000  # bounded relative to the (small) generated input


@fuzz(300)
@given(junk_schemas)
def test_params_survive_arbitrary_schemas(schema: Any) -> None:
    if isinstance(schema, dict):
        assert isinstance(_params(schema, 0), str)


def test_every_sample_signature_is_bounded() -> None:
    for t in [*ALL, blob]:
        assert len(signature(t)) <= 300


@fuzz(300)
@given(st.text(max_size=100), st.lists(st.text(max_size=20), max_size=30))
def test_did_you_mean_never_raises_and_is_bounded(name: str, candidates: list[str]) -> None:
    out = did_you_mean(name, candidates)
    assert len(out) <= 3
    assert all(o in candidates for o in out)
    assert out == did_you_mean(name, candidates)  # deterministic


def test_regression_unhashable_required_entries() -> None:
    # Found by fuzzing: `required` entries that are not strings (so not hashable) crashed _params.
    assert _params({"properties": {}, "required": [[]]}, 0) == ""
    assert _params({"properties": {"a": {"type": "string"}}, "required": [[], {}, 5]}, 0) == (
        "a?: string"
    )
