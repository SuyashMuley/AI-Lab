from hypothesis import given
from hypothesis import strategies as st

from tests.fuzz import fuzz
from toolrouter.results import truncate


@fuzz(500)
@given(st.text(max_size=3000), st.integers(min_value=1, max_value=4000))
def test_truncate_invariants(text: str, limit: int) -> None:
    out, was_truncated = truncate(text, limit)
    assert len(out) <= limit  # the cap always holds, marker included
    if len(text) <= limit:
        assert (out, was_truncated) == (text, False)
    else:
        assert was_truncated
        marker = out.find("\n[truncated ")
        kept = out if marker == -1 else out[:marker]
        assert text.startswith(kept)  # what survives is a true prefix
        if marker != -1:
            dropped = int(out[marker:].split("[truncated ")[1].split(" chars]")[0])
            assert len(kept) + dropped == len(text)
