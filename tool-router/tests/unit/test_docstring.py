from toolrouter.docstring import parse_docstring


def test_empty_and_none() -> None:
    assert parse_docstring(None).summary == ""
    assert parse_docstring("   ").params == {}


def test_summary_only() -> None:
    doc = parse_docstring("Current weather for a city.")
    assert doc.summary == "Current weather for a city."
    assert doc.params == {}


def test_multiline_summary_collapses_lines_but_keeps_paragraphs() -> None:
    doc = parse_docstring(
        """
        First line
        continues here.

        Second paragraph.
        """
    )
    assert doc.summary == "First line continues here.\n\nSecond paragraph."


def test_google_style_args() -> None:
    doc = parse_docstring(
        """Do a thing.

        Args:
            city: Name of the city.
            units (str): Temperature units,
                either c or f.
            *rest: ignored star prefix.

        Returns:
            A report. count: not a param.
        """
    )
    assert doc.summary == "Do a thing."
    assert doc.params == {
        "city": "Name of the city.",
        "units": "Temperature units, either c or f.",
        "rest": "ignored star prefix.",
    }


def test_summary_stops_at_first_section() -> None:
    doc = parse_docstring("Summary.\n\nReturns:\n    x: y\n")
    assert doc.summary == "Summary."
    assert doc.params == {}


def test_params_section_aliases() -> None:
    assert parse_docstring("S.\n\nParameters:\n    a: one\n").params == {"a": "one"}
