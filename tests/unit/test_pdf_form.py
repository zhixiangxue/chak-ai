"""Offline tests for the Pdf tool's form support (metadata/schema/fill).

The fixture form is generated programmatically with PyMuPDF so no network or
binary asset is needed. The radio group is assembled through low-level xref
surgery because PyMuPDF's add_widget cannot create radio groups (it expects an
existing Parent/Kids structure).
"""

import json

import pytest

from chak.tools.std import pdf as pdf_module
from chak.tools.std.pdf import Pdf

pytestmark = pytest.mark.unit

pymupdf = pytest.importorskip("pymupdf")
pytest.importorskip("PyPDFForm")

_RADIO_FLAG = 1 << 15  # /Ff bit 16: Radio


def _add_radio_group(doc, page, name, tooltip, states, y):
    """Build a radio group from checkboxes rewritten into /Kids of a parent."""
    kid_names = [f"__kid_{name}_{i}" for i in range(len(states))]
    for i, kid in enumerate(kid_names):
        w = pymupdf.Widget()
        w.field_name = kid
        w.field_type = pymupdf.PDF_WIDGET_TYPE_CHECKBOX
        w.rect = pymupdf.Rect(150 + i * 60, y, 165 + i * 60, y + 15)
        page.add_widget(w)
    kid_xrefs = [w.xref for w in page.widgets() if w.field_name in kid_names]

    parent = doc.get_new_xref()
    kids = " ".join(f"{x} 0 R" for x in kid_xrefs)
    doc.update_object(
        parent,
        f"<< /FT /Btn /T ({name}) /TU ({tooltip}) /Ff {_RADIO_FLAG} /V /Off "
        f"/Kids [{kids}] >>",
    )
    for xref, state in zip(kid_xrefs, states):
        # Rebuild the kid annot wholesale: nulling keys leaves null objects
        # behind, which pypdf reads back as a NullObject field name.
        _, rect = doc.xref_get_key(xref, "Rect")
        _, on_stream = doc.xref_get_key(xref, "AP/N/Yes")
        _, off_stream = doc.xref_get_key(xref, "AP/N/Off")
        doc.update_object(
            xref,
            f"<< /Type /Annot /Subtype /Widget /Rect {rect} /F 4 /FT /Btn "
            f"/Ff {_RADIO_FLAG} /Parent {parent} 0 R /AS /Off "
            f"/AP << /N << /{state} {on_stream} /Off {off_stream} >> >> >>",
        )

    catalog = doc.pdf_catalog()
    _, fields = doc.xref_get_key(catalog, "AcroForm/Fields")
    for xref in kid_xrefs:
        fields = fields.replace(f"{xref} 0 R", "")
    fields = fields.rstrip("]").rstrip() + f" {parent} 0 R]"
    doc.xref_set_key(catalog, "AcroForm/Fields", " ".join(fields.split()))


@pytest.fixture(scope="module")
def form_pdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("forms") / "fixture_form.pdf"
    doc = pymupdf.open()
    page = doc.new_page()

    # Printed label left of an unlabeled field, for the nearby-text fallback
    page.insert_text((50, 114), "Last Name:", fontsize=10)

    w = pymupdf.Widget()
    w.field_name = "first_name"
    w.field_type = pymupdf.PDF_WIDGET_TYPE_TEXT
    w.field_label = "Borrower first name"
    w.rect = pymupdf.Rect(150, 60, 300, 80)
    page.add_widget(w)

    w = pymupdf.Widget()
    w.field_name = "last_name"
    w.field_type = pymupdf.PDF_WIDGET_TYPE_TEXT
    w.rect = pymupdf.Rect(150, 100, 300, 120)
    w.text_maxlen = 20
    page.add_widget(w)

    w = pymupdf.Widget()
    w.field_name = "us_citizen"
    w.field_type = pymupdf.PDF_WIDGET_TYPE_CHECKBOX
    w.field_label = "Is the borrower a US citizen?"
    w.rect = pymupdf.Rect(150, 140, 165, 155)
    page.add_widget(w)

    w = pymupdf.Widget()
    w.field_name = "loan_purpose"
    w.field_type = pymupdf.PDF_WIDGET_TYPE_COMBOBOX
    w.field_label = "Purpose of the loan"
    w.choice_values = ["Purchase", "Refinance", "Construction"]
    w.rect = pymupdf.Rect(150, 180, 300, 200)
    page.add_widget(w)

    _add_radio_group(
        doc, page, "occupancy", "Occupancy type", ["Primary", "Investment"], 220
    )

    doc.save(str(path))
    doc.close()
    return str(path)


@pytest.fixture(scope="module")
def plain_pdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("forms") / "plain.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 100), "Just some text, no form.", fontsize=11)
    doc.save(str(path))
    doc.close()
    return str(path)


@pytest.fixture()
def fields_by_name(form_pdf):
    payload = json.loads(Pdf().schema(form_pdf))
    return {field["name"]: field for field in payload["fields"]}


# --- metadata ------------------------------------------------------------------


def test_metadata_reports_fillable_form(form_pdf):
    payload = json.loads(Pdf().metadata(form_pdf))
    assert payload["is_fillable_form"] is True
    assert payload["form_field_count"] == 5
    assert payload["form_fields_by_type"] == {
        "text": 2,
        "checkbox": 1,
        "dropdown": 1,
        "radio": 1,
    }
    assert payload["form_pages"] == "1"
    assert "schema" in payload["form_hint"] and "fill" in payload["form_hint"]


def test_metadata_plain_pdf_not_fillable(plain_pdf):
    payload = json.loads(Pdf().metadata(plain_pdf))
    assert payload["is_fillable_form"] is False
    assert "form_hint" not in payload
    assert "form_field_count" not in payload


# --- schema ----------------------------------------------------------------------


def test_schema_field_records(form_pdf, fields_by_name):
    payload = json.loads(Pdf().schema(form_pdf))
    assert payload["total_fields"] == 5

    first = fields_by_name["first_name"]
    assert first["type"] == "text"
    assert first["page"] == 1
    assert first["label"] == "Borrower first name"
    assert "current_value" not in first

    checkbox = fields_by_name["us_citizen"]
    assert checkbox["type"] == "checkbox"
    assert checkbox["label"] == "Is the borrower a US citizen?"


def test_schema_nearby_text_fallback_and_max_length(fields_by_name):
    last = fields_by_name["last_name"]
    assert "label" not in last
    assert "Last Name" in last["nearby_text"]
    assert last["max_length"] == 20


def test_schema_option_mappings(fields_by_name):
    dropdown = fields_by_name["loan_purpose"]
    assert dropdown["options"] == {"0": "Purchase", "1": "Refinance", "2": "Construction"}

    radio = fields_by_name["occupancy"]
    assert radio["type"] == "radio"
    assert radio["label"] == "Occupancy type"
    assert radio["options"] == {"0": "Primary", "1": "Investment"}
    assert "current_value" not in radio


def test_schema_plain_pdf_degrades_gracefully(plain_pdf):
    payload = json.loads(Pdf().schema(plain_pdf))
    assert payload["is_fillable_form"] is False
    assert "read_pages" in payload["hint"]


def test_schema_ambiguous_shared_label_gets_nearby_text(tmp_path):
    # Real-world forms ship copy-paste TU errors (URLA 2021 labels 16
    # declaration radios identically); a label shared by >= 4 fields must be
    # supplemented with a nearby-text hint so the LLM can tell fields apart.
    path = tmp_path / "dup_labels.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    questions = [
        "Do you have judgments?",
        "Are you in bankruptcy?",
        "Any foreclosures?",
        "Party to a lawsuit?",
    ]
    for i, question in enumerate(questions):
        y = 100 + i * 40
        page.insert_text((50, y + 12), question, fontsize=10)
        w = pymupdf.Widget()
        w.field_name = f"q{i}"
        w.field_type = pymupdf.PDF_WIDGET_TYPE_CHECKBOX
        w.field_label = "Copy Paste Error"  # same bogus tooltip on all four
        w.rect = pymupdf.Rect(300, y, 315, y + 15)
        page.add_widget(w)
    doc.save(str(path))
    doc.close()

    fields = {f["name"]: f for f in json.loads(Pdf().schema(str(path)))["fields"]}
    for i, question in enumerate(questions):
        field = fields[f"q{i}"]
        assert field["label"] == "Copy Paste Error"
        # Checkbox falls back to the left-side clip: right side is blank here
        assert question.split()[0] in field["nearby_text"]


def test_schema_unique_label_has_no_nearby_text(fields_by_name):
    # A discriminative tooltip stays authoritative — no hint bloat
    assert "nearby_text" not in fields_by_name["first_name"]


# --- fill ------------------------------------------------------------------------


def test_fill_happy_path_and_current_values(form_pdf, tmp_path):
    out = str(tmp_path / "filled.pdf")
    result = json.loads(
        Pdf().fill(
            form_pdf,
            {
                "first_name": "Jane",
                "last_name": "Doe",
                "us_citizen": True,
                "loan_purpose": "Refinance",  # label resolved to index 1
                "occupancy": 1,
            },
            output_path=out,
        )
    )
    assert result["output_path"] == out
    assert result["filled"] == 5
    assert "errors" not in result

    fields = {f["name"]: f for f in json.loads(Pdf().schema(out))["fields"]}
    assert fields["first_name"]["current_value"] == "Jane"
    assert fields["last_name"]["current_value"] == "Doe"
    assert fields["us_citizen"]["current_value"] is True
    assert fields["loan_purpose"]["current_value"] == "Refinance"
    assert fields["occupancy"]["current_value"] == "Investment"


def test_fill_incremental_second_pass(form_pdf, tmp_path):
    out = str(tmp_path / "step.pdf")
    Pdf().fill(form_pdf, {"first_name": "Jane"}, output_path=out)
    # Second round fills the working copy in place
    result = json.loads(Pdf().fill(out, {"last_name": "Doe"}, output_path=out))
    assert result["filled"] == 1

    fields = {f["name"]: f for f in json.loads(Pdf().schema(out))["fields"]}
    assert fields["first_name"]["current_value"] == "Jane"
    assert fields["last_name"]["current_value"] == "Doe"


def test_fill_rejects_bad_fields_without_blocking_valid_ones(form_pdf, tmp_path):
    out = str(tmp_path / "partial.pdf")
    result = json.loads(
        Pdf().fill(
            form_pdf,
            {
                "first_name": "Jane",
                "no_such_field": "x",
                "loan_purpose": "HELOC",  # not an existing option: rejected
                "occupancy": 5,  # index out of range
            },
            output_path=out,
        )
    )
    assert result["filled"] == 1
    assert set(result["errors"]) == {"no_such_field", "loan_purpose", "occupancy"}
    assert "Purchase" in result["errors"]["loan_purpose"]


def test_fill_all_rejected_returns_no_output(form_pdf):
    result = json.loads(Pdf().fill(form_pdf, {"nope": "x"}))
    assert result["filled"] == 0
    assert "output_path" not in result
    assert "nope" in result["errors"]


def test_fill_accepts_json_string_and_coerces_checkbox(form_pdf, tmp_path):
    out = str(tmp_path / "coerced.pdf")
    result = json.loads(
        Pdf().fill(form_pdf, json.dumps({"us_citizen": "yes"}), output_path=out)
    )
    assert result["filled"] == 1
    fields = {f["name"]: f for f in json.loads(Pdf().schema(out))["fields"]}
    assert fields["us_citizen"]["current_value"] is True


def test_fill_plain_pdf_raises(plain_pdf):
    with pytest.raises(ValueError, match="no fillable form fields"):
        Pdf().fill(plain_pdf, {"a": "b"})


# --- pure helpers ----------------------------------------------------------------


def test_coerce_radio_label_case_insensitive():
    record = {"type": "radio", "options": ["Primary", "Investment"]}
    assert pdf_module._coerce_form_value(record, "investment") == (1, None)
    assert pdf_module._coerce_form_value(record, "0") == (0, None)
    value, reason = pdf_module._coerce_form_value(record, True)
    assert value is None and "index" in reason


def test_decode_pdf_name_hex_escapes():
    # Radio export values are PDF names with #XX hex escapes (URLA 2021 has
    # states like "U.S.#20Citizen"); the LLM must see the readable form.
    assert pdf_module._decode_pdf_name("U.S.#20Citizen") == "U.S. Citizen"
    assert pdf_module._decode_pdf_name("plain") == "plain"


def test_compress_page_ranges():
    assert pdf_module._compress_page_ranges([1, 2, 3, 5, 7, 8]) == "1-3,5,7-8"
    assert pdf_module._compress_page_ranges([4]) == "4"
    assert pdf_module._compress_page_ranges([]) == ""
