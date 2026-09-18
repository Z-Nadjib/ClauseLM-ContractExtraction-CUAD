import json

from contract_extractor.data.bio import LABEL_TO_ID, char_span_ids, decode_spans, token_labels
from contract_extractor.data.chunking import Chunk, chunk_targets, chunk_text
from contract_extractor.data.cuad import Span
from contract_extractor.evaluation.metrics import Counts, count_matches, jaccard, normalize_tokens, span_counts
from contract_extractor.inference.base import ChunkResult, locate, merge_chunk_extractions
from contract_extractor.schema import FIELD_KEYS, empty_extraction, json_schema, parse_output, to_compact_json


def test_schema_has_41_unique_fields():
    assert len(FIELD_KEYS) == 41
    assert len(set(FIELD_KEYS)) == 41
    assert set(json_schema()["properties"]) == set(FIELD_KEYS)


def test_compact_json_round_trip():
    extraction = empty_extraction()
    extraction["parties"] = ["Acme Corp."]
    extraction["governing_law"] = ["laws of the State of Delaware"]
    raw = to_compact_json(extraction)
    assert list(json.loads(raw)) == ["parties", "governing_law"]
    parsed, valid = parse_output(raw)
    assert valid
    assert parsed == extraction


def test_parse_output_rejects_invalid():
    for raw in ['{"parties": "Acme"}', '{"unknown": []}', "{not json", "[]"]:
        parsed, valid = parse_output(raw)
        assert not valid
        assert parsed == empty_extraction()


def test_chunks_cover_text_with_overlap():
    text = " ".join(f"word{i}" for i in range(3000))
    chunks = chunk_text(text, chunk_size=1000, overlap=100)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    for previous, current in zip(chunks[:-1], chunks[1:], strict=True):
        assert current.start < previous.end
        assert current.start > previous.start
    for chunk in chunks:
        assert chunk.text == text[chunk.start : chunk.end]


def test_chunk_targets_clip_boundary_spans():
    text = "a" * 50 + " CLAUSE TEXT HERE " + "b" * 50
    start = text.index("CLAUSE")
    span = Span("audit_rights", start, start + len("CLAUSE TEXT HERE"), "CLAUSE TEXT HERE")
    chunk = chunk_text(text, chunk_size=len(text), overlap=10)[0]
    assert chunk_targets(chunk, [span], 0.5, 300)["audit_rights"] == ["CLAUSE TEXT HERE"]


def test_jaccard_and_matching():
    assert jaccard(normalize_tokens("The Seller, Inc."), normalize_tokens("the seller inc")) == 1.0
    assert count_matches(["alpha beta gamma"], ["alpha beta gamma delta"], 0.5) == 1
    assert count_matches(["alpha"], ["alpha beta gamma delta"], 0.5) == 0
    assert count_matches(["x y", "x y z"], ["x y z"], 0.5) == 1


def test_span_counts():
    gold = empty_extraction()
    gold["parties"] = ["Acme Corp.", "Beta LLC"]
    pred = empty_extraction()
    pred["parties"] = ["Acme Corp.", "Gamma Inc."]
    counts = span_counts(pred, gold)["parties"]
    assert (counts.tp, counts.fp, counts.fn) == (1, 1, 1)
    assert Counts(tp=1, fp=1, fn=1).f1 == 0.5


def test_locate_tolerates_whitespace():
    assert locate("foo bar", "xx foo\n  bar yy") == (3, 12)
    assert locate("absent", "xx foo bar") is None


def test_merge_unions_overlapping_chunk_spans():
    document = "AAAA The licensee shall maintain insurance. BBBB"
    clause_start, clause_end = 5, document.index(" BBBB")
    first = Chunk(index=0, start=0, end=30, text=document[:30])
    second = Chunk(index=1, start=15, end=len(document), text=document[15:])
    results = []
    for chunk in (first, second):
        extraction = empty_extraction()
        lo, hi = max(clause_start, chunk.start), min(clause_end, chunk.end)
        extraction["insurance"] = [document[lo:hi]]
        results.append(ChunkResult(raw="", extraction=extraction, valid=True))
    merged, n_spans, n_grounded = merge_chunk_extractions(document, [first, second], results)
    assert (n_spans, n_grounded) == (2, 2)
    assert merged["insurance"] == [document[clause_start:clause_end]]


def test_bio_round_trip():
    text = "Party A grants a license. Governed by Delaware law."
    license_start = text.index("grants")
    law_start = text.index("Governed")
    spans = [
        Span("license_grant", license_start, license_start + len("grants a license"), "grants a license"),
        Span("governing_law", law_start, law_start + len("Governed by Delaware law"), "Governed by Delaware law"),
    ]
    words, offsets, position = text.split(" "), [], 0
    for word in words:
        offsets.append((position, position + len(word)))
        position += len(word) + 1
    labels = token_labels(offsets, spans, char_span_ids(len(text), spans))
    assert labels[0] == LABEL_TO_ID["O"]
    decoded = decode_spans(offsets, labels)
    assert [(k, text[s:e].rstrip(".")) for k, s, e in decoded] == [
        ("license_grant", "grants a license"),
        ("governing_law", "Governed by Delaware law"),
    ]
