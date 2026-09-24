import json

from contract_extractor.inference.guards import close_truncated_json, is_looping


def test_is_looping_detects_short_cycles():
    assert is_looping([1, 2, 3] + [7] * 60, min_tokens=48, max_period=64)
    assert is_looping(list(range(10)) + [4, 5, 6] * 20, min_tokens=48, max_period=64)
    assert not is_looping(list(range(100)), min_tokens=48, max_period=64)
    assert not is_looping([7] * 20, min_tokens=48, max_period=64)


def test_close_truncated_json_keeps_complete_items():
    raw = '{"parties": ["Acme Corp.", "Beta LLC"], "audit_rights": ["The Licensee shall keep rec'
    closed = close_truncated_json(raw)
    assert json.loads(closed) == {"parties": ["Acme Corp.", "Beta LLC"]}


def test_close_truncated_json_inside_second_list():
    raw = '{"parties": ["Acme"], "insurance": ["Each party shall maintain", "Covera'
    assert json.loads(close_truncated_json(raw)) == {"parties": ["Acme"], "insurance": ["Each party shall maintain"]}


def test_close_truncated_json_handles_escapes_and_complete_input():
    raw = '{"document_name": ["A \\"quoted\\" title"], "parties": ["x\\n'
    assert json.loads(close_truncated_json(raw)) == {"document_name": ['A "quoted" title']}
    assert close_truncated_json('{"parties": []}') == '{"parties": []}'
    assert json.loads(close_truncated_json('{"parti')) == {}
