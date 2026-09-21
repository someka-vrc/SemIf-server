import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from semif_phase1.server import Engine, RequestError, build_answers, build_plan, confidence, make_handler

STATE = "Help! My payouts have been failing for 3 days."


def request_body(**questions):
    return {"state": STATE, "model": "jev-latest", "questions": questions}


NOUL = {"type": "noul", "instructions": "Does this convey urgency?"}
CHOICE = {"type": "choice", "instructions": "Which team?",
          "criteria": {"billing": "Payments", "technical": None, "sales": {"scope": "pricing"}}}
SCORE = {"type": "score", "instructions": "How frustrated?", "criteria": ["Calm", "Frustrated", "Very angry"]}


def fake_result(row, probabilities):
    return {"id": row["id"], "probabilities": probabilities, "input_tokens": 10}


def fake_scorers():
    calls = []

    def direct(model, tokenizer, row, metadata, max_tokens):
        calls.append(("direct", row["id"]))
        return fake_result(row, [0.9] + [0.1 / (len(row["options"]) - 1)] * (len(row["options"]) - 1))

    def shared(model, tokenizer, rows, metadata, max_tokens):
        calls.append(("shared", [row["id"] for row in rows]))
        results = [fake_result(row, [0.9] + [0.1 / (len(row["options"]) - 1)] * (len(row["options"]) - 1))
                   for row in rows]
        return results, {"prefix_tokens": 7, "true_suffix_tokens": 5}

    return direct, shared, calls


def test_plan_maps_each_question_type_to_options():
    rows, plan = build_plan(request_body(u=NOUL, d=CHOICE, f=SCORE))
    by_id = {row["id"]: row for row in rows}
    assert [o["id"] for o in by_id["u"]["options"]] == ["true", "false"]
    assert [o["description"] for o in by_id["d"]["options"]] == [
        "billing: Payments", "technical", 'sales: {"scope": "pricing"}']
    assert [o["id"] for o in by_id["f"]["options"]] == ["0", "1", "2"]
    assert plan["f"]["type"] == "score"


def test_object_instructions_are_serialized():
    question = {"type": "noul", "instructions": {"data": {"a": 1}, "question": "Is `data` big?"}}
    rows, _ = build_plan(request_body(q=question))
    assert json.loads(rows[0]["question"])["question"] == "Is `data` big?"


def test_answers_match_documented_shapes():
    _, plan = build_plan(request_body(u=NOUL, d=CHOICE, f=SCORE))
    answers = build_answers(plan, [
        {"id": "u", "probabilities": [0.95, 0.05]},
        {"id": "d", "probabilities": [0.1, 0.8, 0.1]},
        {"id": "f", "probabilities": [0.0, 0.95, 0.05]},
    ])
    assert answers["u"] == {"type": "noul", "noul": 0.95}
    assert answers["d"]["choice"] == "technical"
    assert answers["d"]["probabilities"] == {"billing": 0.1, "technical": 0.8, "sales": 0.1}
    assert answers["f"]["score"] == pytest.approx(1.05)
    assert answers["f"]["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    assert answers["f"]["probabilities"] == {"0": 0.0, "1": 0.95, "2": 0.05}


def test_confidence_matches_documented_formula():
    assert confidence([1.0, 0.0, 0.0]) == 1.0
    assert confidence([1 / 3] * 3) == pytest.approx(0.0)
    assert confidence([0.9, 0.06, 0.04]) == pytest.approx((3 * 0.9 - 1) / 2)


@pytest.mark.parametrize("body,field", [
    ([], []),
    ({"model": "m", "questions": {"q": NOUL}}, ["state"]),
    ({"state": "s", "questions": {"q": NOUL}}, ["model"]),
    (request_body(), ["questions"]),
    (request_body(q={"type": "rank", "instructions": "x"}), ["questions", "q", "type"]),
    (request_body(q={"type": "noul"}), ["questions", "q", "instructions"]),
    (request_body(q={"type": "choice", "instructions": "x", "criteria": {"only": None}}), ["questions", "q", "criteria"]),
    (request_body(q={"type": "score", "instructions": "x", "criteria": ["one"]}), ["questions", "q", "criteria"]),
    (request_body(q={"type": "choice", "instructions": "x", "criteria": {str(i): None for i in range(17)}}), ["questions", "q", "criteria"]),
    ({"state": "", "model": "m", "questions": {"q": NOUL}}, ["state"]),
])
def test_invalid_requests_name_the_field(body, field):
    with pytest.raises(RequestError) as error:
        build_plan(body)
    assert error.value.status == 422
    assert error.value.body["detail"][0]["loc"] == ["body", *field]


@pytest.fixture
def server():
    direct, shared, calls = fake_scorers()
    engine = Engine(None, None, {"source": "fake"}, direct, shared)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine, "secret"))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", calls
    httpd.shutdown()
    httpd.server_close()


def post(url, body, key="secret"):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url + "/v1/systemone", data, headers)) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def test_single_question_uses_direct_scoring(server):
    url, calls = server
    status, payload = post(url, request_body(u=NOUL))
    assert status == 200 and calls == [("direct", "u")]
    assert payload["answers"]["u"] == {"type": "noul", "noul": 0.9}
    assert payload["usage"] == {"input_tokens": 10, "output_tokens": 0}


def test_multiple_questions_share_one_state_prefill(server):
    url, calls = server
    status, payload = post(url, request_body(u=NOUL, d=CHOICE, f=SCORE))
    assert status == 200 and calls == [("shared", ["u", "d", "f"])]
    assert set(payload["answers"]) == {"u", "d", "f"}
    assert payload["usage"]["input_tokens"] == 12


def test_auth_and_validation_errors(server):
    url, _ = server
    assert post(url, request_body(u=NOUL), key=None)[0] == 401
    assert post(url, request_body(u=NOUL), key="wrong")[0] == 401
    assert post(url, b"not json")[0] == 422
    assert post(url, request_body(u={"type": "noul"}))[0] == 422
