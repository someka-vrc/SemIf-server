"""Jev-compatible HTTP server: POST /v1/systemone over the open scorers.

Request and response shapes follow https://docs.typesafe.ai/api. Every question
is turned into a typed option decision and read from native option logits, so
noul, choice, and score answers cost one forward pass and generate no text.
"""

from __future__ import annotations

import argparse
import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .core import LETTERS, validate_row

MODEL_NAME = "semif-direct-v1"
MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_SCORE_LEVELS = 10


class RequestError(Exception):
    """A 422 validation failure that names the offending field."""

    def __init__(self, loc: list, message: str, status: int = 422):
        super().__init__(message)
        self.status = status
        self.body = {"detail": [{"loc": ["body", *loc], "msg": message}]}


def _text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _described(label: str, description) -> str:
    return label if description is None else f"{label}: {_text(description)}"


def build_plan(body) -> tuple[list[dict], dict]:
    """Validate a request body; return decision rows and per-question metadata."""
    if not isinstance(body, dict):
        raise RequestError([], "Request body must be a JSON object")
    for field in ("state", "model", "questions"):
        if field not in body:
            raise RequestError([field], "Field required")
    state, questions = body["state"], body["questions"]
    if not isinstance(body["model"], str) or not body["model"]:
        raise RequestError(["model"], "model must be a nonempty string")
    if not isinstance(questions, dict) or not questions:
        raise RequestError(["questions"], "questions must be a nonempty object")
    rows, plan = [], {}
    for qid, question in questions.items():
        where = ["questions", qid]
        if not qid:
            raise RequestError(["questions"], "Question ids must be nonempty")
        if not isinstance(question, dict):
            raise RequestError(where, "Question must be an object")
        kind, instructions = question.get("type"), question.get("instructions")
        if kind not in ("noul", "choice", "score"):
            raise RequestError([*where, "type"], "type must be one of noul, choice, score")
        if not instructions:
            raise RequestError([*where, "instructions"], "Field required")
        criteria = question.get("criteria")
        if kind == "noul":
            if criteria is not None and not isinstance(criteria, dict):
                raise RequestError([*where, "criteria"], "criteria must be an object with true/false")
            criteria = criteria or {}
            keys = ["true", "false"]
            options = [
                {"id": "true", "description": _text(criteria.get("true") or "Yes: the condition holds.")},
                {"id": "false", "description": _text(criteria.get("false") or "No: the condition does not hold.")},
            ]
        elif kind == "choice":
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise RequestError([*where, "criteria"], "criteria must be an object with at least two options")
            if len(criteria) > len(LETTERS):
                raise RequestError([*where, "criteria"], f"This server supports at most {len(LETTERS)} options")
            keys = list(criteria)
            options = [{"id": key, "description": _described(key, criteria[key])} for key in keys]
        else:
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= MAX_SCORE_LEVELS:
                raise RequestError([*where, "criteria"], f"criteria must be an array of 2-{MAX_SCORE_LEVELS} levels")
            keys = [str(index) for index in range(len(criteria))]
            options = [{"id": key, "description": _text(level)} for key, level in zip(keys, criteria)]
        row = {"id": qid, "state": state, "question": _text(instructions), "options": options}
        try:
            validate_row(row)
        except ValueError as error:
            raise RequestError(["state"] if "state" in str(error) else where, str(error)) from error
        rows.append(row)
        plan[qid] = {"type": kind, "keys": keys, "levels": criteria if kind == "score" else None}
    return rows, plan


def confidence(probabilities: list[float]) -> float:
    count = len(probabilities)
    return max(0.0, min(1.0, (count * max(probabilities) - 1) / (count - 1)))


def build_answers(plan: dict, results: list[dict]) -> dict:
    answers = {}
    for result in results:
        info, probabilities = plan[result["id"]], result["probabilities"]
        distribution = dict(zip(info["keys"], probabilities))
        if info["type"] == "noul":
            answers[result["id"]] = {"type": "noul", "noul": probabilities[0]}
        elif info["type"] == "choice":
            answers[result["id"]] = {
                "type": "choice",
                "choice": info["keys"][probabilities.index(max(probabilities))],
                "probabilities": distribution,
                "confidence": confidence(probabilities),
            }
        else:
            answers[result["id"]] = {
                "type": "score",
                "score": sum(index * value for index, value in enumerate(probabilities)),
                "legend": {key: _text(level) for key, level in zip(info["keys"], info["levels"])},
                "probabilities": distribution,
                "confidence": confidence(probabilities),
            }
    return answers


class Engine:
    """One loaded model; requests are serialized because the GPU holds one model."""

    def __init__(self, model, tokenizer, metadata: dict, direct, shared, max_tokens: int = 4096):
        self.model, self.tokenizer, self.metadata = model, tokenizer, metadata
        self.direct, self.shared, self.max_tokens = direct, shared, max_tokens
        self.lock = threading.Lock()

    def evaluate(self, body) -> dict:
        rows, plan = build_plan(body)
        try:
            with self.lock:
                if len(rows) == 1:
                    results = [self.direct(self.model, self.tokenizer, rows[0], self.metadata, self.max_tokens)]
                    tokens = results[0]["input_tokens"]
                else:
                    # Independent questions over one state: prefill it once, branch per question.
                    results, timing = self.shared(self.model, self.tokenizer, rows, self.metadata, self.max_tokens)
                    tokens = timing["prefix_tokens"] + timing["true_suffix_tokens"]
        except ValueError as error:  # e.g. prompt exceeds max tokens; never truncated
            raise RequestError(["questions"], str(error)) from error
        return {
            "model": MODEL_NAME,
            "answers": build_answers(plan, results),
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }


def warm_up(engine: Engine) -> None:
    """Run both scoring paths once so the first real request pays no CUDA/kernel setup."""
    question = {"type": "noul", "instructions": "Is this a greeting?"}
    for questions in ({"a": question}, {"a": question, "b": question}):
        engine.evaluate({"state": "Hello there.", "model": MODEL_NAME, "questions": questions})


def make_handler(engine: Engine, api_key: str | None):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "model": engine.metadata})
            else:
                self._send(404, {"detail": "Not found"})

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self._send(404, {"detail": "Not found"})
            if api_key is not None:
                supplied = self.headers.get("Authorization", "")
                if not hmac.compare_digest(supplied.encode(), f"Bearer {api_key}".encode()):
                    return self._send(401, {"detail": "Missing or invalid API key"})
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                return self._send(411, {"detail": "Content-Length required"})
            if not 0 < length <= MAX_BODY_BYTES:
                return self._send(413, {"detail": "Request body too large"})
            try:
                body = json.loads(self.rfile.read(length))
                self._send(200, engine.evaluate(body))
            except json.JSONDecodeError:
                self._send(422, RequestError([], "Body is not valid JSON").body)
            except RequestError as error:
                self._send(error.status, error.body)
            except Exception as error:  # keep the server alive; surface the failure
                self._send(500, {"detail": f"{type(error).__name__}: {error}"})

        def log_message(self, fmt, *args):
            pass

    return Handler


def main() -> None:
    import os

    from .core import load_causal_model
    from .direct import score as direct_score
    from .shared import score_shared

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--backend", choices=("torch", "mlx"), default="torch")
    parser.add_argument("--mlx-bits", type=int, choices=(4, 8))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--api-key-env", default="SEMIF_API_KEY",
                        help="Env var holding a bearer key; no auth when it is unset")
    args = parser.parse_args()
    if args.mlx_bits and args.backend != "mlx":
        parser.error("--mlx-bits requires --backend mlx")
    direct, shared = direct_score, score_shared
    if args.backend == "mlx":
        from . import mlx_backend

        model, tokenizer, metadata = mlx_backend.load_model(
            args.model, args.revision, args.mlx_bits, cache_limit_mib=mlx_backend.DEFAULT_CACHE_LIMIT_MIB)
        direct, shared = mlx_backend.score, mlx_backend.score_shared
    else:
        model, tokenizer, metadata = load_causal_model(args.model, args.revision)
    engine = Engine(model, tokenizer, metadata, direct, shared, args.max_tokens)
    print("Warming up...", flush=True)
    warm_up(engine)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine, os.environ.get(args.api_key_env)))
    print(f"Serving POST http://{args.host}:{args.port}/v1/systemone", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
