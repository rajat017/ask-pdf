#!/usr/bin/env python3
"""Run an ordered conversational RAG evaluation against /askPDF.

The eval JSON stays simple: each row contains only input.question and reference.
After each response, this runner automatically stores:
- the actual user question,
- the actual generated assistant answer,
- the response's selected_tags.

That state is passed to the next request. The backend rewrite model decides whether
that next question is a follow-up or a standalone/topic-change question.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def trim_history(history: list[dict[str, str]], max_messages: int) -> list[dict[str, str]]:
    if max_messages <= 0:
        return history
    return history[-max_messages:]


def parse_sse(body: str) -> dict[str, Any]:
    answer_parts: list[str] = []
    final_result: dict[str, Any] = {}
    errors: list[str] = []

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if "message" in event:
            answer_parts.append(str(event.get("message") or ""))
        elif isinstance(event.get("result"), dict):
            final_result = event["result"]
        elif "error" in event:
            error = event["error"]
            if isinstance(error, dict):
                errors.append(str(error.get("message") or error))
            else:
                errors.append(str(error))

    result = dict(final_result)
    streamed_answer = "".join(answer_parts).strip()
    if streamed_answer:
        result["answer"] = streamed_answer
    if errors:
        result["error"] = "; ".join(errors)
    return result


def call_ask_pdf(url: str, data: dict[str, Any], timeout: float) -> dict[str, Any]:
    payload = json.dumps({"data": data}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed: {exc}") from exc

    if "application/json" in content_type:
        parsed = json.loads(body)
        if isinstance(parsed, dict) and isinstance(parsed.get("result"), dict):
            return parsed["result"]
        if isinstance(parsed, dict):
            return parsed
        raise RuntimeError("Unexpected JSON response shape")

    return parse_sse(body)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sequential conversational RAG evaluator")
    parser.add_argument("eval_file", type=Path, help="Ordered JSON file with input.question and reference")
    parser.add_argument("--url", default="http://127.0.0.1:3400/askPDF")
    parser.add_argument("--output", type=Path, default=Path("conversation_eval_results.json"))
    parser.add_argument("--history-max-messages", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    cases = json.loads(args.eval_file.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("Eval file must contain a JSON array")

    history: list[dict[str, str]] = []
    previous_tags: list[str] = []
    results: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Case {index} is not an object")
        input_obj = case.get("input") or {}
        question = str(input_obj.get("question") or "").strip()
        reference = case.get("reference", "")
        if not question:
            raise ValueError(f"Case {index} has no input.question")

        request_data: dict[str, Any] = {
            "question": question,
            "history": trim_history(history, args.history_max_messages),
            "previous_tags": list(previous_tags),
        }

        # Preserve optional normal request fields while keeping history out of the eval JSON.
        for key in ("tag", "sourceType", "datum"):
            value = input_obj.get(key)
            if value not in (None, "", []):
                request_data[key] = value

        print(f"[{index}/{len(cases)}] {question}", flush=True)

        try:
            response = call_ask_pdf(args.url, request_data, args.timeout)
            answer = str(response.get("answer") or "").strip()
            selected_tags_raw = response.get("selected_tags")
            selected_tags = (
                [str(tag).strip() for tag in selected_tags_raw if str(tag).strip()]
                if isinstance(selected_tags_raw, list)
                else []
            )

            results.append({
                "input": {"question": question},
                "reference": reference,
                "output": answer,
                "conversation": {
                    "history_messages_sent": len(request_data["history"]),
                    "previous_tags_sent": request_data["previous_tags"],
                    "is_follow_up": bool(response.get("is_follow_up", False)),
                    "standalone_question": response.get("standalone_question", question),
                    "question_rewritten": bool(response.get("question_rewritten", False)),
                    "selected_tags": selected_tags,
                    "tag_strategy": response.get("tag_strategy", ""),
                },
                "metrics": {
                    "rewrite_ms": response.get("rewrite_ms", 0),
                    "total_ms": response.get("total_ms", 0),
                    "retrieval_ms": response.get("retrieval_ms", 0),
                    "rerank_ms": response.get("rerank_ms", 0),
                    "answer_generation_ms": response.get("answer_generation_ms", 0),
                },
                "sources": response.get("sources", []),
                **({"error": response["error"]} if response.get("error") else {}),
            })

            # Store the actual model interaction, never the reference answer.
            if answer:
                history.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ])
                history = trim_history(history, args.history_max_messages)
                previous_tags = selected_tags

        except Exception as exc:  # continue so one failed row does not discard the run
            results.append({
                "input": {"question": question},
                "reference": reference,
                "output": "",
                "error": str(exc),
            })
            print(f"  ERROR: {exc}", file=sys.stderr, flush=True)

        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved {len(results)} results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
