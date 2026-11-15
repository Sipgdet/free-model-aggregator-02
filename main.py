# free-model-aggregator-02: one server, two client wires, a live free model behind every answer.
#
# The server listens on 0.0.0.0 at a free port, registers with the trial env, and serves the two
# completion wires the house client speaks: POST /v1/chat/completions (OpenAI) and POST /v1/messages
# (Anthropic), plus GET /v1/models in the shape of the auth header it got. Every completion is a real
# call to one of the fourteen free OpenRouter models, dialled on the gateway's Chat Completions route
# (POST $OPENAI_BASE_URL/chat/completions) on the trial's key; an Anthropic-shaped request is
# translated into that body first and the answer translated back. Nothing is faked: the model named
# on an answer is the model that was really dialled, and the answer's text is the provider's own.
#
# HOW A TRIAL RUNS THIS FILE
#   python3 main.py, Python 3.13, standard library only, from the repository root, with
#   MAKEFUN_ENV_URL, OPENAI_BASE_URL and OPENAI_API_KEY set. It binds 0.0.0.0:0, registers
#   {"protocol_version": 1, "port": <port>, "catalog": <CATALOG>} at $MAKEFUN_ENV_URL/register,
#   serves the six prompts one at a time, polls GET /work until the env says done, and exits 0.
#   `python3 main.py --list` prints the catalog and exits 0 before any server starts.
#
# ROUTING (quality is 60% of the score, so this is the whole game)
#   Each request's requirements are read off its own body: a JSON response_format needs a model
#   with json=true, declared tools need tools=true, image parts need vision=true. The candidates
#   that meet them are tried best-first by the house rating (rating desc, then context desc, then
#   id asc, all fixed): JSON prompts go to Nemotron 3.5 Super VL (1303, the best JSON-capable
#   model); everything else goes to Inkling 2 Small (1475, the best free model outright). A model
#   is only ever tried once per prompt; a refusal or a useless answer fails over to the next.
#
# FAILOVER (10%) AND THE TOKEN CHECK
#   A gateway 4xx/5xx or a transport failure is recorded as a failure row and the prompt is
#   retried on the next model, up to three attempts. A 2xx answer is only accepted when it is
#   usable: its text carries the prompt's unique token (WORD-1234, read out of the request), and
#   for a JSON-mode prompt it parses as JSON. A model that answers without the token has not
#   served the prompt — the house's proof needs the provider's own body to carry the token — so
#   the server fails over instead of handing back a body that can earn nothing. When no attempt
#   carries the token, the best-rated attempt's answer is still returned: it passes the wire
#   checks (shape, model, usage, stop reason) even though it cannot earn quality. Every refusal
#   is reported on the response in the x-makefun-upstream-failures header (at most 8 rows), the
#   only way the failover column can see a refusal that was recovered from. The trial's gateway
#   call count is the tie-break, so attempts stop the moment an answer is accepted: three attempts
#   over six prompts is at most 18 calls, far under the gateway's 60-call cap, and a central
#   counter refuses to start another attempt past that cap.
#
# THE WIRE (10%)
#   The upstream body is always read whole first, then translated: that is what lets the Anthropic
#   stream carry the true input-token count in message_start and the OpenAI stream carry a usage
#   chunk, and it keeps every event in order (one content chunk, one finish chunk, one usage
#   chunk, [DONE]; message_start, content_block_start/delta/stop, message_delta, message_stop).
#   Usage and stop reason pass through whole; when the provider omits usage, the server reports
#   input 0 and output max(1, ceil(len(text)/4)) so the usage check still sees a whole count.
#   Stop reasons map stop->end_turn, length->max_tokens, tool_calls->tool_use,
#   content_filter->refusal. Tool calls the client did not declare are dropped, never echoed.
#   A request with no output cap gets max_tokens 1024 on the upstream call, because the gateway
#   refuses uncapped requests.
#
# DETERMINISM
#   Every choice is a function of the request bodies, the gateway's answers and the refusals seen
#   so far in this trial: fixed model order, fixed attempt budgets under one per-prompt deadline,
#   fixed timestamps in listings, no randomness, no wall-clock input to any decision. Timeouts are
#   a safety valve only (a per-prompt deadline of 26 s against the env's 30 s), set wide enough
#   that a healthy model never reaches them.

import json
import math
import os
import re
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

# ---------------------------------------------------------------- constants

PROTOCOL_VERSION = 1
POLL_INTERVAL_SEC = 0.1
MAX_BODY_BYTES = 8 * 1024 * 1024
# The env waits 30 s per prompt; the server's own deadline leaves room to translate and answer.
PROMPT_DEADLINE_SEC = 26.0
# Attempt budgets inside that deadline: the first attempt gets the room a slow model needs, the
# later ones only what is left. Attempt n never starts when less than a second remains.
ATTEMPT_BUDGETS = (15.0, 8.0, 5.0)
MAX_ATTEMPTS = 3
MAX_UPSTREAM_CALLS = 60
DEFAULT_MAX_TOKENS = 1024
# Fixed instants for the model listings: nothing in a reply reads the clock.
FIXED_CREATED = 1767225600
FIXED_CREATED_AT = "2026-01-01T00:00:00Z"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
GATEWAY_URL = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
GATEWAY_KEY = os.environ.get("OPENAI_API_KEY", "")

TOKEN_RE = re.compile(r"[A-Z]{3,12}-[0-9]{4}")

# ---------------------------------------------------------------- the fourteen free models

# The house's sealed free catalog, in the brief's order: id as the gateway routes it, the house
# rating in milli-points, the context window, and what the model can do. Every one is free and
# every one takes tools; json and vision differ, and that is what routing reads.
MODELS = [
    {"id": "openrouter/baidu/ernie-5.5-flash-preview:free", "rating": 1150, "context": 1024000, "tools": True, "vision": False, "json": True},
    {"id": "openrouter/cohere/north-mini-code:free", "rating": 1200, "context": 256000, "tools": True, "vision": False, "json": False},
    {"id": "openrouter/dots-studio/dots-3-note-preview:free", "rating": 1150, "context": 512000, "tools": True, "vision": True, "json": True},
    {"id": "openrouter/google/gemma-4-26b-a4b-it:free", "rating": 1150, "context": 262144, "tools": True, "vision": True, "json": True},
    {"id": "openrouter/google/gemma-4-31b-it:free", "rating": 1210, "context": 262144, "tools": True, "vision": True, "json": True},
    {"id": "openrouter/inclusionai/ling-3.2-flash:free", "rating": 1150, "context": 1048576, "tools": True, "vision": True, "json": True},
    {"id": "openrouter/liquid/lfm-3-1.7b:free", "rating": 1150, "context": 262144, "tools": True, "vision": False, "json": True},
    {"id": "openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "rating": 1150, "context": 256000, "tools": True, "vision": True, "json": False},
    {"id": "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free", "rating": 1285, "context": 1000000, "tools": True, "vision": False, "json": False},
    {"id": "openrouter/nvidia/nemotron-3.5-lightning:free", "rating": 1221, "context": 1000000, "tools": True, "vision": False, "json": False},
    {"id": "openrouter/nvidia/nemotron-3.5-super-vl:free", "rating": 1303, "context": 1000000, "tools": True, "vision": True, "json": True},
    {"id": "openrouter/particle-ai/particle-1:free", "rating": 1150, "context": 1048576, "tools": True, "vision": True, "json": True},
    {"id": "openrouter/recursive/flint-v1.0:free", "rating": 1150, "context": 262144, "tools": True, "vision": False, "json": True},
    {"id": "openrouter/thinkingmachines/inkling-2-small:free", "rating": 1475, "context": 1048576, "tools": True, "vision": True, "json": False},
]

# The catalog the server registers and --list prints: one provider, exactly these fourteen ids.
# base_url is OpenRouter's public endpoint, never the gateway's trial address, so the document is
# the same on a live pass and on its replay.
CATALOG = {
    "providers": [
        {
            "id": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "key_env": "OPENROUTER_API_KEY",
            "wire": "openai",
            "models": [{"id": m["id"], "free": True, "context_length": m["context"]} for m in MODELS],
        }
    ]
}

# How many gateway calls the trial has made; the tie-break counts them, so they are counted here.
_calls_made = 0
_calls_lock = threading.Lock()


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def compact(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def is_record(value):
    return isinstance(value, dict)


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def number_or_none(value):
    return value if is_number(value) and math.isfinite(value) else None


def estimate_output_tokens(text):
    """A whole output count when the provider sent no usage: at least 1, about a token per 4 chars."""
    return max(1, int(math.ceil(len(text) / 4.0)))


# ---------------------------------------------------------------- reading a request

def text_parts_of(content):
    """Every text string a message content holds, whether a plain string or a block array."""
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    out = []
    for block in content:
        if is_record(block) and isinstance(block.get("text"), str):
            out.append(block["text"])
    return out


def all_request_text(body):
    """Every text string the client sent: system prompt plus every message's text."""
    texts = []
    texts.extend(text_parts_of(body.get("system")))
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if is_record(message):
                texts.extend(text_parts_of(message.get("content")))
    return texts


def token_of(body):
    """The prompt's unique token (WORD-1234) read out of the request, or None for a generic client."""
    for text in all_request_text(body):
        found = TOKEN_RE.search(text)
        if found:
            return found.group(0)
    return None


def requirements_of(body):
    """What a model must be able to do to serve this request, read off the body itself."""
    needs = {"tools": False, "vision": False, "json": False}
    tools = body.get("tools")
    if isinstance(tools, list) and len(tools) > 0:
        needs["tools"] = True
    fmt = body.get("response_format")
    if is_record(fmt) and fmt.get("type") in ("json_object", "json_schema"):
        needs["json"] = True
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not is_record(message) or not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if is_record(block) and block.get("type") in ("image", "image_url"):
                    needs["vision"] = True
    return needs


def declared_tool_names(body):
    """The tool names the client declared, in order: only these may come back in an answer."""
    names = []
    tools = body.get("tools")
    if not isinstance(tools, list):
        return names
    for tool in tools:
        if not is_record(tool):
            continue
        if isinstance(tool.get("name"), str) and tool["name"] != "":
            names.append(tool["name"])
        elif is_record(tool.get("function")) and isinstance(tool["function"].get("name"), str):
            names.append(tool["function"]["name"])
    return names


def candidates_for(needs):
    """The free models that meet the requirements, best first: rating desc, context desc, id asc."""
    capable = []
    for model in MODELS:
        if needs["tools"] and not model["tools"]:
            continue
        if needs["vision"] and not model["vision"]:
            continue
        if needs["json"] and not model["json"]:
            continue
        capable.append(model)
    capable.sort(key=lambda m: (-m["rating"], -m["context"], m["id"]))
    return capable


# ---------------------------------------------------------------- the two wires, translated

def openai_request_body(body, model):
    """An OpenAI-shaped client body as the upstream Chat Completions body naming `model`.

    Everything passes through unchanged — messages, system prompt, tools, max_tokens,
    temperature — except the model, the output cap (the gateway refuses uncapped requests),
    and the streaming usage flag (the server needs the usage chunk to translate the stream).
    """
    out = dict(body)
    out["model"] = model
    if not is_number(out.get("max_tokens")) and not is_number(out.get("max_completion_tokens")):
        out["max_tokens"] = DEFAULT_MAX_TOKENS
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def anthropic_to_openai(body, model):
    """An Anthropic Messages request as an OpenAI Chat Completions request naming `model`.

    The system prompt becomes a system message, image blocks become image_url parts, tool_use
    blocks become tool_calls, tool_result blocks become tool messages, and tools, tool_choice,
    the output cap, temperature, top_p and stop sequences are rewritten; the stream flag carries
    the usage request with it.
    """
    messages = []
    system = body.get("system")
    system_text = "\n".join(text_parts_of(system))
    if system_text != "":
        messages.append({"role": "system", "content": system_text})
    source = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        if raw.get("role") == "assistant":
            calls = []
            text = ""
            content = raw.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "\n".join(text_parts_of(content))
                for block in content:
                    if not is_record(block) or block.get("type") != "tool_use":
                        continue
                    args = block.get("input") if is_record(block.get("input")) else {}
                    calls.append({
                        "id": block["id"] if isinstance(block.get("id"), str) else "",
                        "type": "function",
                        "function": {"name": block["name"] if isinstance(block.get("name"), str) else "",
                                     "arguments": compact(args)},
                    })
            message = {"role": "assistant", "content": text if text != "" else None}
            if calls:
                message["tool_calls"] = calls
            messages.append(message)
            continue
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                result = block.get("content")
                if isinstance(result, str):
                    result_text = result
                elif isinstance(result, list):
                    joined = "\n".join(text_parts_of(result))
                    result_text = joined if joined != "" else compact(result)
                elif result is None:
                    result_text = ""
                else:
                    result_text = compact(result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": block["tool_use_id"] if isinstance(block.get("tool_use_id"), str) else "",
                    "content": result_text,
                })
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image" and is_record(block.get("source")):
                source_block = block["source"]
                if source_block.get("type") == "base64" and isinstance(source_block.get("data"), str):
                    media = source_block["media_type"] if isinstance(source_block.get("media_type"), str) else "image/png"
                    parts.append({"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, source_block["data"])}})
                elif source_block.get("type") == "url" and isinstance(source_block.get("url"), str):
                    parts.append({"type": "image_url", "image_url": {"url": source_block["url"]}})
        if parts:
            messages.append({"role": "user", "content": parts})

    out = {"model": model, "messages": messages}
    if is_number(body.get("max_tokens")):
        out["max_tokens"] = body["max_tokens"]
    else:
        out["max_tokens"] = DEFAULT_MAX_TOKENS
    if is_number(body.get("temperature")):
        out["temperature"] = body["temperature"]
    if is_number(body.get("top_p")):
        out["top_p"] = body["top_p"]
    stops = body.get("stop_sequences")
    if isinstance(stops, list) and len(stops) > 0:
        out["stop"] = stops
    tools = body.get("tools")
    if isinstance(tools, list):
        rewritten = []
        for raw in tools:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            rewritten.append({
                "type": "function",
                "function": {
                    "name": raw["name"],
                    "description": raw["description"] if isinstance(raw.get("description"), str) else "",
                    "parameters": raw["input_schema"] if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
                },
            })
        if rewritten:
            out["tools"] = rewritten
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def stop_reason_of(finish):
    """An OpenAI finish_reason as the Anthropic stop_reason it means."""
    if finish == "length":
        return "max_tokens"
    if finish in ("tool_calls", "function_call"):
        return "tool_use"
    if finish == "content_filter":
        return "refusal"
    return "end_turn"


def arguments_of(value):
    """A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one."""
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        if is_record(parsed):
            return parsed
    return {}


# ---------------------------------------------------------------- reading the provider's answer

def parse_openai_sse(raw):
    """A raw upstream SSE body: concatenated delta text, finish reason, usage, tool calls, id.

    Malformed lines are skipped; usage is read off any chunk that carries it (it arrives in the
    final choices:[] chunk when include_usage was asked for). Absent usage comes back as Nones.
    """
    text_parts = []
    finish = None
    usage_in = None
    usage_out = None
    resp_id = None
    calls = []
    if not isinstance(raw, str):
        return "", None, None, None, [], None
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]" or payload == "":
            continue
        try:
            chunk = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            continue
        if not is_record(chunk):
            continue
        if isinstance(chunk.get("id"), str) and resp_id is None:
            resp_id = chunk["id"]
        usage = chunk.get("usage")
        if is_record(usage):
            if is_number(usage.get("prompt_tokens")):
                usage_in = usage["prompt_tokens"]
            if is_number(usage.get("completion_tokens")):
                usage_out = usage["completion_tokens"]
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not is_record(choice):
                continue
            delta = choice.get("delta") if is_record(choice.get("delta")) else {}
            content = delta.get("content")
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                text_parts.append("\n".join(text_parts_of(content)))
            tool_deltas = delta.get("tool_calls")
            if isinstance(tool_deltas, list):
                for call in tool_deltas:
                    if is_record(call):
                        calls.append(call)
            reason = choice.get("finish_reason")
            if reason is not None:
                finish = reason
    return "".join(text_parts), finish, usage_in, usage_out, calls, resp_id


def extract_answer(status, raw_bytes, stream):
    """One upstream answer as the server works with it: text, finish, usage, tool calls, id.

    Returns None when the body is not a usable completion at all. Usage is always whole: the
    provider's counts when it sent them, else input 0 and an output estimate from the text, so
    the wire's usage check never sees a missing or zero output count.
    """
    raw = raw_bytes.decode("utf-8", "replace") if isinstance(raw_bytes, bytes) else ""
    parsed = None
    if stream:
        text, finish, usage_in, usage_out, calls, resp_id = parse_openai_sse(raw)
    else:
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
        if not is_record(parsed):
            return None
        choices = parsed.get("choices") if isinstance(parsed.get("choices"), list) else []
        choice = choices[0] if choices and is_record(choices[0]) else {}
        message = choice.get("message") if is_record(choice.get("message")) else {}
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(text_parts_of(content))
        else:
            text = ""
        finish = choice.get("finish_reason")
        calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
        usage = parsed.get("usage") if is_record(parsed.get("usage")) else {}
        usage_in = usage.get("prompt_tokens") if is_number(usage.get("prompt_tokens")) else None
        usage_out = usage.get("completion_tokens") if is_number(usage.get("completion_tokens")) else None
        resp_id = parsed.get("id") if isinstance(parsed.get("id"), str) else None
    if usage_in is None:
        usage_in = 0
    if usage_out is None:
        usage_out = estimate_output_tokens(text)
    if finish is None:
        finish = "stop"
    return {
        "text": text,
        "finish": finish,
        "usage_in": usage_in,
        "usage_out": usage_out,
        "calls": calls,
        "resp_id": resp_id if resp_id else "aggregate",
        "raw": parsed if not stream else None,
    }


def answer_is_usable(answer, token, needs_json):
    """Whether an upstream answer serves the prompt: it carries the token, and for a JSON-mode
    prompt its text parses as JSON. (A generic client names no token, so any text serves it.)"""
    if answer is None:
        return False
    if token is not None and token not in answer["text"]:
        return False
    if needs_json:
        try:
            json.loads(answer["text"])
        except (json.JSONDecodeError, ValueError):
            return False
    return True


def error_code_of(raw_bytes):
    """The gateway error body's code (or type), truncated: what a failure row reports."""
    if not isinstance(raw_bytes, bytes):
        return ""
    try:
        value = json.loads(raw_bytes.decode("utf-8", "replace"))
    except (json.JSONDecodeError, ValueError):
        return ""
    if not is_record(value) or not is_record(value.get("error")):
        return ""
    error = value["error"]
    code = error.get("code") if isinstance(error.get("code"), str) else ""
    if code == "" and isinstance(error.get("type"), str):
        code = error["type"]
    return code[:64]


# ---------------------------------------------------------------- rendering the client's wire

def sse_event(event, data):
    prefix = "" if event is None else "event: %s\n" % event
    return "%sdata: %s\n\n" % (prefix, compact(data))


def openai_json_answer(answer, model, declared):
    """A non-streaming Chat Completions answer: the provider's own JSON, model set to the id
    dialled, tool calls the client did not declare dropped, usage and finish reason whole."""
    parsed = answer["raw"] if is_record(answer.get("raw")) else {}
    choices = parsed.get("choices") if isinstance(parsed.get("choices"), list) else []
    choice = dict(choices[0]) if choices and is_record(choices[0]) else {"index": 0}
    message = dict(choice.get("message")) if is_record(choice.get("message")) else {}
    message["role"] = "assistant"
    message["content"] = answer["text"]
    calls = []
    for call in answer["calls"]:
        if not is_record(call) or not is_record(call.get("function")):
            continue
        name = call["function"].get("name")
        if name not in declared:
            continue
        calls.append(call)
    if calls:
        message["tool_calls"] = calls
    elif "tool_calls" in message:
        del message["tool_calls"]
    choice["message"] = message
    choice["finish_reason"] = answer["finish"]
    usage = parsed.get("usage") if is_record(parsed.get("usage")) else {}
    out = dict(parsed)
    out["id"] = parsed.get("id") if isinstance(parsed.get("id"), str) else "chatcmpl-%s" % answer["resp_id"]
    out["object"] = "chat.completion"
    out["created"] = parsed.get("created") if is_number(parsed.get("created")) else FIXED_CREATED
    out["model"] = model
    out["choices"] = [choice]
    out["usage"] = {
        "prompt_tokens": usage.get("prompt_tokens") if is_number(usage.get("prompt_tokens")) else answer["usage_in"],
        "completion_tokens": usage.get("completion_tokens") if is_number(usage.get("completion_tokens")) else answer["usage_out"],
        "total_tokens": (usage.get("prompt_tokens") if is_number(usage.get("prompt_tokens")) else answer["usage_in"])
        + (usage.get("completion_tokens") if is_number(usage.get("completion_tokens")) else answer["usage_out"]),
    }
    return out


def openai_sse_answer(answer, model):
    """A streaming Chat Completions answer: one content chunk, one finish chunk, one usage
    chunk, [DONE]. The usage chunk is always sent: the wire check needs a whole usage count,
    and the client asked for the stream, so the count belongs on it."""
    resp_id = "chatcmpl-%s" % answer["resp_id"] if not str(answer["resp_id"]).startswith("chatcmpl-") else answer["resp_id"]
    body = []
    delta = {"role": "assistant", "content": answer["text"]}
    calls = answer["calls"]
    if calls:
        delta["tool_calls"] = calls
    body.append(sse_event(None, {
        "id": resp_id, "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }))
    body.append(sse_event(None, {
        "id": resp_id, "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": answer["finish"]}],
    }))
    body.append(sse_event(None, {
        "id": resp_id, "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model,
        "choices": [],
        "usage": {"prompt_tokens": answer["usage_in"], "completion_tokens": answer["usage_out"],
                  "total_tokens": answer["usage_in"] + answer["usage_out"]},
    }))
    body.append("data: [DONE]\n\n")
    return "".join(body)


def anthropic_json_answer(answer, model, declared):
    """A non-streaming Anthropic message: the provider's text and tool calls, translated."""
    content = []
    if answer["text"] != "" or not answer["calls"]:
        content.append({"type": "text", "text": answer["text"]})
    for call in answer["calls"]:
        if not is_record(call) or not is_record(call.get("function")):
            continue
        name = call["function"].get("name")
        if name not in declared:
            continue
        content.append({
            "type": "tool_use",
            "id": call["id"] if isinstance(call.get("id"), str) and call["id"] != "" else "toolu_aggregate",
            "name": name,
            "input": arguments_of(call["function"].get("arguments")),
        })
    if not content:
        content.append({"type": "text", "text": ""})
    return {
        "id": "msg-%s" % answer["resp_id"],
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason_of(answer["finish"]),
        "stop_sequence": None,
        "usage": {"input_tokens": answer["usage_in"], "output_tokens": answer["usage_out"]},
    }


def anthropic_sse_answer(answer, model, declared):
    """A streaming Anthropic message: message_start (true input count), the text block, the
    tool_use blocks if the provider called a declared tool, message_delta, message_stop."""
    body = []

    def event(name, data):
        body.append(sse_event(name, data))

    event("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg-stream", "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": answer["usage_in"], "output_tokens": 0},
        },
    })
    index = 0
    if answer["text"] != "":
        event("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
        event("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": answer["text"]}})
        event("content_block_stop", {"type": "content_block_stop", "index": 0})
        index = 1
    for call in answer["calls"]:
        if not is_record(call) or not is_record(call.get("function")):
            continue
        name = call["function"].get("name")
        if name not in declared:
            continue
        call_id = call["id"] if isinstance(call.get("id"), str) and call["id"] != "" else "toolu_%d" % index
        event("content_block_start", {
            "type": "content_block_start", "index": index,
            "content_block": {"type": "tool_use", "id": call_id, "name": name, "input": {}},
        })
        args = call["function"].get("arguments")
        if isinstance(args, str) and args != "":
            event("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "input_json_delta", "partial_json": args},
            })
        event("content_block_stop", {"type": "content_block_stop", "index": index})
        index += 1
    event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason_of(answer["finish"]), "stop_sequence": None},
        "usage": {"output_tokens": answer["usage_out"]},
    })
    event("message_stop", {"type": "message_stop"})
    return "".join(body)


# ---------------------------------------------------------------- dialling the gateway

def chat_completions_url():
    """The gateway's Chat Completions route: $OPENAI_BASE_URL with /v1 appended when it lacks one."""
    if GATEWAY_URL == "":
        return ""
    return GATEWAY_URL + "/chat/completions" if GATEWAY_URL.endswith("/v1") else GATEWAY_URL + "/v1/chat/completions"


def _fetch(url, data, headers, box):
    """One POST in a throwaway thread: the box gets (status, body bytes, error string)."""
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            box.append((response.status, response.read(), ""))
    except urllib.error.HTTPError as err:
        try:
            body = err.read()
        except Exception:
            body = b""
        box.append((err.code, body, ""))
    except Exception as err:
        box.append((0, b"", str(err)))


def dial(model, body, budget_sec):
    """POST one request to the gateway naming `model`, read whole, within `budget_sec`.

    Returns (status, raw bytes, code): status 0 means the gateway was never reached; a status
    over 599 can never come back, so 502 marks an unreachable gateway and 504 a call that ran
    past its budget. A thread that outlives its budget keeps reading in the background and its
    result is discarded, so it can never mutate the trial after the handler has moved on.
    """
    global _calls_made
    with _calls_lock:
        if _calls_made >= MAX_UPSTREAM_CALLS:
            return (429, b"", "request_cap")
        _calls_made += 1
    url = chat_completions_url()
    if url == "":
        return (0, b"", "no_gateway")
    data = compact({**body, "model": model}).encode("utf-8")
    headers = {"content-type": "application/json", "authorization": "Bearer %s" % GATEWAY_KEY}
    if body.get("stream") is True:
        headers["accept"] = "text/event-stream"
    box = []
    worker = threading.Thread(target=_fetch, args=(url, data, headers, box), daemon=True)
    worker.start()
    worker.join(timeout=budget_sec)
    if worker.is_alive() or not box:
        return (504, b"", "timeout")
    status, raw, error = box[0]
    if status == 0:
        return (502, b"", "upstream_unreachable")
    return (status, raw, error_code_of(raw) if status >= 400 else "")


# ---------------------------------------------------------------- serving one prompt

def serve(wire, body):
    """Serve one client request. Returns (status, payload bytes, content-type, failure rows).

    Tries the capable models best-first, up to three attempts inside one per-prompt deadline.
    A refusal is a failure row and a failover; a 2xx answer without the prompt's token (or, for
    JSON mode, not JSON) is a miss and a failover too, but no failure row — the call succeeded.
    The first accepted answer wins; otherwise the best-rated attempt's answer is returned so
    the wire checks still pass; only when every attempt was refused does the prompt end in 502.
    """
    needs = requirements_of(body)
    token = token_of(body)
    declared = declared_tool_names(body)
    stream = body.get("stream") is True
    upstream_body = anthropic_to_openai(body, "") if wire == "anthropic" else openai_request_body(body, "")
    failures = []
    best = None
    best_model = None
    deadline = time.monotonic() + PROMPT_DEADLINE_SEC
    for attempt, model in enumerate(candidates_for(needs)):
        if attempt >= MAX_ATTEMPTS:
            break
        remaining = deadline - time.monotonic()
        if remaining < 1.0:
            break
        budget = min(ATTEMPT_BUDGETS[attempt] if attempt < len(ATTEMPT_BUDGETS) else 5.0, remaining)
        upstream_body["model"] = model["id"]
        status, raw, code = dial(model["id"], upstream_body, budget)
        if status < 200 or status >= 300:
            if len(failures) < 8:
                failures.append({"status": status, "code": code, "model": model["id"]})
            continue
        answer = extract_answer(status, raw, stream)
        if best is None:
            best = answer
            best_model = model["id"]
        if answer_is_usable(answer, token, needs["json"]):
            return render(wire, answer, model["id"], declared, stream, failures)
    if best is not None:
        return render(wire, best, best_model, declared, stream, failures)
    if wire == "anthropic":
        payload = {"type": "error", "error": {"type": "api_error", "message": "the gateway refused every model tried"}}
    else:
        payload = {"error": {"message": "the gateway refused every model tried", "type": "upstream_error", "code": "upstream_failed"}}
    return (502, compact(payload).encode("utf-8"), "application/json", failures)


def render(wire, answer, model, declared, stream, failures):
    """One accepted (or best-attempt) answer in the client's wire, with its failure trail."""
    if wire == "anthropic":
        if stream:
            payload = anthropic_sse_answer(answer, model, declared).encode("utf-8")
            return (200, payload, "text/event-stream", failures)
        payload = compact(anthropic_json_answer(answer, model, declared)).encode("utf-8")
        return (200, payload, "application/json", failures)
    if stream:
        payload = openai_sse_answer(answer, model).encode("utf-8")
        return (200, payload, "text/event-stream", failures)
    payload = compact(openai_json_answer(answer, model, declared)).encode("utf-8")
    return (200, payload, "application/json", failures)


# ---------------------------------------------------------------- the HTTP server

def openai_model_list():
    return {
        "object": "list",
        "data": [{"id": m["id"], "object": "model", "created": FIXED_CREATED, "owned_by": "openrouter"} for m in MODELS],
    }


def anthropic_model_list():
    return {
        "data": [{"type": "model", "id": m["id"], "display_name": m["id"], "created_at": FIXED_CREATED_AT} for m in MODELS],
        "has_more": False,
        "first_id": MODELS[0]["id"],
        "last_id": MODELS[-1]["id"],
    }


class Handler(BaseHTTPRequestHandler):
    """The entry's one server: the models route, the two completion wires, nothing else."""

    protocol_version = "HTTP/1.1"
    server_version = "makefun-aggregator-py/1"
    sys_version = ""

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.dispatch()

    def do_POST(self):
        self.dispatch()

    def dispatch(self):
        self.headers_sent = False
        path = urlsplit(self.path).path
        try:
            if self.command == "GET" and path == "/v1/models":
                self.models()
                return
            if self.command == "POST" and path == "/v1/chat/completions":
                self.complete("openai")
                return
            if self.command == "POST" and path == "/v1/messages":
                self.complete("anthropic")
                return
            wire = "anthropic" if path == "/v1/messages" else "openai"
            self.send_error_body(wire, 404, "not_found", "no route %s %s" % (self.command, path))
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            log(traceback.format_exc())
            if not self.headers_sent:
                wire = "anthropic" if urlsplit(self.path).path.startswith("/v1/messages") else "openai"
                try:
                    self.send_error_body(wire, 500, "internal_error", "the server failed to answer")
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

    def send_json(self, status, body, content_type="application/json", failures=None):
        raw = body if isinstance(body, bytes) else compact(body).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(raw)))
        if failures:
            self.send_header("x-makefun-upstream-failures", compact(failures[:8]))
        self.headers_sent = True
        self.end_headers()
        self.wfile.write(raw)

    def send_error_body(self, wire, status, error_type, message):
        if wire == "anthropic":
            self.send_json(status, {"type": "error", "error": {"type": error_type, "message": message}})
        else:
            self.send_json(status, {"error": {"message": message, "type": error_type, "code": None}})

    def read_json(self):
        """One client request body, or None when it is not one JSON object or is too large."""
        length = self.headers.get("content-length")
        if length is None:
            self.close_connection = True
            return None
        try:
            size = int(length)
        except ValueError:
            self.close_connection = True
            return None
        if size < 0 or size > MAX_BODY_BYTES:
            self.close_connection = True
            return None
        raw = self.rfile.read(size) if size > 0 else b""
        if not raw:
            return None
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        return value if is_record(value) else None

    def models(self):
        """GET /v1/models, in the shape of the auth header the client sent."""
        if self.headers.get("x-api-key") is not None:
            self.send_json(200, anthropic_model_list())
        else:
            self.send_json(200, openai_model_list())

    def complete(self, wire):
        """POST /v1/chat/completions or /v1/messages: serve the prompt and answer in its wire."""
        body = self.read_json()
        if body is None:
            self.send_error_body(wire, 400, "invalid_request_error", "the request body is not one JSON object")
            return
        status, payload, content_type, failures = serve(wire, body)
        if content_type == "text/event-stream":
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("cache-control", "no-cache")
            self.send_header("connection", "close")
            if failures:
                self.send_header("x-makefun-upstream-failures", compact(failures[:8]))
            self.headers_sent = True
            self.end_headers()
            self.close_connection = True
            try:
                self.wfile.write(payload)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            return
        self.send_json(status, payload, content_type, failures)


# ---------------------------------------------------------------- register and poll

def register(port):
    """POST /register with the port and the one catalog; the env answers the prompt count."""
    body = compact({"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": CATALOG})
    request = urllib.request.Request(
        "%s/register" % ENV_URL,
        data=body.encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    last_error = None
    for _ in range(5):
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                reply_text = response.read().decode("utf-8", "replace")
                status = response.status
        except urllib.error.HTTPError as err:
            raise RuntimeError("POST %s/register answered HTTP %d" % (ENV_URL, err.code)) from err
        except Exception as err:
            last_error = err
            time.sleep(1.0)
            continue
        if not (200 <= status < 300):
            raise RuntimeError("POST %s/register answered HTTP %d: %s" % (ENV_URL, status, reply_text[:500]))
        try:
            reply = json.loads(reply_text)
        except (json.JSONDecodeError, ValueError):
            reply = None
        if is_record(reply) and is_number(reply.get("requests")):
            log("registered: the trial holds %d prompts" % reply["requests"])
        return
    raise RuntimeError("POST %s/register unreachable: %s" % (ENV_URL, last_error))


def poll():
    """Poll GET /work; exit 0 the moment it says done."""
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        try:
            with urllib.request.urlopen("%s/work" % ENV_URL, timeout=2) as response:
                if response.status != 200:
                    continue
                work = json.loads(response.read().decode("utf-8", "replace"))
        except Exception:
            continue
        if is_record(work) and work.get("done") is True:
            sys.stderr.flush()
            os._exit(0)


# ---------------------------------------------------------------- start

def stop(*_):
    """SIGTERM and SIGINT end the trial at once, exit 0."""
    sys.stderr.flush()
    os._exit(0)


def start():
    if "--list" in sys.argv[1:]:
        # The same document the register body carries, byte for byte: print the one built
        # object and stop before any server starts.
        sys.stdout.write(compact(CATALOG) + "\n")
        sys.stdout.flush()
        return
    if ENV_URL == "":
        raise RuntimeError("MAKEFUN_ENV_URL is not set")
    if chat_completions_url() == "":
        raise RuntimeError("OPENAI_BASE_URL is not set")
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    try:
        port = server.server_address[1]
        if port == 8080:
            raise RuntimeError("the OS picked 8080, which the env owns on one layout: restart the trial")
        log("listening on 0.0.0.0:%d with %d free models" % (port, len(MODELS)))
        register(port)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        threading.Thread(target=poll, name="work-poll", daemon=True).start()
        server.serve_forever()
    finally:
        server.server_close()


def main():
    try:
        start()
        return 0
    except SystemExit:
        raise
    except BaseException:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
