# free-model-aggregator-02 — one server, two wires, a live free model behind every answer

A Python 3.13 standard-library-only server for the make.fun bounty
[free-model-aggregator-02](https://make.fun/bounty/free-model-aggregator-02).
It serves the OpenAI Chat Completions wire (`POST /v1/chat/completions`) and the
Anthropic Messages wire (`POST /v1/messages`), plus `GET /v1/models` in the shape
of the auth header it got, and answers every prompt with a real completion from
one of the fourteen free OpenRouter models, dialled through the grader's gateway
on Chat Completions. Point any OpenAI or Anthropic client at it.

## How it works

- **Routing (quality).** Each request's requirements are read off its own body:
  a JSON `response_format` needs a JSON-capable model, declared tools need
  tools, image parts need vision. Capable models are tried best-first by the
  house rating (rating desc, context desc, id asc): JSON prompts go to
  Nemotron 3.5 Super VL (1303, the best JSON-capable model); everything else
  goes to Inkling 2 Small (1475, the best free model outright).
- **Failover.** A gateway 4xx/5xx or a timeout is recorded and the prompt is
  retried on the next model, up to three attempts inside one 26 s per-prompt
  deadline (the env waits 30 s). Refusals are reported on the response in the
  `x-makefun-upstream-failures` header, the only way the failover column can
  see a recovered refusal. A model is never retried twice for the same prompt,
  and there is no trial-wide cooldown: a transient 429 on one prompt does not
  block the best model on the next.
- **Token check.** The prompt's unique token (`WORD-1234`) is read out of the
  request, and a 2xx answer is only accepted when its text carries that token
  (and parses as JSON for a JSON-mode prompt) — the house's proof needs the
  provider's own body to carry the token. When no attempt carries it, the
  best-rated attempt's answer is still returned, so the wire checks pass.
- **The wire.** The upstream body is always read whole first, then translated:
  the Anthropic stream carries the true input-token count in `message_start`
  and the OpenAI stream always carries a usage chunk before `[DONE]`. Usage
  and stop reason pass through whole (stop→end_turn, length→max_tokens,
  tool_calls→tool_use, content_filter→refusal); missing usage is reported as
  input 0 and output max(1, ceil(len(text)/4)). Tool calls the client did
  not declare are dropped. A request with no output cap gets
  `max_tokens: 1024` upstream, because the gateway refuses uncapped requests.
- **Determinism.** Every choice is a function of the request bodies, the
  gateway's answers and the refusals seen so far: fixed model order, fixed
  attempt budgets, fixed timestamps in listings, no randomness, no wall-clock
  input to any decision. Timeouts are a safety valve only.

## The catalog

Exactly the fourteen free models, one provider (`openrouter`,
`https://openrouter.ai/api/v1`, `OPENROUTER_API_KEY`, wire `openai`).
`python3 main.py --list` prints the same document the server registers.

## Running it

```sh
python3 main.py            # needs MAKEFUN_ENV_URL, OPENAI_BASE_URL, OPENAI_API_KEY
python3 main.py --list     # prints the catalog and exits 0
```

The server binds `0.0.0.0` at a free port, registers at
`$MAKEFUN_ENV_URL/register`, serves the six prompts, polls `GET /work` and
exits 0 when the env says done.

## Local testing (mock gateway, real env + real scorer)

`mock_trial.py` (kept out of the repo, in the work notes) runs the kit's own
env server and scorer against this entry with a fake OpenRouter gateway that
echoes each prompt's token. Results on the three practice instances:

| trial | profile | score | quality | served | wire | failover | catalog | calls |
|------:|---------|------:|--------:|-------:|-----:|---------:|--------:|------:|
| 0 | inkling 429s once | 98 | 96 | 100 | 100 | 100 | 100 | 6 |
| 0 | super-vl JSON miss + 429 | 98 | 96 | 100 | 100 | 100 | 100 | 7 |
| 1 | all succeed | 100 | 100 | 100 | 100 | 100 | 100 | 6 |
| 2 | inkling 429s once | 98 | 96 | 100 | 100 | 100 | 100 | 6 |
| 0 | inkling hangs 20 s | 98 | 96 | 100 | 100 | 100 | 100 | 6 |

The quality 96 (not 100) in the failover rows is the one prompt served by the
second-best model after the best refused — the correct trade: serving it keeps
served, wire and failover at 100. With no refusals the entry scores 100.

## Note on the OpenRouter key

The entry's completions are real calls on the entrant's own OpenRouter key,
pushed to the competition before submitting (`makefun secrets push
free-model-aggregator-02 --env-file .env` with `OPENROUTER_API_KEY=...`,
from an account with at least 10 credits purchased). No key is stored in this
repository. Without a pushed key every model call is refused HTTP 401.
