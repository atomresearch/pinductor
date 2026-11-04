"""Shared utilities: language-model access, run directories, bookkeeping.

Three groups of helpers live here, used by every approach in the
package:

* **Language model access.** :func:`query_llm` is the single door to a
  provider (OpenRouter, OpenAI, or a local Ollama server). It owns the
  retry policy, the optional prompt-caching and provider-pinning knobs,
  and the token counters that :mod:`main` reports at the end of a run.
* **Run directories and prompt files.** :func:`get_log_dir` resolves the
  Hydra run directory that everything else writes into;
  :func:`parse_prompt` / :func:`write_prompt` read and write the prompt
  transcripts stored there.
* **Small shared computations**, such as :func:`parse_code`, which pulls
  generated Python out of a model reply, and :func:`discounted_reward`.

Credentials come from a ``.env`` file at the repository root (see
``.env.example``) and never from the configuration files.

Logging convention followed throughout: the instrumentation records
metadata and counters only — model identifiers, latencies, token
counts — and never prompt or response content, states, or observations,
which is what the "POMDP-safe" remarks below refer to.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import dotenv
import hydra
import requests  # type: ignore
import httpx

from openai import OpenAI
import wandb

log = logging.getLogger(__name__)

env_file = os.path.join(pathlib.Path(__file__).parent.parent, ".env")
dotenv.load_dotenv(env_file, override=True)
openrouter_api_key = os.environ.get("OPEN_ROUTER_KEY")
# Optional second OpenRouter key. A long sweep can exhaust the credits of
# the primary key mid-run; when that happens the request layer switches to
# this one instead of failing the experiment. The key itself is never
# logged — only the [OPENROUTER_KEY_SWAPPED] marker.
_openrouter_api_key_backup = os.environ.get("OPEN_ROUTER_KEY_BACKUP")
_openrouter_key_swapped = False
openai_api_key = os.environ.get("OPENAI_API_KEY")


class OpenRouterOutOfCreditsError(RuntimeError):
    """Both primary and backup OpenRouter keys returned 402 / are out of
    credits. Raised by ``query_llm`` to fail fast instead of looping in
    a 60s retry that will keep returning 402."""


def _check_openrouter_credits(api_key: Optional[str]) -> Optional[float]:
    """Return remaining USD credits for an OpenRouter key, or ``None``
    if the call failed. POMDP-safe: only the balance is logged."""
    if not api_key:
        return None
    try:
        r = requests.get(
            url="https://openrouter.ai/api/v1/credits",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        data = r.json().get("data", {}) or {}
        total = float(data.get("total_credits", 0) or 0)
        used = float(data.get("total_usage", 0) or 0)
        return round(total - used, 4)
    except Exception:
        return None

# Supported llm_provider values: "openrouter", "openai", "ollama"

PROJECT_ROOT = os.path.dirname(__file__)

# ---- Token tracking ----
# Process-wide counters: `main` zeroes them when a run starts and reports
# the totals when it ends, so one run's cost is one process's counters.
_token_usage: Dict[str, int] = {"input": 0, "output": 0, "calls": 0}


def get_token_usage() -> Dict[str, int]:
    """Return a snapshot copy of the global LLM token counters."""
    return dict(_token_usage)


def reset_token_usage() -> None:
    """Zero the global LLM token counters."""
    _token_usage["input"] = 0
    _token_usage["output"] = 0
    _token_usage["calls"] = 0


def _record_usage(input_tokens: int, output_tokens: int) -> None:
    """Add one call's token counts to the process-wide totals."""
    _token_usage["input"] += input_tokens
    _token_usage["output"] += output_tokens
    _token_usage["calls"] += 1


def parse_prompt(path: str) -> List[Dict[str, Any]]:
    """Parse a ``#define <role>`` formatted prompt file into chat entries.

    Each ``#define`` line opens a new ``{"role": ..., "content": ...}``
    record; everything that follows (until the next ``#define``) becomes
    the content. Returns the list of records in source order.
    """
    entries = []
    current_entry: Optional[Dict] = None

    with open(os.path.join(get_log_dir(), path), "r") as file:
        for line in file:
            line = line.rstrip("\n")  # Remove only trailing newline
            if line.startswith("#define "):
                # Start a new entry
                if current_entry:
                    entries.append(current_entry)
                current_entry = {"role": line[len("#define ") :], "content": ""}
            elif current_entry is not None:
                # Preserve indentation and add newline manually
                current_entry["content"] += line + "\n"

    # Append the last entry if it exists
    if current_entry:
        entries.append(current_entry)

    return entries


def write_prompt(path: str, entries: List[Dict[str, Any]]) -> None:
    """Inverse of :func:`parse_prompt`: serialize chat entries back to disk."""
    with open(os.path.join(get_log_dir(), path), "w") as file:
        for entry in entries:
            # Write the role definition
            file.write(f"#define {entry['role']}\n")
            # Write the content, each line is separated
            content_lines = entry["content"]

            file.write(content_lines + "\n")


def save_log(path: str, text: str) -> None:
    """Write ``text`` to ``path`` resolved against the current log dir."""
    with open(os.path.join(get_log_dir(), path), "w", encoding="utf-8") as f:
        f.write(text)


def get_log_dir() -> str:
    """Return the active Hydra job dir (or a timestamped fallback)."""
    # If not under a Hydra job, fall back to a default
    if not hydra.core.hydra_config.HydraConfig.initialized():
        timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        log_dir = os.path.join("outputs", timestamp)
        # Optionally create the folder if you want to ensure it exists
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    # Otherwise, use the Hydra job's run directory
    return hydra.core.hydra_config.HydraConfig.get().run.dir


# Canonical names of the four POMDP components a generated model must
# define. The prompts ask for these exact names and the code that checks a
# reply looks them up by the same spelling.
REWARD_FUNCTION_NAME = "reward_func"
TRANSITION_FUNCTION_NAME = "transition_func"
INITIAL_FUNCTION_NAME = "initial_func"
OBSERVATION_FUNCTION_NAME = "observation_func"

# ENGINE = "gpt-3.5-turbo-0125"
# ENGINE = "openai/gpt-4-turbo"
# ENGINE_openrouter = "openai/gpt-4o"
# ENGINE = "openai/o1"
# ENGINE = "codellama:34b-instruct-q5_K_M"
# ENGINE = "deepseek-r1-large"
# ENGINE = "qwen-32k"
# ENGINE = "llama33-largecontext"
DEFAULT_MODELS: Dict[str, str] = {
    # Model used for the reported runs. The ``PAPER_LLM_MODEL`` environment
    # variable overrides it, which is how the experiment comparing several
    # language models is run without editing any configuration file.
    "openrouter": os.environ.get("PAPER_LLM_MODEL", "qwen/qwen3.6-plus"),
}

openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None

# Add Ollama client:
# ollama_client = OpenAI(
#     base_url="http://localhost:11434/v1",
#     api_key="ollama",
#     http_client=httpx.Client(),
# )

def parse_code(input_text: str) -> str | None:
    """Extract the concatenated code from all fenced blocks in ``input_text``.

    Looks for ```python``` blocks first, then any ``` blocks. Every block
    is kept and concatenated, because a reply may define the requested
    functions across several blocks. Returns ``None`` if no fenced block
    is present.
    """
    pattern1 = "```python(.*?)```"
    pattern2 = "```(.*?)```"
    matches1 = re.findall(pattern1, input_text, re.DOTALL)

    if not matches1:
        matches = re.findall(pattern2, input_text, re.DOTALL)
        if len(matches) == 0:
            return None
    else:
        matches = matches1

    all_code = ""
    for match in matches:
        all_code += "\n" + match
    return all_code


def discounted_reward(rewards: List[float], gamma: float) -> float:
    """Compute the total discounted return ``sum_t gamma^t * rewards[t]``.

    Mutates ``rewards`` in place to store the suffix-discounted return
    at each index (the original list is overwritten).
    """
    discounted_sum = 0.0
    for t in reversed(range(len(rewards))):
        discounted_sum = rewards[t] + gamma * discounted_sum
        rewards[t] = discounted_sum  # Replace in-place if needed
    return rewards[0]  # Total discounted return for the episode


def query_llm(
    message: List[Dict[str, str]],
    max_retries: int = 5,
    llm_provider: str = "openrouter",
    llm_model: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[str, float]:
    """Send a chat request to a language-model provider and return its reply.

    Args:
        message: chat entries, as produced by :func:`parse_prompt`.
        max_retries: how many times a failed request is retried before the
            exception is propagated.
        llm_provider: ``"openrouter"``, ``"openai"`` or ``"ollama"``.
        llm_model: model identifier; falls back to :data:`DEFAULT_MODELS`.
        temperature, top_p, seed: sampling controls. They are only put in
            the request when set, so that a provider's own defaults apply
            otherwise and the payload stays comparable across conditions.

    Returns:
        The reply text and the wall-clock seconds the call took.

    Raises:
        OpenRouterOutOfCreditsError: both OpenRouter keys are out of
            credits, which no amount of retrying will fix.

    Token counts are added to the process-wide counters for every
    provider, so cost accounting does not depend on which one is used.
    """
    global openrouter_api_key, _openrouter_key_swapped
    model = llm_model or DEFAULT_MODELS.get(llm_provider, "")
    retry_count = 0
    backoff_factor = 60 if llm_provider in ("openrouter", "openai") else 2

    while True:
        try:
            st = time.time()

            if llm_provider == "openai":
                if openai_client is None:
                    raise ValueError("OPENAI_API_KEY not set in .env")
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": message,
                }
                if temperature is not None:
                    kwargs["temperature"] = temperature
                if top_p is not None:
                    kwargs["top_p"] = top_p
                if seed is not None:
                    kwargs["seed"] = seed

                response_openai = openai_client.chat.completions.create(**kwargs)
                content = str(response_openai.choices[0].message.content)
                if response_openai.usage:
                    _record_usage(response_openai.usage.prompt_tokens, response_openai.usage.completion_tokens)

            elif llm_provider == "openrouter":
                # Prompt caching opt-in (OPENROUTER_PROMPT_CACHE=1).
                # OpenRouter support varies by backend: some honour the
                # explicit ``cache_control`` marker on
                # content blocks; DeepSeek auto-caches without a marker;
                # Qwen / Mistral / Llama have no caching layer and may 400
                # on the block-format payload. Hence we (a) gate on env
                # var (zero-regression default), AND (b) gate on the
                # model alias prefix so we never send the block format to
                # a backend that doesn't accept it.
                #
                # Cached tokens are billed at 0.25x input rate, TTL 3-5 min.
                # See docs/guides/best-practices/prompt-caching.
                _CACHE_PREFIXES = ("anthropic/", "google/")
                cache_prefix_on = (
                    os.environ.get("OPENROUTER_PROMPT_CACHE", "0")
                    .strip().lower() in ("1", "true", "yes", "on")
                )
                model_supports_cache = any(
                    model.startswith(p) for p in _CACHE_PREFIXES
                )
                payload_messages = message
                if (
                    cache_prefix_on
                    and model_supports_cache
                    and message
                    and message[0].get("role") == "system"
                ):
                    sys_content = message[0].get("content", "")
                    # Anthropic minimum cacheable prefix is ~1024 tokens
                    # (~4096 chars). Below that bar caching never engages
                    # so the block-format wrapping would be pure overhead.
                    if isinstance(sys_content, str) and len(sys_content) >= 4096:
                        payload_messages = [
                            {
                                "role": "system",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": sys_content,
                                        "cache_control": {"type": "ephemeral"},
                                    }
                                ],
                            },
                            *message[1:],
                        ]
                        log.info(
                            "[PROMPT_CACHE] Wrapped system prompt (%d chars) "
                            "with cache_control=ephemeral for model=%s",
                            len(sys_content), model,
                        )
                payload: Dict[str, Any] = {
                    "model": model,
                    "messages": payload_messages,
                }
                if temperature is not None:
                    payload["temperature"] = temperature
                if top_p is not None:
                    payload["top_p"] = top_p
                if seed is not None:
                    payload["seed"] = seed

                # Opt-in provider pinning. OpenRouter routes the same model
                # alias to different backend hosts from one session to the
                # next, and those hosts do not answer identically, which adds
                # between-session variance to results. Set
                # OPENROUTER_PIN_PROVIDER=<comma-separated list> to force one
                # provider order with no fallback. Unset by default, in which
                # case routing is left to OpenRouter.
                pin_provider = os.environ.get("OPENROUTER_PIN_PROVIDER", "").strip()
                if pin_provider:
                    order = [p.strip() for p in pin_provider.split(",") if p.strip()]
                    if order:
                        payload["provider"] = {
                            "order": order,
                            "allow_fallbacks": False,
                        }

                # Reasoning cap. Recent reasoning models spend on the order
                # of 8-14k tokens of thinking per call by default, which
                # dominates latency on long prompts; capping keeps most of
                # the reasoning at a fraction of the cost. The value is a
                # shared experimental parameter: the baseline package sets
                # the same cap in its own copy of this file, so that both
                # methods are given the same reasoning budget. Override with
                # ``PAPER_REASONING_MAX_TOKENS`` (0 disables reasoning) and
                # keep the two sides in step.
                reasoning_cap_str = os.environ.get(
                    "PAPER_REASONING_MAX_TOKENS", "5000"
                ).strip()
                try:
                    reasoning_cap = int(reasoning_cap_str)
                except ValueError:
                    reasoning_cap = 5000
                if reasoning_cap > 0:
                    payload["reasoning"] = {"max_tokens": reasoning_cap}
                elif reasoning_cap == 0:
                    payload["reasoning"] = {"enabled": False}

                # Capture key used IN this request (local var) for race-
                # condition recovery below.
                key_used_in_request = openrouter_api_key
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key_used_in_request}",
                        "Content-Type": "application/json",
                    },
                    data=json.dumps(payload),
                    timeout=300,
                )
                # OpenRouter answers 402 (Payment Required) once an account
                # runs out of credits, and 403 in related quota cases. Retry
                # the request on the backup key rather than lose the run.
                if (
                    response.status_code in (402, 403)
                    and _openrouter_api_key_backup
                    and not _openrouter_key_swapped
                ):
                    log.warning(
                        "[OPENROUTER_KEY_SWAPPED] status=%d on primary key, "
                        "retrying with backup", response.status_code,
                    )
                    openrouter_api_key = _openrouter_api_key_backup
                    _openrouter_key_swapped = True
                    key_used_in_request = openrouter_api_key
                    response = requests.post(
                        url="https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {key_used_in_request}",
                            "Content-Type": "application/json",
                        },
                        data=json.dumps(payload),
                        timeout=300,
                    )
                elif (
                    response.status_code in (402, 403)
                    and _openrouter_api_key_backup
                    and _openrouter_key_swapped
                    and key_used_in_request != _openrouter_api_key_backup
                ):
                    # Another thread swapped the key while this request was
                    # in flight, so this one went out with the old key and
                    # failed for a reason that is already fixed: retry it.
                    log.info(
                        "[OPENROUTER_RACE_RECOVERY] status=%d on stale primary "
                        "(another thread swapped during in-flight request) — "
                        "retrying with backup", response.status_code,
                    )
                    key_used_in_request = _openrouter_api_key_backup
                    response = requests.post(
                        url="https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {key_used_in_request}",
                            "Content-Type": "application/json",
                        },
                        data=json.dumps(payload),
                        timeout=300,
                    )
                # Both keys are refused: stop here instead of entering the
                # generic retry loop below, which would sleep 60s at a time
                # while every attempt gets the same answer.
                if response.status_code in (402, 403):
                    prim_rem = _check_openrouter_credits(
                        os.environ.get("OPEN_ROUTER_KEY")
                    )
                    back_rem = _check_openrouter_credits(
                        _openrouter_api_key_backup
                    )
                    log.error(
                        "[OPENROUTER_OUT_OF_CREDITS] status=%d after %s "
                        "primary_remaining=%s backup_remaining=%s — "
                        "aborting (top up at "
                        "https://openrouter.ai/settings/credits)",
                        response.status_code,
                        "swap" if _openrouter_key_swapped else "no_swap",
                        f"${prim_rem}" if prim_rem is not None else "?",
                        f"${back_rem}" if back_rem is not None else "?",
                    )
                    raise OpenRouterOutOfCreditsError(
                        f"OpenRouter both keys returned {response.status_code}. "
                        f"primary_remaining=${prim_rem} backup_remaining=${back_rem}"
                    )
                log.info(f"Response status code: {response.status_code}")
                response_json = response.json()
                # Instrumentation: log which model_id/provider OpenRouter
                # actually served. Helps diagnose inter-session variance due
                # to OpenRouter routing a given alias to different hosts.
                # POMDP-safe: only metadata is logged, never prompt/response
                # content, state, or observations.
                try:
                    served_model = response_json.get("model", "unknown")
                    served_provider = response_json.get("provider") or (
                        response.headers.get("openrouter-provider")
                        if hasattr(response, "headers")
                        else None
                    ) or "unknown"
                    latency_ms = int((time.time() - st) * 1000)
                    log.info(
                        f"[OPENROUTER] model_id={served_model} "
                        f"provider={served_provider} latency_ms={latency_ms}"
                    )
                except Exception as _instr_err:
                    log.debug(f"[OPENROUTER] instrumentation failed: {_instr_err}")
                content = str(response_json["choices"][0]["message"]["content"])
                # OpenRouter echoes token counts in the response body. They
                # feed the process-wide counters and are also written out
                # per call, as an [OPENROUTER_TOKENS] line and a W&B metric,
                # so cost can be attributed to individual calls after the
                # fact from the logs alone. Counts only, never content.
                usage = response_json.get("usage", {}) or {}
                ptok = int(usage.get("prompt_tokens", 0))
                ctok = int(usage.get("completion_tokens", 0))
                ttok = int(usage.get("total_tokens", ptok + ctok))
                _record_usage(ptok, ctok)
                try:
                    log.info(
                        f"[OPENROUTER_TOKENS] prompt={ptok} completion="
                        f"{ctok} total={ttok}"
                    )
                    if wandb.run is not None:
                        wandb.log(
                            {
                                "openrouter/prompt_tokens": ptok,
                                "openrouter/completion_tokens": ctok,
                                "openrouter/total_tokens": ttok,
                            }
                        )
                except Exception as _tok_err:
                    log.debug(f"[OPENROUTER_TOKENS] extraction failed: {_tok_err}")

            elif llm_provider == "ollama":
                ollama_client = OpenAI(
                    base_url="http://localhost:11434/v1",
                    api_key="ollama",
                    http_client=httpx.Client(),
                )
                resp = ollama_client.chat.completions.create(
                    model=model,
                    messages=message,
                )
                content = str(resp.choices[0].message.content)
                if resp.usage:
                    _record_usage(resp.usage.prompt_tokens, resp.usage.completion_tokens)

            else:
                raise ValueError(f"Unknown llm_provider: {llm_provider}")

            return (content, time.time() - st)

        except OpenRouterOutOfCreditsError:
            raise
        except Exception as e:
            retry_count += 1
            if retry_count > max_retries:
                raise e
            sleep_time = min(backoff_factor * (2**retry_count), 60)  # cap at 60s
            # Any failure lands here, not only a connection one: an invalid key
            # returns a body without `choices` and raises a KeyError. Naming the
            # exception keeps the retry from describing an authentication
            # problem as a network problem.
            log.info(
                f"Request failed ({type(e).__name__}: {e}). "
                f"Retrying in {sleep_time} seconds..."
            )
            time.sleep(sleep_time)

if __name__ == "__main__":
    pass
