"""Utilities for the baseline package: model access, log dirs, parsing.

This directory holds a self-contained copy of the prior method this work
is compared against, kept apart from the main package so that the
baseline runs with its original behaviour instead of inheriting later
changes. This file is that copy's counterpart to
``uncertain_worms/utils.py``: same helper names, deliberately simpler.
Differences that matter when reading the two side by side:

* a single provider (OpenRouter) and a single model constant, with no
  provider argument, sampling parameters or prompt caching;
* no token counters — usage is only written to the log, where the
  analysis scripts pick it up;
* the same switch to a backup key when API credits run out, and the same
  reasoning-token cap, so both methods are given the same budget.

Everything else — prompt file format, run-directory resolution, code
extraction from a reply — is duplicated verbatim rather than imported,
which is what keeps this baseline independent of the main package.
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

log = logging.getLogger(__name__)

env_file = os.path.join(pathlib.Path(__file__).parent.parent, ".env")
dotenv.load_dotenv(env_file, override=True)
openrouter_api_key = os.environ.get("OPEN_ROUTER_KEY")
# Optional second key, used when the primary one runs out of credits
# mid-run. The key itself is never logged, only the swap marker.
_openrouter_api_key_backup = os.environ.get("OPEN_ROUTER_KEY_BACKUP")
_openrouter_key_swapped = False

PROJECT_ROOT = os.path.dirname(__file__)


def parse_prompt(path: str) -> List[Dict[str, Any]]:
    """Read a ``#define <role>`` prompt file into chat entries.

    Each ``#define`` line opens a ``{"role": ..., "content": ...}`` record
    and every line until the next one is appended to its content. The path
    is resolved inside the current run directory.
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
    """Write chat entries back in the format :func:`parse_prompt` reads.

    Prompts are kept on disk in the run directory so that a run can be
    inspected, and re-read, exactly as it was sent.
    """
    with open(os.path.join(get_log_dir(), path), "w") as file:
        for entry in entries:
            # Write the role definition
            file.write(f"#define {entry['role']}\n")
            # Write the content, each line is separated
            content_lines = entry["content"]

            file.write(content_lines + "\n")


def save_log(path: str, text: str) -> None:
    """Write ``text`` to ``path``, resolved inside the run directory."""
    with open(os.path.join(get_log_dir(), path), "w") as f:
        f.write(text)


def get_log_dir() -> str:
    """Return the directory the current run writes into.

    Under Hydra this is the job's run directory, which is where every
    artefact of a run belongs. Outside Hydra — an interactive session, a
    unit test — a timestamped directory is created instead, so importing
    this module never fails for lack of a Hydra context.
    """
    # If not under a Hydra job, fall back to a default
    if not hydra.core.hydra_config.HydraConfig.initialized():
        timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        log_dir = os.path.join("outputs", timestamp)
        # Optionally create the folder if you want to ensure it exists
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    # Otherwise, use the Hydra job's run directory
    return hydra.core.hydra_config.HydraConfig.get().run.dir


# Canonical names of the model functions the generated code must define.
REWARD_FUNCTION_NAME = "reward_func"
TRANSITION_FUNCTION_NAME = "transition_func"
INITIAL_FUNCTION_NAME = "initial_func"
OBSERVATION_FUNCTION_NAME = "observation_func"

# Model used for the reported runs. It reads the same environment
# variable as the main package, so a sweep over language models moves
# both sides at once and the comparison stays like-for-like.
ENGINE = os.environ.get("PAPER_LLM_MODEL", "qwen/qwen3.6-plus")


def parse_code(input_text: str) -> str | None:
    """Return the code of all fenced blocks in a reply, concatenated.

    ```python``` blocks are preferred; if there are none, any fenced block
    is taken. Returns ``None`` when the reply contains no code block.
    """
    pattern1 = "```python(.*?)```"
    pattern2 = "```(.*?)```"
    matches1 = re.findall(pattern1, input_text, re.DOTALL)

    if len(matches1) == 0:
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
    """Return the discounted return of an episode.

    ``rewards`` is rewritten in place, each index ending up with the
    discounted return from that step onwards; callers that still need the
    raw rewards must pass a copy.
    """
    discounted_sum = 0.0
    for t in reversed(range(len(rewards))):
        discounted_sum = rewards[t] + gamma * discounted_sum
        rewards[t] = discounted_sum  # Replace in-place if needed
    return rewards[0]  # Total discounted return for the episode


def query_llm(message: List[Dict[str, str]], max_retries: int = 5) -> Tuple[str, float]:
    """Send a chat request to OpenRouter and return the reply and latency.

    Failed requests are retried with an exponential backoff. Token counts
    are logged, not accumulated: this package keeps no counters of its
    own, and the analysis reads the log lines instead.
    """
    global openrouter_api_key, _openrouter_key_swapped
    retry_count = 0
    backoff_factor = 60
    # Reasoning cap, held equal to the value used by the main package so
    # that neither method is given a larger thinking budget than the
    # other. Override with PAPER_REASONING_MAX_TOKENS (0 disables
    # reasoning) and change both sides together.
    reasoning_cap_str = os.environ.get(
        "PAPER_REASONING_MAX_TOKENS", "5000"
    ).strip()
    try:
        reasoning_cap = int(reasoning_cap_str)
    except ValueError:
        reasoning_cap = 5000
    while True:
        try:
            st = time.time()

            body: Dict[str, Any] = {"model": ENGINE, "messages": message}
            if reasoning_cap > 0:
                body["reasoning"] = {"max_tokens": reasoning_cap}
            elif reasoning_cap == 0:
                body["reasoning"] = {"enabled": False}

            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": "Bearer {}".format(openrouter_api_key),
                },
                data=json.dumps(body),
            )
            # OpenRouter answers 402 (Payment Required) once an account is
            # out of credits, 403 in related quota cases. Retry once on the
            # backup key rather than lose a run in progress.
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
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": "Bearer {}".format(openrouter_api_key),
                    },
                    data=json.dumps(body),
                )
            # The status is logged before the body is read, because an error
            # body has no `choices` and the KeyError below would otherwise be
            # the only trace of what went wrong.
            log.info(f"Response status code: {response.status_code}")
            response_json = response.json()
            # Token accounting. The [OPENROUTER_TOKENS] line has the same
            # format as in the main package, so a single log parser reads
            # the cost of both methods. Counts only, never content.
            usage = response_json.get("usage", {}) or {}
            ptok = int(usage.get("prompt_tokens", 0))
            ctok = int(usage.get("completion_tokens", 0))
            ttok = int(usage.get("total_tokens", ptok + ctok))
            log.info(
                f"[OPENROUTER_TOKENS] prompt={ptok} completion={ctok} total={ttok}"
            )
            return (
                str(response_json["choices"][0]["message"]["content"]),
                time.time() - st,
            )
        except Exception as e:
            retry_count += 1
            if retry_count > max_retries:
                raise e
            # Capped like the main package: an unbounded backoff means a bad
            # key stalls for minutes before anyone sees the failure.
            sleep_time = min(backoff_factor * (2**retry_count), 60)
            # Any failure lands here, not only throttling: an invalid key
            # returns a body without `choices` and raises a KeyError. Naming
            # the exception keeps the retry from disguising an authentication
            # problem as a rate limit.
            log.info(
                f"Request failed ({type(e).__name__}: {e}). "
                f"Retrying in {sleep_time} seconds..."
            )
            time.sleep(sleep_time)


if __name__ == "__main__":
    pass
