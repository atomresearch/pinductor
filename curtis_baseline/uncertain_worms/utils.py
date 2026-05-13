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
# Fallback backup key : swap auto si credits exhausted. No log de la cle.
_openrouter_api_key_backup = os.environ.get("OPEN_ROUTER_KEY_BACKUP")
_openrouter_key_swapped = False

PROJECT_ROOT = os.path.dirname(__file__)


def parse_prompt(path: str) -> List[Dict[str, Any]]:
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
    with open(os.path.join(get_log_dir(), path), "w") as file:
        for entry in entries:
            # Write the role definition
            file.write(f"#define {entry['role']}\n")
            # Write the content, each line is separated
            content_lines = entry["content"]

            file.write(content_lines + "\n")


def save_log(path: str, text: str) -> None:
    with open(os.path.join(get_log_dir(), path), "w") as f:
        f.write(text)


def get_log_dir() -> str:
    # If not under a Hydra job, fall back to a default
    if not hydra.core.hydra_config.HydraConfig.initialized():
        timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        log_dir = os.path.join("outputs", timestamp)
        # Optionally create the folder if you want to ensure it exists
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    # Otherwise, use the Hydra job's run directory
    return hydra.core.hydra_config.HydraConfig.get().run.dir


REWARD_FUNCTION_NAME = "reward_func"
TRANSITION_FUNCTION_NAME = "transition_func"
INITIAL_FUNCTION_NAME = "initial_func"
OBSERVATION_FUNCTION_NAME = "observation_func"

# Paper runs : Qwen 3.6 Plus uniquement. Override via env var PAPER_LLM_MODEL
# (utilisé par E4 LLM variation, voir scripts/paper/).
ENGINE = os.environ.get("PAPER_LLM_MODEL", "qwen/qwen3.6-plus")


def parse_code(input_text: str) -> str | None:
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
    discounted_sum = 0.0
    for t in reversed(range(len(rewards))):
        discounted_sum = rewards[t] + gamma * discounted_sum
        rewards[t] = discounted_sum  # Replace in-place if needed
    return rewards[0]  # Total discounted return for the episode


def query_llm(message: List[Dict[str, str]], max_retries: int = 5) -> Tuple[str, float]:
    global openrouter_api_key, _openrouter_key_swapped
    retry_count = 0
    backoff_factor = 60
    # Reasoning cap — kept in sync with uncertain_worms/utils.py on the ours
    # side for a fair LLM budget across ours ↔ curtis_their. Override with
    # PAPER_REASONING_MAX_TOKENS (0 disables reasoning).
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
            # Swap vers backup si credits epuises sur la principale (402/403)
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
            response_json = response.json()
            # Token accounting for paper runs — identique à
            # uncertain_worms/utils.py (format `[OPENROUTER_TOKENS]` parsé
            # par scripts/paper/log_parser.py). POMDP-safe : comptes seulement.
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
            sleep_time = backoff_factor * (2**retry_count)
            log.info(f"Rate limit exceeded. Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)


if __name__ == "__main__":
    pass
