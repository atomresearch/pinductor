"""Compact, lossless serialization of the Blocks-rand6 observation-action history
for the VLA baseline's language channel.

Why this exists: OpenVLA's LLaMA-2 backbone has max_position_embeddings = 2048
(verified from the instantiated config), and 256 of those go to visual patches
plus 419 to the byte-identical env description -> ~1358 tokens left. The verbose
format costs 21 tokens/step, so a 120-step episode would need 2520 and would not
fit. This format costs 9 tokens/step -> 1080 for a full episode, total ~1770.

Nothing is dropped: each step records the action and the full BlocksObservation
(current column stack bottom-to-top, held block). Round-trip is exact - see
selftest below.

Rationale for giving the history at all: our own pipeline's belief is updated
with the whole observation history (`update_belief(..., observation_history,
action_history)` in partially_obs_planning_agent.py), so our deployed policy is
pi(a | o_0..o_t). A memoryless VLA baseline would not be the matched condition.
"""

def step_str(action: int, stack, held: int) -> str:
    """One history step, 9 tokens. '-' encodes an empty stack / nothing held."""
    s = "".join(str(int(b)) for b in stack) or "-"
    return f"{int(action)}:{s}:{int(held) if held else '-'};"


def history_str(actions, stacks, helds) -> str:
    return "".join(step_str(a, s, h) for a, s, h in zip(actions, stacks, helds))


def tail(hist: str, k: int) -> str:
    """Last k steps of a serialized history. k <= 0 means "no truncation".

    The window is a HARDWARE bound, not a design choice: fwd+bwd OOMs at 1,652
    positions (measured peak 23.09 GiB of 23.98, unchanged by gradient
    checkpointing), so the history has to be capped at whatever length fits.
    More history can only add information to a POMDP policy, hence the window is
    set to the largest measured feasible value rather than tuned.
    """
    if k <= 0 or not hist:
        return hist
    parts = [p for p in hist.split(";") if p]
    return "".join(p + ";" for p in parts[-k:])


def parse_step(tok: str):
    """Inverse of step_str, for the round-trip test."""
    a, s, h = tok.rstrip(";").split(":")
    stack = () if s == "-" else tuple(int(c) for c in s)
    return int(a), stack, (0 if h == "-" else int(h))


def selftest():
    import pickle, pathlib
    p = (pathlib.Path(__file__).resolve().parents[2] / "uncertain_worms" /
         "environments" / "blocksworld" / "trajectory_data" / "rand6_demos.pkl")
    b = pickle.load(open(p, "rb"))
    n = bad = 0
    for ep in b.episodes:
        for t in range(ep.length):
            o = ep.previous_observations[t]
            a = int(ep.actions[t])
            enc = step_str(a, tuple(o.current_stack), int(o.held))
            a2, st2, h2 = parse_step(enc)
            if (a2, st2, h2) != (a, tuple(o.current_stack), int(o.held)):
                bad += 1
            n += 1
    print(f"round-trip: {n - bad}/{n} exact ({bad} mismatches)")
    assert bad == 0, "serialization is lossy"


if __name__ == "__main__":
    selftest()
