# type:ignore
import copy


def update_code(code, message):
    code_blocks = extract_code_blocks(message)
    code_blocks = remove_noncompilable_code_blocks(code_blocks, prefix=code)
    code = code + "\n" + "\n".join(code_blocks)
    code = remove_duplicate_code(code)
    return code


def extract_code_blocks(message):
    code_blocks = []
    in_code_block = False
    for line in message.split("\n"):
        if line.startswith("```"):
            if in_code_block:
                in_code_block = False
                code_blocks[-1] = "\n".join(code_blocks[-1])
            else:
                in_code_block = True
                code_blocks.append([])
        elif in_code_block:
            code_blocks[-1].append(line)
    if in_code_block:
        code_blocks[-1] = "\n".join(code_blocks[-1])
    return code_blocks


def _eq(a, b):
    try:
        if type(a) != type(b):
            return False
        if "all" in dir(a == b):
            return (a == b).all()
        return a == b
    except Exception as e:
        print(f"Warning: eq error, {e}, {a}, {b}, {type(a)}, {type(b)}")
        if len(a) != len(b):
            return False
        if type(a) != type(b):
            return False
        return all([a_ == b_ for a_, b_ in zip(a, b)])


def remove_duplicate_code(code):
    exec_globals = {}
    exec_globals = eval_code(code, exec_globals=exec_globals)
    if not isinstance(exec_globals, dict):
        return code
    code = "\n".join([l for l in code.split("\n") if l.strip() or l.startswith("#")])

    lines = code.split("\n")
    code_blocks = []
    prv_index = 0
    prv_exec_globals = dict()
    for index in range(len(lines)):
        if index < len(lines) - 1 and len(lines[index + 1].lstrip()) != len(
            lines[index + 1]
        ):
            continue
        if index < len(lines) - 1 and lines[index].startswith("#"):
            continue
        if index < len(lines) - 1 and (
            lines[index + 1].startswith("else:") or lines[index + 1].startswith("elif ")
        ):
            continue
        try:
            _local_exec_globals = copy.deepcopy(prv_exec_globals)
        except Exception as e:
            _local_exec_globals = None
        if _local_exec_globals is not None:
            exec_globals = eval_code(
                "\n".join(lines[: index + 1]),
                exec_globals=copy.deepcopy(prv_exec_globals),
            )
        else:
            exec_globals = eval_code(
                "\n".join(lines[: index + 1]),
                exec_globals=dict(),
            )
        if not isinstance(exec_globals, dict):
            continue
        exec_globals = {k: v for k, v in prv_exec_globals.items()}
        exec_globals = eval_code(
            "\n".join(lines[prv_index : index + 1]),
            exec_globals=exec_globals,
        )
        if not isinstance(exec_globals, dict):
            continue
        for k in exec_globals:
            try:
                bool(
                    not k.startswith("__")
                    and (not isinstance(k, str) or not _eq(k, "evaluate"))
                    and isinstance(k, str)
                    and (
                        all(
                            not _eq(k, kk)
                            for kk in prv_exec_globals
                            if isinstance(kk, str)
                        )
                        or not _eq(exec_globals[k], prv_exec_globals[k])
                    )
                )
            except Exception as e:
                print(k)
                print(type(k), len(k))
                raise e
        update_vars = [
            k
            for k in exec_globals
            if (
                not k.startswith("__")
                and (not isinstance(k, str) or not _eq(k, "evaluate"))
                and isinstance(k, str)
                and (
                    all(
                        not _eq(k, kk) for kk in prv_exec_globals if isinstance(kk, str)
                    )
                    or not _eq(exec_globals[k], prv_exec_globals[k])
                )
            )
        ]
        if len(update_vars) == 0:
            prv_index = index + 1
            continue
        code_blocks.append(
            (
                update_vars,
                "\n".join(lines[prv_index : index + 1]),
            )
        )
        prv_index = index + 1
        prv_exec_globals = exec_globals

    deduplicated_code_blocks = []
    seen_vars = set()
    for var_list, code in code_blocks[::-1]:
        if any([v not in seen_vars for v in var_list]):
            deduplicated_code_blocks.append(code)
            seen_vars.update(var_list)
    deduplicated_code_blocks = deduplicated_code_blocks[::-1]

    return "\n".join(deduplicated_code_blocks)


def remove_unused_code(code, entry_point):
    code = remove_duplicate_code(code)
    exec_globals = {}
    exec_globals = eval_code(code, exec_globals=exec_globals)
    if not isinstance(exec_globals, dict):
        return code
    code = "\n".join([l for l in code.split("\n") if l.strip() or l.startswith("#")])

    lines = code.split("\n")
    code_blocks = {}
    prv_index = 0
    prv_exec_globals = dict()
    for index in range(len(lines)):
        if index < len(lines) - 1 and len(lines[index + 1].lstrip()) != len(
            lines[index + 1]
        ):
            continue
        if index < len(lines) - 1 and lines[index].startswith("#"):
            continue
        if index < len(lines) - 1 and (
            lines[index + 1].startswith("else:") or lines[index + 1].startswith("elif ")
        ):
            continue
        try:
            _local_exec_globals = copy.deepcopy(prv_exec_globals)
        except Exception as e:
            _local_exec_globals = None
        if _local_exec_globals is not None:
            exec_globals = eval_code(
                "\n".join(lines[: index + 1]),
                exec_globals=copy.deepcopy(prv_exec_globals),
            )
        else:
            exec_globals = eval_code(
                "\n".join(lines[: index + 1]),
                exec_globals=dict(),
            )
        if not isinstance(exec_globals, dict):
            assert index != len(lines) - 1, (
                f"index == len(lines)-1, {index} == {len(lines)-1}",
                code,
            )
            continue
        exec_globals = {k: v for k, v in prv_exec_globals.items()}
        exec_globals = eval_code(
            "\n".join(lines[prv_index : index + 1]),
            exec_globals=exec_globals,
        )
        if not isinstance(exec_globals, dict):
            continue
        update_vars = [
            k
            for k in exec_globals
            if (
                not k.startswith("__")
                and k != "evaluate"
                and isinstance(k, str)
                and
                # callable(exec_globals[k]) and
                (
                    k not in [kk for kk in prv_exec_globals if isinstance(kk, str)]
                    or not _eq(exec_globals[k], prv_exec_globals[k])
                )
            )
        ]
        if len(update_vars) == 0:
            prv_index = index + 1
            continue
        assert tuple(sorted(set(update_vars))) not in code_blocks, (
            f"set(update_vars) in code_blocks, {set(update_vars)} in {code_blocks}",
            code,
        )
        code_blocks[tuple(sorted(set(update_vars)))] = "\n".join(
            lines[prv_index : index + 1]
        )
        prv_index = index + 1
        prv_exec_globals = exec_globals

    assert entry_point in prv_exec_globals, (
        f"entry_point not in prv_exec_globals, {entry_point} not in {prv_exec_globals}",
        code,
    )
    useful_vars = set([entry_point])
    while True:
        useful_code = "\n".join(
            [
                code
                for var_list, code in code_blocks.items()
                if set(var_list).intersection(useful_vars)
            ]
        )
        cur_useful_vars = useful_vars.copy()
        for var_list, code in code_blocks.items():
            if set(var_list).intersection(cur_useful_vars) or any(
                [v in useful_code for v in var_list]
            ):
                cur_useful_vars.update(var_list)
        if useful_vars == cur_useful_vars:
            break
        useful_vars = cur_useful_vars
    useful_code = "\n".join(
        [
            code
            for var_list, code in code_blocks.items()
            if code.strip().startswith("class ")
            or set(var_list).intersection(useful_vars)
        ]
    )

    return useful_code


def remove_noncompilable_code_blocks(code_blocks, prefix=""):
    idx = 0
    code_blocks = copy.deepcopy(code_blocks)
    while idx < len(code_blocks):
        code = "\n".join(code_blocks[: idx + 1])
        exec_globals = eval_code(prefix + "\n" + code, exec_globals=dict())
        if not isinstance(exec_globals, dict):
            exec_globals = eval_code(code, exec_globals=dict())
        if not isinstance(exec_globals, dict):
            code_blocks = code_blocks[:idx] + code_blocks[idx + 1 :]
        else:
            idx += 1
    return code_blocks


# ====== Eval Code ======

import contextlib
import io
import signal


@contextlib.contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield stream


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from."""

    def read(self, *args, **kwargs):
        raise OSError

    def readline(self, *args, **kwargs):
        raise OSError

    def readlines(self, *args, **kwargs):
        raise OSError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


def eval_code(line: str, timeout: float = 3.0, exec_globals: bool = None):
    try:
        exec_globals = {} if exec_globals is None else exec_globals
        with swallow_io() as s:
            with time_limit(timeout):
                exec(line, exec_globals)
        return exec_globals
    except TimeoutException:
        return "timed out"
    except BaseException as e:
        return str(e)
