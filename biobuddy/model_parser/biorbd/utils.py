from typing import Callable

import numpy as np


def tokenize_biomod(filepath: str) -> list[str]:
    # Load the model from the filepath
    with open(filepath, "r") as f:
        content = f.read()
    lines = content.splitlines()

    # Do a first pass to remove every commented content
    is_block_commenting = False
    line_index = 0
    for line_index in range(len(lines)):
        line = lines[line_index]
        # Remove everything after // or between /* */ (comments)
        if "/*" in line and "*/" in line:
            # Deal with the case where the block comment is on the same line
            is_block_commenting = False
            line = (line.split("/*")[0] + "" + line.split("*/")[1]).strip()
        if not is_block_commenting and "/*" in line:
            is_block_commenting = True
            line = line.split("/*")[0]
        if is_block_commenting and "*/" in line:
            is_block_commenting = False
            line = line.split("*/")[1]
        if is_block_commenting:
            line = ""

        line = line.split("//")[0]
        line = line.strip()
        lines[line_index] = line
    tokens = lines

    # Make spaces also a separator
    tokens_tp: list[str] = []
    for line in tokens:
        tokens_tp.extend(line.split(" "))
    tokens = [token for token in tokens_tp if token != ""]

    # Make tabs also a separator
    tokens_tp: list[str] = []
    for token in tokens:
        tokens_tp.extend(token.split("\t"))
    tokens = [token for token in tokens_tp if token != ""]

    return tokens


def check_if_version_defined(biomod_version: int):
    if biomod_version is None:
        raise ValueError("Version not defined")
    return


def read_str(next_token: Callable) -> str:
    return next_token()


def read_str_list(next_token: Callable, length: int) -> list[str]:
    str_list = [next_token() for _ in range(length)]
    return str_list


def read_int(next_token: Callable) -> int:
    return int(next_token())


def read_float(next_token: Callable) -> float:
    return float(next_token())


def read_bool(next_token: Callable) -> bool:
    return next_token() == "1"


def read_float_vector(next_token: Callable, length: int) -> np.ndarray:
    return np.array([read_float(next_token=next_token) for _ in range(length)])
