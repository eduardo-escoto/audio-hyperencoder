from typing import Union
from pathlib import Path
from collections import defaultdict
from collections.abc import Generator

from regex import Pattern


def get_file_paths_by_pattern(
    directory: Union[Path, str], filename_pattern: Pattern
) -> Generator[Path]:
    if type(directory) is str:
        directory = Path(directory)

    for file in directory.rglob("*"):
        if filename_pattern.match(file.name):
            yield file


def group_paths_by_pattern(
    file_paths: list[Path], group_pattern: Pattern
) -> dict[str, list[Path]]:
    group_dict: dict[str, list] = defaultdict(list)

    for file_path in file_paths:
        group_key = group_pattern.search(str(file_path)).group()
        group_dict[group_key].append(file_path)

    return group_dict
