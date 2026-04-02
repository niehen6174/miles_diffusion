import itertools
import json
import logging
import os
import random
import re

from miles.utils.types import Sample

__all__ = ["Dataset"]

logger = logging.getLogger(__name__)


def read_file(path):
    path, row_slice = _parse_generalized_path(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt dataset path '{path}' does not exist.")

    if path.endswith(".jsonl"):

        def jsonl_reader(p):
            with open(p, encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error at line {line_num}: {e}")
                        continue

        reader = jsonl_reader(path)

    else:
        raise ValueError(f"Unsupported file format: {path}. Supported format is .jsonl.")

    if row_slice is not None:

        logger.info("read_file path=%s applying slice row_slice=%s", path, row_slice)
        reader = itertools.islice(reader, row_slice.start, row_slice.stop, row_slice.step)

    yield from reader


def _parse_generalized_path(s: str):
    if (m := re.match(r"^(?P<real_path>.*)@\[(?P<start>-?\d*):(?P<end>-?\d*)\]$", s)) is not None:
        path = m.group("real_path")
        start = int(x) if (x := m.group("start")) != "" else None
        end = int(x) if (x := m.group("end")) != "" else None
        return path, slice(start, end)

    return s, None


class Dataset:
    """T2I RL: same loading pattern as :class:`miles.utils.data.Dataset` — ``read_file`` yields row dicts; we build :class:`~miles.utils.types.Sample`."""

    def __init__(
        self,
        path,
        *,
        prompt_key="text",
        metadata_key="metadata",
        seed=42,
    ):
        origin_samples = []
        for data in read_file(path):
            prompt = data.get(prompt_key)
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            metadata = data.get(metadata_key) or {}
            if not isinstance(metadata, dict):
                metadata = {}
            origin_samples.append(Sample(prompt=prompt.strip(), metadata=metadata))

        self.origin_samples = origin_samples
        self.epoch_id = -1
        self.seed = seed
        self.samples = self.origin_samples

    def shuffle(self, new_epoch_id: int) -> None:
        if self.epoch_id == new_epoch_id:
            return
        random.seed(self.seed + new_epoch_id)
        order = list(range(len(self.samples)))
        random.shuffle(order)
        self.samples = [self.origin_samples[i] for i in order]
        self.epoch_id = new_epoch_id

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]

    def __len__(self) -> int:
        return len(self.samples)
