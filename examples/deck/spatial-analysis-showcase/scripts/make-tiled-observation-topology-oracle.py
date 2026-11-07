# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent whole-focus 8-connected and four-edge adjacency reference."""

import json
import math
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "public/data/dixie-spectral"
ORACLE = ROOT / "../../../dev-docs/research/showcase-overhaul-2026-10-25/earth-source-oracles.json"
SOURCE_SIZE = 250
FOCUS_ROW = 82
FOCUS_COLUMN = 92
FOCUS_SIZE = 64
TILE_SIZE = 32
THRESHOLD = 0.2


def read_float_array(path: Path) -> list[float]:
    raw = path.read_bytes()
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def components(mask: list[bool], size: int) -> tuple[list[int], list[int]]:
    labels = [-1] * (size * size)
    sizes: list[int] = []
    for start, foreground in enumerate(mask):
        if not foreground or labels[start] >= 0:
            continue
        label = len(sizes)
        labels[start] = label
        stack = [start]
        population = 0
        while stack:
            index = stack.pop()
            population += 1
            row, column = divmod(index, size)
            for row_delta in (-1, 0, 1):
                for column_delta in (-1, 0, 1):
                    if row_delta == 0 and column_delta == 0:
                        continue
                    next_row = row + row_delta
                    next_column = column + column_delta
                    if not (0 <= next_row < size and 0 <= next_column < size):
                        continue
                    next_index = next_row * size + next_column
                    if mask[next_index] and labels[next_index] < 0:
                        labels[next_index] = label
                        stack.append(next_index)
        sizes.append(population)
    return labels, sizes


def make_record(date: str, filename: str, validity_filename: str) -> dict:
    values = read_float_array(DATA / filename)
    validity = DATA.joinpath(validity_filename).read_bytes()
    focus_values: list[float] = []
    focus_validity: list[bool] = []
    for row in range(FOCUS_SIZE):
        for column in range(FOCUS_SIZE):
            source_index = (FOCUS_ROW + row) * SOURCE_SIZE + FOCUS_COLUMN + column
            value = values[source_index]
            finite = validity[source_index] == 1 and math.isfinite(value)
            focus_values.append(value)
            focus_validity.append(finite)

    class_labels: dict[str, list[int]] = {}
    class_records = {}
    for class_name, predicate in (
        ("low", lambda value: value < THRESHOLD),
        ("high", lambda value: value >= THRESHOLD),
    ):
        mask = [valid and predicate(value) for valid, value in zip(focus_validity, focus_values)]
        labels, populations = components(mask, FOCUS_SIZE)
        class_labels[class_name] = labels
        local_counts = []
        for tile_row in range(2):
            for tile_column in range(2):
                local_mask = []
                for row in range(TILE_SIZE):
                    source_row = tile_row * TILE_SIZE + row
                    for column in range(TILE_SIZE):
                        source_column = tile_column * TILE_SIZE + column
                        source_index = source_row * FOCUS_SIZE + source_column
                        local_mask.append(mask[source_index])
                _, local_populations = components(local_mask, TILE_SIZE)
                local_counts.append(len(local_populations))
        class_records[class_name] = {
            "validPopulation": sum(mask),
            "globalComponentCount": len(populations),
            "globalComponentPopulations": populations,
            "localComponentCounts": local_counts,
            "localComponentCountSum": sum(local_counts),
            "seamMerges": sum(local_counts) - len(populations),
        }

    ids = [-1] * (FOCUS_SIZE * FOCUS_SIZE)
    for index in range(len(ids)):
        if not focus_validity[index]:
            continue
        low_label = class_labels["low"][index]
        high_label = class_labels["high"][index]
        ids[index] = low_label if low_label >= 0 else 128 + high_label

    contacts: dict[tuple[int, int], int] = {}
    horizontal_seam_contacts = 0
    vertical_seam_contacts = 0
    for row in range(FOCUS_SIZE):
        for column in range(FOCUS_SIZE):
            first_index = row * FOCUS_SIZE + column
            first_id = ids[first_index]
            if first_id < 0:
                continue
            for row_delta, column_delta in ((0, 1), (1, 0)):
                next_row = row + row_delta
                next_column = column + column_delta
                if next_row >= FOCUS_SIZE or next_column >= FOCUS_SIZE:
                    continue
                second_index = next_row * FOCUS_SIZE + next_column
                second_id = ids[second_index]
                if second_id < 0 or second_id == first_id:
                    continue
                pair = (min(first_id, second_id), max(first_id, second_id))
                contacts[pair] = contacts.get(pair, 0) + 1
                if column == TILE_SIZE - 1 and column_delta:
                    horizontal_seam_contacts += 1
                if row == TILE_SIZE - 1 and row_delta:
                    vertical_seam_contacts += 1

    return {
        "date": date,
        "source": filename,
        "validPopulation": sum(focus_validity),
        "threshold": THRESHOLD,
        "connectivity": 8,
        "focus": {
            "sourceRowsHalfOpen": [FOCUS_ROW, FOCUS_ROW + FOCUS_SIZE],
            "sourceColumnsHalfOpen": [FOCUS_COLUMN, FOCUS_COLUMN + FOCUS_SIZE],
            "ownedCoreSize": TILE_SIZE,
            "ownedCoreOrigins": [[0, 0], [32, 0], [0, 32], [32, 32]],
        },
        "classes": class_records,
        "adjacency": {
            "connectivity": 4,
            "lowHighRegionPairCount": len(contacts),
            "lowHighSharedEdgeContacts": sum(contacts.values()),
            "focusSeamContacts": horizontal_seam_contacts + vertical_seam_contacts,
            "horizontalFocusSeamContacts": horizontal_seam_contacts,
            "verticalFocusSeamContacts": vertical_seam_contacts,
            "pairs": [
                {"lowRegion": first, "highRegion": second - 128, "sharedEdges": count}
                for (first, second), count in sorted(contacts.items())
            ],
        },
    }


def main() -> None:
    oracle_path = ORACLE.resolve()
    oracle = json.loads(oracle_path.read_text())
    oracle["tiledObservationTopology"] = {
        "method": "independent stdlib flood-fill over bundled calibrated NIR and strict QA validity; 8-neighbor components, four-neighbor unique edge contacts",
        "cellSizeMeters": 60,
        "records": [
            make_record("2021-07-13", "nir_before.bin", "validity_before.bin"),
            make_record("2021-09-21", "nir_after.bin", "validity_after.bin"),
        ],
    }
    oracle_path.write_text(json.dumps(oracle, indent=2) + "\n")
    for record in oracle["tiledObservationTopology"]["records"]:
        low = record["classes"]["low"]
        high = record["classes"]["high"]
        adjacency = record["adjacency"]
        print(
            f"{record['date']}: valid {record['validPopulation']}, "
            f"low {low['validPopulation']}/{low['localComponentCountSum']}→{low['globalComponentCount']}, "
            f"high {high['validPopulation']}/{high['localComponentCountSum']}→{high['globalComponentCount']}, "
            f"pairs {adjacency['lowHighRegionPairCount']}, contacts {adjacency['lowHighSharedEdgeContacts']}, "
            f"focus-seam contacts {adjacency['focusSeamContacts']}"
        )


if __name__ == "__main__":
    main()
