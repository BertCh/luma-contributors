# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Build website samples from the committed showcase and the BTS July 2023 ZIP."""
import base64
import csv
import datetime
import io
import json
import math
from pathlib import Path
import random
import struct
import sys
import zipfile

WEBSITE = Path(__file__).resolve().parents[1]
SHOWCASE = WEBSITE.parent / "examples/deck/spatial-analysis-showcase/public/data"
OUTPUT = WEBSITE / "src/gpu/data"


def column(dataset, filename, format_string):
    return struct.iter_unpack("<" + format_string, (SHOWCASE / dataset / filename).read_bytes())


def write_records(name, format_string, rows):
    records = b"".join(struct.pack("<" + format_string, *row) for row in rows)
    (OUTPUT / (name + ".b64")).write_text(base64.b64encode(records).decode("ascii") + "\n")


def lidar():
    dataset = "lidar-ground"
    origin_longitude, origin_latitude = -122.33976786609728, 47.595888866686835
    candidates = []
    for (longitude, latitude), (height,), (classification,) in zip(
        column(dataset, "position.bin", "dd"),
        column(dataset, "elevation.bin", "f"),
        column(dataset, "classification.bin", "I")
    ):
        # Rotate into a 128 m north by 72 m west window in the structured western strip.
        north = (latitude - origin_latitude) * 111195 + 70
        west = (origin_longitude - longitude) * 111195 * math.cos(math.radians(origin_latitude)) - 15
        if 0 <= north < 128 and 0 <= west < 72:
            candidates.append((round(north * 100), round(west * 100), round(height * 100), classification))
    write_records("king-county", "HHHB", random.Random(2282).sample(candidates, 9000))


def lightning():
    dataset = "poopdeck-goes-glm-lightning"
    candidates = []
    for (longitude, latitude), (seconds,), (energy,), (area,) in zip(
        column(dataset, "position.bin", "ff"),
        column(dataset, "time.bin", "I"),
        column(dataset, "energy.bin", "f"),
        column(dataset, "area.bin", "f")
    ):
        if -99 <= longitude < -83 and 36 <= latitude < 46:
            candidates.append((round((longitude + 99) * 1000), round((latitude - 36) * 1000),
                               seconds, round(energy), round(area)))
    write_records("midwest-flashes", "HHHHH", random.Random(20240521).sample(candidates, 4096))


def departures(archive):
    midnight = datetime.datetime(2023, 7, 1)
    departures = []
    with zipfile.ZipFile(archive) as bundle:
        filename = next(name for name in bundle.namelist() if name.endswith(".csv"))
        with io.TextIOWrapper(bundle.open(filename), encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if row["Origin"] != "ATL" or row["Cancelled"] != "0.00":
                    continue
                # DepDelay includes departures on a different date than scheduled.
                day = (datetime.datetime.fromisoformat(row["FlightDate"]) - midnight).days
                scheduled = int(row["CRSDepTime"])
                minutes = day * 1440 + scheduled // 100 * 60 + scheduled % 100 + round(float(row["DepDelay"]))
                departures.append(minutes)
    sample = random.Random(202307).sample(departures, 4000)
    (OUTPUT / "atlantadepartures.json").write_text(json.dumps(sorted(sample), separators=(",", ":")) + "\n")


if __name__ == "__main__":
    lidar()
    lightning()
    if len(sys.argv) > 1:
        departures(sys.argv[1])
