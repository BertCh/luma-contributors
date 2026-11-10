# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Refresh the statistics fixtures with the Python standard library. See statistics-sources.md."""

import csv
import gzip
import io
import json
import struct
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

WEBSITE = Path(__file__).resolve().parents[2]
DATA = WEBSITE / 'src/gpu/data'
SHOWCASE = WEBSITE.parent / 'examples/deck/spatial-analysis-showcase/public/data'


def write_rows(filename, rows):
    (DATA / filename).write_text(json.dumps(rows, separators=(',', ':')) + '\n')


def prepare_events():
    url = ('https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv'
           '&starttime=2018-05-01&endtime=2018-08-29&minlatitude=19.25&maxlatitude=19.65'
           '&minlongitude=-155.4&maxlongitude=-154.8&minmagnitude=3.5'
           '&orderby=time-asc&eventtype=earthquake&catalog=hv')
    epoch = datetime(2018, 5, 1, tzinfo=timezone.utc)
    with urllib.request.urlopen(url) as response:
        rows = []
        for record in csv.DictReader(io.TextIOWrapper(response, encoding='utf8')):
            if record['locationSource'] == 'hv':
                date = datetime.fromisoformat(record['time'].replace('Z', '+00:00'))
                rows.append([round(float(record['longitude']), 4),
                             round(float(record['latitude']), 4),
                             round((date - epoch).total_seconds() / 86400, 4)])
    write_rows('kilauea-events.json', rows)


def prepare_jobs():
    counts = {tract['geoid']: [0] * 6 for tract in json.loads((DATA / 'chicago-tracts.json').read_text())}
    for period, year in enumerate(range(2018, 2024)):
        url = f'https://lehd.ces.census.gov/data/lodes/LODES8/il/wac/il_wac_S000_JT00_{year}.csv.gz'
        with urllib.request.urlopen(url) as response, gzip.GzipFile(fileobj=response) as archive:
            for block in csv.DictReader(io.TextIOWrapper(archive, encoding='utf8')):
                tract = block['w_geocode'][:11]
                if tract in counts:
                    counts[tract][period] += int(block['C000'])
    write_rows('chicago-workplace-jobs.json', counts)


def prepare_births():
    def read_column(dataset, column, fmt):
        return [value[0] for value in struct.iter_unpack(fmt, (SHOWCASE / dataset / (column + '.bin')).read_bytes())]

    fips = read_column('us-counties', 'fips', '<I')
    centers = read_column('us-counties', 'centroid', '<f')
    births = read_column('us-county-births', 'births', '<f')
    exposure = read_column('us-county-births', 'womenYears', '<f')
    rows = [[identifier, round(centers[index * 2], 4), round(centers[index * 2 + 1], 4),
             int(births[index]), int(exposure[index])]
            for index, identifier in enumerate(fips) if identifier // 1000 == 17]
    write_rows('illinois-births.json', rows)


if __name__ == '__main__':
    prepare_events()
    prepare_jobs()
    prepare_births()
