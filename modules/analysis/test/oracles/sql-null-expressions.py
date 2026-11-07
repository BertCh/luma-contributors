# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Direct reference: duckdb==1.4.1 and pyarrow==22.0.0, matching SQL GPU fixtures.

Run this file to regenerate fixtures/sql-null-expressions.json.
"""
import json
from pathlib import Path
import duckdb
import pyarrow as arrow

assert duckdb.__version__ == '1.4.1'
assert arrow.__version__ == '22.0.0'
connection = duckdb.connect()
connection.register('trips', arrow.table({
    'fare': arrow.array([None, 8, 3, 10, 11], type=arrow.float32()),
    'category': arrow.array([0, 1, 0, 1, 0], type=arrow.uint32())
}))
expressions = [
    'COALESCE(NULL, fare, 7)',
    'CASE WHEN fare >= 8 THEN fare WHEN fare IS NULL THEN 7 ELSE NULL END',
    'CASE fare WHEN NULL THEN 100 WHEN 8 THEN 80 WHEN 3 THEN 30 ELSE 7 END',
    'CASE WHEN NULL THEN 100 WHEN FALSE THEN 200 ELSE 7 END',
    'CASE WHEN fare > 9 THEN fare END',
    'COALESCE(NULL, CASE WHEN fare > 9 THEN fare END)',
    'COALESCE(NULL, NULL)'
]
queries = [
    'SELECT COUNT(*) AS rows, COUNT(fare) AS present FROM trips',
    'SELECT COUNT(*) AS rows, COUNT(fare) AS present FROM trips WHERE fare IS NULL',
    'SELECT COUNT(*) AS rows, COUNT(fare) AS present FROM trips WHERE FALSE',
    'SELECT category, COUNT(*) AS rows, COUNT(fare) AS present FROM trips GROUP BY category ORDER BY category'
]
result = {
    'reference': {'duckdb': duckdb.__version__, 'pyarrow': arrow.__version__},
    'expressions': [{'expression': expression, 'values': [row[0] for row in connection.execute(f'SELECT {expression} FROM trips').fetchall()]} for expression in expressions],
    'counts': [{'query': query, 'rows': connection.execute(query).fetchall()} for query in queries]
}
connection.register('nonfinite', arrow.table({'value': arrow.array([float('nan'), float('inf'), -float('inf'), None], type=arrow.float32())}))
result['nonfiniteCount'] = connection.execute('SELECT COUNT(value) FROM nonfinite').fetchone()[0]
Path(__file__).with_name('fixtures').joinpath('sql-null-expressions.json').write_text(json.dumps(result, indent=2) + '\n')
