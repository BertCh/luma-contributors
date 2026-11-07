# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run with duckdb==1.4.1 and pyarrow==22.0.0; same typed Arrow values as GPU tests."""
import json
import duckdb
import pyarrow as arrow

assert duckdb.__version__ == '1.4.1'
assert arrow.__version__ == '22.0.0'
connection = duckdb.connect()
source = arrow.table({
    'select': arrow.array([2, 4, 8], type=arrow.float32()),
    'value.with.dot': arrow.array([10, 20, 30], type=arrow.float32()),
    'a"b': arrow.array([1, 2, 3], type=arrow.float32())
})
connection.register('trip data.v1', source)
quoted = connection.execute('SELECT "input"."select", "value.with.dot", "a""b", "select" * 2 AS "double ""value""" FROM "trip data.v1" AS "input" WHERE "select" >= 4 ORDER BY "double ""value""" DESC').fetchall()
connection.register('trips', arrow.table({'identifier': arrow.array([10, 20, 30, 40, 50], type=arrow.uint32())}))
connection.register('accounts', arrow.table({'identifier': arrow.array([20, 40], type=arrow.uint32())}))
matched = connection.execute('SELECT trips.identifier FROM trips JOIN accounts USING (identifier) ORDER BY identifier').fetchall()
unmatched = connection.execute('SELECT trips.identifier FROM trips ANTI JOIN accounts USING ("identifier") ORDER BY identifier').fetchall()
print(json.dumps({'duckdb': duckdb.__version__, 'pyarrow': arrow.__version__, 'quotedRows': quoted, 'innerJoinRows': matched, 'antiJoinRows': unmatched}, indent=2))
