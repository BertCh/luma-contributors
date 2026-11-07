# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Direct pinned pgRouting/MobilityDB references; connection uses standard PG* variables.

Run with --psql /path/to/psql after installing the pinned extensions.
All inputs, queries, and normalization are explicit; no Python spatial solver is used.
"""
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--psql', default='psql')
arguments = parser.parse_args()
command = [arguments.psql, '-X', '-A', '-t', '-v', 'ON_ERROR_STOP=1']


def query(sql):
    return json.loads(subprocess.check_output(command, input=sql.encode()))


versions = query("""SELECT json_build_object('postgresql',current_setting('server_version'),
 'postgis',postgis_lib_version(),'pgrouting',pgr_version(),
 'mobilitydb',mobilitydb_version());""")
assert versions == dict(postgresql='16.6', postgis='3.5.3', pgrouting='3.8.0', mobilitydb='MobilityDB 1.2.1')
edges = 'SELECT * FROM (VALUES (1,0,1,4.0,-1.0),(2,1,2,4.0,-1.0),(3,0,1,6.0,-1.0)) AS edges(id,source,target,cost,reverse_cost)'
points = "SELECT * FROM (VALUES (1,1,0.25,'b'::char),(2,1,0.75,'b'::char),(3,2,0.5,'b'::char),(4,3,0.25,'b'::char)) AS points(pid,edge_id,fraction,side)"
routing = []
for start, end, source_edge, end_edge, source_fraction, end_fraction in [
    (1,2,0,0,.25,.75), (1,3,0,1,.25,.5), (2,1,0,0,.75,.25), (4,3,2,1,.25,.5)
]:
    cost = query(f"SELECT coalesce(to_json((SELECT agg_cost FROM pgr_withPointsCost($edges${edges}$edges$, $points${points}$points$, {-start}::bigint, {-end}::bigint, true, 'b'))), 'null'::json) ;")
    routing.append(dict(query=[source_edge,end_edge], fractions=[source_fraction,end_fraction], cost=cost))
assert [entry['cost'] for entry in routing] == [2,5,None,6.5]

epoch = "'2000-01-01 00:00:00+00'::timestamptz"
trajectory = "'[Point(-1 2)@2000-01-01 00:00:00+00, Point(5 2)@2000-01-01 00:00:06+00]'::tgeompoint"
polygon = "'POLYGON((0 0,4 0,4 4,0 4,0 0),(1 1,3 1,3 3,1 3,1 1))'::geometry"
clip = query(f"""WITH clipped AS (SELECT atGeometry({trajectory},{polygon}) AS value),
 parts AS (SELECT unnest(sequences(value)) AS value FROM clipped)
 SELECT json_agg(json_build_object('interval',ARRAY[extract(epoch FROM startTimestamp(value)-{epoch}),extract(epoch FROM endTimestamp(value)-{epoch})],
 'closed',ARRAY[lowerInc(value),upperInc(value)]) ORDER BY startTimestamp(value)) FROM parts;""")
assert [entry['interval'] for entry in clip] == [[1,2],[4,5]]

time_boundaries = []
for policy, lower, upper in [('closed','[',']'),('open','(',')'),('left-closed','[',')'),('right-closed','(',']')]:
    value = f"atTime({trajectory}, '{lower}2000-01-01 00:00:01+00,2000-01-01 00:00:05+00{upper}'::tstzspan)"
    result = query(f"SELECT json_build_object('interval',ARRAY[extract(epoch FROM startTimestamp({value})-{epoch}),extract(epoch FROM endTimestamp({value})-{epoch})], 'closed',ARRAY[lowerInc({value}),upperInc({value})]);")
    time_boundaries.append(dict(policy=policy, **result))

first = "'[Point(-2 0)@2000-01-01 00:00:00+00,Point(2 0)@2000-01-01 00:00:04+00]'::tgeompoint"
second = "'[Point(0 0)@2000-01-01 00:00:00+00,Point(0 0)@2000-01-01 00:00:04+00]'::tgeompoint"
absent = "'[Point(0 0)@2000-01-01 00:00:10+00,Point(1 0)@2000-01-01 00:00:11+00]'::tgeompoint"
pairwise = query(f"""WITH encounter AS (SELECT tDwithin({first},{second},1,true) AS value)
 SELECT json_build_object('interval',ARRAY[extract(epoch FROM startTimestamp(value)-{epoch}),extract(epoch FROM endTimestamp(value)-{epoch})],
 'closed',ARRAY[lowerInc(value),upperInc(value)],'ever',eDwithin({first},{second},1),'always',always_eq(tDwithin({first},{second},1),true),'directADwithin',aDwithin({first},{second},1),
 'disjointEver',eDwithin({first},{absent},1),'disjointAlways',aDwithin({first},{absent},1)) FROM encounter;""")
assert pairwise['interval'] == [1,3] and pairwise['ever'] and not pairwise['always']
assert pairwise['disjointEver'] is None and pairwise['disjointAlways'] is None
result = dict(generator=Path(__file__).name, versions=versions, routing=routing,
              clip=clip, timeBoundaries=time_boundaries, pairwise=pairwise,
              normalization='SQL NULL unreachable cost maps to GPU +Infinity; NULL temporal relation maps to uint32 0xffffffff. Seconds are timestamp differences from explicit UTC epoch.',
              referenceDifference='MobilityDB 1.2.1 aDwithin returns true for this partial crossing; always_eq(tDwithin(first,second,1),true) returns false and is the reference for full-common-support always semantics. The raw aDwithin result is retained, not matched by the GPU.')
destination = Path(__file__).parent / 'fixtures' / 'network-movement-postgres.json'
destination.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(versions=versions, destination=str(destination))))
