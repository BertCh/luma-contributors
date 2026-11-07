# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Generate direct PostGIS ST_Subdivide references using a caller-supplied psql.

The connection comes from PGHOST/PGPORT/PGDATABASE/PGUSER or --connection.
PostGIS must already be installed in this isolated reference database.
Run with --write-sql to inspect the exact query without connecting to PostgreSQL.
The GPU contract uses unclosed piece rings; PostGIS max_vertices includes closure,
so its limits 5/9/17 correspond to GPU limits 4/8/16.
"""
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--psql', default='psql')
parser.add_argument('--connection')
parser.add_argument('--write-sql', type=Path)
arguments = parser.parse_args()
root = Path(__file__).parent
source = json.loads((root / 'fixtures' / 'geometry-advanced.json').read_text())
inputs = []
for source_id, fixture in enumerate(source['subdivision']):
    positions = fixture['positions']
    rings = []
    for first, last in zip(fixture['ringOffsets'], fixture['ringOffsets'][1:]):
        rings.append('(' + ','.join(f'{positions[vertex*2]:.17g} {positions[vertex*2+1]:.17g}' for vertex in range(first, last)) + ')')
    for maximum_vertices in [5, 9, 17]:
        inputs.append(dict(sourceId=source_id, maximumVertices=maximum_vertices,
                           wkt='POLYGON (' + ','.join(rings) + ')', **fixture))
payload = json.dumps(inputs, separators=(',', ':')).replace("'", "''")
sql = f"""-- Direct PostGIS query; all coordinates and sample queries are repository fixture inputs.
WITH inputs AS (
  SELECT input, (input->>'sourceId')::integer AS source_id,
    (input->>'maximumVertices')::integer AS maximum_vertices,
    ST_GeomFromText(input->>'wkt') AS source_geometry
  FROM jsonb_array_elements('{payload}'::jsonb) AS input
), pieces AS (
  SELECT inputs.*, split.geometry AS piece_geometry, split.ordinality AS piece_id
  FROM inputs
  CROSS JOIN LATERAL ST_Subdivide(source_geometry, maximum_vertices)
    WITH ORDINALITY AS split(geometry, ordinality)
), summaries AS (
  SELECT source_id, maximum_vertices, count(*) AS piece_count,
    max(ST_NPoints(piece_geometry)) AS largest_piece_vertices,
    sum(ST_Area(piece_geometry)) AS area,
    bool_and(ST_IsValid(piece_geometry)) AS valid,
    ST_UnaryUnion(ST_Collect(piece_geometry)) AS reconstructed,
    jsonb_agg(jsonb_build_object('pieceId',piece_id,'sourceId',source_id,
      'vertices',ST_NPoints(piece_geometry),'geometry',ST_AsGeoJSON(piece_geometry,17)::jsonb)
      ORDER BY piece_id) AS parts
  FROM pieces GROUP BY source_id, maximum_vertices
), results AS (
  SELECT jsonb_build_object('sourceId',inputs.source_id,
    'maximumVertices',inputs.maximum_vertices,
    'positions',input->'positions','ringOffsets',input->'ringOffsets',
    'queries',input->'queries','area',summaries.area,'parts',summaries.parts,
    'pieceCount',piece_count,'largestPieceVertices',largest_piece_vertices,
    'valid',valid,'reconstructsInput',ST_Equals(reconstructed,source_geometry),
    'contains',(SELECT jsonb_agg(ST_Covers(reconstructed,
       ST_Point((query->>0)::double precision,(query->>1)::double precision)) ORDER BY ordinal)
       FROM jsonb_array_elements(input->'queries') WITH ORDINALITY AS samples(query,ordinal))) AS result,
    inputs.source_id, inputs.maximum_vertices
  FROM inputs JOIN summaries USING (source_id,maximum_vertices)
)
SELECT jsonb_build_object('generator','geometry-subdivision-postgis.py',
  'versions',jsonb_build_object('postgresql',current_setting('server_version'),
    'postgis',postgis_lib_version(),'geos',postgis_geos_version()),
  'cases',(SELECT jsonb_agg(result ORDER BY source_id,maximum_vertices) FROM results));
"""
if arguments.write_sql:
    arguments.write_sql.write_text(sql)
else:
    command = [arguments.psql, '-X', '--quiet', '--no-psqlrc', '-A', '-t', '-v', 'ON_ERROR_STOP=1']
    if arguments.connection:
        command.extend(['--dbname', arguments.connection])
    result = json.loads(subprocess.check_output(command, input=sql.encode()))
    assert result['versions'] == {'postgresql': '16.6', 'postgis': '3.5.3', 'geos': '3.14.1-CAPI-1.20.5'}
    assert len(result['cases']) == len(inputs)
    for fixture in result['cases']:
        assert fixture['valid'] and fixture['reconstructsInput']
        assert fixture['largestPieceVertices'] <= fixture['maximumVertices']
        original = source['subdivision'][fixture['sourceId']]
        assert abs(fixture['area'] - original['area']) < 1e-5
        assert fixture['contains'] == original['contains']
    destination = root / 'fixtures' / 'geometry-subdivision-postgis.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(versions=result['versions'], cases=len(result['cases']), destination=str(destination))))
