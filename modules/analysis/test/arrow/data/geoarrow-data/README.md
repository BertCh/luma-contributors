# Data File Licenses

## geoarrow-data

The files under `geoarrow-data/` were copied from the GeoArrow project sample-data repository:

- Source repository: https://github.com/geoarrow/geoarrow-data
- Source tag: `v0.2.0`
- Source directory: `example/files`

The copied subset covers native GeoArrow separated and interleaved coordinate layouts for point,
linestring, polygon, multipoint, multilinestring, multipolygon, selected Z/M/ZM variants, and WKB/WKT
geometry samples. This directory retains provenance material; the current checkout has no
executable test or loader consuming these assets. The 2026-11-19 consumer audit found 52 `.arrows`
files plus this README, and no loader-to-test path through computed filenames, directory enumeration,
globs, URLs or IPC reads.

Current dense-union conversion tests in `modules/kepler/test/geometry-preparation/` and
`modules/kepler/test/resident-select-geometry.node.spec.ts` construct Arrow vectors from inline WKT,
WKB or their own JSON oracles and pass them to the published converter. They do not load this
sample directory. The fixture audit does not authorize removal or replace the provenance above;
retention or cleanup remains an owner decision.
