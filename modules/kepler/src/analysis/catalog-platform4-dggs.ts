// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCellAggregation,
  GPUCellGeometry,
  GPUCellRollup,
  GPUCellTableCompare,
  GPUPointToCell,
  GPU_CELL_DEFAULT_SUM_SCALE,
  GPU_CELL_INDEX_RESOLUTION_RANGES,
  type GPUCellIndexFamily,
  type GPUCellTable
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {
  CatalogOperationBinding,
  CatalogParameterSpec,
  CatalogRunOutput
} from './catalog-binding-types';
import {
  appendCatalogRow,
  getPointPositions,
  makeCatalogGeometryText,
  makeCatalogTable,
  type CatalogDiagnostic,
  type CatalogFeatureCollection,
  type CatalogOutputTable
} from './catalog-geojson';
import {
  CatalogGpuUnsupportedError,
  withCatalogGpuSession,
  type CatalogGpuOutput,
  type CatalogGpuSession
} from './catalog-gpu-session';
import {
  PLATFORM3_GPU_OR_CPU_CLEANUP,
  makeInputSpec,
  makeNumberParameter,
  makeOutputSpec,
  makeStringParameter,
  readNumericProperty,
  type Values,
  addDroppedDiagnostic as addSharedDroppedDiagnostic
} from './catalog-platform3-b-common';
import {
  DGGS_CPU_FAMILIES,
  formatDggsCellId,
  getDggsCpuBoundary,
  getDggsCpuCell,
  getQuadbinParent,
  joinDggsWords,
  type DggsCpuFamily
} from './catalog-platform4-dggs-cells';

const POINTS_INPUT = makeInputSpec('points', 'Points', ['point']);
const MAXIMUM_CELL_ROWS = 100000;

const AGGREGATION_FAMILIES = ['quadbin', 'h3'] as const;
type AggregationFamily = (typeof AGGREGATION_FAMILIES)[number];

const H3_CPU_NOTE =
  'The H3 family has no CPU engine: it needs a GPU device and throws without one (Quadbin, quadkey, geohash and S2 keep their CPU reference).';

const DGGS_BEHAVIOR = [
  'Points are float32 longitude/latitude degrees; positions outside [-180, 180] x [-90, 90], without a point position or (when a value property is set) without a numeric value are skipped and counted.',
  'Quadbin and quadkey follow the Web Mercator tile grid (latitude clipped to +/-85.051129), geohash bisects the geographic extent, S2 and H3 are spherical grids; the S2 boundary is published with straight edges between its four corners.',
  H3_CPU_NOTE
];

type GeographicSamples = {
  positions: Float32Array;
  values: Float32Array;
  featureIndex: Int32Array;
  partIndex: Int32Array;
  count: number;
  droppedCount: number;
};

/** Longitude/latitude samples of a point collection; invalid rows are dropped and counted. */
function readGeographicSamples(
  collection: CatalogFeatureCollection,
  valueProperty: string
): GeographicSamples {
  const positions: number[] = [];
  const values: number[] = [];
  const featureIndex: number[] = [];
  const partIndex: number[] = [];
  let droppedCount = 0;
  collection.features.forEach((feature, index) => {
    const points = getPointPositions(feature.geometry);
    const numeric = valueProperty ? readNumericProperty(feature, valueProperty) : 1;
    if (!points.length || numeric === null) {
      droppedCount++;
      return;
    }
    points.forEach((position, part) => {
      const [longitude, latitude] = position;
      if (
        !Number.isFinite(longitude) ||
        !Number.isFinite(latitude) ||
        Math.abs(longitude) > 180 ||
        Math.abs(latitude) > 90
      ) {
        droppedCount++;
        return;
      }
      positions.push(longitude, latitude);
      values.push(numeric);
      featureIndex.push(index);
      partIndex.push(part);
    });
  });
  return {
    positions: Float32Array.from(positions),
    values: Float32Array.from(values),
    featureIndex: Int32Array.from(featureIndex),
    partIndex: Int32Array.from(partIndex),
    count: values.length,
    droppedCount
  };
}

const DROPPED_FEATURE_MESSAGE =
  'Points that are outside the longitude/latitude range, have no position or have no numeric value are skipped.';

function addDroppedDiagnostic(diagnostics: CatalogDiagnostic[], droppedCount: number): void {
  addSharedDroppedDiagnostic(diagnostics, droppedCount, DROPPED_FEATURE_MESSAGE);
}

function checkResolution(family: GPUCellIndexFamily, resolution: number): void {
  const {minimum, maximum} = GPU_CELL_INDEX_RESOLUTION_RANGES[family];
  // Resolution is outside the valid range of the chosen grid family.
  if (!Number.isInteger(resolution) || resolution < minimum || resolution > maximum)
    throw new RangeError(`${family} resolution must be an integer in [${minimum}, ${maximum}]`);
}

function requireCpuFamily(family: string): DggsCpuFamily {
  // The H3 family needs the GPU contributors; the CPU engine cannot index it.
  if (!(DGGS_CPU_FAMILIES as readonly string[]).includes(family))
    throw new Error('H3 has no CPU engine; supply a GPU device');
  return family as DggsCpuFamily;
}

function getBoundaryText(boundary: readonly (readonly [number, number])[] | null): string | null {
  return boundary && boundary.length >= 4 ? makeCatalogGeometryText('Polygon', [boundary]) : null;
}

// ---------------------------------------------------------------------------------------------
// Cell tables

export type DggsCellRow = {
  key: bigint;
  count: number;
  sum: number;
  minimum: number;
  maximum: number;
};

/** CPU aggregation of samples per cell, ascending by key. */
function aggregateCpuCells(
  samples: GeographicSamples,
  family: DggsCpuFamily,
  resolution: number
): DggsCellRow[] {
  const cells = new Map<bigint, DggsCellRow>();
  for (let index = 0; index < samples.count; index++) {
    const key = getDggsCpuCell(
      family,
      samples.positions[2 * index],
      samples.positions[2 * index + 1],
      resolution
    );
    if (key === 0n) continue;
    const value = samples.values[index];
    const row = cells.get(key);
    if (row) {
      row.count++;
      row.sum += value;
      row.minimum = Math.min(row.minimum, value);
      row.maximum = Math.max(row.maximum, value);
    } else cells.set(key, {key, count: 1, sum: value, minimum: value, maximum: value});
  }
  return [...cells.values()].sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
}

type TableOutputs = {
  table: GPUCellTable;
  cells: CatalogGpuOutput<'uint32x2'>;
  counts: CatalogGpuOutput<'uint32'>;
  sums: CatalogGpuOutput<'uint32x2'>;
  minimums: CatalogGpuOutput<'float32'>;
  maximums: CatalogGpuOutput<'float32'>;
  count: CatalogGpuOutput<'uint32'>;
  overflow: CatalogGpuOutput<'uint32'>;
  candidateOverflow: CatalogGpuOutput<'uint32'>;
};

function makeTableOutputs(
  session: CatalogGpuSession,
  name: string,
  capacity: number
): TableOutputs {
  const cells = session.output(`${name}-cells`, 'uint32x2', capacity);
  const counts = session.output(`${name}-counts`, 'uint32', capacity);
  const sums = session.output(`${name}-sums`, 'uint32x2', capacity);
  const minimums = session.output(`${name}-minimums`, 'float32', capacity);
  const maximums = session.output(`${name}-maximums`, 'float32', capacity);
  const count = session.output(`${name}-count`, 'uint32', 1);
  const overflow = session.output(`${name}-overflow`, 'uint32', 1);
  const candidateOverflow = session.output(`${name}-candidate-overflow`, 'uint32', 1);
  return {
    table: {
      cells: cells.view,
      counts: counts.view,
      sums: sums.view,
      minimums: minimums.view,
      maximums: maximums.view,
      status: {
        count: count.view,
        overflow: overflow.view,
        candidateOverflow: candidateOverflow.view
      }
    },
    cells,
    counts,
    sums,
    minimums,
    maximums,
    count,
    overflow,
    candidateOverflow
  };
}

async function readTableRows(
  session: CatalogGpuSession,
  outputs: TableOutputs
): Promise<DggsCellRow[]> {
  const rowCount = await session.readScalar(outputs.count);
  if (
    (await session.readScalar(outputs.overflow)) ||
    (await session.readScalar(outputs.candidateOverflow))
  ) {
    throw new CatalogGpuUnsupportedError(
      'Cell table is incomplete; increase the source or destination capacity'
    );
  }
  const cellWords = await session.readUint32(outputs.cells, rowCount);
  const counts = await session.readUint32(outputs.counts, rowCount);
  const sumWords = await session.readUint32(outputs.sums, rowCount);
  const minimums = await session.readFloat32(outputs.minimums, rowCount);
  const maximums = await session.readFloat32(outputs.maximums, rowCount);
  const rows: DggsCellRow[] = [];
  for (let row = 0; row < rowCount; row++)
    rows.push({
      key: joinDggsWords(cellWords[2 * row], cellWords[2 * row + 1]),
      count: counts[row],
      sum:
        Number(BigInt.asIntN(64, joinDggsWords(sumWords[2 * row], sumWords[2 * row + 1]))) /
        GPU_CELL_DEFAULT_SUM_SCALE,
      minimum: minimums[row],
      maximum: maximums[row]
    });
  return rows;
}

/** Adds key and aggregation nodes for one point set; returns the table outputs. */
function addAggregationNodes(
  session: CatalogGpuSession,
  name: string,
  samples: GeographicSamples,
  family: AggregationFamily,
  resolution: number
): TableOutputs {
  const keys = session.output(`${name}-keys`, 'uint32x2', samples.count);
  session.graph.add(
    new GPUPointToCell({
      id: `${name}-point-to-cell`,
      family,
      resolution,
      precision: 'extended',
      positions: session.input(`${name}-positions`, samples.positions, 'float32x2'),
      output: {cells: keys.view}
    })
  );
  const outputs = makeTableOutputs(session, name, samples.count);
  session.graph.add(
    new GPUCellAggregation({
      id: `${name}-aggregation`,
      family,
      resolution,
      cells: keys.view,
      values: session.input(`${name}-values`, samples.values, 'float32'),
      output: outputs.table
    })
  );
  return outputs;
}

type GpuBoundaries = Map<bigint, readonly (readonly [number, number])[] | null>;

/** Adds a GPUCellGeometry node for a table and returns a reader of its closed boundary rings. */
function addBoundaryNodes(
  session: CatalogGpuSession,
  name: string,
  family: AggregationFamily,
  outputs: TableOutputs,
  capacity: number
): (rows: readonly DggsCellRow[]) => Promise<GpuBoundaries> {
  const maximumVertexCount = family === 'h3' ? 10 : 4;
  const boundaries = session.output(
    `${name}-boundaries`,
    'float32x2',
    capacity * maximumVertexCount
  );
  const vertexCounts = session.output(`${name}-vertex-counts`, 'uint32', capacity);
  session.graph.add(
    new GPUCellGeometry({
      id: `${name}-geometry`,
      family,
      cells: outputs.table.cells,
      maximumVertexCount,
      output: {boundaries: boundaries.view, vertexCounts: vertexCounts.view}
    })
  );
  return async rows => {
    const coordinates = await session.readFloat32(boundaries, rows.length * maximumVertexCount);
    const counts = await session.readUint32(vertexCounts, rows.length);
    const result: GpuBoundaries = new Map();
    rows.forEach((row, index) => {
      const ring: [number, number][] = [];
      for (let vertex = 0; vertex < counts[index]; vertex++) {
        const offset = 2 * (index * maximumVertexCount + vertex);
        ring.push([coordinates[offset], coordinates[offset + 1]]);
      }
      if (ring.length) ring.push(ring[0]);
      result.set(row.key, ring.length ? ring : null);
    });
    return result;
  };
}

const CELL_COLUMN_TYPES = {
  cell_id: 'string',
  family: 'string',
  resolution: 'int32',
  count: 'int32',
  sum: 'float64',
  mean: 'float64',
  min: 'float64',
  max: 'float64',
  geometry: 'string'
} as const;

function appendCellRow(
  table: CatalogOutputTable,
  row: DggsCellRow,
  family: string,
  resolution: number,
  hasValues: boolean,
  geometry: string | null,
  extra: Record<string, string | number | null> = {}
): void {
  appendCatalogRow(table, {
    cell_id: formatDggsCellId(row.key),
    family,
    resolution,
    count: row.count,
    sum: hasValues ? row.sum : null,
    mean: hasValues ? row.sum / row.count : null,
    min: hasValues ? row.minimum : null,
    max: hasValues ? row.maximum : null,
    geometry,
    ...extra
  });
}

// ---------------------------------------------------------------------------------------------
// dggs-point-index

const FAMILY_PARAMETER: CatalogParameterSpec = {
  name: 'family',
  label: 'Grid family',
  kind: 'enum',
  options: ['quadbin', 'quadkey', 'geohash', 's2', 'h3'],
  defaultValue: 'quadbin'
};

const AGGREGATION_FAMILY_PARAMETER: CatalogParameterSpec = {
  name: 'family',
  label: 'Grid family',
  kind: 'enum',
  options: [...AGGREGATION_FAMILIES],
  defaultValue: 'quadbin'
};

const RESOLUTION_PARAMETER = makeNumberParameter(
  'resolution',
  'Resolution (Quadbin 0-26, H3 0-15, quadkey 1-29, geohash 1-12, S2 0-30)',
  8,
  0,
  30,
  undefined,
  'integer'
);

function buildPointIndexOutput(
  samples: GeographicSamples,
  keys: readonly bigint[],
  family: string,
  resolution: number,
  invalidCount: number
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  addDroppedDiagnostic(diagnostics, samples.droppedCount + invalidCount);
  const table = makeCatalogTable(
    'point-cells',
    'Point cell keys',
    {
      feature_index: 'int32',
      part_index: 'int32',
      cell_id: 'string',
      family: 'string',
      resolution: 'int32',
      geometry: 'string'
    },
    'geometry'
  );
  keys.forEach((key, index) => {
    if (key === 0n) return;
    appendCatalogRow(table, {
      feature_index: samples.featureIndex[index],
      part_index: samples.partIndex[index],
      cell_id: formatDggsCellId(key),
      family,
      resolution,
      geometry: makeCatalogGeometryText('Point', [
        samples.positions[2 * index],
        samples.positions[2 * index + 1]
      ])
    });
  });
  return {
    tables: [table],
    diagnostics,
    summary: {
      pointCount: table.rowCount,
      cellCount: new Set(table.columns['cell_id']).size,
      family,
      resolution
    }
  };
}

const POINT_INDEX_BINDING: CatalogOperationBinding = {
  id: 'dggs-point-index',
  title: 'Discrete global grid point index',
  domain: 'aggregation',
  summary:
    'Indexes each point into a Quadbin, quadkey, geohash, S2 or H3 cell and publishes the 64-bit cell id (hex).',
  capabilityIds: ['dggs-indexing-and-topology'],
  contributorNames: ['GPUPointToCell'],
  gpuContributorNames: ['GPUPointToCell'],
  engine: 'gpu-or-cpu',
  inputs: [POINTS_INPUT],
  parameters: [FAMILY_PARAMETER, RESOLUTION_PARAMETER],
  outputs: [
    makeOutputSpec(
      'point-cells',
      'Point cell keys',
      'point',
      ['feature_index', 'part_index', 'cell_id'],
      200000
    )
  ],
  cleanup: PLATFORM3_GPU_OR_CPU_CLEANUP,
  declaredBehavior: [
    ...DGGS_BEHAVIOR,
    'The GPU kernels use f32 trigonometry: Quadbin, quadkey and geohash keys are exact, S2 keys match the CPU reference through level 10. H3 keys use double-float (extended) arithmetic and equal h3-js on the same float32 positions at every resolution 0-15 (verified on Metal; the float32 input itself is about a metre coarse, so finest resolutions index the rounded position).'
  ],
  limitation:
    'Indexing only; use dggs-cell-aggregation to count per cell. Cell ids are published as hex text because the table schema has no 64-bit integer.',
  run: (inputs, values) => {
    const family = requireCpuFamily(values['family'] as string);
    const resolution = values['resolution'] as number;
    checkResolution(family, resolution);
    const samples = readGeographicSamples(inputs['points'], '');
    const keys: bigint[] = [];
    for (let index = 0; index < samples.count; index++)
      keys.push(
        getDggsCpuCell(
          family,
          samples.positions[2 * index],
          samples.positions[2 * index + 1],
          resolution
        )
      );
    return buildPointIndexOutput(samples, keys, family, resolution, 0);
  },
  runGpu: async (context, inputs, values) => {
    const family = values['family'] as GPUCellIndexFamily;
    const resolution = values['resolution'] as number;
    checkResolution(family, resolution);
    const samples = readGeographicSamples(inputs['points'], '');
    if (!samples.count) throw new CatalogGpuUnsupportedError('No indexable points');
    const keys = await withCatalogGpuSession(
      context.device,
      'catalog-dggs-point-index',
      async session => {
        const output = session.output('keys', 'uint32x2', samples.count);
        session.graph.add(
          new GPUPointToCell({
            id: 'catalog-dggs-point-index',
            family,
            resolution,
            precision: 'extended',
            positions: session.input('positions', samples.positions, 'float32x2'),
            output: {cells: output.view}
          })
        );
        session.run();
        return session.readUint32(output);
      }
    );
    const joined: bigint[] = [];
    let invalidCount = 0;
    for (let index = 0; index < samples.count; index++) {
      const key = joinDggsWords(keys[2 * index], keys[2 * index + 1]);
      if (key === 0n) invalidCount++;
      joined.push(key);
    }
    return {
      ...buildPointIndexOutput(samples, joined, family, resolution, invalidCount),
      dispatchedContributors: ['GPUPointToCell']
    };
  }
};

// ---------------------------------------------------------------------------------------------
// dggs-cell-aggregation

const AGGREGATION_PARAMETERS: CatalogParameterSpec[] = [
  AGGREGATION_FAMILY_PARAMETER,
  makeNumberParameter(
    'resolution',
    'Resolution (Quadbin 0-26, H3 0-15)',
    8,
    0,
    26,
    undefined,
    'integer'
  ),
  makeStringParameter('valueProperty', 'Value property (empty = count only)')
];

function makeCellTable(
  id: string,
  label: string,
  extraColumns = {}
): CatalogOutputTable & {
  columns: Record<string, (string | number | boolean | null)[]>;
} {
  return makeCatalogTable(id, label, {...CELL_COLUMN_TYPES, ...extraColumns}, 'geometry');
}

const CELL_AGGREGATION_BINDING: CatalogOperationBinding = {
  id: 'dggs-cell-aggregation',
  title: 'DGGS cell aggregation (Quadbin or H3)',
  domain: 'aggregation',
  summary:
    'Aggregates points per Quadbin or H3 cell (count, sum, mean, min, max) and publishes the cell boundary polygons.',
  capabilityIds: ['dggs-aggregation'],
  contributorNames: ['GPUPointToCell', 'GPUCellAggregation', 'GPUCellGeometry'],
  gpuContributorNames: ['GPUPointToCell', 'GPUCellAggregation', 'GPUCellGeometry'],
  engine: 'gpu-or-cpu',
  inputs: [POINTS_INPUT],
  parameters: AGGREGATION_PARAMETERS,
  outputs: [
    makeOutputSpec(
      'cells',
      'Aggregated cells',
      'polygon',
      ['cell_id', 'family', 'resolution'],
      MAXIMUM_CELL_ROWS
    )
  ],
  cleanup: PLATFORM3_GPU_OR_CPU_CLEANUP,
  declaredBehavior: [
    ...DGGS_BEHAVIOR,
    'Only occupied cells are published, ascending by 64-bit key; more than 100,000 cells throws. Sums are exact 16-bit fixed point on the GPU, so they match the CPU float64 sum of float32 values to 2^-17 per row.',
    'This is the equal-area-capable replacement of the square lattice cell-aggregation: Quadbin cells are Web Mercator tiles, H3 cells are hexagons.'
  ],
  limitation:
    'GPUCellAggregation indexes Quadbin and H3 only (quadkey, geohash and S2 keys come from dggs-point-index). H3 runs only on the GPU with extended-precision (double-float) point indexing, and H3 boundaries are float32.',
  run: (inputs, values) => {
    const family = requireCpuFamily(values['family'] as string) as 'quadbin';
    const resolution = values['resolution'] as number;
    checkResolution(family, resolution);
    const valueProperty = values['valueProperty'] as string;
    const samples = readGeographicSamples(inputs['points'], valueProperty);
    const diagnostics: CatalogDiagnostic[] = [];
    addDroppedDiagnostic(diagnostics, samples.droppedCount);
    const table = makeCellTable('cells', 'Aggregated cells');
    for (const row of aggregateCpuCells(samples, family, resolution))
      appendCellRow(
        table,
        row,
        family,
        resolution,
        Boolean(valueProperty),
        getBoundaryText(getDggsCpuBoundary(family, row.key))
      );
    return {
      tables: [table],
      diagnostics,
      summary: {cellCount: table.rowCount, sampleCount: samples.count, family, resolution}
    };
  },
  runGpu: async (context, inputs, values) => {
    const family = values['family'] as AggregationFamily;
    const resolution = values['resolution'] as number;
    checkResolution(family, resolution);
    const valueProperty = values['valueProperty'] as string;
    const samples = readGeographicSamples(inputs['points'], valueProperty);
    if (!samples.count) throw new CatalogGpuUnsupportedError('No indexable points');
    const {rows, boundaries} = await withCatalogGpuSession(
      context.device,
      'catalog-dggs-cell-aggregation',
      async session => {
        const outputs = addAggregationNodes(session, 'aggregate', samples, family, resolution);
        const readBoundaries = addBoundaryNodes(
          session,
          'aggregate',
          family,
          outputs,
          samples.count
        );
        session.run();
        const tableRows = await readTableRows(session, outputs);
        return {rows: tableRows, boundaries: await readBoundaries(tableRows)};
      }
    );
    const diagnostics: CatalogDiagnostic[] = [];
    addDroppedDiagnostic(diagnostics, samples.droppedCount);
    const table = makeCellTable('cells', 'Aggregated cells');
    for (const row of rows)
      appendCellRow(
        table,
        row,
        family,
        resolution,
        Boolean(valueProperty),
        getBoundaryText(boundaries.get(row.key) ?? null)
      );
    return {
      tables: [table],
      diagnostics,
      summary: {cellCount: table.rowCount, sampleCount: samples.count, family, resolution},
      dispatchedContributors: ['GPUPointToCell', 'GPUCellAggregation', 'GPUCellGeometry']
    };
  }
};

// ---------------------------------------------------------------------------------------------
// dggs-cell-rollup

const ROLLUP_PARAMETERS: CatalogParameterSpec[] = [
  ...AGGREGATION_PARAMETERS,
  makeNumberParameter('levels', 'Coarser levels', 3, 1, 6, undefined, 'integer'),
  makeNumberParameter('levelStep', 'Resolution decrease per level', 1, 1, 4, undefined, 'integer')
];

function getRollupResolutions(values: Values): number[] {
  const resolution = values['resolution'] as number;
  const resolutions: number[] = [];
  for (let level = 0; level <= (values['levels'] as number); level++)
    resolutions.push(resolution - level * (values['levelStep'] as number));
  // The coarsest level would drop below resolution zero; reduce levels or levelStep.
  if (resolutions[resolutions.length - 1] < 0)
    throw new RangeError('Rollup levels reach below resolution 0');
  return resolutions;
}

function appendRollupRows(
  table: CatalogOutputTable,
  rows: readonly DggsCellRow[],
  family: AggregationFamily,
  resolutions: readonly number[],
  level: number,
  hasValues: boolean,
  getGeometry: (row: DggsCellRow) => string | null
): void {
  for (const row of rows)
    appendCellRow(table, row, family, resolutions[level], hasValues, getGeometry(row), {
      level,
      parent_cell_id:
        family === 'quadbin' && level + 1 < resolutions.length
          ? formatDggsCellId(getQuadbinParent(row.key, resolutions[level + 1]))
          : null
    });
}

const ROLLUP_EXTRA_COLUMNS = {level: 'int32', parent_cell_id: 'string'} as const;

const CELL_ROLLUP_BINDING: CatalogOperationBinding = {
  id: 'dggs-cell-rollup',
  title: 'DGGS cell rollup pyramid (Quadbin or H3)',
  domain: 'aggregation',
  summary:
    'Aggregates points at a base resolution and rolls the table up to coarser resolutions with parent links.',
  capabilityIds: ['dggs-aggregation'],
  contributorNames: ['GPUPointToCell', 'GPUCellAggregation', 'GPUCellRollup', 'GPUCellGeometry'],
  gpuContributorNames: ['GPUPointToCell', 'GPUCellAggregation', 'GPUCellRollup', 'GPUCellGeometry'],
  engine: 'gpu-or-cpu',
  inputs: [POINTS_INPUT],
  parameters: ROLLUP_PARAMETERS,
  outputs: [
    makeOutputSpec(
      'rollup',
      'Rolled-up cells',
      'polygon',
      ['cell_id', 'level', 'parent_cell_id'],
      MAXIMUM_CELL_ROWS
    )
  ],
  cleanup: PLATFORM3_GPU_OR_CPU_CLEANUP,
  declaredBehavior: [
    ...DGGS_BEHAVIOR,
    'Level 0 is the base resolution and level n is resolution - n * levelStep; the GPU rolls each level up from the finer table, the CPU recomputes each level from the points (equal for the exactly nesting Quadbin pyramid, integer sums are associative).',
    'parent_cell_id is published for Quadbin only; H3 parents are not derived on the CPU. Output over 100,000 rows throws.'
  ],
  limitation:
    'H3 children only roughly nest in their parent (aperture 7 rotation), so H3 rollup counts are those of the GPU cell parent mapping and have no CPU oracle.',
  run: (inputs, values) => {
    const family = requireCpuFamily(values['family'] as string) as 'quadbin';
    checkResolution(family, values['resolution'] as number);
    const resolutions = getRollupResolutions(values);
    const valueProperty = values['valueProperty'] as string;
    const samples = readGeographicSamples(inputs['points'], valueProperty);
    const diagnostics: CatalogDiagnostic[] = [];
    addDroppedDiagnostic(diagnostics, samples.droppedCount);
    const table = makeCellTable('rollup', 'Rolled-up cells', ROLLUP_EXTRA_COLUMNS);
    resolutions.forEach((resolution, level) =>
      appendRollupRows(
        table,
        aggregateCpuCells(samples, family, resolution),
        family,
        resolutions,
        level,
        Boolean(valueProperty),
        row => getBoundaryText(getDggsCpuBoundary(family, row.key))
      )
    );
    return {
      tables: [table],
      diagnostics,
      summary: {cellCount: table.rowCount, sampleCount: samples.count, levels: resolutions.length}
    };
  },
  runGpu: async (context, inputs, values) => {
    const family = values['family'] as AggregationFamily;
    checkResolution(family, values['resolution'] as number);
    const resolutions = getRollupResolutions(values);
    const valueProperty = values['valueProperty'] as string;
    const samples = readGeographicSamples(inputs['points'], valueProperty);
    if (!samples.count) throw new CatalogGpuUnsupportedError('No indexable points');
    const levels = await withCatalogGpuSession(
      context.device,
      'catalog-dggs-cell-rollup',
      async session => {
        const tables = [addAggregationNodes(session, 'level-0', samples, family, resolutions[0])];
        for (let level = 1; level < resolutions.length; level++) {
          const outputs = makeTableOutputs(session, `level-${level}`, samples.count);
          session.graph.add(
            new GPUCellRollup({
              id: `catalog-dggs-rollup-${level}`,
              family,
              sourceResolution: resolutions[level - 1],
              resolution: resolutions[level],
              source: tables[level - 1].table,
              output: outputs.table
            })
          );
          tables.push(outputs);
        }
        const readers = tables.map((outputs, level) =>
          addBoundaryNodes(session, `level-${level}`, family, outputs, samples.count)
        );
        session.run();
        const result: {rows: DggsCellRow[]; boundaries: GpuBoundaries}[] = [];
        for (let level = 0; level < tables.length; level++) {
          const rows = await readTableRows(session, tables[level]);
          result.push({rows, boundaries: await readers[level](rows)});
        }
        return result;
      }
    );
    const diagnostics: CatalogDiagnostic[] = [];
    addDroppedDiagnostic(diagnostics, samples.droppedCount);
    const table = makeCellTable('rollup', 'Rolled-up cells', ROLLUP_EXTRA_COLUMNS);
    levels.forEach(({rows, boundaries}, level) =>
      appendRollupRows(table, rows, family, resolutions, level, Boolean(valueProperty), row =>
        getBoundaryText(boundaries.get(row.key) ?? null)
      )
    );
    return {
      tables: [table],
      diagnostics,
      summary: {cellCount: table.rowCount, sampleCount: samples.count, levels: resolutions.length},
      dispatchedContributors: [
        'GPUPointToCell',
        'GPUCellAggregation',
        'GPUCellRollup',
        'GPUCellGeometry'
      ]
    };
  }
};

// ---------------------------------------------------------------------------------------------
// dggs-cell-compare

type CompareRow = {
  key: bigint;
  before: number;
  after: number;
  presence: number;
};

function getZScores(rows: readonly CompareRow[], measure: string): number[] {
  const deltas = rows.map(row => row.after - row.before);
  if (measure === 'count')
    return rows.map((row, index) =>
      row.before + row.after === 0 ? 0 : deltas[index] / Math.sqrt(row.before + row.after)
    );
  const mean = deltas.reduce((sum, delta) => sum + delta, 0) / (deltas.length || 1);
  const variance =
    deltas.reduce((sum, delta) => sum + (delta - mean) ** 2, 0) / (deltas.length || 1);
  const deviation = Math.sqrt(variance);
  return deltas.map(delta => (deviation > 0 ? (delta - mean) / deviation : 0));
}

function buildCompareOutput(
  rows: readonly CompareRow[],
  zScores: readonly number[],
  family: string,
  resolution: number,
  measure: string,
  droppedCount: number,
  getGeometry: (key: bigint) => string | null,
  dispatchedContributors?: readonly string[]
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  addDroppedDiagnostic(diagnostics, droppedCount);
  const table = makeCatalogTable(
    'cell-comparison',
    'DGGS cell comparison',
    {
      cell_id: 'string',
      family: 'string',
      resolution: 'int32',
      status: 'string',
      before: 'float64',
      after: 'float64',
      delta: 'float64',
      ratio: 'float64',
      percent_change: 'float64',
      z_score: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  rows.forEach((row, index) => {
    const delta = row.after - row.before;
    appendCatalogRow(table, {
      cell_id: formatDggsCellId(row.key),
      family,
      resolution,
      status: row.presence === 1 ? 'disappeared' : row.presence === 2 ? 'appeared' : 'present',
      before: row.before,
      after: row.after,
      delta,
      ratio: row.before === 0 ? null : row.after / row.before,
      percent_change: row.before === 0 ? null : (100 * delta) / row.before,
      z_score: zScores[index],
      geometry: getGeometry(row.key)
    });
  });
  return {
    tables: [table],
    diagnostics,
    summary: {cellCount: table.rowCount, measure, family, resolution},
    ...(dispatchedContributors ? {dispatchedContributors} : {})
  };
}

const COMPARE_PARAMETERS: CatalogParameterSpec[] = [
  AGGREGATION_FAMILY_PARAMETER,
  makeNumberParameter(
    'resolution',
    'Resolution (Quadbin 0-26, H3 0-15)',
    8,
    0,
    26,
    undefined,
    'integer'
  ),
  makeStringParameter('valueProperty', 'Value property (needed for the sum measure)'),
  {
    name: 'measure',
    label: 'Compared measure',
    kind: 'enum',
    options: ['count', 'sum'],
    defaultValue: 'count'
  }
];

function requireCompareMeasure(values: Values): string {
  const measure = values['measure'] as string;
  // The sum measure compares the value property, so the property must be named.
  if (measure === 'sum' && !values['valueProperty'])
    throw new RangeError('The sum measure needs a value property');
  return measure;
}

const CELL_COMPARE_BINDING: CatalogOperationBinding = {
  id: 'dggs-cell-compare',
  title: 'DGGS cell comparison (Quadbin or H3)',
  domain: 'aggregation',
  summary:
    'Aggregates a before and an after point set per cell and compares count or sum: delta, ratio, percent change and a z-score.',
  capabilityIds: ['dggs-outlines-and-change'],
  contributorNames: [
    'GPUPointToCell',
    'GPUCellAggregation',
    'GPUCellTableCompare',
    'GPUCellGeometry'
  ],
  gpuContributorNames: [
    'GPUPointToCell',
    'GPUCellAggregation',
    'GPUCellTableCompare',
    'GPUCellGeometry'
  ],
  engine: 'gpu-or-cpu',
  inputs: [
    makeInputSpec('before', 'Before points', ['point']),
    makeInputSpec('after', 'After points', ['point'])
  ],
  parameters: COMPARE_PARAMETERS,
  outputs: [
    makeOutputSpec(
      'cell-comparison',
      'DGGS cell comparison',
      'polygon',
      ['cell_id', 'family', 'resolution'],
      MAXIMUM_CELL_ROWS
    )
  ],
  cleanup: PLATFORM3_GPU_OR_CPU_CLEANUP,
  declaredBehavior: [
    ...DGGS_BEHAVIOR,
    'The union of occupied cells is published; status is appeared (only after), disappeared (only before) or present, a missing side counts as zero and ratio and percent_change are null when before is zero.',
    'z_score is delta / sqrt(before + after) for counts (0 when both are zero) and the standardized delta over the union rows (population standard deviation) for sums.'
  ],
  limitation:
    'Descriptive change statistic only: the z-scores assume Poisson counts or an exchangeable delta distribution and are not a spatial significance test.',
  run: (inputs, values) => {
    const family = requireCpuFamily(values['family'] as string) as 'quadbin';
    const resolution = values['resolution'] as number;
    checkResolution(family, resolution);
    const measure = requireCompareMeasure(values);
    const valueProperty = measure === 'sum' ? (values['valueProperty'] as string) : '';
    const before = readGeographicSamples(inputs['before'], valueProperty);
    const after = readGeographicSamples(inputs['after'], valueProperty);
    const beforeCells = new Map(
      aggregateCpuCells(before, family, resolution).map(row => [row.key, row])
    );
    const afterCells = new Map(
      aggregateCpuCells(after, family, resolution).map(row => [row.key, row])
    );
    const keys = [...new Set([...beforeCells.keys(), ...afterCells.keys()])].sort((a, b) =>
      a < b ? -1 : a > b ? 1 : 0
    );
    const read = (row: DggsCellRow | undefined) =>
      row ? (measure === 'sum' ? row.sum : row.count) : 0;
    const rows: CompareRow[] = keys.map(key => ({
      key,
      before: read(beforeCells.get(key)),
      after: read(afterCells.get(key)),
      presence: (beforeCells.has(key) ? 1 : 0) | (afterCells.has(key) ? 2 : 0)
    }));
    return buildCompareOutput(
      rows,
      getZScores(rows, measure),
      family,
      resolution,
      measure,
      before.droppedCount + after.droppedCount,
      key => getBoundaryText(getDggsCpuBoundary(family, key))
    );
  },
  runGpu: async (context, inputs, values) => {
    const family = values['family'] as AggregationFamily;
    const resolution = values['resolution'] as number;
    checkResolution(family, resolution);
    const measure = requireCompareMeasure(values);
    const valueProperty = measure === 'sum' ? (values['valueProperty'] as string) : '';
    const before = readGeographicSamples(inputs['before'], valueProperty);
    const after = readGeographicSamples(inputs['after'], valueProperty);
    // The compare kernel needs two non-empty tables.
    if (!before.count || !after.count) throw new CatalogGpuUnsupportedError('A side has no points');
    const capacity = before.count + after.count;
    const result = await withCatalogGpuSession(
      context.device,
      'catalog-dggs-cell-compare',
      async session => {
        const beforeOutputs = addAggregationNodes(session, 'before', before, family, resolution);
        const afterOutputs = addAggregationNodes(session, 'after', after, family, resolution);
        const cells = session.output('compare-cells', 'uint32x2', capacity);
        const presence = session.output('compare-presence', 'uint32', capacity);
        const beforeValues = session.output('compare-before', 'float32', capacity);
        const afterValues = session.output('compare-after', 'float32', capacity);
        const zScores = session.output('compare-z', 'float32', capacity);
        const count = session.output('compare-count', 'uint32', 1);
        const overflow = session.output('compare-overflow', 'uint32', 1);
        const candidateOverflow = session.output('compare-candidate-overflow', 'uint32', 1);
        session.graph.add(
          new GPUCellTableCompare({
            id: 'catalog-dggs-compare',
            before: beforeOutputs.table,
            after: afterOutputs.table,
            measure: measure as 'count' | 'sum',
            output: {
              cells: cells.view,
              presence: presence.view,
              before: beforeValues.view,
              after: afterValues.view,
              zScore: zScores.view,
              status: {
                count: count.view,
                overflow: overflow.view,
                candidateOverflow: candidateOverflow.view
              }
            }
          })
        );
        const geometry = addBoundaryNodes(
          session,
          'compare',
          family,
          {...beforeOutputs, table: {...beforeOutputs.table, cells: cells.view}},
          capacity
        );
        session.run();
        const rowCount = await session.readScalar(count);
        if ((await session.readScalar(overflow)) || (await session.readScalar(candidateOverflow))) {
          throw new CatalogGpuUnsupportedError(
            'Cell comparison is incomplete; increase the source or destination capacity'
          );
        }
        const words = await session.readUint32(cells, rowCount);
        const presenceValues = await session.readUint32(presence, rowCount);
        const beforeRead = await session.readFloat32(beforeValues, rowCount);
        const afterRead = await session.readFloat32(afterValues, rowCount);
        const zRead = await session.readFloat32(zScores, rowCount);
        const rows: CompareRow[] = [];
        for (let row = 0; row < rowCount; row++)
          rows.push({
            key: joinDggsWords(words[2 * row], words[2 * row + 1]),
            before: beforeRead[row],
            after: afterRead[row],
            presence: presenceValues[row]
          });
        const stubRows: DggsCellRow[] = rows.map(row => ({
          key: row.key,
          count: 0,
          sum: 0,
          minimum: 0,
          maximum: 0
        }));
        return {rows, zScores: Array.from(zRead), boundaries: await geometry(stubRows)};
      }
    );
    return buildCompareOutput(
      result.rows,
      result.zScores,
      family,
      resolution,
      measure,
      before.droppedCount + after.droppedCount,
      key => getBoundaryText(result.boundaries.get(key) ?? null),
      ['GPUPointToCell', 'GPUCellAggregation', 'GPUCellTableCompare', 'GPUCellGeometry']
    );
  }
};

export const PLATFORM4_DGGS_BINDINGS: readonly CatalogOperationBinding[] = [
  POINT_INDEX_BINDING,
  CELL_AGGREGATION_BINDING,
  CELL_ROLLUP_BINDING,
  CELL_COMPARE_BINDING
];
