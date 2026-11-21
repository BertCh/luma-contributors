// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUS2CapCovering,
  GPUS2CellHierarchy,
  GPUS2CellMeasures,
  GPUS2PolygonCovering
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {CatalogOperationBinding, CatalogRunOutput} from './catalog-binding-types';
import {EARTH_RADIUS_METERS} from './catalog-geodesy';
import {
  formatDggsCellId,
  getDggsCpuBoundary,
  getDggsCpuCell,
  getS2Cell,
  getS2Face,
  getS2FaceUV,
  getS2FaceXYZ,
  getS2ST,
  getS2UV,
  joinDggsWords,
  readS2Cell
} from './catalog-platform4-dggs-cells';
import {
  cross,
  dot,
  getUnitVector,
  getRingSignedArea,
  isOnArc,
  isRingValid,
  normalize,
  type Vector3
} from './catalog-platform4-spherical';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  getPointPositions,
  getPolygonParts,
  makeCatalogGeometryText,
  makeCatalogTable,
  type CatalogDiagnostic,
  type CatalogFeatureCollection
} from './catalog-geojson';
import {CatalogGpuUnsupportedError, withCatalogGpuSession} from './catalog-gpu-session';
import {
  PLATFORM3_GPU_OR_CPU_CLEANUP,
  makeInputSpec,
  makeOutputSpec,
  type Values,
  addDroppedDiagnostic as addSharedDroppedDiagnostic
} from './catalog-platform3-b-common';

const MAXIMUM_ROWS = 100000;
const CAPABILITY_IDS = ['dggs-indexing-and-topology'] as const;

type LevelCell = {featureIndex: number; coordinates: [number, number]; cell: bigint};

/** S2 cell of every usable point position at `level`; unusable positions are counted. */
function readLevelCells(points: CatalogFeatureCollection, level: number) {
  const entries: LevelCell[] = [];
  let droppedCount = 0;
  points.features.forEach((feature, featureIndex) => {
    const positions = getPointPositions(feature.geometry);
    if (!positions.length) droppedCount++;
    for (const [longitude, latitude] of positions) {
      const cell =
        Number.isFinite(longitude) && Number.isFinite(latitude) && Math.abs(latitude) <= 90
          ? getDggsCpuCell('s2', longitude, latitude, level)
          : 0n;
      if (cell) entries.push({featureIndex, coordinates: [longitude, latitude], cell});
      else droppedCount++;
    }
  });
  return {entries, droppedCount};
}

const DROPPED_FEATURE_MESSAGE =
  'Features without a valid longitude/latitude point position are skipped.';

function addDroppedDiagnostic(diagnostics: CatalogDiagnostic[], droppedCount: number): void {
  addSharedDroppedDiagnostic(diagnostics, droppedCount, DROPPED_FEATURE_MESSAGE);
}

function getBoundaryText(cell: bigint): string {
  return makeCatalogGeometryText('Polygon', [getDggsCpuBoundary('s2', cell)]);
}

function splitWords(cells: readonly bigint[]): Uint32Array {
  const words = new Uint32Array(cells.length * 2);
  cells.forEach((cell, index) => {
    words[2 * index] = Number(cell & 0xffffffffn);
    words[2 * index + 1] = Number(cell >> 32n);
  });
  return words;
}

function joinWords(words: Uint32Array): bigint[] {
  const cells: bigint[] = [];
  for (let index = 0; index < words.length; index += 2)
    cells.push(joinDggsWords(words[index], words[index + 1]));
  return cells;
}

/** Cell corner unit vectors in the order (0,0), (1,0), (1,1), (0,1) of the S2 (i, j) frame. */
function getCellVertices(cell: bigint): Vector3[] {
  const {face, i, j, level} = readS2Cell(cell);
  const scale = 2 ** level;
  return [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
  ].map(([deltaI, deltaJ]) =>
    normalize(
      getS2FaceXYZ(face, getS2UV((i + deltaI) / scale), getS2UV((j + deltaJ) / scale)) as Vector3
    )
  );
}

function getArcAngle(first: Vector3, second: Vector3): number {
  return Math.atan2(Math.hypot(...cross(first, second)), dot(first, second));
}

// ---------------------------------------------------------------------------------------------
// s2-cell-hierarchy
// ---------------------------------------------------------------------------------------------

type HierarchyOperation = 'parent' | 'children' | 'range' | 'neighbors';
const HIERARCHY_STRIDE: Record<HierarchyOperation, number> = {
  parent: 1,
  children: 4,
  range: 2,
  neighbors: 4
};
const HIERARCHY_ROLES: Record<HierarchyOperation, readonly string[]> = {
  parent: ['parent'],
  children: ['child_0', 'child_1', 'child_2', 'child_3'],
  range: ['range_min', 'range_max'],
  neighbors: ['i_minus', 'i_plus', 'j_minus', 'j_plus']
};

/** Edge neighbor in (i, j) face coordinates; off-face neighbors are relocated through the sphere. */
function getS2Neighbor(cell: bigint, slot: number): bigint {
  const {face, i, j, level} = readS2Cell(cell);
  const size = 2 ** level;
  const scale = 2 ** (30 - level);
  let neighborI = i;
  let neighborJ = j;
  if (slot < 2) neighborI += slot === 1 ? 1 : -1;
  else neighborJ += slot === 3 ? 1 : -1;
  if (neighborI >= 0 && neighborI < size && neighborJ >= 0 && neighborJ < size)
    return getS2Cell(face, neighborI * scale, neighborJ * scale, level);
  const [x, y, z] = getS2FaceXYZ(
    face,
    getS2UV((neighborI + 0.5) / size),
    getS2UV((neighborJ + 0.5) / size)
  );
  const neighborFace = getS2Face(x, y, z);
  const [u, v] = getS2FaceUV(neighborFace, x, y, z);
  const clamp = (value: number) => Math.max(0, Math.min(size - 1, Math.floor(size * value)));
  return getS2Cell(neighborFace, clamp(getS2ST(u)) * scale, clamp(getS2ST(v)) * scale, level);
}

type HierarchyResult = {cells: bigint[]; status: number[]};

function computeHierarchyCpu(
  cells: readonly bigint[],
  operation: HierarchyOperation,
  parentLevel: number
): HierarchyResult {
  const stride = HIERARCHY_STRIDE[operation];
  const result: HierarchyResult = {cells: [], status: []};
  for (const cell of cells) {
    const {level} = readS2Cell(cell);
    let failure = 0;
    if (operation === 'parent' && level < parentLevel) failure = 2;
    if (operation === 'children' && level === 30) failure = 2;
    const marker = 1n << BigInt(2 * (30 - level));
    for (let slot = 0; slot < stride; slot++) {
      let value = 0n;
      if (failure === 0) {
        if (operation === 'neighbors') value = getS2Neighbor(cell, slot);
        else if (operation === 'parent') {
          const shift = BigInt(2 * (30 - parentLevel));
          value = ((cell >> shift) << shift) | (1n << shift);
        } else if (operation === 'children') {
          const childMarker = marker >> 2n;
          value = cell - marker + childMarker + BigInt(slot) * (childMarker << 1n);
        } else value = slot === 1 ? cell + (marker - 1n) : cell - (marker - 1n);
      }
      result.cells.push(value);
    }
    result.status.push(failure);
  }
  return result;
}

function buildHierarchyOutput(
  entries: readonly LevelCell[],
  operation: HierarchyOperation,
  result: HierarchyResult,
  droppedCount: number,
  dispatchedContributors?: readonly string[]
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  addDroppedDiagnostic(diagnostics, droppedCount);
  const stride = HIERARCHY_STRIDE[operation];
  const roles = HIERARCHY_ROLES[operation];
  const table = makeCatalogTable(
    's2-hierarchy',
    'S2 cell hierarchy',
    {
      point_feature_index: 'int32',
      relation: 'string',
      source_cell_id: 'string',
      cell_id: 'string',
      level: 'int32',
      geometry: 'string'
    },
    'geometry'
  );
  let unavailableCount = 0;
  entries.forEach((entry, index) => {
    if (result.status[index] !== 0) {
      unavailableCount++;
      return;
    }
    for (let slot = 0; slot < stride; slot++) {
      const cell = result.cells[index * stride + slot];
      appendCatalogRow(table, {
        point_feature_index: entry.featureIndex,
        relation: roles[slot],
        source_cell_id: formatDggsCellId(entry.cell),
        cell_id: formatDggsCellId(cell),
        level: readS2Cell(cell).level,
        // The range bounds are leaf cells; the shape of the source cell is the useful geometry.
        geometry: getBoundaryText(operation === 'range' ? entry.cell : cell)
      });
    }
  });
  if (unavailableCount)
    addCatalogDiagnostic(
      diagnostics,
      's2-level-unavailable',
      'warning',
      'Cells whose requested level does not exist (leaf children, or a parent level below the cell level) produce no rows.',
      unavailableCount
    );
  return {
    tables: [table],
    diagnostics,
    summary: {pointCount: entries.length, rowCount: table.rowCount, unavailableCount},
    ...(dispatchedContributors ? {dispatchedContributors} : {})
  };
}

function readHierarchyValues(values: Values) {
  return {
    level: values['level'] as number,
    operation: values['operation'] as HierarchyOperation,
    parentLevel: values['parentLevel'] as number
  };
}

const HIERARCHY_BINDING: CatalogOperationBinding = {
  id: 's2-cell-hierarchy',
  title: 'S2 cell hierarchy',
  domain: 'geometry',
  summary:
    'Indexes points into S2 cells and publishes the parent at a coarser level, the four children, the inclusive leaf range or the four edge neighbors of every cell, with exact integer arithmetic.',
  capabilityIds: CAPABILITY_IDS,
  contributorNames: ['GPUS2CellHierarchy'],
  gpuContributorNames: ['GPUS2CellHierarchy'],
  engine: 'gpu-or-cpu',
  inputs: [makeInputSpec('points', 'Points', ['point'])],
  parameters: [
    {
      name: 'level',
      label: 'Point cell level (0-30)',
      kind: 'integer',
      minimum: 0,
      maximum: 30,
      defaultValue: 10
    },
    {
      name: 'operation',
      label: 'Relation',
      kind: 'enum',
      options: ['children', 'parent', 'neighbors', 'range'],
      defaultValue: 'children'
    },
    {
      name: 'parentLevel',
      label: 'Parent level (operation parent)',
      kind: 'integer',
      minimum: 0,
      maximum: 30,
      defaultValue: 5
    }
  ],
  outputs: [
    makeOutputSpec(
      's2-hierarchy',
      'S2 cell hierarchy',
      'polygon',
      ['point_feature_index', 'relation'],
      MAXIMUM_ROWS
    )
  ],
  cleanup: PLATFORM3_GPU_OR_CPU_CLEANUP,
  declaredBehavior: [
    'Each point is converted to its S2 cell at the given level on the CPU in float64 (float32 point positions), so both engines transform exactly the same keys; the relation itself is exact 64-bit integer arithmetic.',
    'Children are the four Hilbert-ordered children (child_0..child_3), neighbors are the four edge neighbors in face coordinates (i_minus, i_plus, j_minus, j_plus) across cube-face seams, range is the inclusive leaf id interval (range_min, range_max, the geometry is the source cell).',
    'A parent level finer than the cell level and the children of a level-30 cell do not exist: those points produce no rows and are counted as s2-level-unavailable. Cell boundaries are published with straight edges between the four corners.'
  ],
  limitation:
    'Point input only: cell-id input tables and mixed-level batches (supported by the contributor) are not exposed; the neighbor CPU reference relocates off-face neighbors through float64 cell centers.',
  run: (inputs, values) => {
    const {level, operation, parentLevel} = readHierarchyValues(values);
    const {entries, droppedCount} = readLevelCells(inputs['points'], level);
    return buildHierarchyOutput(
      entries,
      operation,
      computeHierarchyCpu(
        entries.map(entry => entry.cell),
        operation,
        parentLevel
      ),
      droppedCount
    );
  },
  runGpu: async (context, inputs, values) => {
    const {level, operation, parentLevel} = readHierarchyValues(values);
    const {entries, droppedCount} = readLevelCells(inputs['points'], level);
    if (!entries.length) throw new CatalogGpuUnsupportedError('No indexable points');
    const stride = HIERARCHY_STRIDE[operation];
    const result = await withCatalogGpuSession(
      context.device,
      'catalog-s2-hierarchy',
      async session => {
        const cells = session.output('cells', 'uint32x2', entries.length * stride);
        const status = session.output('status', 'uint32', entries.length);
        session.graph.add(
          new GPUS2CellHierarchy({
            id: 'catalog-s2-hierarchy',
            cells: session.input(
              'source',
              splitWords(entries.map(entry => entry.cell)),
              'uint32x2'
            ),
            operation,
            level: parentLevel,
            output: {cells: cells.view, status: status.view}
          })
        );
        session.run();
        return {
          cells: joinWords(await session.readUint32(cells)),
          status: Array.from(await session.readUint32(status))
        };
      }
    );
    return buildHierarchyOutput(entries, operation, result, droppedCount, ['GPUS2CellHierarchy']);
  }
};

// ---------------------------------------------------------------------------------------------
// s2-cell-measures
// ---------------------------------------------------------------------------------------------

/** Spherical area (steradians) and perimeter (radians) of a cell with great-circle edges. */
function getCellMeasures(cell: bigint): [number, number] {
  const [a, b, c, d] = getCellVertices(cell);
  const triangle = (first: Vector3, second: Vector3, third: Vector3) =>
    2 *
    Math.atan2(
      Math.abs(dot(first, cross(second, third))),
      1 + dot(first, second) + dot(second, third) + dot(third, first)
    );
  return [
    triangle(a, b, c) + triangle(a, c, d),
    getArcAngle(a, b) + getArcAngle(b, c) + getArcAngle(c, d) + getArcAngle(d, a)
  ];
}

function buildMeasuresOutput(
  entries: readonly LevelCell[],
  measures: readonly (readonly [number, number])[],
  radius: number,
  droppedCount: number,
  dispatchedContributors?: readonly string[]
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  addDroppedDiagnostic(diagnostics, droppedCount);
  const table = makeCatalogTable(
    's2-measures',
    'S2 cell measures',
    {
      point_feature_index: 'int32',
      cell_id: 'string',
      level: 'int32',
      area: 'float64',
      perimeter: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  let totalArea = 0;
  entries.forEach((entry, index) => {
    const [area, perimeter] = measures[index];
    totalArea += area * radius * radius;
    appendCatalogRow(table, {
      point_feature_index: entry.featureIndex,
      cell_id: formatDggsCellId(entry.cell),
      level: readS2Cell(entry.cell).level,
      area: area * radius * radius,
      perimeter: perimeter * radius,
      geometry: getBoundaryText(entry.cell)
    });
  });
  return {
    tables: [table],
    diagnostics,
    summary: {pointCount: entries.length, totalArea},
    ...(dispatchedContributors ? {dispatchedContributors} : {})
  };
}

const MEASURES_BINDING: CatalogOperationBinding = {
  id: 's2-cell-measures',
  title: 'S2 cell area and perimeter',
  domain: 'geometry',
  summary:
    'Indexes points into S2 cells and publishes the spherical area and perimeter of each cell in the units of the sphere radius.',
  capabilityIds: CAPABILITY_IDS,
  contributorNames: ['GPUS2CellMeasures'],
  gpuContributorNames: ['GPUS2CellMeasures'],
  engine: 'gpu-or-cpu',
  inputs: [makeInputSpec('points', 'Points', ['point'])],
  parameters: [
    {
      name: 'level',
      label: 'Cell level (0-22)',
      kind: 'integer',
      minimum: 0,
      maximum: 22,
      defaultValue: 12
    },
    {
      name: 'radius',
      label: 'Sphere radius',
      kind: 'number',
      unit: 'm',
      minimum: 1,
      maximum: 1e9,
      defaultValue: EARTH_RADIUS_METERS
    }
  ],
  outputs: [
    makeOutputSpec('s2-measures', 'S2 cell measures', 'polygon', ['point_feature_index'], 20000)
  ],
  cleanup: PLATFORM3_GPU_OR_CPU_CLEANUP,
  declaredBehavior: [
    'Area is the spherical area of the cell with great-circle edges times radius squared; perimeter is the sum of the four edge arc lengths times radius. Points outside [-90, 90] latitude or non-finite are skipped and counted.',
    'The GPU contributor uses cancellation-free local increments in float32; the CPU reference sums the two corner triangles in float64, which itself loses accuracy beyond about level 22, hence the level limit.'
  ],
  limitation:
    'Levels 23-30 (supported by the contributor) are not exposed because the float64 corner-triangle oracle cannot verify them; the sphere is spherical, not an ellipsoid.',
  run: (inputs, values) => {
    const {entries, droppedCount} = readLevelCells(inputs['points'], values['level'] as number);
    return buildMeasuresOutput(
      entries,
      entries.map(entry => getCellMeasures(entry.cell)),
      values['radius'] as number,
      droppedCount
    );
  },
  runGpu: async (context, inputs, values) => {
    const {entries, droppedCount} = readLevelCells(inputs['points'], values['level'] as number);
    if (!entries.length) throw new CatalogGpuUnsupportedError('No indexable points');
    const measures = await withCatalogGpuSession(
      context.device,
      'catalog-s2-measures',
      async session => {
        const output = session.output('measures', 'float32x2', entries.length);
        const status = session.output('status', 'uint32', entries.length);
        session.graph.add(
          new GPUS2CellMeasures({
            id: 'catalog-s2-measures',
            cells: session.input('cells', splitWords(entries.map(entry => entry.cell)), 'uint32x2'),
            output: {measures: output.view, status: status.view}
          })
        );
        session.run();
        const [data, statuses] = [
          await session.readFloat32(output),
          await session.readUint32(status)
        ];
        // Status 1 means an invalid key; the CPU keys are valid by construction.
        if (statuses.some(value => value !== 0)) throw new Error('S2 cell measures rejected a key');
        return entries.map((_, index) => [data[2 * index], data[2 * index + 1]] as const);
      }
    );
    return buildMeasuresOutput(entries, measures, values['radius'] as number, droppedCount, [
      'GPUS2CellMeasures'
    ]);
  }
};

// ---------------------------------------------------------------------------------------------
// s2-cap-covering
// ---------------------------------------------------------------------------------------------

type CapCandidate = {cell: bigint; vertices: Vector3[]};

/** Distance from `point` to the cell: zero inside, else the closest angle to the boundary. */
function getCellDistance(vertices: readonly Vector3[], point: Vector3): number {
  let minimum = Math.PI;
  let inside = true;
  for (let edge = 0; edge < 4; edge++) {
    const first = vertices[edge];
    const second = vertices[(edge + 1) % 4];
    const normal = normalize(cross(first, second));
    inside = inside && dot(normal, point) >= 0;
    minimum = Math.min(minimum, getArcAngle(point, first));
    const projected = cross(normal, cross(point, normal));
    if (Math.hypot(...projected) > 1e-10) {
      const nearest = normalize(projected);
      if (dot(cross(first, nearest), normal) >= 0 && dot(cross(nearest, second), normal) >= 0)
        minimum = Math.min(minimum, getArcAngle(point, nearest));
    }
  }
  return inside ? 0 : minimum;
}

function getCapCandidates(level: number): CapCandidate[] {
  const candidates: CapCandidate[] = [];
  const positionCount = 4 ** level;
  for (let candidate = 0; candidate < 6 * positionCount; candidate++) {
    const face = Math.floor(candidate / positionCount);
    const position = candidate % positionCount;
    const cell =
      (BigInt(face) << 61n) |
      (BigInt(position) << BigInt(61 - 2 * level)) |
      (1n << BigInt(60 - 2 * level));
    candidates.push({cell, vertices: getCellVertices(cell)});
  }
  return candidates;
}

const CAP_TOLERANCE = 1e-5;

type CapResult = {cells: bigint[]; required: number};

function computeCapCpu(
  candidates: readonly CapCandidate[],
  center: Vector3,
  radius: number,
  mode: string,
  capacity: number
): CapResult {
  const antipode: Vector3 = [-center[0], -center[1], -center[2]];
  const cells: bigint[] = [];
  let required = 0;
  for (const candidate of candidates) {
    const isSelected =
      mode === 'interior'
        ? Math.PI - getCellDistance(candidate.vertices, antipode) <= radius - CAP_TOLERANCE
        : getCellDistance(candidate.vertices, center) <= radius + CAP_TOLERANCE;
    if (!isSelected) continue;
    if (required < capacity) cells.push(candidate.cell);
    required++;
  }
  return {cells, required};
}

type CapPoint = LevelCell & {center: Vector3};

function readCapPoints(points: CatalogFeatureCollection) {
  const entries: CapPoint[] = [];
  let droppedCount = 0;
  points.features.forEach((feature, featureIndex) => {
    const positions = getPointPositions(feature.geometry);
    if (!positions.length) droppedCount++;
    for (const [longitude, latitude] of positions)
      if (Number.isFinite(longitude) && Number.isFinite(latitude) && Math.abs(latitude) <= 90)
        entries.push({
          featureIndex,
          coordinates: [longitude, latitude],
          cell: 0n,
          center: getUnitVector(longitude, latitude)
        });
      else droppedCount++;
  });
  return {entries, droppedCount};
}

function buildCapOutput(
  entries: readonly CapPoint[],
  results: readonly CapResult[],
  level: number,
  capacity: number,
  droppedCount: number,
  requestedCapacity: number,
  dispatchedContributors?: readonly string[]
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  addDroppedDiagnostic(diagnostics, droppedCount);
  if (requestedCapacity > capacity)
    addCatalogDiagnostic(
      diagnostics,
      's2-maximum-cells-clamped',
      'info',
      `maximumCells ${requestedCapacity} was clamped to the ${capacity} cell limit at level ${level}.`
    );
  const table = makeCatalogTable(
    's2-cap-cover',
    'S2 cap covering',
    {
      point_feature_index: 'int32',
      cell_id: 'string',
      level: 'int32',
      geometry: 'string'
    },
    'geometry'
  );
  let truncatedCount = 0;
  entries.forEach((entry, index) => {
    if (results[index].required > capacity) truncatedCount++;
    for (const cell of results[index].cells)
      appendCatalogRow(table, {
        point_feature_index: entry.featureIndex,
        cell_id: formatDggsCellId(cell),
        level,
        geometry: getBoundaryText(cell)
      });
  });
  if (truncatedCount)
    addCatalogDiagnostic(
      diagnostics,
      's2-cover-truncated',
      'warning',
      'More cells matched than maximumCells; only the first cells in face and Hilbert order are published for those points.',
      truncatedCount
    );
  return {
    tables: [table],
    diagnostics,
    summary: {pointCount: entries.length, cellCount: table.rowCount, truncatedCount},
    ...(dispatchedContributors ? {dispatchedContributors} : {})
  };
}

function readCapValues(values: Values) {
  const level = values['level'] as number;
  const effectiveMaximum = Math.min(6144, 6 * 4 ** level);
  const requestedCapacity = values['maximumCells'] as number;
  return {
    level,
    mode: values['mode'] as string,
    radius: Math.fround((values['radiusMeters'] as number) / EARTH_RADIUS_METERS),
    capacity: Math.min(requestedCapacity, effectiveMaximum),
    requestedCapacity
  };
}

const CAP_BINDING: CatalogOperationBinding = {
  id: 's2-cap-covering',
  title: 'S2 cap covering',
  domain: 'geometry',
  summary:
    'Covers the spherical cap (circle on the sphere) of a radius around each point with S2 cells of one level: every cell touching the cap (exterior) or lying entirely inside it (interior).',
  capabilityIds: CAPABILITY_IDS,
  contributorNames: ['GPUS2CapCovering'],
  gpuContributorNames: ['GPUS2CapCovering'],
  engine: 'gpu-or-cpu',
  inputs: [makeInputSpec('points', 'Cap centers', ['point'], 200)],
  parameters: [
    {
      name: 'radiusMeters',
      label: 'Cap radius',
      kind: 'number',
      unit: 'm',
      minimum: 0,
      maximum: 20015086,
      defaultValue: 500000
    },
    {
      name: 'level',
      label: 'Cell level (0-6)',
      kind: 'integer',
      minimum: 0,
      maximum: 6,
      defaultValue: 3
    },
    {
      name: 'mode',
      label: 'Cover',
      kind: 'enum',
      options: ['exterior', 'interior'],
      defaultValue: 'exterior'
    },
    {
      name: 'maximumCells',
      label: 'Maximum cells per cap',
      kind: 'integer',
      minimum: 1,
      maximum: Number.MAX_SAFE_INTEGER,
      defaultValue: 384
    }
  ],
  outputs: [
    makeOutputSpec(
      's2-cap-cover',
      'S2 cap covering',
      'polygon',
      ['point_feature_index', 'cell_id'],
      MAXIMUM_ROWS
    )
  ],
  cleanup: PLATFORM3_GPU_OR_CPU_CLEANUP,
  declaredBehavior: [
    'The cap radius is meters on a sphere of the mean Earth radius (6371008.8 m); a cell is selected when its closest angular distance to the cap center is within the radius plus 1e-5 radians (exterior) or when its farthest point is inside the radius minus 1e-5 radians (interior).',
    'maximumCells is clamped to min(6,144, 6 * 4^level), with an s2-maximum-cells-clamped diagnostic when reduced; its default is 384 at the default level 3.',
    'The contributor enumerates all 6 * 4^level cells uniformly (not an adaptive coverer), so the level is limited to 6; when more cells match than maximumCells the first cells in enumeration order are kept and the point is counted as s2-cover-truncated.',
    'Cells whose distance to the cap boundary is within about 1e-6 radians can differ between the float32 GPU and the float64 CPU engine.'
  ],
  limitation:
    'Uniform single-level covers only (levels 0-6 of the contributor range 0-8); no mixed-level region coverer, no minimum/maximum level bracket and no polygon union across several caps.',
  run: (inputs, values) => {
    const {level, mode, radius, capacity, requestedCapacity} = readCapValues(values);
    const {entries, droppedCount} = readCapPoints(inputs['points']);
    const candidates = getCapCandidates(level);
    return buildCapOutput(
      entries,
      entries.map(entry => computeCapCpu(candidates, entry.center, radius, mode, capacity)),
      level,
      capacity,
      droppedCount,
      requestedCapacity
    );
  },
  runGpu: async (context, inputs, values) => {
    const {level, mode, radius, capacity, requestedCapacity} = readCapValues(values);
    const {entries, droppedCount} = readCapPoints(inputs['points']);
    if (!entries.length) throw new CatalogGpuUnsupportedError('No cap centers');
    const results = await withCatalogGpuSession(context.device, 'catalog-s2-cap', async session => {
      const cells = session.output('cells', 'uint32x2', entries.length * capacity);
      const counts = session.output('counts', 'uint32x2', entries.length);
      const status = session.output('status', 'uint32', entries.length);
      session.graph.add(
        new GPUS2CapCovering({
          id: 'catalog-s2-cap',
          caps: session.input(
            'caps',
            Float32Array.from(entries.flatMap(entry => [...entry.center, radius])),
            'float32x4'
          ),
          level,
          mode: mode as 'exterior' | 'interior',
          capacity,
          output: {cells: cells.view, counts: counts.view, status: status.view}
        })
      );
      session.run();
      const cellWords = joinWords(await session.readUint32(cells));
      const countWords = await session.readUint32(counts);
      const statuses = await session.readUint32(status);
      // Status 1 is an invalid cap; unit vectors and a radius in [0, pi] are valid by construction.
      if (statuses.some(value => (value & 1) !== 0))
        throw new Error('S2 cap covering rejected a cap');
      return entries.map(
        (_, index): CapResult => ({
          cells: cellWords.slice(index * capacity, index * capacity + countWords[2 * index]),
          required: countWords[2 * index + 1]
        })
      );
    });
    return buildCapOutput(entries, results, level, capacity, droppedCount, requestedCapacity, [
      'GPUS2CapCovering'
    ]);
  }
};

// ---------------------------------------------------------------------------------------------
// s2-polygon-covering
// ---------------------------------------------------------------------------------------------

const COVER_TOLERANCE = 1e-6;

type CoverPolygon = {rings: Vector3[][]; featureIndex: number; partIndex: number};

/** Rings counter-clockwise in their own hemisphere, including concave rings. */
function orientRing(ring: Vector3[]): Vector3[] {
  return getRingSignedArea(ring) < 0 ? [...ring].reverse() : ring;
}

function readCoverPolygons(polygons: CatalogFeatureCollection) {
  const shapes: CoverPolygon[] = [];
  let invalidCount = 0;
  polygons.features.forEach((feature, featureIndex) => {
    getPolygonParts(feature.geometry).forEach((polygon, partIndex) => {
      const rings = polygon.map(ring => {
        const isClosed =
          ring.length > 1 &&
          ring[0][0] === ring[ring.length - 1][0] &&
          ring[0][1] === ring[ring.length - 1][1];
        return (isClosed ? ring.slice(0, -1) : ring).map(([longitude, latitude]) =>
          getUnitVector(longitude, latitude)
        );
      });
      if (
        rings.length &&
        rings.every(ring => {
          if (!isRingValid(ring)) return false;
          const area = getRingSignedArea(ring);
          return Number.isFinite(area) && area !== 0;
        })
      )
        shapes.push({rings: rings.map(orientRing), featureIndex, partIndex});
      else invalidCount++;
    });
  });
  return {shapes, invalidCount};
}

/** Float64 port of the contributor's cell/polygon classification and bounded stack traversal. */
function computePolygonCoverCpu(
  shape: CoverPolygon,
  minimumLevel: number,
  maximumLevel: number,
  isInterior: boolean,
  capacity: number,
  maximumVisitedCells: number
) {
  const {rings} = shape;
  const containsPoint = (point: Vector3): [boolean, boolean] => {
    let inside = false;
    let onBoundary = false;
    rings.forEach((ring, ringIndex) => {
      let winding = 0;
      for (let edge = 0; edge < ring.length; edge++) {
        const first = ring[edge];
        const second = ring[(edge + 1) % ring.length];
        onBoundary = onBoundary || isOnArc(first, second, point, COVER_TOLERANCE);
        const numerator = dot(point, cross(first, second));
        const denominator = dot(first, second) - dot(point, first) * dot(point, second);
        if (Math.abs(numerator) + Math.abs(denominator) > 1e-12)
          winding += Math.atan2(numerator, denominator);
      }
      inside = ringIndex === 0 ? winding > Math.PI : inside && winding <= Math.PI;
    });
    return [inside, onBoundary];
  };
  const classify = (cell: bigint): [boolean, boolean] => {
    const vertices = getCellVertices(cell);
    const crosses = vertices.map((vertex, edge) => cross(vertex, vertices[(edge + 1) % 4]));
    const normals = crosses.map(normalize);
    let intersects = false;
    let contained = true;
    let touched = false;
    for (const ring of rings)
      for (const vertex of ring)
        if (normals.every(normal => dot(normal, vertex) >= -COVER_TOLERANCE))
          touched = intersects = true;
    for (let edge = 0; edge < 4; edge++) {
      const first = vertices[edge];
      const second = vertices[(edge + 1) % 4];
      const [inside, onBoundary] = containsPoint(first);
      intersects = intersects || inside || onBoundary;
      contained = contained && inside && !onBoundary;
      for (const ring of rings)
        for (let vertex = 0; vertex < ring.length; vertex++) {
          const third = ring[vertex];
          const fourth = ring[(vertex + 1) % ring.length];
          const crossing = cross(crosses[edge], cross(third, fourth));
          if (Math.hypot(...crossing) > COVER_TOLERANCE * COVER_TOLERANCE) {
            const point = normalize(crossing);
            const opposite: Vector3 = [-point[0], -point[1], -point[2]];
            if (
              (isOnArc(first, second, point, COVER_TOLERANCE) &&
                isOnArc(third, fourth, point, COVER_TOLERANCE)) ||
              (isOnArc(first, second, opposite, COVER_TOLERANCE) &&
                isOnArc(third, fourth, opposite, COVER_TOLERANCE))
            )
              touched = intersects = true;
          } else if (
            isOnArc(first, second, third, COVER_TOLERANCE) ||
            isOnArc(first, second, fourth, COVER_TOLERANCE)
          )
            touched = intersects = true;
        }
    }
    return [intersects, contained && !touched];
  };
  const stack: bigint[] = [];
  for (let face = 5; face >= 0; face--) stack.push(BigInt(2 * face + 1) << 60n);
  const cells: bigint[] = [];
  let required = 0;
  let visitedCount = 0;
  let isBudgetExhausted = false;
  while (stack.length) {
    if (visitedCount >= maximumVisitedCells) {
      isBudgetExhausted = true;
      break;
    }
    const cell = stack.pop()!;
    visitedCount++;
    const [intersects, contained] = classify(cell);
    if (!intersects) continue;
    const {level} = readS2Cell(cell);
    if (level >= minimumLevel && (contained || (!isInterior && level === maximumLevel))) {
      if (required < capacity) cells.push(cell);
      required++;
    } else if (level < maximumLevel) {
      const marker = 1n << BigInt(2 * (30 - level));
      const childMarker = marker >> 2n;
      const first = cell - marker + childMarker;
      for (let child = 3; child >= 0; child--)
        stack.push(first + BigInt(child) * (childMarker << 1n));
    }
  }
  return {cells, required, isBudgetExhausted};
}

type CoverResult = {cells: bigint[]; required: number; isBudgetExhausted: boolean};

function readCoverValues(values: Values) {
  return {
    minimumLevel: values['minimumLevel'] as number,
    maximumLevel: values['maximumLevel'] as number,
    isInterior: values['mode'] === 'interior',
    capacity: values['maximumCells'] as number,
    maximumVisitedCells: values['maximumVisitedCells'] as number
  };
}

function buildCoverOutput(
  shapes: readonly CoverPolygon[],
  results: readonly CoverResult[],
  invalidCount: number,
  capacity: number,
  dispatchedContributors?: readonly string[]
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  if (invalidCount)
    addCatalogDiagnostic(
      diagnostics,
      'invalid-spherical-polygon',
      'warning',
      'Polygons that are self-crossing, degenerate or wider than a hemisphere are skipped.',
      invalidCount
    );
  const table = makeCatalogTable(
    's2-polygon-cover',
    'S2 polygon covering',
    {
      polygon_feature_index: 'int32',
      polygon_part_index: 'int32',
      cell_id: 'string',
      level: 'int32',
      geometry: 'string'
    },
    'geometry'
  );
  let truncatedCount = 0;
  let exhaustedCount = 0;
  shapes.forEach((shape, index) => {
    if (results[index].required > capacity) truncatedCount++;
    if (results[index].isBudgetExhausted) exhaustedCount++;
    for (const cell of results[index].cells)
      appendCatalogRow(table, {
        polygon_feature_index: shape.featureIndex,
        polygon_part_index: shape.partIndex,
        cell_id: formatDggsCellId(cell),
        level: readS2Cell(cell).level,
        geometry: getBoundaryText(cell)
      });
  });
  if (truncatedCount)
    addCatalogDiagnostic(
      diagnostics,
      's2-cover-truncated',
      'warning',
      'More cells were required than maximumCells; only the first cells of the traversal are published.',
      truncatedCount
    );
  if (exhaustedCount)
    addCatalogDiagnostic(
      diagnostics,
      's2-cover-budget-exhausted',
      'warning',
      'The traversal reached maximumVisitedCells; the covering is incomplete.',
      exhaustedCount
    );
  return {
    tables: [table],
    diagnostics,
    summary: {polygonCount: shapes.length, cellCount: table.rowCount, truncatedCount},
    ...(dispatchedContributors ? {dispatchedContributors} : {})
  };
}

const MAXIMUM_GPU_POLYGON_PARTS = 16;

const POLYGON_COVER_BINDING: CatalogOperationBinding = {
  id: 's2-polygon-covering',
  title: 'S2 polygon covering',
  domain: 'geometry',
  summary:
    'Covers polygons whose edges are great-circle arcs with a mixed-level S2 cell set: exterior covers contain the polygon, interior covers lie inside it.',
  capabilityIds: CAPABILITY_IDS,
  contributorNames: ['GPUS2PolygonCovering'],
  gpuContributorNames: ['GPUS2PolygonCovering'],
  engine: 'gpu-or-cpu',
  inputs: [makeInputSpec('polygons', 'Polygons', ['polygon'], 200, 20000)],
  parameters: [
    {
      name: 'mode',
      label: 'Cover',
      kind: 'enum',
      options: ['exterior', 'interior'],
      defaultValue: 'exterior'
    },
    {
      name: 'minimumLevel',
      label: 'Minimum cell level',
      kind: 'integer',
      minimum: 0,
      maximum: 15,
      defaultValue: 0
    },
    {
      name: 'maximumLevel',
      label: 'Maximum cell level',
      kind: 'integer',
      minimum: 0,
      maximum: 15,
      defaultValue: 6
    },
    {
      name: 'maximumCells',
      label: 'Maximum cells per polygon',
      kind: 'integer',
      minimum: 1,
      maximum: 20000,
      defaultValue: 2048
    },
    {
      name: 'maximumVisitedCells',
      label: 'Traversal budget',
      kind: 'integer',
      minimum: 6,
      maximum: 262144,
      defaultValue: 65536
    }
  ],
  outputs: [
    makeOutputSpec(
      's2-polygon-cover',
      'S2 polygon covering',
      'polygon',
      ['polygon_feature_index', 'polygon_part_index', 'cell_id'],
      MAXIMUM_ROWS
    )
  ],
  cleanup: PLATFORM3_GPU_OR_CPU_CLEANUP,
  declaredBehavior: [
    'Each polygon part (outer ring plus holes, vertices joined by great-circle arcs) is covered independently; rings are reoriented counter-clockwise around their own vertex mean and parts with a self-crossing, degenerate or wider-than-hemisphere ring are skipped and counted.',
    'Traversal starts at the six face cells and subdivides cells that cross the boundary; contained cells at or above the minimum level are published as they are, and in exterior mode cells that still touch the boundary at the maximum level are kept too.',
    'The result is capped at maximumCells (s2-cover-truncated) and the traversal at maximumVisitedCells (s2-cover-budget-exhausted); both engines keep the same prefix of the deterministic traversal order.',
    'The GPU engine dispatches one contributor per polygon part, at most 16 parts.'
  ],
  limitation:
    'No complement mode, no shared hemisphere pole and no multi-polygon union; cells within about 1e-6 radians of the boundary can classify differently between float32 and float64.',
  run: (inputs, values) => {
    const {minimumLevel, maximumLevel, isInterior, capacity, maximumVisitedCells} =
      readCoverValues(values);
    const {shapes, invalidCount} = readCoverPolygons(inputs['polygons']);
    return buildCoverOutput(
      shapes,
      shapes.map(shape =>
        computePolygonCoverCpu(
          shape,
          minimumLevel,
          maximumLevel,
          isInterior,
          capacity,
          maximumVisitedCells
        )
      ),
      invalidCount,
      capacity
    );
  },
  runGpu: async (context, inputs, values) => {
    const {minimumLevel, maximumLevel, isInterior, capacity, maximumVisitedCells} =
      readCoverValues(values);
    const {shapes, invalidCount} = readCoverPolygons(inputs['polygons']);
    if (!shapes.length || shapes.length > MAXIMUM_GPU_POLYGON_PARTS)
      throw new CatalogGpuUnsupportedError('Between 1 and 16 valid polygon parts are required');
    const results: CoverResult[] = [];
    for (const shape of shapes)
      results.push(
        await withCatalogGpuSession(context.device, 'catalog-s2-polygon-cover', async session => {
          const positions: number[] = [];
          const ringOffsets = [0];
          for (const ring of shape.rings) {
            for (const vertex of ring) positions.push(...vertex);
            ringOffsets.push(positions.length / 3);
          }
          const cells = session.output('cells', 'uint32x2', capacity);
          const counts = session.output('counts', 'uint32x2', 1);
          const status = session.output('status', 'uint32', 1);
          session.graph.add(
            new GPUS2PolygonCovering({
              id: 'catalog-s2-polygon-cover',
              positions: session.input('positions', Float32Array.from(positions), 'float32x3'),
              ringOffsets: session.input('ring-offsets', Uint32Array.from(ringOffsets), 'uint32'),
              mode: isInterior ? 'interior' : 'exterior',
              minimumLevel,
              maximumLevel,
              capacity,
              maximumVisitedCells,
              output: {cells: cells.view, counts: counts.view, status: status.view}
            })
          );
          session.run();
          const words = await session.readUint32(counts);
          const flags = (await session.readUint32(status))[0];
          // Status 1 is invalid input; the rings were validated on the CPU first.
          if (flags & 1) throw new Error('S2 polygon covering rejected a polygon');
          return {
            cells: joinWords(await session.readUint32(cells)).slice(0, words[0]),
            required: words[1],
            isBudgetExhausted: (flags & 4) !== 0
          };
        })
      );
    return buildCoverOutput(shapes, results, invalidCount, capacity, ['GPUS2PolygonCovering']);
  }
};

export const PLATFORM5_S2_BINDINGS: readonly CatalogOperationBinding[] = [
  HIERARCHY_BINDING,
  MEASURES_BINDING,
  CAP_BINDING,
  POLYGON_COVER_BINDING
];
