// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {PathLayer, ScatterplotLayer, TextLayer} from '@deck.gl/layers';
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPUSphericalArcOverlay,
  GPUSphericalConvexContains,
  GPUSphericalConvexIntersection,
  GPU_EARTH_MEAN_RADIUS_METERS
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {
  loadPacificBoundaryInputs,
  type PacificBoundaryInputs,
  type PacificWindow
} from './pacific-boundary-logic-data';
import {unwrapLongitudeNear} from './ocean-drifters-data';
import {clamp, toDegrees} from '@math.gl/core';

export type PacificBoundaryLogicOptions = {
  analysis: 'contains' | 'convex' | 'arc';
  window: 'A' | 'B';
  boundary: 'closed' | 'open';
  complement: boolean;
  operation: 'intersection' | 'union' | 'difference' | 'symmetric-difference';
  capacity: '3' | '16';
};

type ContainsOutput = {
  values: Buffer;
  statuses: Buffer;
  recordCount: number;
};

type IntersectionOutput = {
  positions: Buffer;
  counts: Buffer;
  status: Buffer;
  measures: Buffer;
  capacity: number;
};

type ArcOutput = {
  positions: Buffer;
  ringOffsets: Buffer;
  counts: Buffer;
  status: Buffer;
};

type ContainsResult = {values: Uint32Array; statuses: Uint32Array};
type IntersectionResult = {
  positions: Float32Array;
  counts: Uint32Array;
  status: number;
  measures: Float32Array;
  capacity: number;
};
type ArcResult = {
  positions: Float32Array;
  ringOffsets: Uint32Array;
  counts: Uint32Array;
  status: number;
};

const OPERATIONS = ['intersection', 'union', 'difference', 'symmetric-difference'] as const;
function sphericalAreaKm2(steradians: number): number {
  return (steradians * GPU_EARTH_MEAN_RADIUS_METERS * GPU_EARTH_MEAN_RADIUS_METERS) / 1_000_000;
}

function makeGeoPosition(vector: ArrayLike<number>, previousLongitude = 180): [number, number] {
  const longitude = toDegrees(Math.atan2(vector[1], vector[0]));
  const latitude = toDegrees(Math.asin(clamp(vector[2], -1, 1)));
  return [unwrapLongitudeNear(previousLongitude, longitude), latitude];
}

/** GPU vectors are converted to map coordinates only for spherical display interpolation. */
function densifyUnitRing(positions: Float32Array, begin: number, end: number): [number, number][] {
  if (end - begin < 2) return [];
  const path: [number, number][] = [];
  let previousLongitude = 180;
  for (let vertex = begin; vertex < end; vertex++) {
    const nextVertex = vertex + 1 === end ? begin : vertex + 1;
    const firstOffset = vertex * 3;
    const secondOffset = nextVertex * 3;
    const first: [number, number, number] = [
      positions[firstOffset],
      positions[firstOffset + 1],
      positions[firstOffset + 2]
    ];
    const second: [number, number, number] = [
      positions[secondOffset],
      positions[secondOffset + 1],
      positions[secondOffset + 2]
    ];
    const angle = Math.acos(
      clamp(first[0] * second[0] + first[1] * second[1] + first[2] * second[2], -1, 1)
    );
    const steps = Math.max(1, Math.ceil(toDegrees(angle) / 2));
    for (let step = 0; step < steps; step++) {
      const fraction = step / steps;
      const denominator = Math.sin(angle);
      const firstWeight =
        denominator > 1e-8 ? Math.sin((1 - fraction) * angle) / denominator : 1 - fraction;
      const secondWeight = denominator > 1e-8 ? Math.sin(fraction * angle) / denominator : fraction;
      const point = makeGeoPosition(
        [
          first[0] * firstWeight + second[0] * secondWeight,
          first[1] * firstWeight + second[1] * secondWeight,
          first[2] * firstWeight + second[2] * secondWeight
        ],
        previousLongitude
      );
      previousLongitude = point[0];
      path.push(point);
    }
  }
  if (path.length > 0) path.push(path[0]);
  return path;
}

function densifyUnitSegment(
  positions: Float32Array,
  firstVertex: number,
  secondVertex: number
): [number, number][] {
  const firstOffset = firstVertex * 3;
  const secondOffset = secondVertex * 3;
  const first = [positions[firstOffset], positions[firstOffset + 1], positions[firstOffset + 2]];
  const second = [
    positions[secondOffset],
    positions[secondOffset + 1],
    positions[secondOffset + 2]
  ];
  const dot = first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
  const angle = Math.acos(clamp(dot, -1, 1));
  const steps = Math.max(1, Math.ceil(toDegrees(angle) / 2));
  const path: [number, number][] = [];
  for (let step = 0; step <= steps; step++) {
    const fraction = step / steps;
    const denominator = Math.sin(angle);
    const firstWeight =
      denominator > 1e-8 ? Math.sin((1 - fraction) * angle) / denominator : 1 - fraction;
    const secondWeight = denominator > 1e-8 ? Math.sin(fraction * angle) / denominator : fraction;
    path.push(
      makeGeoPosition(
        [
          first[0] * firstWeight + second[0] * secondWeight,
          first[1] * firstWeight + second[1] * secondWeight,
          first[2] * firstWeight + second[2] * secondWeight
        ],
        path.at(-1)?.[0] ?? 180
      )
    );
  }
  return path;
}

function makeWindowPath(window: PacificWindow): [number, number][] {
  const dense: [number, number][] = [];
  for (let edge = 0; edge < window.unitPositions.length / 3; edge++) {
    const next = (edge + 1) % (window.unitPositions.length / 3);
    const firstOffset = edge * 3;
    const secondOffset = next * 3;
    const first = [
      window.unitPositions[firstOffset],
      window.unitPositions[firstOffset + 1],
      window.unitPositions[firstOffset + 2]
    ];
    const second = [
      window.unitPositions[secondOffset],
      window.unitPositions[secondOffset + 1],
      window.unitPositions[secondOffset + 2]
    ];
    const dot = first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
    const angle = Math.acos(clamp(dot, -1, 1));
    const steps = Math.max(1, Math.ceil(toDegrees(angle) / 2));
    for (let step = 0; step < steps; step++) {
      const fraction = step / steps;
      const denominator = Math.sin(angle);
      const firstWeight =
        denominator > 1e-8 ? Math.sin((1 - fraction) * angle) / denominator : 1 - fraction;
      const secondWeight = denominator > 1e-8 ? Math.sin(fraction * angle) / denominator : fraction;
      const point = makeGeoPosition([
        first[0] * firstWeight + second[0] * secondWeight,
        first[1] * firstWeight + second[1] * secondWeight,
        first[2] * firstWeight + second[2] * secondWeight
      ]);
      while (point[0] < 0) point[0] += 360;
      dense.push(point);
    }
  }
  if (dense.length) dense.push(dense[0]);
  return dense;
}

function formatStatus(status: number, kind: 'intersection' | 'arc'): string {
  if (status === 0) return 'status 0 · complete';
  if (kind === 'intersection') {
    return `status ${status} · ${status & 1 ? 'invalid or unsupported input' : ''}${status & 2 ? ' capacity reached' : ''}${status & 4 ? ' numerical ambiguity' : ''}`;
  }
  return `status ${status} · ${status & 1 ? 'invalid topology/input/reference' : ''}${status & 2 ? ' capacity' : ''}${status & 4 ? ' unresolved junction/separation' : ''}`;
}

function makeContainsVariant(
  window: 'A' | 'B',
  boundary: 'closed' | 'open',
  complement: boolean
): string {
  return `${window}-${boundary}-${String(complement)}`;
}

function makeOutputViews<Format extends 'float32x3' | 'float32x4' | 'uint32' | 'uint32x2'>(
  graph: GPUCommandGraph<void>,
  name: string,
  buffer: Buffer,
  format: Format,
  length: number
): GraphDataView<Format> {
  return importGraphBuffer(graph, name, buffer, format, length);
}

/** Spherical boundary lab: full archive analysis, GPU containment/intersection/arc arrangement. */
export async function createPacificBoundaryLogic(
  ctx: SceneContext<PacificBoundaryLogicOptions>
): Promise<SceneInstance<PacificBoundaryLogicOptions>> {
  const inputs = await loadPacificBoundaryInputs(
    ctx.datasets.get('poopdeck-earthquakes'),
    ctx.datasets.get('natural-earth'),
    ctx.signal
  );
  const resources = new SpatialAnalysisResources(ctx.device, 'pacific-boundary-logic');
  const graph = new GPUCommandGraph<void>(ctx.device, {id: 'pacific-boundary-logic'});
  const importView = <Format extends 'float32x3' | 'float32x4' | 'uint32' | 'uint32x2'>(
    name: string,
    buffer: Buffer,
    format: Format,
    length: number
  ) => makeOutputViews(graph, name, buffer, format, length);
  const eventCount = inputs.eventCount;
  const queryCount = eventCount + 4;
  const windowPositionsBuffers = inputs.windows.map((window, index) =>
    resources.createBuffer(`window-${index}-positions`, window.unitPositions)
  ) as [Buffer, Buffer];
  const windowOffsetBuffers = inputs.windows.map((_, index) =>
    resources.createBuffer(`window-${index}-offsets`, Uint32Array.of(0, 4))
  ) as [Buffer, Buffer];
  const queryBuffers = inputs.windows.map((window, index) =>
    resources.createBuffer(
      `queries-${index}`,
      Float32Array.from([...inputs.eventUnitPositions, ...window.unitPositions])
    )
  ) as [Buffer, Buffer];
  const polygonIndexBuffers = inputs.windows.map((_, index) =>
    resources.createBuffer(`polygon-indices-${index}`, new Uint32Array(queryCount))
  ) as [Buffer, Buffer];
  const windowPositions = inputs.windows.map((window, index) =>
    importView(`window-${index}-positions`, windowPositionsBuffers[index], 'float32x3', 4)
  ) as [GraphDataView<'float32x3'>, GraphDataView<'float32x3'>];
  const windowOffsets = inputs.windows.map((_, index) =>
    importView(`window-${index}-offsets`, windowOffsetBuffers[index], 'uint32', 2)
  ) as [GraphDataView<'uint32'>, GraphDataView<'uint32'>];
  const queryViews = inputs.windows.map((_, index) =>
    importView(`queries-${index}`, queryBuffers[index], 'float32x3', queryCount)
  ) as [GraphDataView<'float32x3'>, GraphDataView<'float32x3'>];
  const polygonIndexViews = inputs.windows.map((_, index) =>
    importView(`polygon-indices-${index}`, polygonIndexBuffers[index], 'uint32', queryCount)
  ) as [GraphDataView<'uint32'>, GraphDataView<'uint32'>];

  const containsOutputs = new Map<string, ContainsOutput>();
  for (const windowIndex of [0, 1] as const) {
    const window = inputs.windows[windowIndex];
    for (const boundary of ['closed', 'open'] as const) {
      for (const complement of [false, true]) {
        const variant = makeContainsVariant(window.id, boundary, complement);
        const values = resources.createBuffer(`contains-${variant}-values`, queryCount * 4);
        const statuses = resources.createBuffer(`contains-${variant}-status`, queryCount * 4);
        graph.add(
          new GPUSphericalConvexContains({
            id: `contains-${variant}`,
            positions: windowPositions[windowIndex],
            offsets: windowOffsets[windowIndex],
            queries: queryViews[windowIndex],
            polygonIndices: polygonIndexViews[windowIndex],
            boundary,
            complement,
            tolerance: 1e-6,
            output: {
              contains: importView(`contains-${variant}-values`, values, 'uint32', queryCount),
              status: importView(`contains-${variant}-status`, statuses, 'uint32', queryCount)
            }
          })
        );
        containsOutputs.set(variant, {values, statuses, recordCount: queryCount});
      }
    }
  }

  const intersectionOutputs = new Map<number, IntersectionOutput>();
  for (const capacity of [3, 16]) {
    const positions = resources.createBuffer(`intersection-${capacity}-positions`, capacity * 12);
    const counts = resources.createBuffer(`intersection-${capacity}-counts`, 8);
    const status = resources.createBuffer(`intersection-${capacity}-status`, 4);
    const measures = resources.createBuffer(`intersection-${capacity}-measures`, 16);
    graph.add(
      new GPUSphericalConvexIntersection({
        id: `intersection-${capacity}`,
        firstPositions: windowPositions[0],
        secondPositions: windowPositions[1],
        firstOffsets: windowOffsets[0],
        secondOffsets: windowOffsets[1],
        capacity,
        tolerance: 1e-6,
        output: {
          positions: importView(
            `intersection-${capacity}-positions`,
            positions,
            'float32x3',
            capacity
          ),
          counts: importView(`intersection-${capacity}-counts`, counts, 'uint32x2', 1),
          status: importView(`intersection-${capacity}-status`, status, 'uint32', 1),
          measures: importView(`intersection-${capacity}-measures`, measures, 'float32x4', 1)
        }
      })
    );
    intersectionOutputs.set(capacity, {positions, counts, status, measures, capacity});
  }

  const arcOutputs = new Map<string, ArcOutput>();
  for (const complement of [false, true]) {
    for (const operation of OPERATIONS) {
      const key = `${operation}-${String(complement)}`;
      const positions = resources.createBuffer(`arc-${key}-positions`, 64 * 12);
      const ringOffsets = resources.createBuffer(`arc-${key}-rings`, 65 * 4);
      const counts = resources.createBuffer(`arc-${key}-counts`, 16);
      const status = resources.createBuffer(`arc-${key}-status`, 4);
      graph.add(
        new GPUSphericalArcOverlay({
          id: `arc-${key}`,
          left: {
            positions: windowPositions[0],
            ringOffsets: windowOffsets[0],
            referenceInside: complement
          },
          right: {
            positions: windowPositions[1],
            ringOffsets: windowOffsets[1],
            referenceInside: false
          },
          referencePoint: [1, 0, 0],
          operation,
          tolerance: 1e-6,
          output: {
            positions: importView(`arc-${key}-positions`, positions, 'float32x3', 64),
            ringOffsets: importView(`arc-${key}-rings`, ringOffsets, 'uint32', 65),
            counts: importView(`arc-${key}-counts`, counts, 'uint32', 4),
            status: importView(`arc-${key}-status`, status, 'uint32', 1)
          }
        })
      );
      arcOutputs.set(key, {positions, ringOffsets, counts, status});
    }
  }

  let compiled: CompiledGPUCommandGraph<void>;
  try {
    compiled = resources.track(await graph.compileAsync());
  } catch (error) {
    resources.destroy();
    throw error;
  }

  // Fixed, finite variant keys bound these readback caches; current options choose only the key drawn.
  const containsResults = new Map<string, ContainsResult>();
  const intersectionResults = new Map<number, IntersectionResult>();
  const arcResults = new Map<string, ArcResult>();
  const readers: SummaryReader[] = [];
  let destroyed = false;
  let generation = 1;
  let requestedGeneration = 0;
  const renderCurrent = () => ctx.requestLayers();
  for (const [key, output] of containsOutputs) {
    const requestGeneration = generation;
    const reader = new SummaryReader(
      resources,
      `contains-${key}`,
      [
        {buffer: output.values, size: output.recordCount * 4},
        {buffer: output.statuses, size: output.recordCount * 4}
      ],
      bytes => {
        if (destroyed || requestGeneration !== generation) return;
        const values = new Uint32Array(bytes.slice(0, output.recordCount * 4));
        const statuses = new Uint32Array(bytes.slice(output.recordCount * 4));
        containsResults.set(key, {values, statuses});
        publishContainsReadouts(ctx, inputs, key, values, statuses);
        renderCurrent();
      }
    );
    readers.push(reader);
  }
  for (const [capacity, output] of intersectionOutputs) {
    const requestGeneration = generation;
    const reader = new SummaryReader(
      resources,
      `intersection-${capacity}`,
      [
        {buffer: output.positions, size: capacity * 12},
        {buffer: output.counts, size: 8},
        {buffer: output.status, size: 4},
        {buffer: output.measures, size: 16}
      ],
      bytes => {
        if (destroyed || requestGeneration !== generation) return;
        const positions = new Float32Array(bytes.slice(0, capacity * 12));
        const counts = new Uint32Array(bytes.slice(capacity * 12, capacity * 12 + 8));
        const status = new Uint32Array(bytes.slice(capacity * 12 + 8, capacity * 12 + 12))[0];
        const measures = new Float32Array(bytes.slice(capacity * 12 + 12, capacity * 12 + 28));
        intersectionResults.set(capacity, {positions, counts, status, measures, capacity});
        renderCurrent();
      }
    );
    readers.push(reader);
  }
  for (const [key, output] of arcOutputs) {
    const requestGeneration = generation;
    const reader = new SummaryReader(
      resources,
      `arc-${key}`,
      [
        {buffer: output.positions, size: 64 * 12},
        {buffer: output.ringOffsets, size: 65 * 4},
        {buffer: output.counts, size: 16},
        {buffer: output.status, size: 4}
      ],
      bytes => {
        if (destroyed || requestGeneration !== generation) return;
        const positions = new Float32Array(bytes.slice(0, 64 * 12));
        const ringOffsets = new Uint32Array(bytes.slice(64 * 12, 64 * 12 + 65 * 4));
        const counts = new Uint32Array(bytes.slice(64 * 12 + 65 * 4, 64 * 12 + 65 * 4 + 16));
        const status = new Uint32Array(
          bytes.slice(64 * 12 + 65 * 4 + 16, 64 * 12 + 65 * 4 + 20)
        )[0];
        arcResults.set(key, {positions, ringOffsets, counts, status});
        renderCurrent();
      }
    );
    readers.push(reader);
  }

  const earthEvents = Array.from({length: eventCount}, (_, row) => ({
    position: [
      unwrapMapLongitude(inputs.eventLongitudeLatitude[row * 2]),
      inputs.eventLongitudeLatitude[row * 2 + 1]
    ] as [number, number],
    row
  }));
  const windowPaths = inputs.windows.map(window => ({
    path: makeWindowPath(window),
    label: window.label
  }));
  const coastPaths = inputs.coastlinePaths
    .filter(path =>
      path.some(
        ([longitude, latitude]) =>
          longitude > 115 && longitude < 245 && latitude > -55 && latitude < 70
      )
    )
    .map(path => ({path}));
  const cpuCombinedEventCount = (
    operation: PacificBoundaryLogicOptions['operation'],
    complementA: boolean
  ) => {
    const left = containsResults.get(`A-closed-${String(complementA)}`);
    const right = containsResults.get('B-closed-false');
    if (!left || !right) return 'waiting for GPU';
    let count = 0;
    for (let row = 0; row < eventCount; row++) {
      const inA = left.values[row] === 1 && left.statuses[row] !== 1;
      const inB = right.values[row] === 1 && right.statuses[row] !== 1;
      const member =
        operation === 'intersection'
          ? inA && inB
          : operation === 'union'
            ? inA || inB
            : operation === 'difference'
              ? inA && !inB
              : inA !== inB;
      if (member) count++;
    }
    return formatCount(count);
  };
  const getSelectedEventData = (): {position: [number, number]; row: number}[] => {
    if (ctx.options.analysis === 'contains') {
      const variant = makeContainsVariant(
        ctx.options.window,
        ctx.options.boundary,
        ctx.options.complement
      );
      const result = containsResults.get(variant);
      if (!result) return [];
      const selected: {position: [number, number]; row: number}[] = [];
      for (let row = 0; row < eventCount; row++) {
        if (result.statuses[row] !== 1 && result.values[row] === 1) selected.push(earthEvents[row]);
      }
      return selected;
    }
    const complementA = ctx.options.analysis === 'arc' && ctx.options.complement;
    const operation = ctx.options.analysis === 'convex' ? 'intersection' : ctx.options.operation;
    const left = containsResults.get(`A-closed-${String(complementA)}`);
    const right = containsResults.get('B-closed-false');
    if (!left || !right) return [];
    const selected: {position: [number, number]; row: number}[] = [];
    for (let row = 0; row < eventCount; row++) {
      const inA = left.values[row] === 1 && left.statuses[row] !== 1;
      const inB = right.values[row] === 1 && right.statuses[row] !== 1;
      const member =
        operation === 'intersection'
          ? inA && inB
          : operation === 'union'
            ? inA || inB
            : operation === 'difference'
              ? inA && !inB
              : inA !== inB;
      if (member) selected.push(earthEvents[row]);
    }
    return selected;
  };
  const publishSelectedReadouts = () => {
    if (ctx.options.analysis === 'contains') {
      const variant = makeContainsVariant(
        ctx.options.window,
        ctx.options.boundary,
        ctx.options.complement
      );
      const result = containsResults.get(variant);
      if (!result) return;
      let accepted = 0;
      let boundary = 0;
      let invalid = 0;
      for (let row = 0; row < eventCount; row++) {
        if (result.statuses[row] === 1) invalid++;
        else if (result.values[row] === 1) accepted++;
        if (result.statuses[row] === 2) boundary++;
      }
      let syntheticBoundary = 0;
      for (let row = eventCount; row < eventCount + 4; row++)
        syntheticBoundary += Number(result.statuses[row] === 2);
      ctx.setReadout('acceptedEvents', formatCount(accepted));
      ctx.setReadout('boundaryEvents', formatCount(boundary));
      ctx.setReadout('invalidEvents', formatCount(invalid));
      ctx.setReadout(
        'probeStatus',
        `${syntheticBoundary}/4 exact ${ctx.options.window} vertices are boundary-band probes; excluded from earthquake totals`
      );
      ctx.setReadout(
        'computeStatus',
        `GPUSphericalConvexContains · ${variant} · status 0 valid, 2 boundary band · 1e-6`
      );
    } else if (ctx.options.analysis === 'convex') {
      const result = intersectionResults.get(Number(ctx.options.capacity));
      if (!result) return;
      ctx.setReadout(
        'intersectionVertices',
        `${formatCount(result.counts[0])} written / ${formatCount(result.counts[1])} required`
      );
      ctx.setReadout(
        'intersectionArea',
        result.status === 0
          ? `${sphericalAreaKm2(result.measures[0]).toLocaleString('en-US', {maximumFractionDigits: 1})} km² · ${result.measures[0].toPrecision(7)} sr`
          : 'withheld: incomplete or invalid GPU result'
      );
      ctx.setReadout(
        'intersectionFractions',
        result.status === 0
          ? `A ${(result.measures[1] * 100).toFixed(2)}% · B ${(result.measures[2] * 100).toFixed(2)}% · union ${sphericalAreaKm2(result.measures[3]).toLocaleString('en-US', {maximumFractionDigits: 1})} km²`
          : 'withheld with measures'
      );
      ctx.setReadout(
        'combinedEvents',
        `${cpuCombinedEventCount('intersection', false)} archive earthquakes by CPU intersection of GPU A/B containment arrays`
      );
      ctx.setReadout(
        'computeStatus',
        `GPUSphericalConvexIntersection · capacity ${result.capacity} · ${formatStatus(result.status, 'intersection')}`
      );
    } else {
      const key = `${ctx.options.operation}-${String(ctx.options.complement)}`;
      const result = arcResults.get(key);
      if (!result) return;
      ctx.setReadout(
        'arcCounts',
        `${formatCount(result.counts[0])} vertices · ${formatCount(result.counts[1])} rings · ${formatCount(result.counts[2])} required segments`
      );
      ctx.setReadout(
        'arcReference',
        `0°E, 0°N: A ${ctx.options.complement ? 'inside complement' : 'outside'}, B outside → ${result.counts[3] ? 'inside' : 'outside'} ${ctx.options.operation}`
      );
      ctx.setReadout(
        'combinedEvents',
        `${cpuCombinedEventCount(ctx.options.operation, ctx.options.complement)} archive earthquakes by CPU Boolean combination of GPU A/B results`
      );
      ctx.setReadout(
        'computeStatus',
        `GPUSphericalArcOverlay · ${ctx.options.operation} · ${formatStatus(result.status, 'arc')}`
      );
    }
  };
  const getOverlayPaths = (): [number, number][][] => {
    if (ctx.options.analysis === 'convex') {
      const result = intersectionResults.get(Number(ctx.options.capacity));
      if (!result || result.status !== 0 || result.counts[0] < 2) return [];
      if (result.counts[0] === 2) return [densifyUnitSegment(result.positions, 0, 1)];
      return [densifyUnitRing(result.positions, 0, result.counts[0])];
    }
    if (ctx.options.analysis !== 'arc') return [];
    const result = arcResults.get(`${ctx.options.operation}-${String(ctx.options.complement)}`);
    if (!result || result.status !== 0 || result.counts[1] === 0) return [];
    const paths: [number, number][][] = [];
    for (let ring = 0; ring < result.counts[1]; ring++) {
      const path = densifyUnitRing(
        result.positions,
        result.ringOffsets[ring],
        result.ringOffsets[ring + 1]
      );
      if (path.length > 1) paths.push(path);
    }
    return paths;
  };
  const getLayers = (): Layer[] => {
    publishSelectedReadouts();
    const analysis = ctx.options.analysis;
    const selected = getSelectedEventData();
    const outputColor =
      analysis === 'convex'
        ? ([76, 235, 196, 255] as [number, number, number, number])
        : ctx.options.complement
          ? ([231, 90, 222, 255] as [number, number, number, number])
          : ([255, 189, 74, 255] as [number, number, number, number]);
    const outputPaths = getOverlayPaths().map(path => ({path}));
    const intersection =
      analysis === 'convex' ? intersectionResults.get(Number(ctx.options.capacity)) : undefined;
    const validEmptyIntersection = Boolean(
      intersection && intersection.status === 0 && intersection.counts[0] === 0
    );
    const intersectionPoint =
      intersection && intersection.status === 0 && intersection.counts[0] === 1
        ? makeGeoPosition(intersection.positions.subarray(0, 3))
        : null;
    return [
      new PathLayer({
        id: 'pacific-natural-earth-coastline',
        data: coastPaths,
        getPath: item => item.path,
        getColor: [73, 111, 126, 185],
        getWidth: 1,
        widthUnits: 'pixels',
        pickable: false
      }),
      new ScatterplotLayer({
        id: 'pacific-earthquake-archive',
        data: earthEvents,
        getPosition: item => item.position,
        getRadius: 1.8,
        radiusUnits: 'pixels',
        getFillColor: [103, 123, 131, 85],
        stroked: false,
        pickable: false
      }),
      ...(selected.length
        ? [
            new ScatterplotLayer({
              id: 'pacific-selected-earthquakes',
              data: selected,
              getPosition: item => item.position,
              getRadius: 3.2,
              radiusUnits: 'pixels',
              getFillColor: outputColor,
              getLineColor: [11, 22, 28, 210],
              lineWidthUnits: 'pixels',
              lineWidthMinPixels: 0.4,
              stroked: true,
              pickable: false
            })
          ]
        : []),
      new PathLayer({
        id: 'pacific-synthetic-windows',
        data: windowPaths,
        getPath: item => item.path,
        getColor: item =>
          item.label.startsWith('Window A') ? [61, 213, 232, 230] : [238, 123, 81, 225],
        getWidth: 2.3,
        widthUnits: 'pixels',
        pickable: false
      }),
      ...(outputPaths.length
        ? [
            new PathLayer({
              id: 'pacific-gpu-boundary-output',
              data: outputPaths,
              getPath: item => item.path,
              getColor: outputColor,
              getWidth: 4,
              widthUnits: 'pixels',
              pickable: false
            })
          ]
        : []),
      ...(intersectionPoint
        ? [
            new ScatterplotLayer({
              id: 'pacific-gpu-boundary-contact-point',
              data: [{position: intersectionPoint}],
              getPosition: item => item.position,
              getRadius: 5,
              radiusUnits: 'pixels',
              getFillColor: outputColor,
              pickable: false
            })
          ]
        : []),
      ...(analysis === 'convex' && validEmptyIntersection
        ? [
            new TextLayer({
              id: 'pacific-valid-empty-intersection',
              data: [{position: [180, 7], text: 'valid empty intersection'}],
              getPosition: item => item.position,
              getText: item => item.text,
              getColor: [76, 235, 196, 255],
              getSize: 12,
              sizeUnits: 'pixels',
              getTextAnchor: 'middle',
              getAlignmentBaseline: 'center'
            })
          ]
        : []),
      new TextLayer({
        id: 'pacific-window-labels',
        data: [
          {position: [175, 58], text: 'A · 145°E–150°W'},
          {position: [223, -39], text: 'B · 170°E–115°W'},
          {position: [0, 0], text: 'reference · 0°E 0°N'}
        ],
        getPosition: item => item.position,
        getText: item => item.text,
        getColor: [190, 212, 216, 220],
        getSize: 10,
        sizeUnits: 'pixels',
        getTextAnchor: 'middle',
        getAlignmentBaseline: 'center',
        pickable: false
      })
    ];
  };
  const metadata = `Archive rows: ${formatCount(inputs.archiveRowCount)} · type-0 earthquakes: ${formatCount(inputs.earthquakeArchiveCount)} · analysis rows: ${formatCount(eventCount)} · fixed display: every earthquake · source float32 lon/lat → unit XYZ.`;
  ctx.setReadout('archiveEvidence', metadata);
  ctx.setReadout('combinedEvents', 'waiting for GPU');
  return {
    getCompiledGraphs: () => [compiled],
    encode(commandEncoder) {
      if (destroyed) return;
      if (requestedGeneration !== generation) {
        compiled.encode(commandEncoder, {parameters: undefined});
        requestedGeneration = generation;
        for (const reader of readers) reader.request(commandEncoder);
      } else {
        for (const reader of readers) reader.flush(commandEncoder);
      }
    },
    getLayers,
    setOption() {
      ctx.requestLayers();
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      generation++;
      for (const reader of readers) reader.stop();
      resources.destroy();
    }
  };
}

function unwrapMapLongitude(longitude: number): number {
  let unwrapped = longitude;
  while (unwrapped < 0) unwrapped += 360;
  return unwrapped;
}

function publishContainsReadouts(
  ctx: SceneContext<PacificBoundaryLogicOptions>,
  inputs: PacificBoundaryInputs,
  key: string,
  values: Uint32Array,
  statuses: Uint32Array
): void {
  const match = /^(A|B)-(closed|open)-(true|false)$/.exec(key);
  if (!match) return;
  const selectedWindow = match[1];
  const selectedBoundary = match[2];
  const selectedComplement = match[3] === 'true';
  let accepted = 0;
  let boundary = 0;
  let invalid = 0;
  for (let row = 0; row < inputs.eventCount; row++) {
    if (statuses[row] === 1) invalid++;
    else if (values[row] === 1) accepted++;
    if (statuses[row] === 2) boundary++;
  }
  let syntheticBoundary = 0;
  for (let row = inputs.eventCount; row < inputs.eventCount + 4; row++)
    syntheticBoundary += Number(statuses[row] === 2);
  if (
    ctx.options.analysis === 'contains' &&
    ctx.options.window === selectedWindow &&
    ctx.options.boundary === selectedBoundary &&
    ctx.options.complement === selectedComplement
  ) {
    ctx.setReadout('acceptedEvents', formatCount(accepted));
    ctx.setReadout('boundaryEvents', formatCount(boundary));
    ctx.setReadout('invalidEvents', formatCount(invalid));
    ctx.setReadout(
      'probeStatus',
      `${formatCount(syntheticBoundary)}/4 exact source-vertex probes are in the boundary tolerance band; not earthquake rows`
    );
  }
}
