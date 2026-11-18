// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {readFileSync} from 'node:fs';
import {expect, test} from 'vitest';
import {GPUTimeWindowFilter} from '../../src/dataframe-analysis/time-window-filter/gpu-time-window-filter';
import {GPUSpatialPredicateJoin} from '../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import {GPUTrajectoryMetrics} from '../../src/gpu-spatial-analysis/trajectory-analysis/gpu-trajectory-metrics';
import {GPUTrajectorySegments} from '../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-segments';
import {GPUBufferSelection} from '../../src/gpu-spatial-analysis/spatial-join/gpu-buffer-selection';
import {computeProfilePeaksOnCPU} from '../../src/gpu-terrain/terrain-features/cpu-profile-peaks-operation';
import {generateDotCountsOnCPU} from '../gpu-spatial-analysis/dot-density/dot-density-cpu';
import {GPURandomPointsOnLine} from '../../src/gpu-spatial-analysis/dot-density/gpu-random-points-on-line';
import {GPURandomPointsInPolygon} from '../../src/gpu-spatial-analysis/dot-density/gpu-random-points-in-polygon';
import {getGPUDotDensityParameterValues} from '../../src/gpu-spatial-analysis/dot-density/dot-density-parameters';
import {computeStreamlinesOnCPU} from '../../src/raster-analysis/flow-texture/cpu-streamlines-operation';
import {
  getGPUStreamlinesParameterValues,
  getGPUStreamlinesUint32ParameterValues
} from '../../src/raster-analysis/flow-texture/streamlines-parameters';
import {createNullWebGPUDevice} from './gpu-contributor-test-utils';
import {GPUProfilePeaks} from '../../src/gpu-terrain/terrain-features/gpu-profile-peaks';
import {GPUStreamlines} from '../../src/raster-analysis/flow-texture/gpu-streamlines';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {getZoneEventLossCaptures} from '../../../lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/zone-event-loss-captures';
import {getNearTransitSnippetProps} from './join-example-constructors';

function fixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(format: Format, length = 1) =>
    createTransientView(graph, `final1-${serial++}`, format, length);
  const scalar = () => view('uint32');
  const compact = (length = 3) => ({
    ids: view('uint32', length),
    status: {count: scalar(), overflow: scalar(), candidateOverflow: scalar()}
  });
  return {device, graph, view, scalar, compact};
}

test.each([
  'ids',
  'input',
  'foreign'
] as const)('core:F6 time filter rejects candidate destination alias/owner %s', role => {
  const {device, graph, view, compact} = fixture();
  try {
    const output = compact();
    const sourceIds = view('uint32', 3);
    const foreignGraph = new GPUCommandGraph(device);
    output.status.candidateOverflow =
      role === 'foreign'
        ? createTransientView(foreignGraph, 'foreign', 'uint32', 1)
        : graph.createDataView((role === 'ids' ? output.ids : sourceIds).buffer, {
            format: 'uint32',
            length: 1
          });
    expect(() =>
      new GPUTimeWindowFilter({
        timestamps: view('float32', 3),
        sourceIds,
        window: view('float32', 8),
        output: {...output, rows: view('uint32', 3)}
      }).getCommandNodes(graph)
    ).toThrow(/overlap|share buffers|belong|graph/);
  } finally {
    device.destroy();
  }
});

test.each([
  false,
  true
])('spatial-3:F5 anti candidate destination rejects ID alias and accepts disjoint (%s)', alias => {
  const {device, graph, view, compact} = fixture();
  try {
    const unmatched = compact();
    if (alias)
      unmatched.status.candidateOverflow = graph.createDataView(unmatched.ids.buffer, {
        format: 'uint32',
        length: 1
      });
    const construct = () =>
      new GPUSpatialPredicateJoin({
        left: {kind: 'points', positions: view('float32x2', 2)},
        right: {kind: 'points', positions: view('float32x2', 1)},
        predicate: 'intersects',
        how: 'anti',
        output: {unmatched}
      }).getCommandNodes(graph);
    if (alias) expect(construct).toThrow(/overlap/);
    else expect(construct).not.toThrow();
  } finally {
    device.destroy();
  }
});

test.each([
  'metrics',
  'segments'
] as const)('spatial-4:F1 %s candidate publication cannot target track offsets', kind => {
  const {device, graph, view, compact} = fixture();
  try {
    const trackOffsets = view('uint32', 2);
    const output = compact();
    output.status.candidateOverflow = graph.createDataView(trackOffsets.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 4
    });
    const positions = view('float32x2', 3),
      timestamps = view('float32', 3),
      parameters = view('float32', 4);
    expect(() =>
      kind === 'metrics'
        ? new GPUTrajectoryMetrics({
            positions,
            timestamps,
            trackOffsets,
            parameters,
            output: {stops: output}
          })
        : new GPUTrajectorySegments({
            positions,
            timestamps,
            trackOffsets,
            parameters,
            staypointIds: view('uint32', 3),
            output: {triplegs: output}
          })
    ).toThrow(/overlap|share buffers/);
  } finally {
    device.destroy();
  }
});

test.each([
  'safe',
  'unsafe'
] as const)('spatial-3:F6 validates the emitted scalar groups (%s adjacent words)', layout => {
  const {device, graph, view, scalar} = fixture();
  try {
    const words = view('uint32', 65);
    const first = graph.createDataView(words.buffer, {format: 'uint32', length: 1});
    const second = graph.createDataView(words.buffer, {format: 'uint32', length: 1, byteOffset: 4});
    const pairs = {
      ...{leftRows: view('uint32', 3), rightRows: view('uint32', 3)},
      status: {count: scalar(), overflow: scalar(), candidateOverflow: scalar()}
    };
    if (layout === 'unsafe') {
      pairs.status.candidateOverflow = first;
      pairs.status.count = second;
    } else pairs.status.overflow = second;
    const construct = () =>
      new GPUSpatialPredicateJoin({
        left: {kind: 'points', positions: view('float32x2', 2)},
        right: {kind: 'points', positions: view('float32x2', 1)},
        predicate: 'intersects',
        output: {
          pairs,
          status: {overflow: scalar()},
          weights: {
            offsets: view('uint32', 3),
            neighbors: view('uint32', 3),
            weights: view('float32', 3),
            status: {
              count: scalar(),
              requiredCount: scalar(),
              uncertainCount: layout === 'safe' ? first : scalar()
            }
          }
        }
      }).getCommandNodes(graph);
    if (layout === 'safe') expect(construct).not.toThrow();
    else expect(construct).toThrow(/aligned.*overlap/);
  } finally {
    device.destroy();
  }
});

test('core:F7 CPU peaks honor a zero suppression budget', () => {
  const values = new Float32Array(80);
  for (let row = 0; row < 80; row++)
    values[row] = [
      [30, 10],
      [38, 8],
      [46, 6]
    ].reduce(
      (sum, [center, height]) => sum + height * Math.exp(-0.5 * ((row - center) / 2) ** 2),
      0
    );
  const result = computeProfilePeaksOnCPU(values, new Uint32Array([0, 80]), new Float32Array([1]), {
    window: 12,
    suppressionRadius: 10,
    capacity: 80,
    maximumSuppressionIterations: 0
  });
  expect([
    result.count[0],
    result.converged[0],
    result.iterationLimitReached[0],
    result.overflow[0]
  ]).toEqual([0, 0, 1, 0]);
});

test.each([
  'peaks',
  'streamlines'
] as const)('core:F7 %s publication separates iteration loss and clears it', kind => {
  const {device, graph, view, scalar, compact} = fixture();
  try {
    recordWGSLKernelNodes(graph);
    const output = compact(4);
    const iterationLimitReached = scalar();
    const nodes =
      kind === 'peaks'
        ? new GPUProfilePeaks({
            id: 'peaks',
            values: view('float32', 5),
            offsets: view('uint32', 2),
            parameters: view('float32'),
            window: 1,
            minimumSide: 0,
            maximumSuppressionIterations: 0,
            output: {peaks: output, status: {iterationLimitReached}}
          }).getCommandNodes(graph)
        : new GPUStreamlines({
            id: 'flow',
            velocities: view('float32x2'),
            fieldWidth: 1,
            fieldHeight: 1,
            gridWidth: 1,
            gridHeight: 1,
            seedColumns: 1,
            seedRows: 1,
            stepsPerDirection: 1,
            maximumIterations: 1,
            parameters: view('float32', 12),
            uint32Parameters: view('uint32', 4),
            output: {
              lines: output,
              lineOffsets: view('uint32', 5),
              positions: view('float32x2', 12),
              positionCount: scalar(),
              status: {iterationLimitReached}
            }
          }).getCommandNodes(graph);
    const publication = nodes.filter(node =>
      kind === 'peaks'
        ? /-publish$|-publish-unconverged$/.test(node.id)
        : /-finalize$|-candidate-status$/.test(node.id)
    );
    expect(publication).toHaveLength(2);
    const memory = createCPUGraphMemory();
    const record = getWGSLKernelNodeRecord(graph, publication[1].id)!;
    const loss = record.bindings.find(
      binding => binding.name === (kind === 'peaks' ? 'incomplete' : 'state')
    )!.view;
    memory.write(loss, new Uint32Array(kind === 'peaks' ? [1] : [0, 1]));
    executeCommandNodesOnCPU(publication, memory);
    expect([
      memory.readUint32(output.status.overflow)[0],
      memory.readUint32(output.status.candidateOverflow)[0]
    ]).toEqual([0, 1]);
    memory.write(loss, new Uint32Array(kind === 'peaks' ? [0] : [0, 0]));
    executeCommandNodesOnCPU(publication, memory);
    expect([
      memory.readUint32(output.status.overflow)[0],
      memory.readUint32(output.status.candidateOverflow)[0]
    ]).toEqual([0, 0]);
  } finally {
    device.destroy();
  }
});

test.each([
  'peaks',
  'streamlines'
] as const)('core:F7 %s candidate destination cannot overwrite a compact ID', kind => {
  const {device, graph, view, scalar, compact} = fixture();
  try {
    const output = compact();
    output.status.candidateOverflow = graph.createDataView(output.ids.buffer, {
      format: 'uint32',
      length: 1
    });
    expect(() =>
      kind === 'peaks'
        ? new GPUProfilePeaks({
            values: view('float32', 5),
            offsets: view('uint32', 2),
            parameters: view('float32'),
            window: 1,
            output: {peaks: output}
          })
        : new GPUStreamlines({
            velocities: view('float32x2'),
            fieldWidth: 1,
            fieldHeight: 1,
            gridWidth: 1,
            gridHeight: 1,
            seedColumns: 1,
            seedRows: 1,
            stepsPerDirection: 1,
            maximumIterations: 1,
            parameters: view('float32', 12),
            uint32Parameters: view('uint32', 4),
            output: {
              lines: output,
              lineOffsets: view('uint32', 4),
              positions: view('float32x2', 9),
              positionCount: scalar()
            }
          })
    ).toThrow(/overlap/);
  } finally {
    device.destroy();
  }
});

test('spatial-2:F4 fleet consumers capture scratch and event-list completeness separately', () => {
  const lens = readFileSync(
    'modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/fleet-dwell-zone-events-recipe.lens.ts',
    'utf8'
  );
  expect(lens).toContain('getZoneEventLossCaptures');
  const captures = getZoneEventLossCaptures({overflow: 0, candidateOverflow: 1}, 0);
  expect(captures.eventCandidateOverflow.view).toBe(1);
  expect(captures.scratchCandidateOverflow.view).toBe(0);
  expect(captures.eventOverflow.view).toBe(0);
  expect(lens).not.toContain('0 means every event fit and dwell totals are complete');
  const explorer = readFileSync(
    'examples/deck/spatial-analysis-explorer/modes/recipes-scenes-fleet.ts',
    'utf8'
  );
  const expression = explorer.match(
    /'Any overflow(?: \(combined flag\))?',\s*([\s\S]*?)\?\s*'YES'/
  )![1];
  const aggregate = new Function(
    'candidateFlag',
    'trackFlag',
    'eventCapacityFlag',
    'eventFlag',
    `return Boolean(${expression});`
  );
  const flag = (value: number) => ({u32: [value]});
  expect(aggregate(flag(1), flag(0), flag(0), flag(0))).toBe(true);
  expect(aggregate(flag(0), flag(1), flag(0), flag(0))).toBe(true);
  expect(aggregate(flag(0), flag(0), flag(1), flag(0))).toBe(true);
  expect(aggregate(flag(0), flag(0), flag(0), flag(0))).toBe(false);
});

test('core:F7 CPU streamline pruning honors a one-round budget', () => {
  const result = computeStreamlinesOnCPU(
    {
      velocities: new Float32Array([3, 0]),
      parameters: getGPUStreamlinesParameterValues({
        fieldExtent: [0, 0, 10, 10],
        gridExtent: [0, 0, 10, 10],
        stepLength: 0.125
      }),
      uint32Parameters: getGPUStreamlinesUint32ParameterValues({seed: 7})
    },
    {
      fieldWidth: 1,
      fieldHeight: 1,
      gridWidth: 1,
      gridHeight: 1,
      seedColumns: 4,
      seedRows: 1,
      stepsPerDirection: 2,
      lineCapacity: 4,
      pointCapacity: 20,
      maximumIterations: 1
    }
  );
  expect([result.count[0], result.overflow[0], result.iterationLimitReached[0]]).toEqual([1, 0, 1]);
});

test('spatial-1:F6 oracle splits slot loss from spare final capacity', () => {
  const counts = new Uint32Array(65536);
  counts[0] = 65536;
  const featureOffsets = new Uint32Array(65537).fill(1);
  featureOffsets[0] = 0;
  const result = generateDotCountsOnCPU({
    positions: new Float32Array([0, 0, 1, 0, 0, 1]),
    featureOffsets,
    polygonOffsets: new Uint32Array([0, 1]),
    ringOffsets: new Uint32Array([0, 3]),
    counts,
    parameters: getGPUDotDensityParameterValues({seed: 1}),
    capacity: 65536
  });
  expect([result.count, result.requiredCount, result.overflow, result.candidateOverflow]).toEqual([
    65535, 65535, 0, 1
  ]);
});

test.each([
  'line',
  'polygon'
] as const)('spatial-1:F6 %s refuses hidden slot loss without a destination', kind => {
  const {device, view, compact} = fixture();
  try {
    const {candidateOverflow, ...status} = compact(65536).status;
    const output = {
      positions: view('float32x2', 65536),
      points: {ids: view('uint32', 65536), status},
      dots: {ids: view('uint32', 65536), status}
    };
    expect(() =>
      kind === 'line'
        ? new GPURandomPointsOnLine({
            positions: view('float32x2', 2),
            lineOffsets: view('uint32', 65537),
            counts: view('uint32', 65536),
            parameters: view('uint32', 8),
            output
          })
        : new GPURandomPointsInPolygon({
            positions: view('float32x2', 3),
            featureOffsets: view('uint32', 65537),
            polygonOffsets: view('uint32', 2),
            ringOffsets: view('uint32', 2),
            counts: view('uint32', 65536),
            parameters: view('uint32', 8),
            output
          })
    ).toThrow(/candidateOverflow.*required/);
  } finally {
    device.destroy();
  }
});

test('spatial-3:F14 rendered buffer-selection snippet uses row names and distinct status destinations', () => {
  const scene = readFileSync(
    'examples/deck/spatial-analysis-showcase/src/scenes/joins/near-transit.scene.ts',
    'utf8'
  );
  const snippet = scene.slice(scene.lastIndexOf('new GPUBufferSelection'));
  expect(snippet).not.toContain('nearestFeatureIds');
  expect(snippet).toContain('nearestFeatureRows');
  expect(snippet).toContain('matchOverflow');
  const nearestName = snippet.match(/distances,\s*(nearestFeature\w+)/)![1];
  const {device, graph, view, scalar, compact} = fixture();
  try {
    expect(() =>
      new GPUBufferSelection({
        points: {kind: 'points', positions: view('float32x2')},
        features: {kind: 'points', positions: view('float32x2')},
        distance: view('float32'),
        candidateCapacity: 1,
        output: {[nearestName]: scalar(), distances: view('float32'), matches: compact()}
      }).getCommandNodes(graph)
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

test.each([
  'stations',
  'bus',
  'lines'
] as const)('final2 G12/G11 rendered %s operands keep distant observations unselected', features => {
  for (const [separation, threshold] of [
    [1000, 400],
    [10, 1]
  ]) {
    const {device, graph, view, scalar} = fixture();
    try {
      const observations = view('float32x2');
      const stations = view('float32x2');
      const buses = view('float32x2');
      const starts = view('float32x2');
      const ends = view('float32x2');
      const coordinates = new Map<object, number[]>([
        [observations, [0, 0]],
        [stations, [separation, 0]],
        [buses, [separation, 0]],
        [starts, [separation, -1]],
        [ends, [separation, 1]]
      ]);
      const props = getNearTransitSnippetProps(features, threshold, {
        projection: {spatialContext: {coordinateSpace: 'planar', metric: 'native'}},
        positions: observations,
        observationPositions: observations,
        stationPositions: stations,
        busPositions: buses,
        starts,
        ends,
        transitStarts: starts,
        transitEnds: ends,
        points: {length: 7},
        distance: {importToGraph: () => view('float32')},
        graph: {importGPUData: () => scalar()},
        drawCommands: {getInstanceCountData: () => null},
        outputMask: scalar(),
        ids: scalar(),
        count: scalar(),
        requiredCount: scalar(),
        matchOverflow: scalar(),
        matchCandidateOverflow: scalar(),
        distances: view('float32'),
        nearestFeatureRows: scalar(),
        joinOverflow: scalar(),
        joinCandidateOverflow: scalar()
      });
      expect(props.features.kind).toBe(features === 'lines' ? 'segments' : 'points');
      const query = coordinates.get(props.points.positions)!;
      const reference =
        props.features.kind === 'points'
          ? coordinates.get(props.features.positions)!
          : coordinates.get(props.features.starts)!;
      const distance =
        props.features.kind === 'points'
          ? Math.hypot(query[0] - reference[0], query[1] - reference[1])
          : Math.abs(query[0] - reference[0]); // vertical segment spanning query y=0
      expect(Number.isFinite(distance)).toBe(true);
      expect(Number(distance <= threshold)).toBe(0);
      expect(props.points.positions === observations).toBe(true);
      if (props.features.kind === 'points') {
        expect(props.features.positions === (features === 'stations' ? stations : buses)).toBe(
          true
        );
      } else {
        expect(props.features.starts === starts && props.features.ends === ends).toBe(true);
      }
      expect(props.candidateCapacity).toBe(observations.length * (features === 'lines' ? 40 : 24));
      expect(() => new GPUBufferSelection(props).getCommandNodes(graph)).not.toThrow();
    } finally {
      device.destroy();
    }
  }
});
