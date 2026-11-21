// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  GPUMaximumInscribedCircle,
  GPUNormalizePolygons,
  GPUNormalizeGeometry,
  GPUSegmentSnap,
  GPURatioConcaveHull,
  GPULinePrecisionReduction,
  GPUSubdividePolygons,
  GPUConvexStraightSkeleton,
  GPULineLoopRemoval,
  GPUPolygonPrecisionReduction,
  GPUCanonicalPolygonEquality,
  GPUCanonicalGeometryEquality
} from '../../../src/gpu-spatial-analysis/geometry-refinement/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {createTopologyOutput} from './topology-output';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {createEqualityMetricFixture, type EqualityView} from './canonical-equality-fixture';

useCPUExecutorTestHooks();

for (const kind of ['tree', 'polygon'] as const) {
  it(`${kind} equality retains float32 upper-edge tolerance acceptance`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const view: EqualityView = (format, values) =>
      createTransientView(
        graph,
        `upper-${serial++}`,
        format,
        values.length / Number(format.match(/x(\d)$/)?.[1] ?? 1)
      );
    try {
      const fixture = createEqualityMetricFixture(
        kind,
        'euclidean',
        view,
        [0.75, 0.75],
        3.4028234663852894e38
      );
      expect(fixture.contributor.getCommandNodes(graph).length).toBeGreaterThan(0);
    } finally {
      device.destroy();
    }
  });
  it.each([
    {name: 'interior', delta: [3, 4] as const, tolerance: 6, expected: 1},
    {
      name: 'exact Pythagorean boundary',
      delta: [3683, 7644] as const,
      tolerance: 8485,
      expected: 1
    },
    {name: 'exterior', delta: [3, 4] as const, tolerance: 4, expected: 0}
  ])(`${kind} Euclidean equality accepts inclusive boundaries: $name`, fixture => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    const view: EqualityView = (format, values) => {
      const result = createTransientView(
        graph,
        `boundary-${serial++}`,
        format,
        values.length / Number(format.match(/x(\d)$/)?.[1] ?? 1)
      );
      memory.write(
        result,
        format.startsWith('float') ? Float32Array.from(values) : Uint32Array.from(values)
      );
      return result;
    };
    try {
      const comparison = createEqualityMetricFixture(
        kind,
        'euclidean',
        view,
        fixture.delta,
        fixture.tolerance
      );
      executeCommandNodesOnCPU(comparison.contributor.getCommandNodes(graph), memory);
      // 3683^2 + 7644^2 = 71,995,225 = 8485^2 exactly; (3,4) has distance five.
      expect(Array.from(memory.readUint32(comparison.output))).toEqual([fixture.expected]);
    } finally {
      device.destroy();
    }
  });
}

for (const kind of ['tree', 'polygon'] as const) {
  it.each([
    undefined,
    'per-coordinate',
    'euclidean'
  ] as const)(`${kind} equality names its tolerance metric: %s`, metric => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const memory = createCPUGraphMemory();
    const view: EqualityView = (format, values) => {
      const width = Number(format.match(/x(\d)$/)?.[1] ?? 1);
      const result = createTransientView(
        graph,
        `equality-${serial++}`,
        format,
        values.length / width
      );
      memory.write(
        result,
        format.startsWith('float') ? Float32Array.from(values) : Uint32Array.from(values)
      );
      return result;
    };
    const fixture = createEqualityMetricFixture(kind, metric, view);
    executeCommandNodesOnCPU(fixture.contributor.getCommandNodes(graph), memory);
    // sqrt(.75^2 + .75^2) = sqrt(1.125) > 1; each coordinate separately is < 1.
    expect(Array.from(memory.readUint32(fixture.output))).toEqual([metric === 'euclidean' ? 0 : 1]);
    expect(() =>
      fixture.contributor instanceof GPUCanonicalGeometryEquality
        ? new GPUCanonicalGeometryEquality({
            ...fixture.contributor.props,
            toleranceMetric: 'taxicab' as never
          })
        : new GPUCanonicalPolygonEquality({
            ...fixture.contributor.props,
            toleranceMetric: 'taxicab' as never
          })
    ).toThrow(/metric-test.*toleranceMetric.*per-coordinate.*euclidean.*taxicab/);
  });
}
let serial = 0;
function createFixture(coordinates = [0, 0, 4, 0, 4, 4, 0, 4]) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const memory = createCPUGraphMemory();
  const positions = createTransientView(
    graph,
    `positions-${serial++}`,
    'float32x2',
    coordinates.length / 2
  );
  const ringOffsets = createTransientView(graph, `rings-${serial++}`, 'uint32', 2);
  const polygonOffsets = createTransientView(graph, `polygons-${serial++}`, 'uint32', 2);
  const featureOffsets = createTransientView(graph, `features-${serial++}`, 'uint32', 2);
  memory.write(positions, Float32Array.from(coordinates));
  memory.write(ringOffsets, Uint32Array.from([0, coordinates.length / 2]));
  memory.write(polygonOffsets, Uint32Array.from([0, 1]));
  memory.write(featureOffsets, Uint32Array.from([0, 1]));
  return {graph, memory, geometry: {positions, ringOffsets, polygonOffsets, featureOffsets}};
}

it('geometry refinement kernels parse and respect the baseline storage binding limit', () => {
  const {graph, geometry} = createFixture();
  const contributors = [
    new GPUMaximumInscribedCircle({
      ...geometry,
      tolerance: 0.01,
      output: {
        circles: createTransientView(graph, 'circles', 'float32x4', 1),
        witnesses: createTransientView(graph, 'witnesses', 'float32x2', 1),
        status: createTransientView(graph, 'status', 'uint32', 1)
      }
    }),
    new GPUNormalizePolygons({
      ...geometry,
      output: {
        positions: createTransientView(graph, 'normalized', 'float32x2', 4),
        ringOffsets: createTransientView(graph, 'normal-rings', 'uint32', 2),
        polygonOffsets: createTransientView(graph, 'normal-polygons', 'uint32', 2)
      }
    }),
    new GPUSegmentSnap({
      // Exercise the explicit reference route in the serial WGSL interpreter.
      searchStructure: 'brute-force',
      positions: geometry.positions,
      lineOffsets: geometry.ringOffsets,
      referencePositions: geometry.positions,
      tolerance: 1,
      maximumVerticesPerLine: 8,
      output: {
        positions: createTransientView(graph, 'snapped', 'float32x2', 8),
        sourceEdges: createTransientView(graph, 'source-edges', 'uint32', 8),
        counts: createTransientView(graph, 'snap-counts', 'uint32x4', 1)
      }
    })
  ];
  for (const contributor of contributors) {
    for (const {source, constants} of getGPUCommandNodeKernelSources(
      contributor.getCommandNodes(graph)
    )) {
      expect(auditWGSLModule(source, constants)).toEqual([]);
      const reflected = new WgslReflect(source);
      expect(reflected.storage.length).toBeLessThanOrEqual(8);
    }
  }
});

it('maximum inscribed circle bounds a square and reports forced exhaustion', () => {
  for (const maximumIterations of [0, 64]) {
    const {graph, memory, geometry} = createFixture();
    const output = {
      circles: createTransientView(graph, 'circles', 'float32x4', 1),
      witnesses: createTransientView(graph, 'witnesses', 'float32x2', 1),
      status: createTransientView(graph, 'status', 'uint32', 1)
    };
    const nodes = new GPUMaximumInscribedCircle({
      ...geometry,
      maximumIterations,
      cellCapacity: 256,
      tolerance: 0.01,
      output
    }).getCommandNodes(graph);
    executeCommandNodesOnCPU(nodes, memory);
    const circle = memory.readFloat32(output.circles);
    expect(Array.from(circle.slice(0, 3))).toEqual([2, 2, 2]);
    expect(memory.readUint32(output.status)[0]).toBe(maximumIterations === 0 ? 1 : 0);
    expect(circle[3]).toBeGreaterThanOrEqual(0);
    if (maximumIterations) {
      expect(circle[3]).toBeLessThanOrEqual(0.01);
    }
    const witness = memory.readFloat32(output.witnesses);
    expect(Math.hypot(witness[0] - circle[0], witness[1] - circle[1])).toBeCloseTo(2);
  }
});

it('normalization is invariant to ring reversal and rotation and idempotent', () => {
  for (const coordinates of [
    [0, 0, 4, 0, 4, 4, 0, 4],
    [4, 4, 4, 0, 0, 0, 0, 4],
    [0, 0, 0, 4, 4, 4, 4, 0]
  ]) {
    const {graph, memory, geometry} = createFixture(coordinates);
    const output = {
      positions: createTransientView(graph, 'normalized', 'float32x2', 4),
      ringOffsets: createTransientView(graph, 'normal-rings', 'uint32', 2),
      polygonOffsets: createTransientView(graph, 'normal-polygons', 'uint32', 2)
    };
    executeCommandNodesOnCPU(
      new GPUNormalizePolygons({...geometry, output}).getCommandNodes(graph),
      memory
    );
    expect(Array.from(memory.readFloat32(output.positions))).toEqual([0, 0, 0, 4, 4, 4, 4, 0]);
    expect(Array.from(memory.readUint32(output.ringOffsets))).toEqual([0, 4]);
  }
});

it('segment snap inserts ordered, unique references and reports exact capacity demand', () => {
  const {graph, memory, geometry} = createFixture([0, 0, 6, 0]);
  const references = createTransientView(graph, 'references', 'float32x2', 4);
  memory.write(references, Float32Array.from([4, 0.1, 2, 0, 2, 0, 3, 5]));
  const output = {
    positions: createTransientView(graph, 'snapped', 'float32x2', 3),
    sourceEdges: createTransientView(graph, 'source', 'uint32', 3),
    counts: createTransientView(graph, 'counts', 'uint32x4', 1)
  };
  executeCommandNodesOnCPU(
    new GPUSegmentSnap({
      // Exercise the explicit reference route in the serial WGSL interpreter.
      searchStructure: 'brute-force',
      positions: geometry.positions,
      lineOffsets: geometry.ringOffsets,
      referencePositions: references,
      tolerance: 0.2,
      maximumVerticesPerLine: 3,
      output
    }).getCommandNodes(graph),
    memory
  );
  expect(Array.from(memory.readFloat32(output.positions))).toEqual([
    0,
    0,
    2,
    0,
    4,
    Math.fround(0.1)
  ]);
  expect(Array.from(memory.readUint32(output.counts))).toEqual([3, 4, 1, 0]);
});

it('line precision distinguishes valid-output from keep-collapsed', () => {
  for (const mode of ['valid-output', 'keep-collapsed'] as const) {
    const {graph, memory, geometry} = createFixture([0.1, 0.1, 0.2, 0.2]);
    const output = {
      positions: createTransientView(graph, 'rounded', 'float32x2', 2),
      lineOffsets: createTransientView(graph, 'offsets', 'uint32', 2),
      status: createTransientView(graph, 'status', 'uint32', 1)
    };
    executeCommandNodesOnCPU(
      new GPULinePrecisionReduction({
        positions: geometry.positions,
        lineOffsets: geometry.ringOffsets,
        gridSize: 1,
        mode,
        output
      }).getCommandNodes(graph),
      memory
    );
    expect(Array.from(memory.readUint32(output.lineOffsets))).toEqual([
      0,
      mode === 'keep-collapsed' ? 2 : 0
    ]);
    expect(memory.readUint32(output.status)[0]).toBe(mode === 'keep-collapsed' ? 2 : 1);
  }
});

it('convex subdivision merges a square and honors a triangle vertex limit', () => {
  for (const maximumVertices of [3, 4]) {
    const {graph, memory, geometry} = createFixture();
    const output = {
      positions: createTransientView(graph, 'pieces-positions', 'float32x2', 2 * maximumVertices),
      pieces: createTransientView(graph, 'pieces', 'uint32x4', 2),
      status: createTransientView(graph, 'status', 'uint32x4', 1)
    };
    const nodes = new GPUSubdividePolygons({...geometry, maximumVertices, output}).getCommandNodes(
      graph
    );
    for (const {source, constants} of getGPUCommandNodeKernelSources(nodes)) {
      expect(auditWGSLModule(source, constants)).toEqual([]);
    }
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(output.status))).toEqual(
      maximumVertices === 3 ? [2, 2, 0, 0] : [1, 1, 0, 0]
    );
    const pieces = memory.readUint32(output.pieces);
    const positions = memory.readFloat32(output.positions);
    let area = 0;
    for (let piece = 0; piece < memory.readUint32(output.status)[0]; piece++) {
      expect(pieces[piece * 4]).toBeLessThanOrEqual(maximumVertices);
      const base = piece * maximumVertices * 2;
      for (let vertex = 0; vertex < pieces[piece * 4]; vertex++) {
        const next = (vertex + 1) % pieces[piece * 4];
        area +=
          positions[base + vertex * 2] * positions[base + next * 2 + 1] -
          positions[base + next * 2] * positions[base + vertex * 2 + 1];
      }
    }
    expect(Math.abs(area) / 2).toBe(16);
  }
});

it('ratio hull kernels pass the WGSL audit (execution is covered by the headless Shapely comparison)', () => {
  const {graph, memory, geometry} = createFixture([0, 0, 0, 3, 1, 1, 3, 0, 3, 3]);
  const capacity = 5;
  const output = {
    triangles: createTransientView(graph, 'triangles', 'uint32x3', capacity),
    triangleValidity: createTransientView(graph, 'triangle-validity', 'uint32', capacity),
    boundaryEdges: createTransientView(graph, 'edges', 'uint32x2', capacity * 3),
    edgeValidity: createTransientView(graph, 'edge-validity', 'uint32', capacity * 3),
    status: {converged: createTransientView(graph, 'converged', 'uint32', 1)}
  };
  const nodes = new GPURatioConcaveHull({
    positions: geometry.positions,
    ratio: 1,
    output
  }).getCommandNodes(graph);
  for (const {source, constants} of getGPUCommandNodeKernelSources(nodes)) {
    expect(auditWGSLModule(source, constants)).toEqual([]);
  }
  // The erosion kernel is a cooperative workgroup (barriers), which the CPU interpreter cannot run.
  expect(nodes.length).toBeGreaterThan(0);
  expect(memory).toBeDefined();
});

it('convex skeleton emits a rectangle ridge and rejects a concave source', () => {
  for (const coordinates of [
    [0, 0, 4, 0, 4, 2, 0, 2],
    [0, 0, 4, 0, 1, 1, 0, 2]
  ]) {
    const {graph, memory, geometry} = createFixture(coordinates);
    const output = {
      segments: createTransientView(graph, 'segments', 'float32x4', 8),
      sourceEdges: createTransientView(graph, 'edges', 'uint32x2', 8),
      offsetPositions: createTransientView(graph, 'offset', 'float32x2', 4),
      status: createTransientView(graph, 'status', 'uint32x4', 1)
    };
    const nodes = new GPUConvexStraightSkeleton({
      positions: geometry.positions,
      ringOffsets: geometry.ringOffsets,
      maximumVertices: 4,
      offsetDistance: 0.5,
      output
    }).getCommandNodes(graph);
    for (const {source, constants} of getGPUCommandNodeKernelSources(nodes)) {
      expect(auditWGSLModule(source, constants)).toEqual([]);
    }
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(output.status))).toEqual(
      coordinates[4] === 4 ? [5, 5, 4, 0] : [0, 0, 0, 1]
    );
  }
});

it('polygon precision and loop-removal composition kernels pass static WGSL audit', () => {
  const {graph, geometry} = createFixture([0, 0, 4, 0, 4, 4, 0, 4, 0, 0]);
  const view = <Format extends 'uint32' | 'float32' | 'float32x2' | 'float32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `topology-${serial++}`, format, length);
  const contributors = [
    new GPUPolygonPrecisionReduction({
      polygons: {kind: 'polygons', ...geometry},
      gridSize: 1,
      lattice: view('float32', 4),
      intersectionCapacity: 8,
      output: createTopologyOutput(view)
    }),
    new GPULineLoopRemoval({
      lines: {kind: 'lines', positions: geometry.positions, lineOffsets: geometry.ringOffsets},
      intersectionCapacity: 8,
      output: {
        positions: view('float32x2', 16),
        lineOffsets: view('uint32', 2),
        sourceEdges: view('uint32', 16),
        status: createTransientView(graph, 'loop-status', 'uint32x4', 1)
      }
    })
  ];
  for (const contributor of contributors) {
    for (const {source, constants} of getGPUCommandNodeKernelSources(
      contributor.getCommandNodes(graph)
    )) {
      expect(auditWGSLModule(source, constants)).toEqual([]);
    }
  }
});

it('canonical equality compares structure and coordinates without hash collisions', () => {
  const {graph, memory, geometry} = createFixture();
  const otherPositions = createTransientView(graph, 'other-positions', 'float32x2', 4);
  const output = createTransientView(graph, 'equality', 'uint32', 1);
  const contributor = new GPUCanonicalPolygonEquality({
    left: geometry,
    right: {...geometry, positions: otherPositions},
    output
  });
  const nodes = contributor.getCommandNodes(graph);
  for (const {source, constants} of getGPUCommandNodeKernelSources(nodes)) {
    expect(auditWGSLModule(source, constants)).toEqual([]);
  }
  for (const [coordinates, expected] of [
    [[0, 0, 4, 0, 4, 4, 0, 4], 1],
    [[0, 0, 4, 0, 4, 4, 0, 3], 0],
    [[0, 0, 4, 0, 4, 4, 0, Number.NaN], 2]
  ] as const) {
    memory.write(otherPositions, Float32Array.from(coordinates));
    executeCommandNodesOnCPU(nodes, memory);
    expect(memory.readUint32(output)[0]).toBe(expected);
  }
});

it('MIC empty and nonfinite features are unknown rather than radius zero', () => {
  for (const nonfinite of [false, true]) {
    const {graph, memory, geometry} = createFixture(nonfinite ? [Number.NaN, 0] : [0, 0]);
    if (!nonfinite) {
      memory.write(geometry.ringOffsets, Uint32Array.from([0, 0]));
    }
    const output = {
      circles: createTransientView(graph, 'circles', 'float32x4', 1),
      witnesses: createTransientView(graph, 'witnesses', 'float32x2', 1),
      status: createTransientView(graph, 'status', 'uint32', 1)
    };
    executeCommandNodesOnCPU(
      new GPUMaximumInscribedCircle({...geometry, tolerance: 0.01, output}).getCommandNodes(graph),
      memory
    );
    expect(memory.readUint32(output.status)[0]).toBe(nonfinite ? 3 : 2);
    expect(Number.isNaN(memory.readFloat32(output.circles)[2])).toBe(true);
  }
});

it('segment snap removes duplicate references separated by projection-tied references', () => {
  const {graph, memory, geometry} = createFixture([0, 0, 6, 0]);
  const references = createTransientView(graph, 'references', 'float32x2', 3);
  memory.write(references, Float32Array.from([2, 0, 2, 0.1, 2, 0]));
  const output = {
    positions: createTransientView(graph, 'snapped', 'float32x2', 8),
    sourceEdges: createTransientView(graph, 'source', 'uint32', 8),
    counts: createTransientView(graph, 'counts', 'uint32x4', 1),
    validity: createTransientView(graph, 'validity', 'uint32', 1)
  };
  executeCommandNodesOnCPU(
    new GPUSegmentSnap({
      // Exercise the explicit reference route in the serial WGSL interpreter.
      searchStructure: 'brute-force',
      positions: geometry.positions,
      lineOffsets: geometry.ringOffsets,
      referencePositions: references,
      tolerance: 0.2,
      maximumVerticesPerLine: 8,
      output
    }).getCommandNodes(graph),
    memory
  );
  expect(Array.from(memory.readUint32(output.counts))).toEqual([4, 4, 0, 0]);
  expect(Array.from(memory.readFloat32(output.positions)).slice(0, 8)).toEqual([
    0,
    0,
    2,
    0,
    2,
    Math.fround(0.1),
    6,
    0
  ]);
  expect(memory.readUint32(output.validity)[0]).toBe(1);
});

it('tree normalization exact winding respects the baseline binding and WGSL contracts', () => {
  const {graph, geometry} = createFixture();
  const contributor = new GPUNormalizeGeometry({
    positions: geometry.positions,
    nodes: createTransientView(graph, 'tree-nodes', 'uint32x4', 2),
    children: createTransientView(graph, 'tree-children', 'uint32', 1),
    output: {
      positions: createTransientView(graph, 'normalized-tree', 'float32x2', 4),
      children: createTransientView(graph, 'normalized-children', 'uint32', 1),
      status: createTransientView(graph, 'normalized-status', 'uint32', 2)
    }
  });
  for (const {source, constants} of getGPUCommandNodeKernelSources(
    contributor.getCommandNodes(graph)
  )) {
    expect(auditWGSLModule(source, constants)).toEqual([]);
    expect(new WgslReflect(source).storage.length).toBeLessThanOrEqual(8);
  }
});
