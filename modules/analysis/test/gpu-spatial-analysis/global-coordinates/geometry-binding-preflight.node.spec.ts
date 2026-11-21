// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUSphericalArcOverlay} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-arc-overlay';
import {GPUSphericalConvexIntersection} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-convex-intersection';
import {GPUSphericalPolygonOverlay} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-polygon-overlay';
import {GPUGroupGeometry} from '../../../src/gpu-spatial-analysis/group-geometry/gpu-group-geometry';
import {GPULonLatPolygonPreparation} from '../../../src/gpu-spatial-analysis/lon-lat-polygons/gpu-lon-lat-polygon-preparation';
import {GPUMapColoring} from '../../../src/gpu-spatial-analysis/map-coloring/gpu-map-coloring';
import {GPUPairGather} from '../../../src/gpu-spatial-analysis/pair-gather/gpu-pair-gather';
import {GPUCoverageDissolve} from '../../../src/gpu-spatial-analysis/polygon-coverage-topology/gpu-coverage-dissolve';
import {GPUCoverageValidity} from '../../../src/gpu-spatial-analysis/polygon-coverage-topology/gpu-coverage-validity';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createViews(graph: GPUCommandGraph, packed = false, prefix = 'field') {
  let serial = 0;
  let offset = 0;
  // Import/declare one physical input buffer once, then create every read-only field as a view.
  const backing = packed
    ? createTransientView(graph, `${prefix}-packed-inputs`, 'uint32', 4096).buffer
    : undefined;
  const views: GraphDataView[] = [];
  const view = <
    Format extends 'uint32' | 'uint32x2' | 'float32' | 'float32x2' | 'float32x3' | 'float32x4'
  >(
    format: Format,
    length: number,
    input = false
  ) => {
    let result: GraphDataView<Format>;
    if (input && backing) {
      result = graph.createDataView(backing, {format, byteOffset: offset, length});
      offset += Math.ceil(Math.max(length * result.rowByteLength, 4) / 256) * 256;
    } else result = createTransientView(graph, `${prefix}-${serial++}`, format, length);
    views.push(result);
    return result;
  };
  const polygons = (vertices = 4) => ({
    positions: view('float32x2', vertices, true),
    ringOffsets: view('uint32', 2, true),
    polygonOffsets: view('uint32', 2, true),
    featureOffsets: view('uint32', 2, true)
  });
  return {view, polygons, views};
}

type Fixture = {producer: GPUCommandNodeProducer; views: GraphDataView[]};
type Factory = (graph: GPUCommandGraph, packed?: boolean, scratch?: boolean) => Fixture;

const factories: Record<string, Factory> = {
  GPUSphericalArcOverlay(graph, packed, scratch) {
    const {view, views} = createViews(graph, packed);
    return {
      views,
      producer: new GPUSphericalArcOverlay({
        left: {
          positions: view('float32x3', 3, true),
          ringOffsets: view('uint32', 2, true),
          referenceInside: false
        },
        right: {
          positions: view('float32x3', 3, true),
          ringOffsets: view('uint32', 2, true),
          referenceInside: false
        },
        referencePoint: [0, 0, 1],
        operation: 'union',
        output: {
          positions: view('float32x3', scratch ? 32 : 8),
          ringOffsets: view('uint32', 3),
          counts: view('uint32', 4),
          status: view('uint32', 1)
        }
      })
    };
  },
  GPUSphericalConvexIntersection(graph, packed, scratch) {
    const {view, views} = createViews(graph, packed);
    const capacity = scratch ? 43 : 8;
    return {
      views,
      producer: new GPUSphericalConvexIntersection({
        firstPositions: view('float32x3', 3, true),
        secondPositions: view('float32x3', 3, true),
        firstOffsets: view('uint32', 2, true),
        secondOffsets: view('uint32', 2, true),
        capacity,
        output: {
          positions: view('float32x3', capacity),
          counts: view('uint32x2', 1),
          status: view('uint32', 1),
          measures: view('float32x4', 1)
        }
      })
    };
  },
  GPUSphericalPolygonOverlay(graph, packed, scratch) {
    const {view, views} = createViews(graph, packed);
    const operand = () => ({
      kind: 'polygons' as const,
      positions: view('float32x3', 3, true),
      ringOffsets: view('uint32', 2, true),
      polygonOffsets: view('uint32', 2, true),
      featureOffsets: view('uint32', 2, true)
    });
    return {
      views,
      producer: new GPUSphericalPolygonOverlay({
        left: operand(),
        right: operand(),
        chartCenter: [0, 0, 1],
        operation: 'union',
        vertexTolerance: 0.01,
        capacity: {intersections: scratch ? 4 : 8, nodedSegments: 16},
        output: {
          geometry: {
            kind: 'polygons',
            positions: view('float32x3', 16),
            ringOffsets: view('uint32', 3),
            polygonOffsets: view('uint32', 3),
            featureOffsets: view('uint32', 3)
          },
          boundary: {
            endpoints: view('float32x4', 8),
            operandIds: view('uint32', 8),
            sourceFeatureIds: view('uint32', 8),
            status: {
              count: view('uint32', 1),
              overflow: view('uint32', 1),
              requiredCount: view('uint32', 1),
              candidateOverflow: view('uint32', 1)
            }
          },
          status: {
            count: view('uint32', 1),
            overflow: view('uint32', 1),
            candidateOverflow: view('uint32', 1)
          },
          complement: view('uint32', 1)
        }
      })
    };
  },
  GPUGroupGeometry(graph, packed, scratch) {
    const {view, views} = createViews(graph, packed);
    const groupCount = scratch ? 64 : 2;
    return {
      views,
      producer: new GPUGroupGeometry({
        positions: view('float32x2', 16, true),
        labels: view('uint32', 16, true),
        parameters: view('float32', 8, true),
        groupCount,
        output: {counts: view('uint32', groupCount), meanCenters: view('float32x2', groupCount)}
      })
    };
  },
  GPULonLatPolygonPreparation(graph, packed, scratch) {
    const {view, views} = createViews(graph, packed);
    const rings = scratch ? 33 : 1;
    return {
      views,
      producer: new GPULonLatPolygonPreparation({
        positions: view('float32x2', 4, true),
        ringOffsets: view('uint32', rings + 1, true),
        polygonOffsets: view('uint32', 2, true),
        featureOffsets: view('uint32', 2, true),
        parameters: view('float32', 4, true),
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'none',
          edgeModel: 'linear'
        },
        output: {
          positions: view('float32x2', 16),
          ringOffsets: view('uint32', 2 * rings + 1),
          polygonOffsets: view('uint32', 3),
          featureOffsets: view('uint32', 2),
          ringFlags: view('uint32', rings),
          status: {
            count: view('uint32', 1),
            overflow: view('uint32', 1),
            requiredCount: view('uint32', 1),
            invalidCount: view('uint32', 1),
            approximation: view('uint32', 1)
          }
        }
      })
    };
  },
  GPUMapColoring(graph, packed) {
    const {view, views} = createViews(graph, packed);
    return {
      views,
      producer: new GPUMapColoring({
        weights: {
          offsets: view('uint32', 2, true),
          neighbors: view('uint32', 1, true),
          weights: createTransientView(graph, 'unused-weights', 'float32', 1)
        },
        parameters: view('uint32', 1, true),
        maximumIterations: 1,
        output: {
          colors: view('uint32', 1),
          colorCount: view('uint32', 1),
          conflictCount: view('uint32', 1),
          status: {converged: view('uint32', 1), iterationCount: view('uint32', 1)}
        }
      })
    };
  },
  GPUPairGather(graph, packed, scratch) {
    const {view, views} = createViews(graph, packed);
    return {
      views,
      producer: new GPUPairGather({
        neighbors: {
          rows: view('uint32', scratch ? 129 : 4, true),
          counts: view('uint32', 1, true),
          capacity: scratch ? 129 : 4
        },
        how: 'left',
        leftColumns: [{source: view('float32', 1, true), output: view('float32', 4)}],
        output: {
          leftRows: view('uint32', 4),
          rightRows: view('uint32', 4),
          status: {
            count: view('uint32', 1),
            overflow: view('uint32', 1),
            matchedCount: view('uint32', 1)
          }
        }
      })
    };
  },
  GPUCoverageDissolve(graph, packed, scratch) {
    const {view, polygons, views} = createViews(graph, packed);
    return {
      views,
      producer: new GPUCoverageDissolve({
        ...polygons(scratch ? 33 : 4),
        labels: view('uint32', 1, true),
        output: {
          positions: view('float32x2', 48),
          ringOffsets: view('uint32', 3),
          ringShells: view('uint32', 2),
          ringAreas: view('float32', 2),
          polygons: {
            kind: 'polygons',
            positions: view('float32x2', 48),
            ringOffsets: view('uint32', 3),
            polygonOffsets: view('uint32', 3),
            featureOffsets: view('uint32', 3),
            sourceIds: view('uint32', 2)
          },
          status: {
            count: view('uint32', 1),
            overflow: view('uint32', 1),
            boundarySegmentCount: view('uint32', 1),
            uncertainCount: view('uint32', 1)
          }
        }
      })
    };
  },
  GPUCoverageValidity(graph, packed, scratch) {
    const {view, polygons, views} = createViews(graph, packed);
    return {
      views,
      producer: new GPUCoverageValidity({
        ...polygons(),
        parameters: view('float32', 4, true),
        leafCapacity: scratch ? 64 : 4,
        spatialSort: true,
        output: {
          segmentFlags: view('uint32', 4),
          featureInvalidCounts: view('uint32', 1),
          isValid: view('uint32', 1),
          status: {invalidCount: view('uint32', 1)}
        }
      })
    };
  }
};

function expectUnchangedFailure(
  graph: GPUCommandGraph,
  producer: GPUCommandNodeProducer,
  message: RegExp
) {
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const add = vi.spyOn(graph, 'add');
  expect(() => producer.getCommandNodes(graph)).toThrow(message);
  expect(allocate).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
  expect(add).not.toHaveBeenCalled();
  vi.restoreAllMocks();
}

for (const [name, factory] of Object.entries(factories)) {
  it(`${name} checks the aligned binding prefix rather than only logical bytes`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const allocate = graph.createTransientBuffer.bind(graph);
    const createView = graph.createDataView.bind(graph);
    vi.spyOn(graph, 'createTransientBuffer').mockImplementation(props =>
      allocate({...props, byteLength: props.byteLength + 4})
    );
    vi.spyOn(graph, 'createDataView').mockImplementation((buffer, props) =>
      createView(buffer, {...props, byteOffset: 4})
    );
    const {producer, views} = factory(graph);
    vi.restoreAllMocks();
    const logicalBytes = Math.max(...views.map(view => view.length * view.rowByteLength));
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: logicalBytes});
    expectUnchangedFailure(graph, producer, /needs \d+ bytes/);
    device.destroy();
  });

  it(`${name} checks STORAGE usage of every caller field before mutation`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const {producer, views} = factory(graph);
    for (const view of views) {
      const descriptor = Object.getOwnPropertyDescriptor(view.buffer, 'usage')!;
      Object.defineProperty(view.buffer, 'usage', {
        ...descriptor,
        value: Buffer.COPY_SRC | Buffer.COPY_DST
      });
      expectUnchangedFailure(graph, producer, /buffer usage must include Buffer.STORAGE/);
      Object.defineProperty(view.buffer, 'usage', descriptor);
    }
    device.destroy();
  });

  it(`${name} preflights caller or nested scratch limits without graph mutation`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const {producer} = factory(graph, false, true);
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {
      value: name === 'GPUMapColoring' ? 12 : 512
    });
    expectUnchangedFailure(graph, producer, /needs \d+ bytes/);
    device.destroy();
  });

  it(`${name} accepts disjoint read-only fields packed in one physical input buffer`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const {producer} = factory(graph, true);
    recordWGSLKernelNodes(graph);
    const nodes = producer.getCommandNodes(graph);
    expect(() => graph.add(nodes)).not.toThrow();
    for (const node of nodes) {
      const record = getWGSLKernelNodeRecord(graph, node.id);
      if (!record) continue;
      const readers = new Set(
        record.bindings
          .filter(binding => binding.access === 'read')
          .map(binding => binding.view.buffer)
      );
      for (const binding of record.bindings.filter(binding => binding.access === 'read_write'))
        expect(readers.has(binding.view.buffer), `${node.id}.${binding.name} physical access`).toBe(
          false
        );
      if (Object.hasOwn(factories, record.operation)) {
        const code = record.source.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '');
        expect(
          (code.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []).filter(word =>
            WGSL_RESERVED_WORDS.has(word)
          )
        ).toEqual([]);
      }
    }
    device.destroy();
  });
}

it('preflights the dissolve assembly shell hierarchy before any declarations', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const {producer} = factories['GPUCoverageDissolve'](graph);
  const props = (producer as GPUCoverageDissolve).props;
  const {view} = createViews(graph, false, 'large-shell');
  const output = {
    ...props.output,
    ringOffsets: view('uint32', 34),
    ringShells: view('uint32', 33),
    ringAreas: view('float32', 33),
    polygons: {
      ...props.output.polygons!,
      ringOffsets: view('uint32', 34),
      polygonOffsets: view('uint32', 34),
      featureOffsets: view('uint32', 34),
      sourceIds: view('uint32', 33)
    }
  };
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 768});
  expectUnchangedFailure(
    graph,
    new GPUCoverageDissolve({...props, output}),
    /assembly shell hierarchy scratch needs 1016 bytes/
  );
  device.destroy();
});

it('accepts packed scalar coloring outputs across passes and ignores unused edge values', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const {producer} = factories['GPUMapColoring'](graph);
  const props = (producer as GPUMapColoring).props;
  const packed = createTransientView(graph, 'packed-results', 'uint32', 2);
  Object.defineProperty((props.weights as {weights: GraphDataView}).weights.buffer, 'usage', {
    value: Buffer.COPY_SRC
  });
  const output = {
    ...props.output,
    colorCount: graph.createDataView(packed.buffer, {format: 'uint32', length: 1}),
    conflictCount: graph.createDataView(packed.buffer, {format: 'uint32', byteOffset: 4, length: 1})
  };
  expect(() =>
    graph.add(new GPUMapColoring({...props, output}).getCommandNodes(graph))
  ).not.toThrow();
  // A disjoint, aligned slice still conflicts if the same dispatch reads colors and writes it.
  const colors = graph.createDataView(packed.buffer, {format: 'uint32', byteOffset: 0, length: 1});
  expect(
    () =>
      new GPUMapColoring({
        ...props,
        output: {...props.output, colors, conflictCount: output.conflictCount}
      })
  ).toThrow(/read-only and writable storage/);
  device.destroy();
});

it('accepts packed gathered columns written separately, but rejects a column beside its read-only row IDs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const {producer} = factories['GPUPairGather'](graph);
  const props = (producer as GPUPairGather).props;
  const packed = createTransientView(graph, 'packed-columns', 'uint32', 8);
  const first = graph.createDataView(packed.buffer, {format: 'float32', length: 4});
  const second = graph.createDataView(packed.buffer, {
    format: 'float32',
    byteOffset: 16,
    length: 4
  });
  const leftColumns = [
    {source: props.leftColumns![0].source, output: first},
    {source: props.leftColumns![0].source, output: second}
  ];
  expect(() =>
    graph.add(new GPUPairGather({...props, leftColumns}).getCommandNodes(graph))
  ).not.toThrow();
  expect(
    () =>
      new GPUPairGather({
        ...props,
        leftColumns: [leftColumns[1]],
        output: {
          ...props.output,
          leftRows: graph.createDataView(packed.buffer, {format: 'uint32', length: 4})
        }
      })
  ).toThrow(/read-only and writable storage/);
  device.destroy();
});

it('publishes a packed boundary count separately from the assembly reader', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = (factories['GPUCoverageDissolve'](graph).producer as GPUCoverageDissolve).props;
  const packed = createTransientView(graph, 'boundary-and-ring-counts', 'uint32', 2);
  const output = {
    ...props.output,
    status: {
      ...props.output.status,
      boundarySegmentCount: graph.createDataView(packed.buffer, {format: 'uint32', length: 1}),
      count: graph.createDataView(packed.buffer, {format: 'uint32', byteOffset: 4, length: 1})
    }
  };
  recordWGSLKernelNodes(graph);
  const nodes = new GPUCoverageDissolve({...props, output}).getCommandNodes(graph);
  expect(() => graph.add(nodes)).not.toThrow();
  expect(nodes.some(node => node.id.endsWith('-publish-boundary-count'))).toBe(true);
  for (const node of nodes) {
    const record = getWGSLKernelNodeRecord(graph, node.id);
    if (!record) continue;
    const reads = record.bindings.filter(binding => binding.access === 'read');
    for (const write of record.bindings.filter(binding => binding.access === 'read_write'))
      expect(reads.some(read => read.view.buffer === write.view.buffer)).toBe(false);
  }
  device.destroy();
});

it('rejects dissolve tolerances that cannot be represented by generated f32 values', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = (factories['GPUCoverageDissolve'](graph).producer as GPUCoverageDissolve).props;
  for (const value of [Infinity, 1e40])
    expect(() => new GPUCoverageDissolve({...props, snapGridSize: value})).toThrow(/snapGridSize/);
  for (const value of [Infinity, 1e40, 1e-100])
    expect(() => new GPUCoverageDissolve({...props, vertexTolerance: value})).toThrow(
      /vertexTolerance/
    );
  device.destroy();
});

it('checks only the scratch actually used by group standard distances', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const {view} = createViews(graph);
  const producer = new GPUGroupGeometry({
    positions: view('float32x2', 16),
    labels: view('uint32', 16),
    parameters: view('float32', 8),
    groupCount: 65,
    output: {standardDistances: view('float32', 65)}
  });
  // Sorted geography needs 520-byte means, but no 780-byte ellipse table.
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 600});
  expect(() => graph.add(producer.getCommandNodes(graph))).not.toThrow();
  device.destroy();
});

it('ignores unused group weights when only an unweighted mean is requested', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = (factories['GPUGroupGeometry'](graph).producer as GPUGroupGeometry).props;
  const weights = createTransientView(graph, 'unused-group-weights', 'float32', 16);
  Object.defineProperty(weights.buffer, 'usage', {value: Buffer.COPY_SRC});
  expect(() =>
    graph.add(new GPUGroupGeometry({...props, weights}).getCommandNodes(graph))
  ).not.toThrow();
  device.destroy();
});

it('NAM-09 rejects unknown spherical operations before reference membership and preserves XOR', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const arc = factories['GPUSphericalArcOverlay'](graph).producer as GPUSphericalArcOverlay;
    const polygon = factories['GPUSphericalPolygonOverlay'](new GPUCommandGraph(device))
      .producer as GPUSphericalPolygonOverlay;
    for (const operation of ['Union', 'unrelated']) {
      expect(
        () => new GPUSphericalArcOverlay({...arc.props, operation: operation as 'union'})
      ).toThrow(/operation.*intersection.*union.*difference.*symmetric-difference.*received/);
      expect(
        () => new GPUSphericalPolygonOverlay({...polygon.props, operation: operation as 'union'})
      ).toThrow(/operation.*intersection.*union.*difference.*symmetric-difference.*received/);
    }
    // XOR truth table is independent of the arrangement algorithm.
    for (const left of [false, true])
      for (const right of [false, true]) {
        expect(
          new GPUSphericalArcOverlay({
            ...arc.props,
            operation: 'symmetric-difference',
            left: {...arc.props.left, referenceInside: left},
            right: {...arc.props.right, referenceInside: right}
          }).referenceInside
        ).toBe(left !== right);
        expect(
          new GPUSphericalPolygonOverlay({
            ...polygon.props,
            operation: 'symmetric-difference',
            leftComplement: left,
            rightComplement: right
          }).complement
        ).toBe(left !== right);
      }
  } finally {
    device.destroy();
  }
});
