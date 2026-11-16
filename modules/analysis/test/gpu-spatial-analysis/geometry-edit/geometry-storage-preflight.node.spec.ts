// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  GPUCommandGraph,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {afterAll, expect, it, vi} from 'vitest';
import {GPUGeometryCleanup} from '../../../src/gpu-spatial-analysis/geometry-edit/gpu-geometry-cleanup';
import {GPUGeometryOrientation} from '../../../src/gpu-spatial-analysis/geometry-edit/gpu-geometry-orientation';
import {GPUGeometryMeasures} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geometry-measures';
import {GPUGeometryPredicates} from '../../../src/gpu-spatial-analysis/geometry-predicates/gpu-geometry-predicates';
import {GPUGeometryValidity} from '../../../src/gpu-spatial-analysis/geometry-validity/gpu-geometry-validity';
import {GPULineDensity} from '../../../src/gpu-spatial-analysis/line-density/gpu-line-density';
import {GPULineMerge} from '../../../src/gpu-spatial-analysis/line-merge/gpu-line-merge';
import {GPUGreatCircleArcs} from '../../../src/gpu-spatial-analysis/line-segmentize/gpu-great-circle-arcs';
import {GPULineChunk} from '../../../src/gpu-spatial-analysis/line-segmentize/gpu-line-chunk';
import {GPULineSegmentize} from '../../../src/gpu-spatial-analysis/line-segmentize/gpu-line-segmentize';
import {GPULineSimplification} from '../../../src/gpu-spatial-analysis/line-simplification/gpu-line-simplification';
import {GPUOffsetExpansion} from '../../../src/gpu-spatial-analysis/offset-expansion/gpu-offset-expansion';
import {GPUSegmentIntersection} from '../../../src/gpu-spatial-analysis/segment-intersection/gpu-segment-intersection';
import {GPUVertexSnap} from '../../../src/gpu-spatial-analysis/vertex-snap/gpu-vertex-snap';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

const device = createNullWebGPUDevice();
afterAll(() => device.destroy());
const kinds = [
  'cleanup',
  'orientation',
  'measures',
  'predicates',
  'validity',
  'density',
  'merge',
  'arcs',
  'chunk',
  'segmentize',
  'simplify',
  'expansion',
  'intersection',
  'snap'
] as const;
type Kind = (typeof kinds)[number];
function fixture(
  kind: Kind,
  options: {
    storage?: boolean;
    rows?: number;
    capacity?: number;
    leafCapacity?: number;
    alignedPrefix?: boolean;
    packed?: boolean;
  } = {}
) {
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const column = (length: number) =>
    createTransientView(graph, `column-${serial++}`, 'uint32', length);
  const rows = options.rows ?? 2;
  const capacity = options.capacity ?? 4;
  const positions = graph.createDataView(
    graph.createTransientBuffer({
      id: 'input',
      byteLength: rows * 8 + Number(Boolean(options.alignedPrefix)) * 4,
      usage: options.storage === false ? Buffer.COPY_DST : Buffer.STORAGE
    }),
    {format: 'float32x2', length: rows, byteOffset: options.alignedPrefix ? 4 : 0}
  );
  const offsets = column(2);
  const parameters = createTransientView(graph, 'parameters', 'float32', 4);
  const outputPositions = createTransientView(graph, 'output', 'float32x2', rows);
  const scalarStatus = () => ({count: column(1), overflow: column(1)});
  const lineOutput = () => ({
    positions: outputPositions,
    lineOffsets: column(2),
    status: scalarStatus()
  });
  const lines = {kind: 'lines' as const, positions, lineOffsets: offsets};
  let contributor: GPUCommandNodeProducer;
  switch (kind) {
    case 'cleanup':
      contributor = new GPUGeometryCleanup({
        geometryType: 'lines',
        positions,
        lineOffsets: offsets,
        parameters,
        output: lineOutput()
      });
      break;
    case 'orientation':
      contributor = new GPUGeometryOrientation({
        mode: 'reverse',
        positions,
        lineOffsets: offsets,
        output: {positions: outputPositions}
      });
      break;
    case 'measures':
      contributor = new GPUGeometryMeasures({
        geometryType: 'points',
        positions,
        output: {centroids: outputPositions}
      });
      break;
    case 'predicates':
      contributor = new GPUGeometryPredicates({
        geometry: lines,
        intersectionCapacity: capacity,
        leafCapacity: options.leafCapacity,
        output: {isSimple: column(1), status: {overflow: column(1)}}
      });
      break;
    case 'validity':
      contributor = new GPUGeometryValidity({
        polygons: {
          kind: 'polygons',
          positions,
          ringOffsets: offsets,
          polygonOffsets: column(2),
          featureOffsets: column(2)
        },
        intersectionCapacity: capacity,
        leafCapacity: options.leafCapacity,
        output: {mask: column(1), status: {overflow: column(1)}}
      });
      break;
    case 'density':
      contributor = new GPULineDensity({
        positions,
        lineOffsets: offsets,
        columns: 1,
        rows: 1,
        recordCapacity: capacity,
        parameters,
        output: {
          lengths: createTransientView(graph, 'lengths', 'float32', 1),
          status: {overflow: column(1)}
        }
      });
      break;
    case 'merge':
      contributor = new GPULineMerge({
        positions,
        lineOffsets: offsets,
        output: {positions: outputPositions, chainOffsets: column(2), status: {count: column(1)}}
      });
      break;
    case 'arcs':
      contributor = new GPUGreatCircleArcs({
        sources: positions,
        targets: createTransientView(graph, 'targets', 'float32x2', rows),
        parameters,
        output: {...lineOutput(), lineOffsets: column(rows + 1)}
      });
      break;
    case 'chunk':
      contributor = new GPULineChunk({
        positions,
        lineOffsets: offsets,
        mode: 'chunk',
        parameters,
        output: {...lineOutput(), lineOffsets: column(capacity + 1)}
      });
      break;
    case 'segmentize':
      contributor = new GPULineSegmentize({
        positions,
        lineOffsets: offsets,
        parameters,
        output: lineOutput()
      });
      break;
    case 'simplify':
      contributor = new GPULineSimplification({
        positions,
        lineOffsets: offsets,
        maximumIterations: 1,
        importance: createTransientView(graph, 'importance', 'float32', rows)
      });
      break;
    case 'expansion': {
      const packed = createTransientView(graph, 'packed', 'uint32', 256);
      const part = (byteOffset: number, length: number) =>
        graph.createDataView(packed.buffer, {format: 'uint32', byteOffset, length});
      contributor = new GPUOffsetExpansion({
        offsets: graph.createDataView(positions.buffer, {format: 'uint32', length: 2}),
        output: options.packed
          ? {
              owners: part(0, 2),
              localIndex: part(256, 2),
              status: {count: part(512, 1), overflow: part(768, 1)}
            }
          : {owners: column(2), status: scalarStatus()}
      });
      break;
    }
    case 'intersection':
      contributor = new GPUSegmentIntersection({
        left: lines,
        leafCapacity: options.leafCapacity,
        output: {
          pairs: {leftIds: column(capacity), rightIds: column(capacity), status: scalarStatus()}
        }
      });
      break;
    case 'snap':
      contributor = new GPUVertexSnap({
        positions,
        referencePositions: createTransientView(graph, 'references', 'float32x2', rows),
        parameters,
        output: {positions: outputPositions}
      });
      break;
  }
  return {graph, contributor};
}

it.each(kinds)('%s rejects caller buffers without STORAGE before allocating scratch', kind => {
  const {graph, contributor} = fixture(kind, {storage: false});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/STORAGE/);
  expect(allocate).not.toHaveBeenCalled();
});
it.each(kinds)('%s accepts ordinary layouts and checks all generated identifiers', kind => {
  const {graph, contributor} = fixture(kind);
  const nodes = contributor.getCommandNodes(graph);
  for (const {source} of getGPUCommandNodeKernelSources(nodes))
    expect(findReservedIdentifiers(stripWGSLComments(source))).toEqual([]);
});
it.each([
  'predicates',
  'validity',
  'intersection'
] as const)('%s checks nested BVH scratch before declaration', kind => {
  const {graph, contributor} = fixture(kind, {leafCapacity: 16_777_216});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/bvhNodes.*bytes/);
  expect(allocate).not.toHaveBeenCalled();
});
it('checks private feature statistics before declaring the first node', () => {
  const {graph, contributor} = fixture('measures', {rows: 2_097_153});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/featureStatistics.*bytes/);
  expect(allocate).not.toHaveBeenCalled();
});
it('checks the line chunk record stride before declaring prefix scratch', () => {
  const {graph, contributor} = fixture('chunk', {capacity: 3_355_444});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/pieceRecords.*bytes/);
  expect(allocate).not.toHaveBeenCalled();
});
it('accepts disjoint writable packed ranges imported as one graph buffer', () => {
  const {graph, contributor} = fixture('expansion', {packed: true});
  expect(() => graph.add(contributor)).not.toThrow();
});
it('includes a caller binding alignment prefix in preflight', () => {
  const {graph, contributor} = fixture('orientation', {alignedPrefix: true});
  const originalLimits = device.limits;
  Object.defineProperty(device, 'limits', {
    configurable: true,
    value: {...originalLimits, maxStorageBufferBindingSize: 16}
  });
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  try {
    expect(() => contributor.getCommandNodes(graph)).toThrow(/20 bytes/);
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    Object.defineProperty(device, 'limits', {configurable: true, value: originalLimits});
  }
});

it('ignores optional views that are not bound by the selected predicate, validity or simplification route', () => {
  const graph = new GPUCommandGraph(device);
  const foreign = new GPUCommandGraph(device);
  const unused = foreign.createDataView(
    foreign.createTransientBuffer({id: 'unused', byteLength: 16, usage: Buffer.COPY_DST}),
    {format: 'float32', length: 4}
  );
  const positions = createTransientView(graph, 'positions', 'float32x2', 2);
  const offsets = createTransientView(graph, 'offsets', 'uint32', 2);
  const mask = createTransientView(graph, 'mask', 'uint32', 2);
  const importance = createTransientView(graph, 'importance', 'float32', 2);
  const ignoredStatus = foreign.createDataView(unused.buffer, {format: 'uint32', length: 1});
  expect(() =>
    new GPUGeometryPredicates({
      geometry: {kind: 'points', positions},
      other: {
        kind: 'points',
        positions: foreign.createDataView(unused.buffer, {format: 'float32x2', length: 2})
      },
      parameters: unused,
      output: {numCoordinates: mask, status: {overflow: ignoredStatus}}
    }).getCommandNodes(graph)
  ).not.toThrow();
  expect(() =>
    new GPUGeometryValidity({
      points: {kind: 'points', positions},
      output: {mask, status: {overflow: ignoredStatus}}
    }).getCommandNodes(graph)
  ).not.toThrow();
  expect(() =>
    new GPULineSimplification({
      positions,
      lineOffsets: offsets,
      timestamps: unused,
      parameters: unused,
      importance,
      maximumIterations: 1
    }).getCommandNodes(graph)
  ).not.toThrow();
});
