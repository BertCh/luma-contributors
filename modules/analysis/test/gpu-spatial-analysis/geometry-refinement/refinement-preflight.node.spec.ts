// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPULineLoopRemoval} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-line-loop-removal';
import {GPUOffsetCurveComponents} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-offset-curve-components';
import {GPUPolygonPrecisionReduction} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-precision-reduction';
import {createTopologyOutput} from './topology-output';
import {GPULinePrecisionReduction} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-precision-reduction';
import {GPURatioConcaveHull} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-ratio-concave-hull';
import {GPUSubdividePolygons} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-subdivide-polygons';
import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {afterAll, expect, it, vi} from 'vitest';
import {GPUCanonicalGeometryEquality} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-canonical-geometry-equality';
import {GPUCanonicalPolygonEquality} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-canonical-polygon-equality';
import {GPUNormalizeGeometry} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-normalize-geometry';
import {GPUNormalizePolygons} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-normalize-polygons';
import {GPUConvexStraightSkeleton} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-convex-straight-skeleton';
import {GPUVectorStraightSkeleton} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-vector-straight-skeleton';
import {GPUMaximumInscribedCircle} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-maximum-inscribed-circle';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';

const device = createNullWebGPUDevice();
afterAll(() => device.destroy());
type Kind =
  | 'polygon-precision'
  | 'loop-removal'
  | 'offset-components'
  | 'line-precision'
  | 'ratio-hull'
  | 'subdivide'
  | 'rectilinear'
  | 'tree-equality'
  | 'polygon-equality'
  | 'tree-normalize'
  | 'polygon-normalize'
  | 'convex-skeleton'
  | 'vector-skeleton'
  | 'circle';
function fixture(
  kind: Kind,
  options: {
    storage?: boolean;
    features?: number;
    tolerance?: number;
    packed?: boolean;
    rows?: number;
    intersections?: number;
    segmentCapacity?: number;
  } = {}
): {graph: GPUCommandGraph; contributor: GPUCommandNodeProducer} {
  const graph = new GPUCommandGraph(device);
  const features = options.features ?? 1;
  const positions =
    options.storage === false
      ? graph.createDataView(
          graph.createTransientBuffer({id: 'positions', byteLength: 32, usage: Buffer.COPY_DST}),
          {format: 'float32x2', length: 4}
        )
      : createTransientView(graph, 'positions', 'float32x2', options.rows ?? 4);
  const column = (id: string, length: number) => createTransientView(graph, id, 'uint32', length);
  const geometry = {
    positions,
    ringOffsets: column('rings', 2),
    polygonOffsets: column('polygons', features + 1),
    featureOffsets: column('features', features + 1)
  };
  const tolerance = options.tolerance ?? 0.01;
  const tree = {
    positions,
    nodes: createTransientView(graph, 'nodes', 'uint32x4', 1),
    children: column('children', 1)
  };
  let contributor: GPUCommandNodeProducer;
  if (kind === 'loop-removal')
    contributor = new GPULineLoopRemoval({
      lines: {kind: 'lines', positions, lineOffsets: geometry.ringOffsets},
      intersectionCapacity: options.intersections ?? 8,
      output: {
        positions: createTransientView(graph, 'output', 'float32x2', 4),
        lineOffsets: column('output-offsets', 2),
        sourceEdges: column('source-edges', 4),
        status: createTransientView(graph, 'status', 'uint32x4', 1)
      }
    });
  else if (kind === 'offset-components')
    contributor = new GPUOffsetCurveComponents({
      original: {positions, lineOffsets: geometry.ringOffsets},
      raw: {positions, lineOffsets: geometry.ringOffsets},
      distance: tolerance,
      tolerance,
      intersectionCapacity: options.intersections ?? 8,
      segmentCapacity: options.segmentCapacity ?? 32,
      output: {
        positions: createTransientView(graph, 'output', 'float32x2', 4),
        components: createTransientView(graph, 'components', 'uint32x4', 2),
        sourceEdges: column('source-edges', 4),
        status: createTransientView(graph, 'status', 'uint32x4', 2)
      }
    });
  else if (kind === 'polygon-precision') {
    let serial = 0;
    contributor = new GPUPolygonPrecisionReduction({
      polygons: {kind: 'polygons', ...geometry},
      gridSize: tolerance,
      lattice: createTransientView(graph, 'lattice', 'float32', 4),
      intersectionCapacity: 8,
      output: createTopologyOutput((format, length) =>
        createTransientView(graph, `topology-${serial++}`, format, length)
      )
    });
  } else if (kind === 'line-precision')
    contributor = new GPULinePrecisionReduction({
      positions,
      lineOffsets: geometry.ringOffsets,
      gridSize: tolerance,
      mode: 'valid-output',
      output: {
        positions: createTransientView(graph, 'output', 'float32x2', 4),
        lineOffsets: column('output-offsets', 2),
        status: column('status', 1)
      }
    });
  else if (kind === 'ratio-hull')
    contributor = new GPURatioConcaveHull({
      positions,
      ratio: 0.5,
      output: {
        triangles: createTransientView(graph, 'triangles', 'uint32x3', 3),
        triangleValidity: column('triangle-validity', 3),
        boundaryEdges: createTransientView(graph, 'edges', 'uint32x2', 9),
        edgeValidity: column('edge-validity', 9),
        status: {converged: column('converged', 1)}
      }
    });
  else if (kind === 'subdivide' || kind === 'rectilinear')
    contributor = new GPUSubdividePolygons({
      ...geometry,
      method: kind === 'subdivide' ? 'convex' : 'rectilinear',
      maximumVertices: 4,
      output: {
        positions: createTransientView(graph, 'output', 'float32x2', 16),
        pieces: createTransientView(graph, 'pieces', 'uint32x4', 4),
        status: createTransientView(graph, 'status', 'uint32x4', 1)
      }
    });
  else if (kind === 'tree-equality')
    contributor = new GPUCanonicalGeometryEquality({
      left: tree,
      right: tree,
      tolerance,
      pairs: createTransientView(graph, 'pairs', 'uint32x2', 1),
      output: column('equal', 1)
    });
  else if (kind === 'polygon-equality')
    contributor = new GPUCanonicalPolygonEquality({
      left: geometry,
      right: geometry,
      tolerance,
      output: column('equal', features)
    });
  else if (kind === 'tree-normalize') {
    const packed = createTransientView(graph, 'packed', 'uint32', 65);
    contributor = new GPUNormalizeGeometry({
      ...tree,
      output: {
        positions: options.packed
          ? graph.createDataView(packed.buffer, {format: 'float32x2', length: 4, byteOffset: 4})
          : createTransientView(graph, 'output', 'float32x2', 4),
        children: options.packed
          ? graph.createDataView(packed.buffer, {format: 'uint32', length: 1})
          : column('output-children', 1),
        status: column('status', 1)
      }
    });
  } else if (kind === 'polygon-normalize')
    contributor = new GPUNormalizePolygons({
      ...geometry,
      output: {
        positions: createTransientView(graph, 'output', 'float32x2', 4),
        ringOffsets: column('output-rings', 2),
        polygonOffsets: column('output-polygons', features + 1)
      }
    });
  else if (kind === 'circle')
    contributor = new GPUMaximumInscribedCircle({
      ...geometry,
      tolerance,
      cellCapacity: 4096,
      output: {
        circles: createTransientView(graph, 'circles', 'float32x4', features),
        witnesses: createTransientView(graph, 'witnesses', 'float32x2', features),
        status: column('status', features)
      }
    });
  else if (kind === 'convex-skeleton')
    contributor = new GPUConvexStraightSkeleton({
      positions,
      ringOffsets: geometry.ringOffsets,
      offsetDistance: tolerance,
      maximumVertices: 4,
      segmentCapacity: 8,
      output: {
        segments: createTransientView(graph, 'segments', 'float32x4', 8),
        sourceEdges: createTransientView(graph, 'sources', 'uint32x2', 8),
        offsetPositions: createTransientView(graph, 'offset', 'float32x2', 4),
        status: createTransientView(graph, 'status', 'uint32x4', 1)
      }
    });
  else
    contributor = new GPUVectorStraightSkeleton({
      ...geometry,
      maximumVertices: 4,
      maximumEvents: 4096,
      segmentCapacity: 8,
      offsetDistance: tolerance,
      output: {
        segments: createTransientView(graph, 'segments', 'float32x4', features * 8),
        sourceEdges: createTransientView(graph, 'sources', 'uint32x2', features * 8),
        offsetPositions: createTransientView(graph, 'offset', 'float32x2', features * 8196),
        offsetLinks: createTransientView(graph, 'links', 'uint32x2', features * 8196),
        status: createTransientView(graph, 'status', 'uint32x4', features)
      }
    });
  return {graph, contributor};
}

it.each<Kind>([
  'polygon-precision',
  'loop-removal',
  'offset-components',
  'line-precision',
  'ratio-hull',
  'subdivide',
  'rectilinear',
  'tree-equality',
  'polygon-equality',
  'tree-normalize',
  'polygon-normalize',
  'convex-skeleton',
  'vector-skeleton',
  'circle'
])('%s rejects missing STORAGE before any resource is declared', kind => {
  const {graph, contributor} = fixture(kind, {storage: false});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/STORAGE/);
  expect(allocate).not.toHaveBeenCalled();
});
it.each([
  ['circle', 8193],
  ['vector-skeleton', 500]
] as const)('%s preflights row-proportional nested scratch', (kind, features) => {
  const {graph, contributor} = fixture(kind, {features});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/(?:cells|wavefront).*bytes/);
  expect(allocate).not.toHaveBeenCalled();
});
it.each<Kind>([
  'tree-equality',
  'polygon-equality',
  'convex-skeleton',
  'vector-skeleton',
  'circle'
])('%s rejects f32 literal overflow at construction', kind => {
  expect(() => fixture(kind, {tolerance: 1e100})).toThrow(
    /(?:tolerance|offsetDistance).*3\.402823/
  );
});
it('accepts valid packed positions and children in separate normalization passes', () => {
  const {graph, contributor} = fixture('tree-normalize', {packed: true});
  expect(() => contributor.getCommandNodes(graph)).not.toThrow();
});

it('keeps normalization round shaders constant as nesting depth grows', () => {
  const counts = [];
  for (const maximumNestingDepth of [1, 4, 64]) {
    const {graph, contributor} = fixture('tree-normalize');
    const nodes = new GPUNormalizeGeometry({
      ...(contributor as GPUNormalizeGeometry).props,
      maximumNestingDepth
    }).getCommandNodes(graph);
    const rounds = nodes.filter(node => /round-\d+-(?:nodes|rank)$/.test(node.id));
    const sources = getGPUCommandNodeKernelSources(rounds);
    counts.push(new Set(sources.map(({source}) => source)).size);
    expect(new Set(getGPUCommandNodePipelineKeys(rounds)).size).toBe(2);
    for (const {source} of sources)
      expect(findReservedIdentifiers(stripWGSLComments(source))).toEqual([]);
  }
  expect(counts).toEqual([2, 2, 2]);
});

it('preflights the loop erasure stack after validating its nested splitter tables', () => {
  const {graph, contributor} = fixture('loop-removal', {rows: 1_000_000, intersections: 1_000_000});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/stack.*bytes/);
  expect(allocate).not.toHaveBeenCalled();
});
it('preflights the offset fragment matrix before noding declares anything', () => {
  const {graph, contributor} = fixture('offset-components', {segmentCapacity: 1_000_000});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/fragmentsAndMetadata.*bytes/);
  expect(allocate).not.toHaveBeenCalled();
});
it.each<Kind>([
  'line-precision',
  'polygon-precision'
])('%s rejects a grid flushed to zero by f32 hardware', kind => {
  expect(() => fixture(kind, {tolerance: 1e-44})).toThrow(/gridSize/);
});

it.each<Kind>([
  'polygon-precision',
  'loop-removal',
  'offset-components',
  'line-precision',
  'ratio-hull',
  'subdivide',
  'rectilinear',
  'tree-equality',
  'polygon-equality',
  'tree-normalize',
  'polygon-normalize',
  'convex-skeleton',
  'vector-skeleton',
  'circle'
])('%s emits identifiers accepted by the complete WGSL reserved-word list', kind => {
  const {graph, contributor} = fixture(kind);
  for (const {source} of getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph))) {
    const code = stripWGSLComments(source);
    expect(findReservedIdentifiers(code)).toEqual([]);
    expect(code).not.toMatch(/\b(?:NaN|Infinity)\b/);
  }
});

it('bounds every scalar coordinate of the offset fragment matrix at construction', () => {
  expect(() => fixture('offset-components', {segmentCapacity: 42_949_673})).toThrow(
    /fragment metadata words.*4294967295/
  );
  expect(() => fixture('offset-components', {segmentCapacity: 42_949_672})).not.toThrow();
});

it.each([
  ['polygons', 'positions'],
  ['nodedSegments', 'endpoints'],
  ['directedEdges', 'flags'],
  ['diagnostics', 'dangleCount'],
  ['status', 'count']
] as const)('preflights nested polygon precision output %s.%s STORAGE usage', (port, field) => {
  const {graph, contributor} = fixture('polygon-precision');
  const props = (contributor as GPUPolygonPrecisionReduction).props;
  const views = props.output[port] as unknown as Record<string, GraphDataView>;
  const original = views[field];
  const nonstorage = graph.createDataView(
    graph.createTransientBuffer({
      id: 'nonstorage-output',
      byteLength: Math.max(4, original.length * original.rowByteLength),
      usage: Buffer.COPY_DST
    }),
    {format: original.format, length: original.length}
  );
  const changed = new GPUPolygonPrecisionReduction({
    ...props,
    output: {...props.output, [port]: {...props.output[port], [field]: nonstorage}}
  });
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => changed.getCommandNodes(graph)).toThrow(/STORAGE/);
  expect(allocate).not.toHaveBeenCalled();
});

it('rejects unsupported loop-removal coordinates at construction before nested split allocation', () => {
  const {contributor} = fixture('loop-removal');
  expect(
    () =>
      new GPULineLoopRemoval({
        ...(contributor as GPULineLoopRemoval).props,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
      })
  ).toThrow(/project positions to a planar frame first/);
});

it.each([
  16, 256
])('rejects physical read/write precision output aliasing at byte %s before declaration', byteOffset => {
  const graph = new GPUCommandGraph(device);
  const packed = createTransientView(graph, 'packed', 'uint32', 128);
  expect(
    () =>
      new GPULinePrecisionReduction({
        positions: createTransientView(graph, 'input', 'float32x2', 2),
        lineOffsets: createTransientView(graph, 'offsets', 'uint32', 2),
        gridSize: 1,
        mode: 'valid-output',
        output: {
          positions: graph.createDataView(packed.buffer, {format: 'float32x2', length: 2}),
          lineOffsets: graph.createDataView(packed.buffer, {
            format: 'uint32',
            length: 2,
            byteOffset
          }),
          status: createTransientView(graph, 'status', 'uint32', 1)
        }
      })
  ).toThrow(/overlap|separate physical buffers/);
});

it.each([
  'positions',
  'ringOffsets',
  'polygonOffsets',
  'featureOffsets'
] as const)('rejects substituted subdivision %s formats before declaration', name => {
  const {graph, contributor} = fixture('subdivide');
  const props = (contributor as GPUSubdividePolygons).props;
  const format = name === 'positions' ? 'uint32x2' : 'float32';
  const wrong = createTransientView(graph, 'wrong-format', format, props[name].length);
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => new GPUSubdividePolygons({...props, [name]: wrong} as never)).toThrow(
    new RegExp(`subdivide-polygons ${name}.*received ${format}`)
  );
  expect(allocate).not.toHaveBeenCalled();
});

it('rejects child work before scratch and permits a same-ID subdivision retry', () => {
  const {graph, contributor} = fixture('subdivide');
  const props = (contributor as GPUSubdividePolygons).props;
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => new GPUSubdividePolygons({...props, maximumWork: NaN})).toThrow(
    /subdivide-polygons maximumWork/
  );
  expect(allocate).not.toHaveBeenCalled();
  const correctedProps = {...props, maximumWork: 4000000};
  const corrected = new GPUSubdividePolygons(correctedProps);
  correctedProps.maximumWork = NaN;
  expect(corrected.props.maximumWork).toBe(4000000);
  const nodes = corrected.getCommandNodes(graph);
  expect(nodes.map(node => node.id)).toEqual(
    expect.arrayContaining([
      'subdivide-polygons-triangulation-earcut',
      'subdivide-polygons-merge-count',
      'subdivide-polygons-merge-write'
    ])
  );
  const identifiers = allocate.mock.calls.map(([descriptor]) => descriptor.id);
  expect(identifiers).toEqual(
    expect.arrayContaining([
      'subdivide-polygons-indices',
      'subdivide-polygons-valid',
      'subdivide-polygons-triangle-offsets',
      'subdivide-polygons-polygon-info',
      'subdivide-polygons-consumed',
      'subdivide-polygons-piece-vertex-counts',
      'subdivide-polygons-piece-flags',
      'subdivide-polygons-piece-offsets'
    ])
  );
  expect(new Set(identifiers).size).toBe(identifiers.length);
});

it('declares zero polygons with resident position capacity without leaking scratch', () => {
  const {graph, contributor} = fixture('subdivide');
  const original = (contributor as GPUSubdividePolygons).props;
  const props = {
    ...original,
    positions: createTransientView(graph, 'resident-position-capacity', 'float32x2', 2),
    ringOffsets: createTransientView(graph, 'empty-rings', 'uint32', 1),
    polygonOffsets: createTransientView(graph, 'empty-polygons', 'uint32', 1),
    featureOffsets: createTransientView(graph, 'empty-features', 'uint32', 1)
  };
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  // No polygon can own an index, irrespective of the unused resident position capacity.
  expect(() => new GPUSubdividePolygons(props).getCommandNodes(graph)).not.toThrow();
  expect(allocate).not.toHaveBeenCalled();
  const identifiers = allocate.mock.calls.map(([descriptor]) => descriptor.id);
  expect(new Set(identifiers).size).toBe(identifiers.length);
});

it('rejects an explicit null subdivision method instead of defaulting it', () => {
  const {graph, contributor} = fixture('subdivide');
  const props = (contributor as GPUSubdividePolygons).props;
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => new GPUSubdividePolygons({...props, method: null as never})).toThrow(
    /subdivide-polygons method.*convex.*rectilinear.*received null/
  );
  expect(allocate).not.toHaveBeenCalled();
});
