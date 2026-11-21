// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCommandGraph,
  GPUVisibilityWorkflow,
  createTransientView,
  type GraphDataView,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPULineClipByPolygon} from '../../../src/gpu-spatial-analysis/line-clip/gpu-line-clip-by-polygon';
import {GPUSharedPaths} from '../../../src/gpu-spatial-analysis/line-clip/gpu-shared-paths';
import {GPUSnapRoundedNoding} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-snap-rounded-noding';
import {GPUPolygonize} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-polygonize';
import {GPULineNoding} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-line-noding';
import {GPULineSplit} from '../../../src/gpu-spatial-analysis/line-split/gpu-line-split';
import {GPULineLoopRemoval} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-line-loop-removal';
import {GPUTrackSimilarity} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity';
import {GPUTrackSimilaritySearch} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity-search';
import {GPUBufferSurface} from '../../../src/gpu-spatial-analysis/polygon-buffer/gpu-buffer-surface';
import {GPUSegmentIntersection} from '../../../src/gpu-spatial-analysis/segment-intersection/gpu-segment-intersection';
import {addClusterAndOutlineRecipe} from '../../../src/gpu-spatial-analysis/recipes/cluster-and-outline-recipe';
import {addStraightLineCatchmentsRecipe} from '../../../src/gpu-spatial-analysis/recipes/straight-line-catchments-recipe';
import {GPUGeometryValidity} from '../../../src/gpu-spatial-analysis/geometry-validity/gpu-geometry-validity';
import {GPUPointInPolygonJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join';
import {GPUBufferSelection} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-buffer-selection';
import type {GPUSpatialJoinGeometry} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-join-types';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {CPUGPUCompaction} from '../segment-intersection/cpu-gpu-core-emulations';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {adaptUniforms} from './multipart-point-joins-executor';
import {
  createMultipartClipCase,
  createMultipartSharedPathsCase,
  createMultipartNodingCase,
  createMultipartPolygonizeCase,
  createMultipartSplitCase,
  createMultipartLoopCase,
  createMultipartTrackCase,
  createMultipartTrackSearchCase,
  createMultipartBufferCase,
  getMultipartNestedView,
  MULTIPART_NODED_EXPECTED,
  MULTIPART_TRACK_INPUT,
  type MultipartTestAllocate
} from '../geometry-topology/multipart-line-forms-fixtures';
import {computeTrackSimilarityOnCPU} from '../../../src/gpu-spatial-analysis/track-similarity/cpu-track-similarity-operation';
import {createMultipartSegmentCase} from './multipart-consumer-contracts-fixture';

useCPUExecutorTestHooks();

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const allocate: MultipartTestAllocate = (name, format, length, values) => {
    const view = createTransientView(graph, name, format, length);
    const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
    memory.write(
      view,
      format.startsWith('float')
        ? Float32Array.from(values ?? Array(length * width).fill(NaN))
        : Uint32Array.from(values ?? Array(length * width).fill(99))
    );
    return view;
  };
  return {device, graph, memory, allocate};
}

for (const layout of ['single', 'grouped union', 'known multipart'] as const)
  it(`point consumers preserve logical rows for ${layout}`, () => {
    const {device, graph, memory, allocate} = createFixture();
    const positions = allocate('union-positions', 'float32x2', 2, [1, 1, 2, 2]);
    const featureOffsets = allocate('union-features', 'uint32', 2, [0, 2]);
    const readGeometry = (grouped: boolean): GPUSpatialJoinGeometry =>
      grouped ? {kind: 'points', positions, featureOffsets} : {kind: 'points', positions};
    const points =
      layout === 'known multipart'
        ? {kind: 'points' as const, positions, featureOffsets}
        : readGeometry(layout !== 'single');
    if (points.kind !== 'points') throw new Error('expected point geometry');
    const polygons = {
      kind: 'polygons' as const,
      positions: allocate('polygon-positions', 'float32x2', 4, [0, 0, 3, 0, 3, 3, 0, 3]),
      featureOffsets: allocate('polygon-features', 'uint32', 2, [0, 1]),
      polygonOffsets: allocate('polygons', 'uint32', 2, [0, 1]),
      ringOffsets: allocate('rings', 'uint32', 2, [0, 4])
    };
    const count = layout === 'single' ? 2 : 1;
    const assignment = allocate('assignment', 'uint32', count);
    const featureCounts = allocate('feature-counts', 'uint32', 1);
    const overflow = allocate('overflow', 'uint32', 1);
    const pipOutput = {
      pointFeatureIds: assignment,
      ...(points.featureOffsets
        ? {polygonMembershipCounts: featureCounts}
        : {polygonAssignmentCounts: featureCounts}),
      status: {overflow}
    };
    const radius = allocate('radius', 'float32', 1, [0]);
    const selectionOutput = {outputMask: allocate('mask', 'uint32', count)};
    // Visibility expands the upstream scan internally, bypassing the primitive prototype
    // hook. Emulate only that trusted compaction; execute the selection/join kernels unchanged.
    const visibility = vi
      .spyOn(GPUVisibilityWorkflow.prototype, 'getCommandNodes')
      .mockImplementation(function <Parameters>(
        this: GPUVisibilityWorkflow,
        graph: GPUCommandGraph<Parameters>
      ) {
        const mask = this.predicates[0].mask,
          output = this.output;
        const source =
          this.sourceIds ??
          createTransientView(graph, `${this.id}-identity`, 'uint32', mask.length);
        if (!('buffer' in source) || !('buffer' in mask) || !('buffer' in output))
          throw new Error('point consumer fixture requires packed visibility views');
        const identity = this.sourceIds
          ? []
          : [
              createWGSLKernelNode(graph, {
                id: `${this.id}-identity-fill`,
                operation: 'VisibilityIdentity',
                invocationCount: source.length,
                bindings: [{name: 'ids', view: source, type: 'u32', access: 'read_write'}],
                body: `ids[idsOffset + index] = ${this.firstSourceIndex}u + index;`
              })
            ];
        return [
          ...identity,
          ...new CPUGPUCompaction({
            id: this.id,
            input: source,
            flags: mask,
            output,
            count: this.count
          }).getCommandNodes()
        ] as readonly GPUCommandNode<Parameters>[];
      });
    try {
      const join = new GPUPointInPolygonJoin({
        points,
        polygons,
        candidateCapacity: 4,
        spatialSort: false,
        querySort: false,
        edgeSlabs: false,
        output: pipOutput
      });
      const selection = new GPUBufferSelection({
        points,
        features: {kind: 'points', positions},
        candidateCapacity: 4,
        radius: radius,
        spatialSort: false,
        querySort: false,
        output: selectionOutput
      });
      executeCommandNodesOnCPU(
        adaptUniforms([...join.getCommandNodes(graph), ...selection.getCommandNodes(graph)]),
        memory
      );
      // Hand oracle: both points lie strictly inside the square; grouping makes one feature.
      expect(Array.from(memory.readUint32(assignment))).toEqual(layout === 'single' ? [0, 0] : [0]);
      expect(Array.from(memory.readUint32(featureCounts))).toEqual([count]);
      expect(Array.from(memory.readUint32(overflow))).toEqual([0]);
      expect(Array.from(memory.readUint32(selectionOutput.outputMask))).toEqual(
        layout === 'single' ? [1, 1] : [1]
      );
    } finally {
      visibility.mockRestore();
      device.destroy();
    }
  });

it('X3 grouped validity snapshots inherited positions and preserves feature rows', () => {
  const {device, graph, memory, allocate} = createFixture();
  let positions = allocate('positions', 'float32x2', 2, [1, 1, 2, 2]);
  const original = positions;
  let positionReads = 0;
  const points = Object.assign(
    Object.create({
      get positions() {
        positionReads++;
        return positions;
      }
    }),
    {kind: 'points' as const, featureOffsets: allocate('features', 'uint32', 2, [0, 2])}
  );
  const output = {
    issueCode: allocate('issues', 'uint32', 1),
    validity: allocate('validity', 'uint32', 1)
  };
  try {
    const contributor = new GPUGeometryValidity({points, output});
    positions = allocate('replacement', 'float32x2', 1, [NaN, NaN]);
    expect(positionReads).toBe(1);
    expect(contributor.props.points!.positions).toBe(original);
    executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
    // Two finite point parts are one valid feature, independent of a later caller mutation.
    expect(Array.from(memory.readUint32(output.issueCode))).toEqual([0]);
    expect(Array.from(memory.readUint32(output.validity))).toEqual([1]);
  } finally {
    device.destroy();
  }
});

it('X3 the single-part snapshot captures inherited position getters once', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let positions = createTransientView(graph, 'positions', 'float32x2', 1);
  const original = positions;
  let positionReads = 0;
  const point = Object.assign(
    Object.create({
      get positions() {
        positionReads++;
        return positions;
      }
    }),
    {kind: 'points'}
  );
  try {
    const contributor = new GPUGeometryValidity({
      points: point,
      output: {issueCode: createTransientView(graph, 'issues', 'uint32', 1)}
    });
    positions = createTransientView(graph, 'replacement', 'float32x2', 2);
    expect(contributor.props.points!.positions).toBe(original);
    expect(positionReads).toBe(1);
  } finally {
    device.destroy();
  }
});

function createNodingCase(
  graph: GPUCommandGraph,
  allocate: MultipartTestAllocate,
  snapped = false
) {
  const fixture = createMultipartNodingCase(graph, allocate, true, snapped);
  return {
    ...fixture,
    expected: {
      ...MULTIPART_NODED_EXPECTED,
      'status.count': [5],
      'status.requiredCount': [5],
      'status.overflow': [0],
      'status.candidateOverflow': [0],
      'status.uncertainCount': [0],
      ...(snapped
        ? {lattice: [0, 0, 1, 0], latticeEndpoints: MULTIPART_NODED_EXPECTED.endpoints}
        : {})
    },
    output: {
      ...fixture.output,
      ...(snapped ? {lattice: fixture.lattice!, latticeEndpoints: fixture.latticeEndpoints!} : {})
    }
  };
}

type ContractFixture = {
  output: object;
  nodes: readonly GPUCommandNode[];
  expected?: Record<string, number[]>;
};
type ContractCase = {
  name: string;
  Constructor: {prototype: object};
  create: (graph: GPUCommandGraph, allocate: MultipartTestAllocate) => ContractFixture;
  offsets: Record<string, number[]>;
};

// Numerical literals come from multipart-line-forms.oracle.py (Shapely plus the documented
// provenance derivations). Construction assertions ensure no consumer drops the grouping.
const contractCases: ContractCase[] = [
  {
    name: 'line clip',
    Constructor: GPULineClipByPolygon,
    create: createMultipartClipCase,
    offsets: {lines: [0, 0, 2]}
  },
  {
    name: 'shared paths',
    Constructor: GPUSharedPaths,
    create: createMultipartSharedPathsCase,
    offsets: {left: [0, 0, 2], right: [0, 1, 1, 2]}
  },
  {
    name: 'snap rounded noding',
    Constructor: GPUSnapRoundedNoding,
    create: (graph, allocate) => createNodingCase(graph, allocate, true),
    offsets: {lines: [0, 0, 2, 3]}
  },
  {
    name: 'polygonize',
    Constructor: GPUPolygonize,
    create: createMultipartPolygonizeCase,
    offsets: {lines: [0, 0, 3, 6]}
  },
  {
    name: 'line noding',
    Constructor: GPULineNoding,
    create: createNodingCase,
    offsets: {lines: [0, 0, 2, 3]}
  },
  {
    name: 'line split',
    Constructor: GPULineSplit,
    create: createMultipartSplitCase,
    offsets: {lines: [0, 0, 2, 3]}
  },
  {
    name: 'line loop removal',
    Constructor: GPULineLoopRemoval,
    create: createMultipartLoopCase,
    offsets: {lines: [0, 0, 2, 3]}
  },
  {
    name: 'track similarity',
    Constructor: GPUTrackSimilarity,
    create: createMultipartTrackCase,
    offsets: {linesA: [0, 0, 2, 3], linesB: [0, 3]}
  },
  {
    name: 'track similarity search',
    Constructor: GPUTrackSimilaritySearch,
    create: createMultipartTrackSearchCase,
    offsets: {linesA: [0, 0, 2, 3], linesB: [0, 3]}
  },
  {
    name: 'buffer surface',
    Constructor: GPUBufferSurface,
    create: (graph, allocate) => {
      const fixture = createMultipartBufferCase(graph, allocate);
      return {output: fixture.output, nodes: fixture.contributor.getCommandNodes(graph)};
    },
    offsets: {geometry: [0, 0, 2]}
  },
  {
    name: 'segment intersection',
    Constructor: GPUSegmentIntersection,
    create: createMultipartSegmentCase,
    offsets: {left: [0, 0, 2], right: [0, 0, 2]}
  }
];

for (const {name, Constructor, create, offsets} of contractCases)
  it(`X3 ${name} retains grouped operands and feature-level results`, () => {
    const {device, graph, memory, allocate} = createFixture();
    type Producer = {
      props: Record<string, {featureOffsets?: GraphDataView<'uint32'>}>;
      getCommandNodes(graph: GPUCommandGraph): readonly GPUCommandNode[];
    };
    const prototype = Constructor.prototype as Producer;
    const getCommandNodes = prototype.getCommandNodes;
    let captured: Producer | undefined;
    const declaration = vi.spyOn(prototype, 'getCommandNodes').mockImplementation(function (
      this: Producer,
      targetGraph
    ) {
      captured ??= this;
      return getCommandNodes.call(this, targetGraph);
    });
    try {
      const fixture = create(graph, allocate);
      for (const [field, expected] of Object.entries(offsets))
        expect(
          Array.from(memory.readUint32(captured!.props[field].featureOffsets!)),
          field
        ).toEqual(expected);
      // Ring assembly and dissolve require cooperative barriers. Existing
      // multipart-line-forms.spec.ts executes all of their oracle outputs on WebGPU.
      if (name === 'polygonize' || name === 'buffer surface') return;
      if (name === 'track similarity') {
        executeCommandNodesOnCPU(
          fixture.nodes.filter(
            node => node.id.endsWith('-layout') || node.id === 'track-similarity-status'
          ),
          memory
        );
        expect(
          Array.from(memory.readUint32(getMultipartNestedView(fixture.output, 'pairStatus')))
        ).toEqual([0, 0, 1, 4, 4]);
        const result = computeTrackSimilarityOnCPU(MULTIPART_TRACK_INPUT, {
          densify: 0.5,
          measures: ['hausdorff', 'symmetricSegmentPathDistance', 'maximumDistance']
        });
        for (const [path, expected] of Object.entries(fixture.expected!))
          for (const [row, value] of expected.entries()) {
            const actual = result[path as keyof typeof result][row];
            if (Number.isNaN(value)) expect(actual, path).toBeNaN();
            else {
              expect(Number.isFinite(actual), path).toBe(true);
              expect(Math.abs(actual - value), path).toBeLessThanOrEqual(2e-5);
            }
          }
        return;
      }
      const searchStart =
        name === 'track similarity search'
          ? fixture.nodes.findIndex(node => node.id.startsWith('track-similarity-search-metric-'))
          : -1;
      executeCommandNodesOnCPU(
        adaptUniforms(searchStart >= 0 ? fixture.nodes.slice(0, searchStart) : fixture.nodes),
        memory
      );
      for (const [path, expected] of Object.entries(fixture.expected!)) {
        // Exact metric evaluation uses cooperative workgroups. Check the original pruning
        // kernels here; the shared headless fixture checks exact values and top-k as well.
        if (
          name === 'track similarity search' &&
          !['lowerBound', 'status', 'survivorSummary'].includes(path)
        )
          continue;
        const view = getMultipartNestedView(fixture.output, path);
        const actual = view.format.startsWith('float')
          ? memory.readFloat32(view)
          : memory.readUint32(view);
        if (name.startsWith('track')) {
          // At most twelve f32 distances/sums on coordinates <=10: absolute error <=2e-5.
          for (const [row, value] of expected.entries()) {
            if (Number.isNaN(value)) expect(actual[row], path).toBeNaN();
            else {
              expect(Number.isFinite(actual[row]), path).toBe(true);
              expect(Math.abs(actual[row] - value), path).toBeLessThanOrEqual(2e-5);
            }
          }
        } else expect(Array.from(actual), path).toEqual(expected);
      }
    } finally {
      declaration.mockRestore();
      device.destroy();
    }
  });

for (const field of ['linesA', 'linesB'] as const)
  it(`X3 sequence search refuses grouped ${field} and names explicit traversal alternatives`, () => {
    const {device, graph, allocate} = createFixture();
    let captured: GPUTrackSimilaritySearch | undefined;
    const declaration = vi
      .spyOn(GPUTrackSimilaritySearch.prototype, 'getCommandNodes')
      .mockImplementation(function (this: GPUTrackSimilaritySearch) {
        captured = this;
        return [];
      });
    try {
      createMultipartTrackSearchCase(graph, allocate);
      const props = captured!.props;
      const ungroup = (lines: typeof props.linesA) => ({...lines, featureOffsets: undefined});
      expect(
        () =>
          new GPUTrackSimilaritySearch({
            ...props,
            metric: 'dynamic-time-warping',
            linesA: field === 'linesA' ? props.linesA : ungroup(props.linesA),
            linesB: field === 'linesB' ? props.linesB : ungroup(props.linesB!)
          })
      ).toThrow(/featureOffsets.*ordered traversal.*select hausdorff.*explicit single-part tracks/);
    } finally {
      declaration.mockRestore();
      device.destroy();
    }
  });

for (const [name, create, field] of [
  ['cluster outline', addClusterAndOutlineRecipe, 'points'],
  ['straight line catchments', addStraightLineCatchmentsRecipe, 'facilities']
] as const)
  it(`X3 ${name} refuses multipart points before graph.add`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const points = {
      kind: 'points' as const,
      positions: createTransientView(graph, 'positions', 'float32x2', 2),
      featureOffsets: createTransientView(graph, 'features', 'uint32', 2)
    };
    try {
      expect(() => create(graph, {[field]: {...points}} as never)).toThrow(
        /featureOffsets.*multipart/
      );
    } finally {
      device.destroy();
    }
  });
