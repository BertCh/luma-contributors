// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  computeTrackSimilarityOnCPU,
  CPUTrackSimilarityOperation
} from '../../../src/gpu-spatial-analysis/track-similarity/cpu-track-similarity-operation';
import {GPUTrackSimilarityOperation} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity-operation';
import {GPUGeometryPredicates} from '../../../src/gpu-spatial-analysis/geometry-predicates/gpu-geometry-predicates';
import {GPUGeometryMeasures} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geometry-measures';
import {expect, it, vi} from 'vitest';
import {GPUPolygonOverlay} from '../../../src/gpu-spatial-analysis/polygon-overlay/gpu-polygon-overlay';
import {GPUPointInPolygonJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createNullWebGPUDevice, createOutputBuffer} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {
  createMultipartNodingCase,
  createMultipartSplitCase,
  createMultipartLoopCase,
  createMultipartValidityCase,
  createMultipartBufferCase,
  createMultipartSharedPathsCase,
  createMultipartClipCase,
  createMultipartTrackCase,
  MULTIPART_TRACK_INPUT,
  MULTIPART_TRACK_EXPECTED,
  createMultipartTrackOperationCase,
  createMultipartTrackSearchCase,
  createEmptyMultipartCase,
  createMultipartPolygonizeCase,
  getMultipartNestedView,
  MULTIPART_NODED_EXPECTED
} from './multipart-line-forms-fixtures';

useCPUExecutorTestHooks();

it('R10 buffer keeps logical source rows distinct from caller IDs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const spy = vi.spyOn(GPUPolygonOverlay.prototype, 'getCommandNodes').mockImplementation(function (
    this: GPUPolygonOverlay,
    overlayGraph
  ) {
    return [
      createWGSLKernelNode(overlayGraph, {
        id: 'independent-boundary-ownership',
        operation: 'BoundaryOwnershipOracle',
        invocationCount: 8,
        bindings: [
          {
            name: 'rows',
            view: this.props.output.boundary.sourceFeatureIds,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'count',
            view: this.props.output.boundary.status.count,
            type: 'u32',
            access: 'read_write'
          }
        ],
        body: 'rows[rowsOffset+index]=1u;if(index==0u){count[countOffset]=8u;}'
      })
    ];
  });
  try {
    const fixture = createMultipartBufferCase(graph, (name, format, length, values) => {
      const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
      const view = createTransientView(graph, name, format, length);
      memory.write(
        view,
        format.startsWith('float')
          ? Float32Array.from(values ?? new Array(length * width).fill(12345))
          : Uint32Array.from(values ?? new Array(length * width).fill(99))
      );
      return view;
    });
    executeCommandNodesOnCPU(fixture.contributor.getCommandNodes(graph), memory);
    expect(Array.from(memory.readUint32(fixture.output.boundary.sourceFeatureIds))).toEqual(
      new Array(8).fill(17)
    );
    expect(Array.from(memory.readUint32(fixture.output.boundary.sourceFeatureRows))).toEqual(
      new Array(8).fill(1)
    );
  } finally {
    spy.mockRestore();
    device.destroy();
  }
});

it.each([
  'predicates',
  'measures'
] as const)('empty %s inputs still require one full physical storage row', kind => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const buffer = createOutputBuffer(device, 1);
  try {
    const positions = importGraphBuffer(graph, 'undersized-empty-input', buffer, 'float32x2', 0);
    const featureOffsets = createTransientView(graph, 'empty-feature-offsets', 'uint32', 3);
    const contributor =
      kind === 'predicates'
        ? new GPUGeometryPredicates({
            geometry: {kind: 'points', positions, featureOffsets},
            output: {numCoordinates: createTransientView(graph, 'coordinate-counts', 'uint32', 2)}
          })
        : new GPUGeometryMeasures({
            geometryType: 'points',
            positions,
            featureOffsets,
            output: {lengths: createTransientView(graph, 'lengths', 'float32', 2)}
          });
    // Published graph getViewBindingRange: max(logical byte length, rowByteLength).
    // One float32x2 row is 8 bytes; this imported buffer deliberately holds only 4.
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /aligned binding must fit its buffer byteLength \(4\)/
    );
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it.each([
  ['points', false],
  ['lines', false],
  ['points', true],
  ['lines', true]
] as const)('entire empty multipart %s columns produce GEOS empty values predicatesOnly=%s', (kind, predicatesOnly) => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  try {
    const fixture = createEmptyMultipartCase(
      graph,
      (name, format, length, values) => {
        // Match imported GPU storage: upstream getViewBindingRange requires one complete
        // backing row even for zero logical rows (8 bytes for float32x2).
        const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
        const view =
          length === 0
            ? importGraphBuffer(graph, name, createOutputBuffer(device, width), format, length)
            : createTransientView(graph, name, format, length);
        memory.write(
          view,
          format.startsWith('float')
            ? Float32Array.from(values ?? new Array(length * width).fill(NaN))
            : Uint32Array.from(values ?? new Array(length * width).fill(99))
        );
        return view;
      },
      kind,
      predicatesOnly
    );
    // Group reduction uses cooperative workgroups; the identical headless fixture checks
    // every group port. Node executes all feature outputs and predicates without emulation.
    executeCommandNodesOnCPU(
      fixture.nodes.filter(node => !node.id.startsWith('geometry-measures-group-')),
      memory
    );
    for (const [path, expected] of Object.entries(fixture.expected)) {
      if (path.startsWith('groups.')) continue;
      const view = getMultipartNestedView(fixture.output, path);
      expect(
        Array.from(
          view.format.startsWith('float') ? memory.readFloat32(view) : memory.readUint32(view)
        ),
        path
      ).toEqual(expected);
    }
  } finally {
    device.destroy();
  }
});

it.each([
  'hausdorff',
  'symmetric-segment-path'
] as const)('multipart track search %s bounds and survivors use logical features', metric => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  try {
    const fixture = createMultipartTrackSearchCase(
      graph,
      (name, format, length, values) => {
        const view = createTransientView(graph, name, format, length);
        const width = format.endsWith('x2') ? 2 : 1;
        memory.write(
          view,
          format.startsWith('float')
            ? Float32Array.from(values ?? new Array(length * width).fill(NaN))
            : Uint32Array.from(values ?? new Array(length * width).fill(99))
        );
        return view;
      },
      metric
    );
    const metricStart = fixture.nodes.findIndex(node =>
      node.id.startsWith('track-similarity-search-metric-')
    );
    executeCommandNodesOnCPU(fixture.nodes.slice(0, metricStart), memory);
    for (const [row, expected] of fixture.expected['lowerBound'].entries()) {
      const actual = memory.readFloat32(fixture.output.lowerBound)[row];
      if (Number.isNaN(expected)) expect(actual).toBeNaN();
      else {
        expect(Number.isFinite(actual)).toBe(true);
        expect(actual).toBeCloseTo(expected, 5);
      }
    }
    expect(Array.from(memory.readUint32(fixture.output.status))).toEqual([0, 1, 2, 2]);
    expect(Array.from(memory.readUint32(fixture.output.survivorSummary))).toEqual([1, 1, 0]);
  } finally {
    device.destroy();
  }
});

it('multipart track metrics use disconnected unions and logical feature pairs', async () => {
  const result = computeTrackSimilarityOnCPU(MULTIPART_TRACK_INPUT, {
    densify: 0.5,
    measures: ['hausdorff', 'symmetricSegmentPathDistance', 'maximumDistance']
  });
  for (const [port, values] of Object.entries(MULTIPART_TRACK_EXPECTED))
    for (const [row, expected] of values.entries()) {
      const actual = result[port as keyof typeof result][row];
      if (Number.isNaN(expected)) expect(actual).toBeNaN();
      else {
        expect(Number.isFinite(actual)).toBe(true);
        expect(actual).toBeCloseTo(expected, 5);
      }
    }
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  try {
    const fixture = createMultipartTrackCase(graph, (name, format, length, values) => {
      const view = createTransientView(graph, name, format, length);
      memory.write(
        view,
        format.startsWith('float')
          ? Float32Array.from(values ?? new Array(length).fill(NaN))
          : Uint32Array.from(values ?? new Array(length).fill(99))
      );
      return view;
    });
    executeCommandNodesOnCPU(
      fixture.nodes.filter(
        node => node.id.endsWith('-layout') || node.id === 'track-similarity-status'
      ),
      memory
    );
    expect(Array.from(memory.readUint32(fixture.output.pairStatus))).toEqual([0, 0, 1, 4, 4]);
  } finally {
    device.destroy();
  }
  for (const Operation of [GPUTrackSimilarityOperation, CPUTrackSimilarityOperation])
    expect(
      () =>
        new Operation({
          linesA: {positions: 'positions', lineOffsets: 'parts', featureOffsets: 'features'},
          pairA: 'a',
          pairB: 'b',
          pairCount: 1,
          output: {frechet: 'frechet'}
        })
    ).toThrow(/featureOffsets.*ordered traversal.*single-part/);
  expect(() =>
    computeTrackSimilarityOnCPU(MULTIPART_TRACK_INPUT, {measures: ['dynamicTimeWarping']})
  ).toThrow(/featureOffsets.*ordered traversal.*single-part/);
  const operationCase = createMultipartTrackOperationCase();
  const operation = new CPUTrackSimilarityOperation(operationCase.props);
  operationCase.props.linesA.featureOffsets = 'mutated-after-construction';
  const operationResult = await operation.execute({
    inputs: Object.fromEntries(
      Object.entries(operationCase.inputs).map(([name, input]) => {
        const length = input.values.length / (input.format === 'float32x2' ? 2 : 1);
        return [
          name,
          {
            format: input.format,
            length,
            chunks: [{values: input.values, length}],
            pack: () => input.values
          }
        ];
      })
    )
  });
  for (const [port, values] of Object.entries(MULTIPART_TRACK_EXPECTED))
    for (const [row, expected] of values.entries()) {
      const actual = (operationResult[port] as Float32Array | Uint32Array)[row];
      if (Number.isNaN(expected)) expect(actual).toBeNaN();
      else {
        expect(Number.isFinite(actual)).toBe(true);
        expect(actual).toBeCloseTo(expected, 5);
      }
    }
});

it.each([
  ['shared paths', createMultipartSharedPathsCase],
  [
    'single-part shared paths with caller IDs',
    (graph: GPUCommandGraph, allocate: Parameters<typeof createMultipartSharedPathsCase>[1]) =>
      createMultipartSharedPathsCase(graph, allocate, false)
  ],
  ['inside clip', createMultipartClipCase],
  [
    'outside clip',
    (graph: GPUCommandGraph, allocate: Parameters<typeof createMultipartClipCase>[1]) =>
      createMultipartClipCase(graph, allocate, 'outside')
  ]
] as const)('multipart %s retains feature identities and every oracle port', (_name, createCase) => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  // The Node interpreter supports storage bindings only; the chunked join uses uniforms.
  // Substitute the independent closed-form rectangle classifier only in this Node case.
  // Headless executes the real containment graph, without this substitution.
  const containment = _name.includes('clip')
    ? vi.spyOn(GPUPointInPolygonJoin.prototype, 'getCommandNodes').mockImplementation(function (
        this: GPUPointInPolygonJoin,
        containmentGraph
      ) {
        const {points, output} = this.props;
        return [
          createWGSLKernelNode(containmentGraph, {
            id: 'rectangle-oracle',
            operation: 'rectangle-oracle',
            invocationCount: points.positions.length,
            bindings: [
              {
                name: 'points',
                view: points.positions as GraphDataView<'float32x2'>,
                type: 'f32',
                access: 'read'
              },
              {
                name: 'features',
                view: output.pointFeatureIds! as GraphDataView<'uint32'>,
                type: 'u32',
                access: 'read_write'
              },
              {name: 'overflow', view: output.status!.overflow!, type: 'u32', access: 'read_write'},
              {
                name: 'uncertain',
                view: output.status!.uncertainCount!,
                type: 'u32',
                access: 'read_write'
              }
            ],
            body: 'let x=points[pointsOffset+index*2u];let y=points[pointsOffset+index*2u+1u];features[featuresOffset+index]=select(0xffffffffu,0u,x>=1.0 && x<=6.0 && y>=-1.0 && y<=1.0);if(index==0u){overflow[overflowOffset]=0u;uncertain[uncertainOffset]=0u;}'
          })
        ];
      })
    : undefined;
  try {
    const fixture = createCase(graph, (name, format, length, values) => {
      const view = createTransientView(graph, name, format, length);
      const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
      memory.write(
        view,
        format.startsWith('float')
          ? Float32Array.from(values ?? new Array(length * width).fill(Number.NaN))
          : Uint32Array.from(values ?? new Array(length * width).fill(99))
      );
      return view;
    });
    executeCommandNodesOnCPU(fixture.nodes, memory);
    for (const [path, expected] of Object.entries(fixture.expected)) {
      const view = getMultipartNestedView(fixture.output, path);
      expect(
        Array.from(
          view.format.startsWith('float') ? memory.readFloat32(view) : memory.readUint32(view)
        ),
        path
      ).toEqual(expected);
    }
  } finally {
    containment?.mockRestore();
    device.destroy();
  }
});

it('multipart buffer preparation keeps all part strokes in their logical feature', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  let captured: GPUPolygonOverlay['props']['left'] | undefined;
  const spy = vi.spyOn(GPUPolygonOverlay.prototype, 'getCommandNodes').mockImplementation(function (
    this: GPUPolygonOverlay
  ) {
    captured = this.props.left;
    return [];
  });
  try {
    const fixture = createMultipartBufferCase(graph, (name, format, length, values) => {
      const view = createTransientView(graph, name, format, length);
      const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
      memory.write(
        view,
        format.startsWith('float')
          ? Float32Array.from(values ?? new Array(length * width).fill(Number.NaN))
          : Uint32Array.from(values ?? new Array(length * width).fill(99))
      );
      return view;
    });
    // The mocked dissolve has no writer for boundary ownership/count; its dependent ID
    // publication is tested separately by R10 with an independent ownership stub.
    executeCommandNodesOnCPU(
      fixture.contributor
        .getCommandNodes(graph)
        .filter(node => !node.id.endsWith('-boundary-identities')),
      memory
    );
    expect(captured).toBeDefined();
    expect(Array.from(memory.readUint32(captured!.featureOffsets))).toEqual([0, 0, 2]);
    expect(Array.from(memory.readUint32(captured!.polygonOffsets))).toEqual([0, 1, 2]);
    expect(Array.from(memory.readUint32(captured!.ringOffsets))).toEqual([0, 12, 24]);
    expect(Array.from(memory.readFloat32(captured!.positions))).toEqual(
      [
        [0, 0.5],
        [2, 0.5],
        [2, -0.5],
        [0, -0.5],
        [5, 0.5],
        [7, 0.5],
        [7, -0.5],
        [5, -0.5]
      ].flatMap(([x, y]) => [x, y, x, y, x, y])
    );
  } finally {
    spy.mockRestore();
    device.destroy();
  }
});

it.each([
  'points',
  'lines'
] as const)('multipart %s validity matches Shapely including empty and invalid parts', kind => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  try {
    const fixture = createMultipartValidityCase(
      graph,
      (name, format, length, values) => {
        const view = createTransientView(graph, name, format, length);
        const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
        memory.write(
          view,
          format.startsWith('float')
            ? Float32Array.from(values ?? new Array(length * width).fill(Number.NaN))
            : Uint32Array.from(values ?? new Array(length * width).fill(99))
        );
        return view;
      },
      kind
    );
    executeCommandNodesOnCPU(fixture.nodes, memory);
    for (const [path, expected] of Object.entries(fixture.expected))
      expect(
        Array.from(memory.readUint32(getMultipartNestedView(fixture.output, path))),
        path
      ).toEqual(expected);
  } finally {
    device.destroy();
  }
});

it('multipart loop erasure preserves the feature hierarchy and never bridges parts', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  try {
    const fixture = createMultipartLoopCase(graph, (name, format, length, values) => {
      const view = createTransientView(graph, name, format, length);
      const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
      memory.write(
        view,
        format.startsWith('float')
          ? Float32Array.from(values ?? new Array(length * width).fill(Number.NaN))
          : Uint32Array.from(values ?? new Array(length * width).fill(99))
      );
      return view;
    });
    executeCommandNodesOnCPU(fixture.nodes, memory);
    for (const [path, expected] of Object.entries(fixture.expected)) {
      const view = getMultipartNestedView(fixture.output, path);
      expect(
        Array.from(
          view.format.startsWith('float') ? memory.readFloat32(view) : memory.readUint32(view)
        ),
        path
      ).toEqual(expected);
    }
  } finally {
    device.destroy();
  }
});

it.each([
  5, 3
])('multipart line split preserves feature offsets and provenance with capacity=%s', capacity => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  try {
    const fixture = createMultipartSplitCase(
      graph,
      (name, format, length, values) => {
        const view = createTransientView(graph, name, format, length);
        const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
        memory.write(
          view,
          format.startsWith('float')
            ? Float32Array.from(values ?? new Array(length * width).fill(Number.NaN))
            : Uint32Array.from(values ?? new Array(length * width).fill(99))
        );
        return view;
      },
      capacity
    );
    executeCommandNodesOnCPU(fixture.nodes, memory);
    for (const [path, expected] of Object.entries(fixture.expected)) {
      const view = getMultipartNestedView(fixture.output, path);
      expect(
        Array.from(
          view.format.startsWith('float') ? memory.readFloat32(view) : memory.readUint32(view)
        ),
        path
      ).toEqual(expected);
    }
    // Last-vertex edge rows are unspecified. Each outgoing segment belongs to its original edge.
    expect(
      Array.from(memory.readUint32(fixture.output.pieces.vertexSourceEdges)).filter(
        (_, row) => row % 2 === 0
      )
    ).toEqual([0, 0, 2, 2, 4].slice(0, capacity));
  } finally {
    device.destroy();
  }
});

it.each([
  false,
  true
])('multipart polygonize preserves provenance for feature groups=%s', grouped => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  try {
    const fixture = createMultipartPolygonizeCase(
      graph,
      (name, format, length, values) => {
        const view = createTransientView(graph, name, format, length);
        const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
        memory.write(
          view,
          format.startsWith('float')
            ? Float32Array.from(values ?? new Array(length * width).fill(Number.NaN))
            : Uint32Array.from(values ?? new Array(length * width).fill(99))
        );
        return view;
      },
      grouped
    );
    // The interpreter cannot execute workgroup barriers in ring assembly. Execute every noding
    // and directed-edge provenance kernel; the headless counterpart covers assembly/diagnostics.
    const end = fixture.nodes.findIndex(node => node.id.startsWith('polygonize-faces-'));
    executeCommandNodesOnCPU(
      [
        ...fixture.nodes.slice(0, end),
        ...fixture.nodes.filter(node => node.id.startsWith('polygonize-publish-uncertain-'))
      ],
      memory
    );
    for (const [path, expected] of Object.entries(fixture.expected)) {
      if (!path.startsWith('nodedSegments.') && !path.startsWith('directedEdges.')) continue;
      if (path === 'directedEdges.ringIds' || path === 'directedEdges.flags') continue;
      const view = getMultipartNestedView(fixture.output, path);
      expect(
        Array.from(
          view.format.startsWith('float') ? memory.readFloat32(view) : memory.readUint32(view)
        ),
        path
      ).toEqual(expected);
    }
  } finally {
    device.destroy();
  }
});

it.each([
  {sourceIds: true, snapped: false},
  {sourceIds: false, snapped: false},
  {sourceIds: true, snapped: true},
  {sourceIds: false, snapped: true}
])('multipart noding publishes provenance with identities=$sourceIds snap=$snapped', ({
  sourceIds,
  snapped
}) => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  try {
    const fixture = createMultipartNodingCase(
      graph,
      (name, format, length, values) => {
        const view = createTransientView(graph, name, format, length);
        const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
        memory.write(
          view,
          format.startsWith('float')
            ? Float32Array.from(values ?? new Array(length * width).fill(Number.NaN))
            : Uint32Array.from(values ?? new Array(length * width).fill(99))
        );
        return view;
      },
      sourceIds,
      snapped
    );
    executeCommandNodesOnCPU(fixture.nodes, memory);
    if (snapped) {
      expect(Array.from(memory.readFloat32(fixture.lattice!))).toEqual([0, 0, 1, 0]);
      expect(Array.from(memory.readFloat32(fixture.latticeEndpoints!))).toEqual(
        MULTIPART_NODED_EXPECTED.endpoints
      );
    }
    // multipart-line-forms.oracle.py: Shapely node; provenance, parameters and capacities by hand.
    for (const [name, expected] of Object.entries(MULTIPART_NODED_EXPECTED)) {
      const view = fixture.output[name as keyof typeof MULTIPART_NODED_EXPECTED];
      const actual = view.format.startsWith('float')
        ? memory.readFloat32(view)
        : memory.readUint32(view);
      expect(Array.from(actual), name).toEqual(
        name === 'sourceFeatureIds' && !sourceIds ? [1, 1, 1, 1, 2] : expected
      );
    }
    for (const [name, view] of Object.entries(fixture.output.status))
      expect(Array.from(memory.readUint32(view)), name).toEqual([
        name === 'count' || name === 'requiredCount' ? 5 : 0
      ]);
  } finally {
    device.destroy();
  }
});
