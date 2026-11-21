// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, beforeEach, afterEach, vi} from 'vitest';
import {ScalarData, WgslExec} from 'wgsl_reflect';
import {readFileSync} from 'node:fs';
it('R11 catalog accepts multipart operands and publishes uint32x4 component metadata', () => {
  const source = readFileSync(
    new URL('../../../../../website/src/catalog/entries/geometry-processing.ts', import.meta.url),
    'utf8'
  ).replaceAll("\\'", "'");
  const predicates = source.slice(
    source.indexOf("id: 'geometry-predicates'"),
    source.indexOf("id: 'geometry-predicates'") + 3000
  );
  expect(predicates).not.toContain('multipart input is rejected');
  const components = source.slice(
    source.indexOf("id: 'offset-curve-components'"),
    source.indexOf("id: 'offset-curve-components'") + 2500
  );
  expect(components).toMatch(/makeOutput\(\s*'components',\s*'GraphDataView<"uint32x4">'/);
  expect(components).toContain("GPUOffsetCurveComponentsProps['original']");
  expect(components).toContain("GPUOffsetCurveComponentsProps['raw']");
  expect(components).toContain('sourcePartRow');
  expect(components).toContain('sourceFeatureRow');
});
beforeEach(() => {
  const original = WgslExec.prototype._callBuiltinFunction;
  vi.spyOn(WgslExec.prototype, '_callBuiltinFunction').mockImplementation(function (
    this: WgslExec,
    node,
    context
  ) {
    if (node.name === 'ldexp') {
      const value = this.evalExpression(node.args![0], context) as ScalarData;
      const exponent = this.evalExpression(node.args![1], context) as ScalarData;
      return new ScalarData(
        Math.fround(value.value * 2 ** exponent.value),
        this.getTypeInfo('f32')!
      );
    }
    return original.call(this, node, context);
  });
});
afterEach(() => vi.restoreAllMocks());
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {findConstantNonFiniteBitcasts} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {getMultipartNestedView} from '../geometry-topology/multipart-line-forms-fixtures';
import {createAlignedRelateReviewCase} from './multipart-review-fixtures';
import {createEmptyPointRelateReviewCase} from './multipart-review-fixtures';
it.each([
  false,
  true
])('review2 R1 empty multipoints differ from invalid coordinates invalid=%s', invalid => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  try {
    const fixture = createEmptyPointRelateReviewCase(
      graph,
      (name, format, length, values) => {
        const width = format.endsWith('x2') ? 2 : 1;
        const view = createTransientView(graph, name, format, length);
        memory.write(
          view,
          format.startsWith('float')
            ? Float32Array.from(values ?? new Array(length * width).fill(12345))
            : Uint32Array.from(values ?? new Array(length * width).fill(99))
        );
        return view;
      },
      invalid
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
import {createPaddedSplitReviewCase} from './multipart-review-fixtures';
import {createEmptyTrackReviewCase} from './multipart-review-fixtures';
import {createMeasureReviewCase} from './multipart-review-fixtures';
import {createCollapsedClearanceReviewCase} from './multipart-review-fixtures';
import {createZeroAreaReviewCase} from './multipart-review-fixtures';
import {createIdentityLoopReviewCase} from './multipart-review-fixtures';
import {createExtremeBoundsReviewCase} from './multipart-review-fixtures';
it('v3-1 feature, cooperative and group bounds kernels contain no non-finite constant expressions', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const cooperative of [false, true]) {
      const graph = new GPUCommandGraph(device);
      const fixture = createExtremeBoundsReviewCase(
        graph,
        (name, format, length) => createTransientView(graph, name, format, length),
        'lines',
        cooperative
      );
      const sources = getGPUCommandNodeKernelSources(
        fixture.nodes.filter(
          node =>
            node.id.endsWith('-features') ||
            node.id.includes('cooperative') ||
            node.id.endsWith('-group-reduce')
        )
      );
      expect(sources.length).toBe(cooperative ? 3 : 2);
      for (const kernel of sources)
        expect(findConstantNonFiniteBitcasts(kernel.source), kernel.id).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});
it.each([
  'points',
  'lines'
] as const)('review2 R4 finite float32 extremes are preserved for %s', kind => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  try {
    const fixture = createExtremeBoundsReviewCase(
      graph,
      (name, format, length, values) => {
        const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
        const view = createTransientView(graph, name, format, length);
        memory.write(
          view,
          format.startsWith('float')
            ? Float32Array.from(values ?? new Array(length * width).fill(12345))
            : Uint32Array.from(values ?? new Array(length * width).fill(99))
        );
        return view;
      },
      kind
    );
    executeCommandNodesOnCPU(
      fixture.nodes.filter(node => !node.id.startsWith('geometry-measures-group-')),
      memory
    );
    for (const path of ['output.bounds', 'output.extremeVertices'] as const) {
      const view = getMultipartNestedView(fixture.output, path);
      expect(
        Array.from(
          view.format.startsWith('float') ? memory.readFloat32(view) : memory.readUint32(view)
        ),
        path
      ).toEqual(fixture.expected[path]);
    }
  } finally {
    device.destroy();
  }
});
import {GPULineSplit} from '../../../src/gpu-spatial-analysis/line-split/gpu-line-split';
import {GPULineClipByPolygon} from '../../../src/gpu-spatial-analysis/line-clip/gpu-line-clip-by-polygon';
function createReviewConstructionProps(graph: GPUCommandGraph) {
  const view = (name: string, length = 1) => createTransientView(graph, name, 'uint32', length);
  return {
    lines: {
      kind: 'lines' as const,
      positions: createTransientView(graph, 'construction-lines', 'float32x2', 2),
      lineOffsets: view('construction-line-offsets', 2)
    },
    polygons: {
      kind: 'polygons' as const,
      positions: createTransientView(graph, 'construction-polygons', 'float32x2', 4),
      featureOffsets: view('construction-features', 2),
      polygonOffsets: view('construction-polygon-offsets', 2),
      ringOffsets: view('construction-rings', 2)
    },
    intersectionCapacity: 4,
    candidateCapacity: 8,
    output: {
      pieces: {
        geometry: {
          kind: 'lines' as const,
          positions: createTransientView(graph, 'construction-output', 'float32x2', 2),
          lineOffsets: view('construction-output-offsets', 2)
        },
        sourceIds: view('construction-ids'),
        status: {count: view('construction-count'), overflow: view('construction-overflow')}
      },
      status: {uncertainCount: view('construction-uncertainty')}
    }
  };
}
it.each([
  {name: 'split', Contributor: GPULineSplit},
  {name: 'clip', Contributor: GPULineClipByPolygon}
])('review2 R14 $name snapshots alternating root and nested descriptors once', ({Contributor}) => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const base = createReviewConstructionProps(graph);
    const reads = {output: 0, pieces: 0, geometry: 0, pieceStatus: 0, status: 0, sourceIds: 0};
    const geometry = Object.defineProperty({...base.output.pieces.geometry}, 'sourceIds', {
      get() {
        reads.sourceIds++;
        return reads.sourceIds === 1 ? undefined : base.output.pieces.sourceIds;
      }
    });
    const pieces = Object.defineProperties(
      {...base.output.pieces},
      {
        geometry: {
          get() {
            reads.geometry++;
            return reads.geometry === 1
              ? geometry
              : {...base.output.pieces.geometry, sourceIds: base.output.pieces.sourceIds};
          }
        },
        status: {
          get() {
            reads.pieceStatus++;
            return base.output.pieces.status;
          }
        }
      }
    );
    const output = Object.defineProperties(
      {},
      {
        pieces: {
          get() {
            reads.pieces++;
            return pieces;
          }
        },
        status: {
          get() {
            reads.status++;
            return base.output.status;
          }
        }
      }
    );
    const props = Object.defineProperty({...base}, 'output', {
      get() {
        reads.output++;
        return reads.output === 1
          ? output
          : {
              ...base.output,
              pieces: {
                ...base.output.pieces,
                geometry: {...base.output.pieces.geometry, sourceIds: base.output.pieces.sourceIds}
              }
            };
      }
    });
    const contributor = new Contributor(props);
    expect(reads).toEqual({
      output: 1,
      pieces: 1,
      geometry: 1,
      pieceStatus: 1,
      status: 1,
      sourceIds: 1
    });
    expect(contributor.props.output.pieces.geometry.sourceIds).toBeUndefined();
  } finally {
    device.destroy();
  }
});
it.each([
  {name: 'split', Contributor: GPULineSplit},
  {name: 'clip', Contributor: GPULineClipByPolygon}
])('review2 R14 $name validates captured refused geometry and diagnostic getters', ({
  Contributor
}) => {
  for (const field of ['sourceIds', 'invalidCount']) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    try {
      const base = createReviewConstructionProps(graph);
      let reads = 0;
      const target =
        field === 'sourceIds' ? base.output.pieces.geometry : base.output.pieces.status;
      Object.defineProperty(target, field, {
        get() {
          reads++;
          return reads === 1 ? base.output.pieces.sourceIds : undefined;
        }
      });
      expect(() => new Contributor(base)).toThrow(field);
      expect(reads).toBe(1);
    } finally {
      device.destroy();
    }
  }
});
it('review2 R14 predicates snapshot secondary context and other operand once', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const geometry = {
      kind: 'points' as const,
      positions: createTransientView(graph, 'context-points', 'float32x2', 1)
    };
    let contextReads = 0;
    let otherReads = 0;
    const props = {
      geometry,
      output: {topological: {equals: createTransientView(graph, 'context-equality', 'uint32', 1)}},
      get other() {
        otherReads++;
        return otherReads === 1
          ? geometry
          : {
              ...geometry,
              positions: createTransientView(graph, `other-${otherReads}`, 'float32x2', 2)
            };
      },
      get otherSpatialContext() {
        contextReads++;
        return contextReads === 1
          ? {coordinateSpace: 'planar' as const, metric: 'native' as const}
          : {coordinateSpace: 'longitude-latitude' as const, metric: 'great-circle' as const};
      }
    };
    const contributor = new GPUGeometryPredicates(props);
    expect(contextReads).toBe(1);
    expect(otherReads).toBe(1);
    expect(contributor.props.other?.positions).toBe(geometry.positions);
  } finally {
    device.destroy();
  }
});
import {GPUGeometryPredicates} from '../../../src/gpu-spatial-analysis/geometry-predicates/gpu-geometry-predicates';
it('review2 R13 aligned writable predicate columns cannot alias', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const geometry = {
      kind: 'points' as const,
      positions: createTransientView(graph, 'alias-points', 'float32x2', 1)
    };
    const shared = createTransientView(graph, 'alias-booleans', 'uint32', 1);
    expect(
      () =>
        new GPUGeometryPredicates({
          geometry,
          other: geometry,
          output: {topological: {equals: shared, touches: shared}}
        })
    ).toThrow(
      /output\.topological\.equals.*output\.topological\.touches.*logical byte ranges must not overlap/
    );
  } finally {
    device.destroy();
  }
});
it.each(
  ['split', 'clip'].flatMap(name =>
    ['diagnostic', 'vertex', 'featureRow'].map(port => ({name, port}))
  )
)('review2 R13 $name rows cannot alias $port output', ({name, port}) => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = (name: string, length = 1) => createTransientView(graph, name, 'uint32', length);
  try {
    const capacity = port === 'vertex' ? 2 : 1;
    const shared = view('aliased-row', capacity);
    const pieces = {
      geometry: {
        kind: 'lines' as const,
        positions: createTransientView(graph, 'alias-output-positions', 'float32x2', 2),
        lineOffsets: view('alias-output-offsets', capacity + 1)
      },
      sourceIds: view('alias-output-ids', capacity),
      sourcePartRows: shared,
      sourceFeatureRows: port === 'featureRow' ? shared : undefined,
      vertexSourceEdges: port === 'vertex' ? shared : undefined,
      status: {count: view('alias-count'), overflow: view('alias-overflow')}
    };
    const Contributor = name === 'split' ? GPULineSplit : GPULineClipByPolygon;
    const props = {
      polygons: createReviewConstructionProps(graph).polygons,
      lines: {
        kind: 'lines' as const,
        positions: createTransientView(graph, 'alias-input-positions', 'float32x2', 2),
        lineOffsets: view('alias-input-offsets', 2)
      },
      intersectionCapacity: 4,
      candidateCapacity: 8,
      output: {
        pieces,
        status: {uncertainCount: port === 'diagnostic' ? shared : view('alias-uncertain')}
      }
    };
    expect(() => new Contributor(props)).toThrow(/conflicts|share buffers/);
  } finally {
    device.destroy();
  }
});
it('review2 R10 network split provenance distinguishes caller IDs and part rows', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const allocate = <Format extends 'float32x2' | 'uint32'>(
    name: string,
    format: Format,
    length: number,
    values?: number[]
  ) => {
    const view = createTransientView(graph, name, format, length);
    memory.write(
      view,
      format === 'float32x2'
        ? Float32Array.from(values ?? new Array(length * 2).fill(12345))
        : Uint32Array.from(values ?? new Array(length).fill(99))
    );
    return view;
  };
  try {
    const pieces = {
      geometry: {
        kind: 'lines' as const,
        positions: allocate('network-pieces', 'float32x2', 8),
        lineOffsets: allocate('network-offsets', 'uint32', 5)
      },
      sourceIds: allocate('network-ids', 'uint32', 4),
      sourcePartRows: allocate('network-rows', 'uint32', 4),
      status: {
        count: allocate('network-count', 'uint32', 1),
        overflow: allocate('network-overflow', 'uint32', 1)
      }
    };
    executeCommandNodesOnCPU(
      new GPULineSplit({
        lines: {
          kind: 'lines',
          positions: allocate('network-input', 'float32x2', 4, [0, 1, 2, 1, 1, 0, 1, 2]),
          lineOffsets: allocate('network-input-parts', 'uint32', 3, [0, 2, 4]),
          sourceIds: allocate('network-input-ids', 'uint32', 2, [91, 17])
        },
        intersectionCapacity: 8,
        output: {pieces}
      }).getCommandNodes(graph),
      memory
    );
    expect(Array.from(memory.readUint32(pieces.sourceIds))).toEqual([91, 91, 17, 17]);
    expect(Array.from(memory.readUint32(pieces.sourcePartRows))).toEqual([0, 0, 1, 1]);
  } finally {
    device.destroy();
  }
});
it('R7 direct CPU and GPU contributor publish identity grouping without grouped input', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  try {
    const fixture = createIdentityLoopReviewCase(graph, (name, format, length, values) => {
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
    expect(
      Array.from(
        computeLineLoopRemovalOnCPU(
          {positions: Float32Array.of(0, 0, 2, 0), lineOffsets: Uint32Array.of(0, 2)},
          {vertexCapacity: 2, includeFeatureOffsets: true}
        ).featureOffsets!
      )
    ).toEqual([0, 1]);
  } finally {
    device.destroy();
  }
});
it.each(['points', 'lines'] as const)('R9 grouped %s publishes zero area', kind => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  try {
    const fixture = createZeroAreaReviewCase(
      graph,
      (name, format, length, values) => {
        const width = format.endsWith('x2') ? 2 : 1;
        const view = createTransientView(graph, name, format, length);
        memory.write(
          view,
          format.startsWith('float')
            ? Float32Array.from(values ?? new Array(length * width).fill(12345))
            : Uint32Array.from(values ?? new Array(length * width).fill(99))
        );
        return view;
      },
      kind
    );
    executeCommandNodesOnCPU(
      fixture.nodes.filter(node => !node.id.startsWith('geometry-measures-group-')),
      memory
    );
    expect(Array.from(memory.readFloat32(fixture.output.output.areas))).toEqual([0, 0]);
  } finally {
    device.destroy();
  }
});
it.each([
  [false, false, false],
  [true, false, false],
  [false, true, false],
  [false, true, true]
])('R6 clearance probe includes collapsed parts single=%s tie=%s sort=%s', (single, tie, spatialSort) => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  try {
    const fixture = createCollapsedClearanceReviewCase(
      graph,
      (name, format, length, values) => {
        const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
        const view = createTransientView(graph, name, format, length);
        memory.write(
          view,
          format.startsWith('float')
            ? Float32Array.from(values ?? new Array(length * width).fill(12345))
            : Uint32Array.from(values ?? new Array(length * width).fill(99))
        );
        return view;
      },
      single,
      tie,
      spatialSort
    );
    const reduction = fixture.nodes.findIndex(node => node.id.endsWith('-feature-minimum'));
    executeCommandNodesOnCPU(fixture.nodes.slice(0, reduction), memory);
    const resources = fixture.nodes[reduction].resources as unknown as {
      buffer?: import('@luma.gl/graph/gpu-core').GraphDataView;
    }[];
    const nearest = resources.find(resource => resource.buffer?.format === 'float32x4')!.buffer!;
    const values = memory.readFloat32(nearest);
    for (let row = 0; row < (single ? 2 : 6); row++)
      expect(values[row * 4]).toBe(single ? Infinity : tie ? 1 : 9);
    if (tie) expect(Array.from(values.subarray(0, 3))).toEqual([1, 1, 0]);
  } finally {
    device.destroy();
  }
});
it.each([
  'bounds-points',
  'bounds-lines',
  'collapsed'
] as const)('R4/R5 serial measures %s match original-coordinate and part-weighted GEOS values', mode => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  try {
    for (const repeats of mode === 'collapsed' ? [4, 5] : [4]) {
      const fixture = createMeasureReviewCase(
        graph,
        (name, format, length, values) => {
          const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
          const view = createTransientView(graph, `${name}-${repeats}`, format, length);
          memory.write(
            view,
            format.startsWith('float')
              ? Float32Array.from(values ?? new Array(length * width).fill(12345))
              : Uint32Array.from(values ?? new Array(length * width).fill(99))
          );
          return view;
        },
        mode,
        false,
        repeats
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
    }
  } finally {
    device.destroy();
  }
});
it('R3 all-empty production WGSL writes every track, search and clearance output', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  try {
    const fixture = createEmptyTrackReviewCase(graph, (name, format, length, values) => {
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
import {computeTrackSimilarityOnCPU} from '../../../src/gpu-spatial-analysis/track-similarity/cpu-track-similarity-operation';
import {computeLineLoopRemovalOnCPU} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-line-loop-removal-operation';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {GPUTrackSimilarity} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity';
import {GPUTrackSimilaritySearch} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity-search';
import {GPUMinimumClearance} from '../../../src/gpu-spatial-analysis/minimum-clearance/gpu-minimum-clearance';
import {GPUGeometryMeasures} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geometry-measures';
it('R8 measures captures inherited grouping and output getters exactly once', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  try {
    const positions = createTransientView(graph, 'inherited-positions', 'float32x2', 4);
    const lineOffsets = createTransientView(graph, 'inherited-parts', 'uint32', 3);
    const featureOffsets = createTransientView(graph, 'inherited-features', 'uint32', 2);
    const lengths = createTransientView(graph, 'inherited-length', 'float32', 1);
    memory.write(positions, Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0));
    memory.write(lineOffsets, Uint32Array.of(0, 2, 4));
    memory.write(featureOffsets, Uint32Array.of(0, 2));
    memory.write(lengths, Float32Array.of(12345));
    let inputReads = 0;
    let outputReads = 0;
    const props = Object.assign(
      Object.create({
        get featureOffsets() {
          inputReads++;
          return featureOffsets;
        }
      }),
      {
        geometryType: 'lines' as const,
        positions,
        lineOffsets,
        output: Object.create({
          get lengths() {
            outputReads++;
            return lengths;
          }
        })
      }
    );
    const contributor = new GPUGeometryMeasures(props);
    expect(inputReads).toBe(1);
    expect(outputReads).toBe(1);
    executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
    expect(Array.from(memory.readFloat32(lengths))).toEqual([2]);
    expect(inputReads).toBe(1);
    expect(outputReads).toBe(1);
  } finally {
    device.destroy();
  }
});
import {GPULineLoopRemovalOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-line-loop-removal-operation';
import {CPULineLoopRemovalOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-line-loop-removal-operation';
import {CPUTrackSimilarityOperation} from '../../../src/gpu-spatial-analysis/track-similarity/cpu-track-similarity-operation';
it('review2 R2 preserves terminal-only CPU loop removal and refuses nonzero grouped starts', () => {
  const empty = computeLineLoopRemovalOnCPU(
    {positions: new Float32Array(0), lineOffsets: Uint32Array.of(0)},
    {vertexCapacity: 0}
  );
  expect(Array.from(empty.lineOffsets)).toEqual([0]);
  expect(Array.from(empty.status)).toEqual([0, 0, 0, 0]);
  expect(Array.from(empty.positions)).toEqual([]);
  expect(Array.from(empty.sourceEdges)).toEqual([]);
});
it('review2 R2 grouped offsets must start at zero in both CPU consumers', () => {
  const lines = {
    positions: Float32Array.of(9, 0, 0, 0, 2, 0),
    lineOffsets: Uint32Array.of(1, 3, 3),
    featureOffsets: Uint32Array.of(0, 1)
  };
  expect(() => computeLineLoopRemovalOnCPU(lines, {vertexCapacity: 3})).toThrow(
    'lines.lineOffsets'
  );
  expect(() =>
    computeTrackSimilarityOnCPU(
      {linesA: lines, pairA: Uint32Array.of(0), pairB: Uint32Array.of(0)},
      {measures: ['hausdorff']}
    )
  ).toThrow('linesA.lineOffsets');
});
it('R2 CPU consumers reject nonconstant capacity padding and invalid active offsets', () => {
  for (const lineOffsets of [Uint32Array.of(0, 2, 3), Uint32Array.of(0, 4, 4)]) {
    const lines = {
      positions: Float32Array.of(0, 0, 2, 0, 9, 0),
      lineOffsets,
      featureOffsets: Uint32Array.of(0, 1)
    };
    expect(() =>
      computeTrackSimilarityOnCPU(
        {linesA: lines, pairA: Uint32Array.of(0), pairB: Uint32Array.of(0)},
        {measures: ['hausdorff']}
      )
    ).toThrow('linesA.lineOffsets');
    expect(() => computeLineLoopRemovalOnCPU(lines, {vertexCapacity: 3})).toThrow(
      'lines.lineOffsets'
    );
  }
});
it('R3/R7 CPU operation values match empty-track and identity-hierarchy literals', async () => {
  const input = (format: 'float32x2' | 'uint32', values: Float32Array | Uint32Array) => {
    const length = values.length / (format === 'float32x2' ? 2 : 1);
    return {format, length, chunks: [{values, length}], pack: () => values};
  };
  const empty = await new CPUTrackSimilarityOperation({
    linesA: {positions: 'positions', lineOffsets: 'parts', featureOffsets: 'features'},
    pairA: 'a',
    pairB: 'b',
    pairCount: 1,
    output: {
      hausdorff: 'h',
      maximumDistance: 'max',
      symmetricSegmentPathDistance: 'path',
      pairStatus: 'status'
    }
  }).execute({
    inputs: {
      positionsA: input('float32x2', new Float32Array(0)),
      offsetsA: input('uint32', Uint32Array.of(0)),
      featuresA: input('uint32', Uint32Array.of(0, 0, 0)),
      pairA: input('uint32', Uint32Array.of(0)),
      pairB: input('uint32', Uint32Array.of(1))
    }
  });
  expect(Array.from(empty['status'] as Uint32Array)).toEqual([1]);
  for (const name of ['h', 'max', 'path']) expect((empty[name] as Float32Array)[0]).toBeNaN();
  const loop = await new CPULineLoopRemovalOperation({
    lines: {positions: 'positions', lineOffsets: 'parts'},
    lineCount: 1,
    vertexCapacity: 2,
    intersectionCapacity: 4,
    output: {
      positions: 'outPositions',
      lineOffsets: 'outParts',
      featureOffsets: 'outFeatures',
      sourceEdges: 'outEdges',
      status: 'status'
    }
  }).execute({
    inputs: {
      inputPositions: input('float32x2', Float32Array.of(0, 0, 2, 0)),
      inputOffsets: input('uint32', Uint32Array.of(0, 2))
    }
  });
  for (const [name, expected] of Object.entries({
    outPositions: [0, 0, 2, 0],
    outParts: [0, 2],
    outFeatures: [0, 1],
    outEdges: [0, 0],
    status: [2, 2, 0, 0]
  }))
    expect(Array.from(loop[name] as Float32Array | Uint32Array), name).toEqual(expected);
});
it.each([
  GPULineLoopRemovalOperation,
  CPULineLoopRemovalOperation
])('R7 %s accepts output-only identity grouping', Operation => {
  expect(
    () =>
      new Operation({
        lines: {positions: 'positions', lineOffsets: 'parts'},
        lineCount: 1,
        vertexCapacity: 2,
        intersectionCapacity: 4,
        output: {
          positions: 'outPositions',
          lineOffsets: 'outParts',
          featureOffsets: 'outFeatures',
          sourceEdges: 'outEdges',
          status: 'status'
        }
      })
  ).not.toThrow();
});
it('R3 direct CPU all-empty track features publish emptyTrack', () => {
  const result = computeTrackSimilarityOnCPU(
    {
      linesA: {
        positions: new Float32Array(0),
        lineOffsets: Uint32Array.of(0),
        featureOffsets: Uint32Array.of(0, 0, 0)
      },
      pairA: Uint32Array.of(0),
      pairB: Uint32Array.of(1)
    },
    {measures: ['hausdorff', 'maximumDistance', 'symmetricSegmentPathDistance']}
  );
  expect(Array.from(result.pairStatus)).toEqual([1]);
  expect(result.hausdorff[0]).toBeNaN();
  expect(result.maximumDistance[0]).toBeNaN();
  expect(result.symmetricSegmentPathDistance[0]).toBeNaN();
});
it.each([
  'track',
  'search',
  'clearance'
] as const)('R3 all-empty %s graph construction accepts two logical features', kind => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const positions = createTransientView(graph, 'empty-positions', 'float32x2', 0);
    const lineOffsets = createTransientView(graph, 'empty-parts', 'uint32', 1);
    const featureOffsets = createTransientView(graph, 'empty-features', 'uint32', 3);
    const lines = {kind: 'lines' as const, positions, lineOffsets, featureOffsets};
    const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
    const output = createTransientView(
      graph,
      'empty-distance',
      'float32',
      kind === 'clearance' ? 2 : 1
    );
    const contributor =
      kind === 'track'
        ? new GPUTrackSimilarity({
            linesA: lines,
            pairA: scalar('a'),
            pairB: scalar('b'),
            output: {hausdorff: output, pairStatus: scalar('status')}
          })
        : kind === 'search'
          ? new GPUTrackSimilaritySearch({
              linesA: lines,
              pairA: scalar('a'),
              pairB: scalar('b'),
              metric: 'hausdorff',
              threshold: createTransientView(graph, 'threshold', 'float32', 1),
              output: {
                distance: output,
                lowerBound: createTransientView(graph, 'bound', 'float32', 1),
                matched: scalar('matched'),
                status: scalar('status')
              }
            })
          : new GPUMinimumClearance({geometry: lines, output: {clearances: output}});
    expect(contributor.getCommandNodes(graph).length).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});
useCPUExecutorTestHooks();
it('R2 complete generated geometry with spare part capacity flows into CPU consumers', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  try {
    const fixture = createPaddedSplitReviewCase(graph, (name, format, length, values) => {
      const width = format.endsWith('x2') ? 2 : 1;
      const view = createTransientView(graph, name, format, length);
      memory.write(
        view,
        format.startsWith('float')
          ? Float32Array.from(values ?? new Array(length * width).fill(12345))
          : Uint32Array.from(values ?? new Array(length * width).fill(99))
      );
      return view;
    });
    executeCommandNodesOnCPU(fixture.nodes, memory);
    const geometry = fixture.output.pieces.geometry;
    const lines = {
      positions: memory.readFloat32(geometry.positions),
      lineOffsets: memory.readUint32(geometry.lineOffsets),
      featureOffsets: memory.readUint32(geometry.featureOffsets)
    };
    expect(Array.from(lines.lineOffsets)).toEqual([0, 2, 2, 2, 2]);
    expect(Array.from(lines.featureOffsets)).toEqual([0, 1]);
    expect(
      Array.from(
        computeTrackSimilarityOnCPU(
          {linesA: lines, pairA: Uint32Array.of(0), pairB: Uint32Array.of(0)},
          {measures: ['hausdorff']}
        ).hausdorff
      )
    ).toEqual([0]);
    const result = computeLineLoopRemovalOnCPU(lines, {vertexCapacity: 8});
    expect(Array.from(result.positions.subarray(0, 4))).toEqual([0, 0, 2, 0]);
    expect(Array.from(result.lineOffsets)).toEqual([0, 2, 2, 2, 2]);
    expect(Array.from(result.status)).toEqual([2, 2, 0, 0]);
  } finally {
    device.destroy();
  }
});
it.each([
  0, 1, 2, 3, 4, 5
])('R1 aligned topology variant %s matches Shapely without changing structural equality', variant => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  try {
    const fixture = createAlignedRelateReviewCase(
      graph,
      (name, format, length, values) => {
        const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
        const view = createTransientView(graph, name, format, length);
        memory.write(
          view,
          format.startsWith('float')
            ? Float32Array.from(values ?? new Array(length * width).fill(12345))
            : Uint32Array.from(values ?? new Array(length * width).fill(99))
        );
        return view;
      },
      variant
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
