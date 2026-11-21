// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {getUndeclaredCalls} from '../segment-intersection/wgsl-call-sites';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {createReviewPolygonizeProps} from './polygonize-review-output-oracles-fixtures';
import {getOracleView, type OracleAllocate} from './noding-polygonize-output-oracles-fixtures';
import {GPUPolygonize} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-polygonize';
import {GPUMakeValid} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-make-valid';
import {
  computeOffsetCurveComponentsOnCPU,
  CPUOffsetCurveComponentsOperation
} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-offset-curve-components-operation';
import {
  createReview2OffsetCase,
  createReview2StackedSquaresCase,
  REVIEW2_OFFSET_ORIGINAL,
  REVIEW2_OFFSET_RAW,
  REVIEW2_OFFSET_OPTIONS
} from './review2-output-oracles-fixtures';

useCPUExecutorTestHooks();

function harness() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const allocate: OracleAllocate = (name, format, length, values) => {
    const view = createTransientView(graph, name, format, length);
    const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
    memory.write(
      view,
      format.startsWith('uint')
        ? Uint32Array.from(values ?? new Array(length * width).fill(777))
        : Float32Array.from(values ?? new Array(length * width).fill(-777))
    );
    return view;
  };
  return {device, graph, memory, allocate};
}

function constructorFixture(kind: 'polygonize' | 'make-valid', allocate: OracleAllocate) {
  const props = createReviewPolygonizeProps(allocate);
  const makeValidProps = {
    output: props.output,
    precision: props.precision,
    intersectionCapacity: props.intersectionCapacity,
    polygons: {
      kind: 'polygons' as const,
      positions: allocate('review2-square', 'float32x2', 4, [0, 0, 4, 0, 4, 4, 0, 4]),
      ringOffsets: allocate('review2-rings', 'uint32', 2, [0, 4]),
      polygonOffsets: allocate('review2-polygons', 'uint32', 2, [0, 1]),
      featureOffsets: allocate('review2-features', 'uint32', 2, [0, 1])
    }
  };
  return {
    output: props.output,
    build: () =>
      kind === 'polygonize' ? new GPUPolygonize(props) : new GPUMakeValid(makeValidProps)
  };
}

it.each([
  'polygonize',
  'make-valid'
] as const)('review 2 %s rejects inherited malformed nested status', kind => {
  const {device, allocate} = harness();
  try {
    const fixture = constructorFixture(kind, allocate);
    fixture.output.nodedSegments.status = Object.create({
      ...fixture.output.nodedSegments.status,
      uncertainCount: allocate('review2-bad', 'float32', 0)
    });
    expect(() => fixture.build()).toThrow('uncertainCount');
  } finally {
    device.destroy();
  }
});

it.each([
  'polygonize',
  'make-valid'
] as const)('review 2 %s refuses inherited retired status names', kind => {
  const {device, allocate} = harness();
  try {
    const fixture = constructorFixture(kind, allocate);
    fixture.output.status = Object.create({
      ...fixture.output.status,
      requiredCount: allocate('review2-retired', 'uint32', 1)
    });
    expect(() => fixture.build()).toThrow('requiredRingCount');
  } finally {
    device.destroy();
  }
});

it.each([
  'polygonize',
  'make-valid'
] as const)('review 2 %s captures status getter once and binds the validated view', kind => {
  const {device, allocate} = harness();
  try {
    const fixture = constructorFixture(kind, allocate);
    const valid = allocate('review2-good', 'uint32', 1);
    const invalid = allocate('review2-bad', 'float32', 0);
    let reads = 0;
    Object.defineProperty(fixture.output.nodedSegments.status, 'uncertainCount', {
      enumerable: true,
      get() {
        reads++;
        return reads === 1 ? valid : invalid;
      }
    });
    const contributor = fixture.build();
    expect(reads).toBe(1);
    expect(contributor.props.output.nodedSegments.status.uncertainCount).toBe(valid);
  } finally {
    device.destroy();
  }
});

it('review 2 make-valid rejects an ordinary zero-row delegated uncertainty scalar', () => {
  const {device, allocate} = harness();
  try {
    const fixture = constructorFixture('make-valid', allocate);
    fixture.output.directedEdges.status.uncertainCount = allocate('review2-empty', 'uint32', 0);
    expect(() => fixture.build()).toThrow('uncertainCount');
  } finally {
    device.destroy();
  }
});

it.each([
  'polygonize',
  'make-valid'
] as const)('review 2 %s refuses cross-meaning status alias', kind => {
  const {device, allocate} = harness();
  try {
    const fixture = constructorFixture(kind, allocate);
    fixture.output.directedEdges.status.uncertainCount = fixture.output.status.candidateOverflow;
    expect(() => fixture.build()).toThrow('must not overlap');
  } finally {
    device.destroy();
  }
});

it.each([
  'polygonize',
  'make-valid'
] as const)('review 2 %s permits intentional uncertainty sharing', kind => {
  const {device, allocate} = harness();
  try {
    const fixture = constructorFixture(kind, allocate);
    const shared = allocate('review2-shared', 'uint32', 1);
    fixture.output.status.uncertainCount = shared;
    fixture.output.nodedSegments.status.uncertainCount = shared;
    fixture.output.directedEdges.status.uncertainCount = shared;
    const contributor = fixture.build();
    expect(contributor.props.output.status.uncertainCount).toBe(shared);
    expect(contributor.props.output.directedEdges.status.uncertainCount).toBe(shared);
  } finally {
    device.destroy();
  }
});

it('review 2 CPU offset function preserves surviving traversal row 2', () => {
  const result = computeOffsetCurveComponentsOnCPU(
    {
      original: {
        positions: Float32Array.from(REVIEW2_OFFSET_ORIGINAL),
        lineOffsets: Uint32Array.of(0, 4)
      },
      raw: {positions: Float32Array.from(REVIEW2_OFFSET_RAW), lineOffsets: Uint32Array.of(0, 4)}
    },
    REVIEW2_OFFSET_OPTIONS
  );
  expect(Array.from(result.positions)).toEqual([0, 1, 2, 1]);
  expect(Array.from(result.status)).toEqual([1, 2, 1, 2, 0, 0, 0, 0]);
  expect(Array.from(result.sourceEdges)).toEqual([2, 2]);
});

it('review 2 CPU offset operation preserves surviving traversal row 2', async () => {
  const input = (format: 'float32x2' | 'uint32', values: Float32Array | Uint32Array) => {
    const length = values.length / (format === 'float32x2' ? 2 : 1);
    return {format, length, chunks: [{values, length}], pack: () => values};
  };
  const result = await new CPUOffsetCurveComponentsOperation({
    ...REVIEW2_OFFSET_OPTIONS,
    lineCount: 1,
    original: {positions: 'originalPositions', lineOffsets: 'originalOffsets'},
    raw: {positions: 'rawPositions', lineOffsets: 'rawOffsets'},
    output: {
      positions: 'positions',
      components: 'components',
      sourceEdges: 'sourceEdges',
      status: 'status'
    }
  }).execute({
    inputs: {
      originalPositions: input('float32x2', Float32Array.from(REVIEW2_OFFSET_ORIGINAL)),
      originalOffsets: input('uint32', Uint32Array.of(0, 4)),
      rawPositions: input('float32x2', Float32Array.from(REVIEW2_OFFSET_RAW)),
      rawOffsets: input('uint32', Uint32Array.of(0, 4))
    }
  });
  expect(Array.from(result['positions'] as Float32Array)).toEqual([0, 1, 2, 1]);
  expect(Array.from(result['sourceEdges'] as Uint32Array)).toEqual([2, 2]);
});

it('review 2 GPU offset contributor preserves surviving traversal row 2 on CPU executor', () => {
  const {device, graph, memory, allocate} = harness();
  try {
    const fixture = createReview2OffsetCase(graph, allocate);
    executeCommandNodesOnCPU(fixture.nodes, memory);
    for (const [path, expected] of Object.entries(fixture.expected)) {
      const view = getOracleView(fixture.output, path);
      expect(
        Array.from(
          view.format.startsWith('uint')
            ? memory.readUint32(view, expected.length)
            : memory.readFloat32(view, expected.length)
        ),
        path
      ).toEqual(expected);
    }
  } finally {
    device.destroy();
  }
});

it.each([
  8, 32
])('review 2 stacked %i squares inspect only two coincident candidates per endpoint', squareCount => {
  const {device, graph, memory, allocate} = harness();
  try {
    const fixture = createReview2StackedSquaresCase(graph, allocate, squareCount);
    executeCommandNodesOnCPU(fixture.nodes, memory);
    expect(Array.from(memory.readUint32(fixture.output.candidateCounts, 8 * squareCount))).toEqual(
      fixture.expected.candidateCounts
    );
  } finally {
    device.destroy();
  }
});

it('review 2 instrumented spatial lookup parses within eight storage bindings', () => {
  const {device, graph, allocate} = harness();
  try {
    const fixture = createReview2StackedSquaresCase(graph, allocate, 8);
    for (const {id, source} of getParsedKernelSources(fixture.nodes).configurations) {
      expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
      expect(getContactUndeclaredCalls(source), id).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});

// The shared call-site utility predates this WGSL builtin; it is supplied by the
// shader language and needs no local function declaration. Other missing calls still fail.
function getContactUndeclaredCalls(source: string): string[] {
  return getUndeclaredCalls(source).filter(name => name !== 'workgroupUniformLoad');
}
it('contact call audit recognizes uniform loads without accepting missing helpers', () => {
  expect(
    getContactUndeclaredCalls('fn main() { workgroupUniformLoad(&value); missingContactHelper(); }')
  ).toEqual(['missingContactHelper']);
});
