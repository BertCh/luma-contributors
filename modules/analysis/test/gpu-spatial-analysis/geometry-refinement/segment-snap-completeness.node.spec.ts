// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSegmentSnap} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-segment-snap';
import {
  GPUSegmentSnapOperation,
  type SegmentSnapOperationProps
} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-segment-snap-operation';
import {
  CPUSegmentSnapOperation,
  computeSegmentSnapOnCPU
} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-segment-snap-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

function fixture() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  recordWGSLKernelNodes(graph);
  const props = {
    id: 'completeness',
    tolerance: 0,
    maximumVerticesPerLine: 8,
    candidateCapacity: 1,
    candidateOverflowPolicy: 'incomplete' as const,
    positions: createTransientView(graph, 'positions', 'float32x2', 2),
    lineOffsets: createTransientView(graph, 'offsets', 'uint32', 2),
    referencePositions: createTransientView(graph, 'references', 'float32x2', 2),
    output: {
      positions: createTransientView(graph, 'out', 'float32x2', 8),
      sourceEdges: createTransientView(graph, 'edges', 'uint32', 8),
      counts: createTransientView(graph, 'counts', 'uint32x4', 1)
    }
  };
  return {graph, props};
}
function operationProps(): SegmentSnapOperationProps {
  return {
    id: 'completeness',
    positions: 'positions',
    lineOffsets: 'offsets',
    referencePositions: 'references',
    rowCount: 2,
    referenceCount: 2,
    lineCount: 1,
    tolerance: 0,
    maximumVerticesPerLine: 8,
    candidateCapacity: 1,
    candidateOverflowPolicy: 'incomplete',
    output: {positions: 'out', sourceEdges: 'edges', counts: 'counts'}
  };
}

it('refuses required-outputs-only suppression in the contributor and both operations', () => {
  const input = {
    positions: Float32Array.of(0, 0, 3, 0),
    lineOffsets: Uint32Array.of(0, 2),
    referencePositions: Float32Array.of(1, 0, 2, 0)
  };
  const suppressed = computeSegmentSnapOnCPU(input, operationProps());
  const empty = computeSegmentSnapOnCPU(
    {...input, positions: new Float32Array(), lineOffsets: Uint32Array.of(0, 0)},
    operationProps()
  );
  expect(suppressed.counts).toEqual(empty.counts);
  expect(Array.from(suppressed.issueCode)).toEqual([4]);
  expect(Array.from(empty.issueCode)).toEqual([0]);
  expect(() => new GPUSegmentSnap(fixture().props)).toThrow(
    /output.*(validity|issueCode|candidateOverflow)/
  );
  for (const Operation of [GPUSegmentSnapOperation, CPUSegmentSnapOperation]) {
    expect(() => new Operation(operationProps())).toThrow(
      /output.*(validity|issueCode|candidateOverflow)/
    );
    expect(() => new Operation({...operationProps(), rowCount: 0})).not.toThrow();
    expect(() => new Operation({...operationProps(), candidateCapacity: 2})).not.toThrow();
    expect(
      () => new Operation({...operationProps(), candidateOverflowPolicy: 'bounded-brute-force'})
    ).not.toThrow();
  }
});

it('requires a signal when the default policy exceeds its exhaustive work bound', () => {
  for (const Operation of [GPUSegmentSnapOperation, CPUSegmentSnapOperation])
    expect(
      () =>
        new Operation({
          ...operationProps(),
          candidateOverflowPolicy: 'bounded-brute-force',
          referenceCount: 300
        })
    ).toThrow(/output.*(validity|issueCode|candidateOverflow)/);
});

it('captures inherited getter-backed contributor fields and nested status exactly once', () => {
  const {graph, props} = fixture();
  const diagnostic = createTransientView(graph, 'candidate', 'uint32', 1);
  let reads = 0;
  let current = diagnostic;
  const status = Object.create({
    get candidateOverflow() {
      reads++;
      return current;
    }
  });
  const options = {...props, output: {...props.output, status}};
  const contributor = new GPUSegmentSnap(options);
  const countAfterConstruction = reads;
  current = createTransientView(graph, 'replacement', 'uint32', 1);
  options.candidateCapacity = 2;
  options.output.status = {};
  expect(contributor.props.candidateCapacity).toBe(1);
  expect(contributor.props.output.status!.candidateOverflow).toBe(diagnostic);
  contributor.getCommandNodes(graph);
  expect(getWGSLKernelNodeRecord(graph, 'completeness-insert-0')!.declarations).toContain(
    'CANDIDATE_CAPACITY: u32 = 1u'
  );
  expect(
    getWGSLKernelNodeRecord(graph, 'completeness-candidate-status-0')!.bindings.find(
      binding => binding.name === 'candidateOverflow'
    )!.view
  ).toBe(diagnostic);
  expect(reads).toBe(countAfterConstruction);
  expect(countAfterConstruction).toBe(1);
});

it.each([
  'validity',
  'issueCode',
  'candidateOverflow'
] as const)('a single %s signal distinguishes suppression from empty input', async signal => {
  const vector = (format: 'float32x2' | 'uint32', values: Float32Array | Uint32Array) => {
    const length = values.length / (format === 'float32x2' ? 2 : 1);
    return {format, length, chunks: [{values, length}], pack: () => values};
  };
  for (const empty of [false, true]) {
    const props = operationProps();
    props.rowCount = empty ? 0 : 2;
    if (signal === 'candidateOverflow') props.output.status = {candidateOverflow: 'signal'};
    else props.output[signal] = 'signal';
    const operation = new CPUSegmentSnapOperation(props);
    const result = await operation.execute({
      inputs: {
        positions: vector('float32x2', empty ? new Float32Array() : Float32Array.of(0, 0, 3, 0)),
        lineOffsets: vector('uint32', Uint32Array.of(0, empty ? 0 : 2)),
        referencePositions: vector('float32x2', Float32Array.of(1, 0, 2, 0))
      }
    } as Parameters<typeof operation.execute>[0]);
    expect(result).toHaveProperty('counts', new Uint32Array(4));
    expect(result).toHaveProperty(
      'signal',
      Uint32Array.of(
        signal === 'validity' ? Number(empty) : empty ? 0 : signal === 'issueCode' ? 4 : 1
      )
    );
  }
});

it('GPU operation lowering retains construction-time policy and output signals', () => {
  const props = operationProps();
  props.output.status = {candidateOverflow: 'signal'};
  const operation = new GPUSegmentSnapOperation(props);
  props.candidateCapacity = 2;
  props.output.status = {};
  const {graph, props: views} = fixture();
  const signal = createTransientView(graph, 'signal', 'uint32', 1);
  const ports = {
    ...views,
    positionsOutput: views.output.positions,
    sourceEdges: views.output.sourceEdges,
    counts: views.output.counts,
    candidateOverflow: signal
  };
  const getView = ((name: keyof typeof ports) => ports[name]) as Parameters<
    typeof operation.createContributor
  >[0];
  const contributor = operation.createContributor(getView) as GPUSegmentSnap;
  expect(contributor.props.candidateCapacity).toBe(1);
  expect(contributor.props.output.status!.candidateOverflow).toBe(signal);
  contributor.getCommandNodes(graph);
  expect(getWGSLKernelNodeRecord(graph, 'completeness-insert-0')!.declarations).toContain(
    'CANDIDATE_CAPACITY: u32 = 1u'
  );
});

it('captures CPU operation outputs and policy before the caller mutates them', async () => {
  const props = operationProps();
  props.output.issueCode = 'issues';
  const operation = new CPUSegmentSnapOperation(props);
  props.candidateCapacity = 2;
  props.output.issueCode = 'replacement';
  const result = await operation.execute({
    inputs: {
      positions: {
        format: 'float32x2',
        length: 2,
        chunks: [{values: Float32Array.of(0, 0, 3, 0), length: 2}],
        pack: () => Float32Array.of(0, 0, 3, 0)
      },
      lineOffsets: {
        format: 'uint32',
        length: 2,
        chunks: [{values: Uint32Array.of(0, 2), length: 2}],
        pack: () => Uint32Array.of(0, 2)
      },
      referencePositions: {
        format: 'float32x2',
        length: 2,
        chunks: [{values: Float32Array.of(1, 0, 2, 0), length: 2}],
        pack: () => Float32Array.of(1, 0, 2, 0)
      }
    }
  } as Parameters<typeof operation.execute>[0]);
  expect(result).toHaveProperty('issues', Uint32Array.of(4));
  expect(result).not.toHaveProperty('replacement');
});
