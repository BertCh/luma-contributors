// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPULineSimplification,
  GPULineSimplificationOperation,
  getGPULineSimplificationParameterValues
} from '../../../src/gpu-spatial-analysis/line-simplification/index';
import type {LineSimplificationOperationProps} from '../../../src/gpu-spatial-analysis/line-simplification/gpu-line-simplification-operation';
import type {OracleView} from '../line-merge/line-merge-output-oracle-fixture';

// The middle vertex is 1/8 above the endpoint chord. Tolerance 1/4 removes it;
// endpoints are retained, so selected rows are [0,2], independently of the algorithm.
export const SIMPLIFICATION_INPUT = {
  positions: Float32Array.of(0, 0, 1, 0.125, 2, 0),
  lineOffsets: Uint32Array.of(0, 3),
  parameters: getGPULineSimplificationParameterValues({tolerance: 0.25})
};

export function getSimplificationOperationProps(
  selection = true,
  capacity = 2,
  includeCandidate = false,
  getterCandidate = false
): LineSimplificationOperationProps {
  const status = {
    count: 'count',
    overflow: 'overflow',
    ...(includeCandidate ? {candidateOverflow: 'candidateOverflow'} : {})
  };
  if (getterCandidate) {
    delete status.candidateOverflow;
    Object.setPrototypeOf(
      status,
      Object.defineProperty({}, 'candidateOverflow', {
        get() {
          return 'candidateOverflow';
        }
      })
    );
  }
  return {
    positions: 'positions',
    lineOffsets: 'lineOffsets',
    importance: 'importance',
    rowCount: 3,
    lineCount: 1,
    capacity,
    maximumIterations: 2,
    ...(selection
      ? {
          parameters: 'parameters',
          output: {ids: 'ids', status}
        }
      : {})
  };
}

export function createSimplificationOracleFixture(
  view: OracleView,
  mode: 'operation' | 'candidate' | 'absent',
  capacity = 2,
  includeOperationCandidate = false,
  getterCandidate = false
) {
  const positions = view('float32x2', 3, Array.from(SIMPLIFICATION_INPUT.positions));
  const lineOffsets = view('uint32', 2, [0, 3]);
  const parameters = view('float32', 4, Array.from(SIMPLIFICATION_INPUT.parameters));
  const importance = view('float32', 3);
  const ids = view('uint32', capacity);
  const count = view('uint32', 1);
  const overflow = view('uint32', 1);
  const candidateOverflow =
    mode === 'candidate' || includeOperationCandidate || getterCandidate
      ? view('uint32', 1)
      : undefined;
  const ports = {
    positions,
    lineOffsets,
    parameters,
    importance,
    ids,
    count,
    overflow,
    candidateOverflow
  };
  const operationProps = getSimplificationOperationProps(
    true,
    capacity,
    includeOperationCandidate,
    getterCandidate
  );
  const operation =
    mode === 'operation' ? new GPULineSimplificationOperation(operationProps) : undefined;
  const contributor = operation
    ? operation.createContributor(port => ports[port as keyof typeof ports] as never)
    : new GPULineSimplification({
        positions,
        lineOffsets,
        parameters,
        importance,
        maximumIterations: 2,
        output: {ids, status: {count, overflow, candidateOverflow}}
      });
  return {
    contributor,
    operation,
    operationProps,
    importance,
    ids,
    count,
    overflow,
    candidateOverflow
  };
}
