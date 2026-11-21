// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {
  computeTrajectoryThinningOnCPU,
  CPUTrajectoryThinningOperation
} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/cpu-trajectory-thinning-operation';
import {GPUTrajectoryThinningOperation} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-thinning-operation';
it('names compact source rows separately from the original-row selection mask', () => {
  for (const capacity of [1, 2]) {
    const result = computeTrajectoryThinningOnCPU(
      {
        positions: Float32Array.of(0, 0, 0.25, 0, 1, 0),
        timestamps: Float32Array.of(0, 1, 2),
        trackOffsets: Uint32Array.of(0, 3),
        parameters: Float32Array.of(1, 0, 0, 0)
      },
      {mode: 'minimum-distance', capacity}
    );
    expect(Array.from(result.sourceRows)).toEqual(capacity === 1 ? [0] : [0, 2]);
    expect(result.count[0]).toBe(capacity);
    expect(Array.from(result.sourceSelectionMask)).toEqual([1, 0, 1]);
  }
});
it('rejects the removed output names', () => {
  const props = {
    positions: 'positions',
    timestamps: 'times',
    trackOffsets: 'offsets',
    parameters: 'parameters',
    mode: 'minimum-distance' as const,
    rowCount: 3,
    trackCount: 1,
    capacity: 2,
    output: {
      sourceRows: 'rows',
      sourceSelectionMask: 'selection',
      trackOffsets: 'newOffsets',
      status: {count: 'count', overflow: 'overflow', candidateOverflow: 'candidateOverflow'}
    }
  };
  expect(() => new GPUTrajectoryThinningOperation(props)).not.toThrow();
  expect(
    () =>
      new GPUTrajectoryThinningOperation({
        ...props,
        output: {
          ...props.output,
          // @ts-expect-error The retired output key is also rejected for JavaScript callers.
          rowIds: props.output.sourceRows
        }
      })
  ).toThrow(/output\.rowIds was removed.*output\.sourceRows/);
  expect(
    () =>
      new GPUTrajectoryThinningOperation({
        ...props,
        output: {
          ...props.output,
          // @ts-expect-error The retired output key is also rejected for JavaScript callers.
          validity: props.output.sourceSelectionMask
        }
      })
  ).toThrow(/output\.validity was removed.*output\.sourceSelectionMask/);
  const outputWithRetiredGetter = new (class {
    sourceRows = props.output.sourceRows;
    trackOffsets = props.output.trackOffsets;
    status = props.output.status;
    get validity() {
      return props.output.sourceSelectionMask;
    }
  })();
  expect(
    () => new CPUTrajectoryThinningOperation({...props, output: outputWithRetiredGetter})
  ).toThrow(/output\.validity was removed.*output\.sourceSelectionMask/);
});
