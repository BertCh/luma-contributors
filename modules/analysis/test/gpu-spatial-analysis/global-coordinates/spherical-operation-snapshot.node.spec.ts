// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {
  CPUSphericalArcOverlayOperation,
  computeSphericalArcOverlayOnCPU
} from '../../../src/gpu-spatial-analysis/global-coordinates/cpu-spherical-arc-overlay-operation';
import {
  GPUSphericalArcOverlayOperation,
  captureSphericalArcOverlayOperationProps,
  type SphericalArcOverlayOperationProps
} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-arc-overlay-operation';

function getProps(): SphericalArcOverlayOperationProps {
  return {
    referencePoint: [0, 0, 1],
    operation: 'intersection',
    capacity: 6,
    ringCapacity: 2,
    left: {positions: 'left', ringOffsets: 'leftOffsets', referenceInside: false},
    right: {positions: 'right', ringOffsets: 'rightOffsets', referenceInside: false},
    output: {positions: 'positions', ringOffsets: 'offsets', counts: 'counts', status: 'status'}
  };
}

it('AMG-51 captures the first referencePoint exactly once in both operation forms', () => {
  for (const Operation of [CPUSphericalArcOverlayOperation, GPUSphericalArcOverlayOperation]) {
    const props = getProps();
    let reads = 0;
    Object.defineProperty(props, 'referencePoint', {
      enumerable: true,
      get: () => (++reads === 1 ? [0, 0, 1] : [0, 0, 0])
    });
    expect(() => new Operation(props)).not.toThrow();
    expect(reads).toBe(1);
  }
});

it('AMG-52 preserves inherited capacities and non-enumerable port names', () => {
  for (const Operation of [CPUSphericalArcOverlayOperation, GPUSphericalArcOverlayOperation]) {
    const props = getProps();
    Reflect.deleteProperty(props, 'capacity');
    let reads = 0;
    Object.setPrototypeOf(props, {
      get capacity() {
        reads++;
        return 6;
      }
    });
    Object.defineProperty(props.left, 'positions', {value: 'hiddenLeft', enumerable: false});
    Object.defineProperty(props.output, 'counts', {value: 'hiddenCounts', enumerable: false});
    expect(() => new Operation(props)).not.toThrow();
    expect(reads).toBe(1);
    const captured = captureSphericalArcOverlayOperationProps(props);
    expect(captured.capacity).toBe(6);
    expect(captured.left.positions).toBe('hiddenLeft');
    expect(captured.output.counts).toBe('hiddenCounts');
  }
});

it('NAM-09 rejects misspelled operations in both named forms before lowering', () => {
  for (const Operation of [CPUSphericalArcOverlayOperation, GPUSphericalArcOverlayOperation]) {
    for (const operation of ['Union', 'unrelated']) {
      expect(() => new Operation({...getProps(), operation: operation as 'union'})).toThrow(
        /operation.*intersection.*union.*difference.*symmetric-difference.*received/
      );
    }
    expect(() => new Operation({...getProps(), operation: 'symmetric-difference'})).not.toThrow();
  }
});

it('B3 captures CPU function options once before validation and membership', () => {
  let reads = 0;
  const options = {
    ...getProps(),
    get operation() {
      return ++reads === 1 ? ('union' as const) : ('intersection' as const);
    }
  };
  const empty = {
    positions: new Float32Array(0),
    ringOffsets: Uint32Array.of(0),
    referenceInside: false
  };
  const result = computeSphericalArcOverlayOnCPU(
    {left: {...empty, referenceInside: true}, right: empty},
    options
  );
  // Empty boundaries encode constant regions: true UNION false = true everywhere.
  expect(Array.from(result.counts)).toEqual([0, 0, 0, 1]);
  expect(Array.from(result.status)).toEqual([0]);
  expect(reads).toBe(1);
});
