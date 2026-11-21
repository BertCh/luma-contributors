// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  computePettittTestOnCPU,
  CPUPettittTestOperation
} from '../../../src/gpu-spatial-analysis/change-points/cpu-pettitt-test-operation';
import {
  GPUPettittTestOperation,
  type PettittTestOperationProps
} from '../../../src/gpu-spatial-analysis/change-points/gpu-pettitt-test-operation';
import {CPULocationChangePointsOperation} from '../../../src/gpu-spatial-analysis/change-points/cpu-location-change-points-operation';
import {
  GPULocationChangePointsOperation,
  type LocationChangePointsOperationProps
} from '../../../src/gpu-spatial-analysis/change-points/gpu-location-change-points-operation';
import type {GPUPettittTest} from '../../../src/gpu-spatial-analysis/change-points/gpu-pettitt-test';
import type {GPULocationChangePoints} from '../../../src/gpu-spatial-analysis/change-points/gpu-location-change-points';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOperationTestViewGetter} from '../../utils/statistics-dynamics-operation-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {snapshotChangePointProps} from '../../../src/gpu-spatial-analysis/change-points/change-point-construction';

it('TSC-01 reads inherited output and optional overflow getters once and retains identities', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getView = createOperationTestViewGetter(graph, {overflow: {format: 'uint32', length: 1}});
    for (const overflow of ['overflow', getView('overflow')]) {
      let outputReads = 0;
      let statusReads = 0;
      let overflowReads = 0;
      const status = Object.create({
        get overflow() {
          overflowReads++;
          return overflow;
        }
      }) as {overflow: typeof overflow};
      const output = Object.create({
        get status() {
          statusReads++;
          return status;
        }
      }) as {status: typeof status};
      const captured = snapshotChangePointProps({
        get output() {
          outputReads++;
          return output;
        }
      });
      expect(captured.output.status.overflow).toBe(overflow);
      expect([outputReads, statusReads, overflowReads]).toEqual([1, 1, 1]);
      expect(Object.isFrozen(captured.output.status)).toBe(true);
    }
    expect(snapshotChangePointProps({output: {status: {}}}).output.status).toEqual({});
  } finally {
    device.destroy();
  }
});

it('W2-15 captures Pettitt direct and operation descriptors and replays the CPU operation', async () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props: PettittTestOperationProps = {
      id: 'capture-pettitt',
      values: 'values',
      layout: 'cell-major',
      cellCount: 1,
      sliceCount: 4,
      output: {statistic: 'statistic', changeSlice: 'changeSlice', pValues: 'pValues'}
    };
    // Public property descriptors are supported even if fields are inherited/non-enumerable.
    Object.defineProperty(props, 'sliceCount', {value: 4, writable: true, enumerable: false});
    const operation = new GPUPettittTestOperation(props);
    const cpuOperation = new CPUPettittTestOperation(props);
    const getView = createOperationTestViewGetter(graph, {
      values: {format: 'float32', length: 4},
      statistic: {format: 'uint32', length: 1},
      changeSlice: {format: 'uint32', length: 1},
      pValues: {format: 'float32', length: 1}
    });
    props.sliceCount = 1;
    props.layout = 'period-major';
    props.output.pValues = 'changed';
    const contributor = operation.createContributor(getView) as GPUPettittTest;
    expect(contributor.props.sliceCount).toBe(4);
    expect(contributor.props.values).toBe(getView('values'));
    expect(contributor.props.output.pValues).toBe(getView('pValues'));
    for (const values of [
      Float32Array.of(0, 0, 10, 10),
      Float32Array.of(2, 2, 2, 2),
      Float32Array.of(0, 0, 10, 10)
    ]) {
      const result = await runOperationGraph(device, {
        inputs: {values: {format: 'float32', values}},
        operations: [cpuOperation],
        outputs: ['statistic', 'changeSlice', 'pValues']
      });
      const constant = values[0] === 2;
      // Four cross-split comparisons at split 2; p=2 exp(-6*16/(4^3+4^2)).
      expect(Array.from(result['statistic'])).toEqual([constant ? 0 : 4]);
      expect(Array.from(result['changeSlice'])).toEqual([constant ? 1 : 2]);
      expect(Number.isFinite(result['pValues'][0])).toBe(true);
      expect(result['pValues'][0]).toBeCloseTo(constant ? 1 : 0.6023884238244043, 6);
    }
  } finally {
    device.destroy();
  }
});

it('W2-15 captures location-change status maps and preserves GPU identities through deferred lowering', async () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props: LocationChangePointsOperationProps = {
      id: 'capture-location',
      values: 'values',
      parameters: 'parameters',
      layout: 'cell-major',
      cellCount: 1,
      sliceCount: 4,
      maximumChangePoints: 2,
      cost: 'mean',
      output: {
        changePointCount: 'counts',
        changePoints: 'changes',
        optimalCost: 'cost',
        status: {overflow: 'overflow'}
      }
    };
    Object.defineProperty(props.output.status!, 'overflow', {
      value: 'overflow',
      writable: true,
      enumerable: false
    });
    const operation = new GPULocationChangePointsOperation(props);
    const cpuOperation = new CPULocationChangePointsOperation(props);
    const getView = createOperationTestViewGetter(graph, {
      values: {format: 'float32', length: 4},
      parameters: {format: 'float32', length: 2},
      changePointCount: {format: 'uint32', length: 1},
      changePoints: {format: 'uint32', length: 2},
      optimalCost: {format: 'float32', length: 1},
      overflow: {format: 'uint32', length: 1}
    });
    props.cost = 'poisson';
    props.sliceCount = 1;
    props.output.status!.overflow = 'changed';
    const contributor = operation.createContributor(getView) as GPULocationChangePoints;
    expect(contributor.props.cost).toBe('mean');
    expect(contributor.props.sliceCount).toBe(4);
    expect(contributor.props.values).toBe(getView('values'));
    expect(contributor.props.output.status!.overflow).toBe(getView('overflow'));
    const direct = contributor.props;
    const originalOverflow = direct.output.status!.overflow;
    // Snapshot maps stay frozen; the GPU resource itself is the original identity.
    expect(Object.isFrozen(direct.output.status)).toBe(true);
    expect(direct.output.status!.overflow).toBe(originalOverflow);
    for (const values of [
      Float32Array.of(0, 0, 10, 10),
      Float32Array.of(2, 2, 2, 2),
      Float32Array.of(0, 0, 10, 10)
    ]) {
      const result = await runOperationGraph(device, {
        inputs: {
          values: {format: 'float32', values},
          parameters: {format: 'float32', values: Float32Array.of(1, 1)}
        },
        operations: [cpuOperation],
        outputs: ['counts', 'changes', 'cost', 'overflow']
      });
      const constant = values[0] === 2;
      // SSE is 100 without a split, zero within each half; one split costs penalty 1.
      expect(Array.from(result['counts'])).toEqual([constant ? 0 : 1]);
      expect(Array.from(result['changes'])).toEqual(
        constant ? [0xffffffff, 0xffffffff] : [2, 0xffffffff]
      );
      expect(Array.from(result['cost'])).toEqual([constant ? 0 : 1]);
      expect(Array.from(result['overflow'])).toEqual([0]);
    }
  } finally {
    device.destroy();
  }
});

it('W2-15 CPU Pettitt dimensions retain the caller ID', () => {
  expect(() =>
    computePettittTestOnCPU(Float32Array.of(1), {
      id: 'my-pettitt',
      cellCount: 1,
      sliceCount: 2,
      layout: 'cell-major'
    })
  ).toThrow(/^my-pettitt.*values.*2.*1/);
});
