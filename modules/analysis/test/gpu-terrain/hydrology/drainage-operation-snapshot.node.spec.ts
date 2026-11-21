// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  createTransientView,
  GPUCommandGraph,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {
  GPUTerrainFlowOperation,
  type TerrainFlowOperationProps
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow-operation';
import {CPUTerrainFlowOperation} from '../../../src/gpu-terrain/hydrology/cpu-terrain-flow-operation';
import {
  GPUTerrainWatershedsOperation,
  type TerrainWatershedsOperationProps
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-watersheds-operation';
import {
  CPUTerrainWatershedsOperation,
  computeTerrainWatershedsOnCPU
} from '../../../src/gpu-terrain/hydrology/cpu-terrain-watersheds-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const grid = {
  width: 5,
  height: 1,
  affine: [1, 0, 0, 0, -1, 0] as const,
  pixelInterpretation: 'area' as const
};
const flowProps = {
  grid,
  elevation: {
    id: 'dem',
    format: 'float32' as const,
    storage: {kind: 'buffer' as const, values: 'elevation'}
  },
  parameters: 'parameters',
  output: {flowDirections: 'directions'}
};

it('TSC-01 requires a per-cell flow writer in both operation forms', () => {
  for (const Operation of [GPUTerrainFlowOperation, CPUTerrainFlowOperation]) {
    expect(() => new Operation(flowProps)).not.toThrow();
    for (const output of [{}, {status: {accumulationConverged: 'done'}}]) {
      // Runtime JavaScript admission remains guarded even when the named selection is typed.
      const malformed = {...flowProps, output} as TerrainFlowOperationProps;
      expect(() => new Operation(malformed)).toThrow(/per-cell output/);
    }
    expect(
      () =>
        new Operation({
          ...flowProps,
          output: {flowDirections: 'directions', status: {accumulationConverged: 'done'}}
        })
    ).toThrow(/accumulation/);
    expect(() => new Operation({...flowProps, output: {filledElevation: 'filled'}})).toThrow(
      /fillDepressions/
    );
    expect(
      () =>
        new Operation({
          ...flowProps,
          output: {flowDirections: 'directions', status: {flatsConverged: 'done'}}
        })
    ).toThrow(/resolveFlats/);
  }
});

it('TSC-01 captures inherited flow writers and status getters once before mutation and lowering', () => {
  for (const Operation of [GPUTerrainFlowOperation, CPUTerrainFlowOperation]) {
    let writer = 'directions',
      diagnostic = 'done',
      writerReads = 0,
      diagnosticReads = 0;
    const status = Object.create({
      get fillConverged() {
        diagnosticReads++;
        return diagnostic;
      }
    });
    const output = Object.create({
      get flowDirections() {
        writerReads++;
        return writer;
      },
      status
    });
    const operation = new Operation({
      ...flowProps,
      fillDepressions: true,
      maximumFillIterations: 1,
      output
    });
    writer = 'changed';
    diagnostic = 'changed';
    if (operation instanceof GPUTerrainFlowOperation)
      expect(operation.outputs).toMatchObject({
        flowDirections: {value: 'directions'},
        fillConverged: {value: 'done'}
      });
    else expect(Object.keys(operation.outputs)).toEqual(['directions', 'done']);
    expect([writerReads, diagnosticReads]).toEqual([1, 1]);
    if (operation instanceof GPUTerrainFlowOperation) {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      const views = Object.fromEntries(
        [...Object.keys(operation.inputs), ...Object.keys(operation.outputs)].map(port => [
          port,
          createTransientView(
            graph,
            port,
            port === 'flowDirections' || port === 'fillConverged' ? 'uint32' : 'float32',
            port === 'parameters' ? 8 : port === 'fillConverged' ? 1 : 5
          )
        ])
      );
      const contributor = operation.createContributor(
        port => views[port] as never
      ) as GPUCommandNodeProducer;
      expect(() => contributor.getCommandNodes(graph)).not.toThrow();
      expect([writerReads, diagnosticReads]).toEqual([1, 1]);
      device.destroy();
    }
  }
});

it('B2 resolves every selected flow view once, including the definite writer', () => {
  for (const output of [
    {cellClasses: 'classes'},
    {
      filledElevation: 'filled',
      flowDirections: 'directions',
      cellClasses: 'classes',
      accumulation: 'accumulation',
      streams: 'streams',
      status: {fillConverged: 'done'}
    }
  ]) {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const operation = new GPUTerrainFlowOperation({
        ...flowProps,
        fillDepressions: true,
        maximumFillIterations: 1,
        output
      });
      const calls: Record<string, number> = {};
      const contributor = operation.createContributor(port => {
        calls[port] = (calls[port] ?? 0) + 1;
        const outputPort = operation.outputs[port];
        return createTransientView(
          graph,
          port,
          outputPort?.format ?? 'float32',
          outputPort?.length ?? (port === 'parameters' ? 8 : 5)
        ) as never;
      }) as GPUCommandNodeProducer;
      expect(() => contributor.getCommandNodes(graph)).not.toThrow();
      expect(calls).toEqual(
        Object.fromEntries(
          [...Object.keys(operation.inputs), ...Object.keys(operation.outputs)].map(port => [
            port,
            1
          ])
        )
      );
    } finally {
      device.destroy();
    }
  }
});

it('S1 rejects invalid captured option values at GPU and CPU operation construction', () => {
  for (const Operation of [GPUTerrainFlowOperation, CPUTerrainFlowOperation]) {
    for (const [option, accepted] of [
      ['flowRouting', 'd8'],
      ['cellSizeMode', 'uniform'],
      ['accumulationUnits', 'cells']
    ]) {
      expect(() =>
        Reflect.construct(Operation, [{...flowProps, id: 'review-option', [option]: 'typo'}])
      ).toThrow(new RegExp(`^review-option.*${option}.*${accepted}`));
    }
  }
  for (const Operation of [GPUTerrainWatershedsOperation, CPUTerrainWatershedsOperation]) {
    expect(() =>
      Reflect.construct(Operation, [
        {
          id: 'review-direction',
          grid,
          flowDirections: 'directions',
          directionEncoding: 'typo',
          output: {labels: 'labels'}
        }
      ])
    ).toThrow(/^review-direction.*directionEncoding.*esri.*whitebox.*taudem/);
  }
});

it('R10/R11 capture inherited watershed budgets and diagnostic destinations once', async () => {
  for (const Operation of [GPUTerrainWatershedsOperation, CPUTerrainWatershedsOperation]) {
    let budget = 1,
      budgetReads = 0,
      diagnostic = 'rounds',
      diagnosticReads = 0;
    const props: TerrainWatershedsOperationProps = Object.create({
      id: 'captured-watershed',
      grid,
      flowDirections: 'directions',
      get maximumIterations() {
        budgetReads++;
        return budget;
      },
      output: {
        labels: 'labels',
        status: Object.create({
          get iterationCount() {
            diagnosticReads++;
            return diagnostic;
          },
          converged: 'done',
          iterationLimitReached: 'limited'
        })
      }
    });
    const operation = new Operation(props);
    budget = 1025;
    diagnostic = 'changed';
    if (operation instanceof GPUTerrainWatershedsOperation)
      expect(operation.outputs['iterationCount']?.value).toBe('rounds');
    else expect(Object.keys(operation.outputs)).toContain('rounds');
    expect([budgetReads, diagnosticReads]).toEqual([1, 1]);
    if (operation instanceof GPUTerrainWatershedsOperation) {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      const contributor = operation.createContributor(
        port =>
          createTransientView(
            graph,
            port,
            'uint32',
            port === 'labels' || port === 'flowDirections' ? 5 : 1
          ) as never
      ) as GPUCommandNodeProducer;
      const ids = contributor.getCommandNodes(graph).map(node => node.id);
      expect(ids).toContain('captured-watershed-pointer-round-0-snapshot');
      expect(ids).not.toContain('captured-watershed-pointer-round-1');
      device.destroy();
    } else {
      const values = Uint32Array.of(1, 1, 1, 1, 0);
      const result = await operation.execute({
        inputs: {
          flowDirections: {
            format: 'uint32',
            length: 5,
            chunks: [{values, length: 5}],
            pack: () => values
          }
        }
      });
      // One synchronous doubling: [1,2,3,4,4] -> [2,3,4,4,4].
      expect(Array.from(result['labels'] as Uint32Array)).toEqual([
        0xfffffffe, 0xfffffffe, 4, 4, 4
      ]);
      expect(Array.from(result['rounds'] as Uint32Array)).toEqual([1]);
      expect(Array.from(result['done'] as Uint32Array)).toEqual([0]);
      expect(Array.from(result['limited'] as Uint32Array)).toEqual([1]);
    }
    expect([budgetReads, diagnosticReads]).toEqual([1, 1]);
    expect(
      () =>
        new Operation(
          Object.create({
            grid,
            flowDirections: 'directions',
            output: {labels: 'labels'},
            maximumIterations: 1025
          })
        )
    ).toThrow(/maximumIterations.*1024/);
  }
});

it('R11 validates and executes the same pure CPU getter budget', () => {
  let reads = 0;
  const result = computeTerrainWatershedsOnCPU(Uint32Array.of(1, 1, 1, 1, 0), {
    grid,
    get maximumIterations() {
      return ++reads === 1 ? 1 : 1025;
    }
  });
  expect(reads).toBe(1);
  expect(Array.from(result.labels)).toEqual([0xfffffffe, 0xfffffffe, 4, 4, 4]);
  expect(Array.from(result.iterationCount)).toEqual([1]);
  expect(() =>
    computeTerrainWatershedsOnCPU(
      Uint32Array.of(1, 1, 1, 1, 0),
      Object.create({grid, maximumIterations: 1025})
    )
  ).toThrow(/maximumIterations.*1024/);
  const unbounded = computeTerrainWatershedsOnCPU(Uint32Array.of(1, 1, 1, 1, 0), {grid});
  expect(Array.from(unbounded.labels)).toEqual([4, 4, 4, 4, 4]);
  expect(Array.from(unbounded.iterationCount)).toEqual([0]);
});
