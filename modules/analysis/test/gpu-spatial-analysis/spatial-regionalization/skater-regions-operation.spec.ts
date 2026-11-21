// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUGraph, gpuGraph} from '@luma.gl/graph/gpu-core';
import {CPUVector} from '@luma.gl/graph/gpu-data';
import {expect, it} from 'vitest';
import {GPU_CONTRIBUTOR_OPERATION_LOWERING} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {CPUSkaterRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-skater-regions-operation';
import {
  GPUSkaterRegionsOperation,
  type SkaterRegionsOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-skater-regions-operation';
import {getGPUSkaterRegionsParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-skater-regions';
import {GPU_MINIMUM_SPANNING_TREE_STATUS_LAYOUT} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-spatial-weights-minimum-spanning-tree';
import {
  getCPUVectorRows,
  makeCPUVector,
  runOperationGraph,
  type OperationGraphInput
} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const flagName of ['overflow', 'candidateOverflow'] as const)
  it(`SKATER operations suppress the lost third chain link after ${flagName} and recover on one compiled graph`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: SkaterRegionsOperationProps = {
      rowCount: 3,
      edgeCount: 4,
      maximumRegionCount: 3,
      weights: {
        offsets: 'offsets',
        neighbors: 'neighbors',
        weights: 'weights',
        status: {[flagName]: 'loss'}
      },
      treeEdgeFlags: 'treeEdgeFlags',
      componentLabels: 'componentLabels',
      values: 'values',
      parameters: 'parameters',
      output: {
        labels: 'labels',
        regionCount: 'regionCount',
        cutEdges: 'cutEdges',
        cutGains: 'cutGains'
      }
    };
    const createInputs = (incomplete: number): Record<string, OperationGraphInput> => ({
      offsets: {
        format: 'uint32',
        values: Uint32Array.from(incomplete ? [0, 1, 2, 2] : [0, 1, 3, 4])
      },
      neighbors: {format: 'uint32', values: Uint32Array.of(1, 0, 2, 1)},
      weights: {format: 'float32', values: Float32Array.of(1, 1, 1, 1)},
      treeEdgeFlags: {
        format: 'uint32',
        values: Uint32Array.from(incomplete ? [0, 0, 0, 0] : [1, 0, 1, 0])
      },
      componentLabels: {
        format: 'uint32',
        values: Uint32Array.from(incomplete ? [0xffffffff, 0xffffffff, 0xffffffff] : [0, 0, 0])
      },
      values: {format: 'float32', values: Float32Array.of(1, 2, 4)},
      parameters: {format: 'uint32', values: Uint32Array.of(1, 1)},
      loss: {format: 'uint32', values: Uint32Array.of(incomplete)}
    });
    const outputs = ['labels', 'regionCount', 'cutEdges', 'cutGains'];
    for (const Backend of [GPUSkaterRegionsOperation, CPUSkaterRegionsOperation]) {
      const unflagged = await runOperationGraph(device, {
        inputs: createInputs(0),
        operations: [new Backend({...props, weights: {...props.weights, status: undefined}})],
        outputs
      });
      const graph = new GPUGraph({id: `skater-reused-${Backend.name}-${flagName}`});
      for (const [name, input] of Object.entries(createInputs(0)))
        graph.addInput(name, gpuGraph.vector(input.format, {length: input.values.length}));
      const operation = new Backend(props);
      graph.add(operation);
      const compiled = graph.compile<Record<string, CPUVector>>(device, {
        outputs,
        outputLocations: Object.fromEntries(outputs.map(name => [name, 'cpu' as const])),
        lowerings: [GPU_CONTRIBUTOR_OPERATION_LOWERING]
      });
      try {
        for (const incomplete of [0, 1, 0]) {
          const values = await compiled.run({
            inputs: Object.fromEntries(
              Object.entries(createInputs(incomplete)).map(([name, input]) => [
                name,
                makeCPUVector(input.format, input.values)
              ])
            )
          });
          const result = Object.fromEntries(
            outputs.map(name => {
              expect(values[name]).toBeInstanceOf(CPUVector);
              return [name, getCPUVectorRows(values[name])];
            })
          );
          expect(Array.from(result['labels'])).toEqual(
            incomplete ? [0xffffffff, 0xffffffff, 0xffffffff] : [0, 0, 0]
          );
          expect(Array.from(result['regionCount'])).toEqual([incomplete ? 0 : 1]);
          expect(Array.from(result['cutEdges'])).toEqual([0xffffffff, 0xffffffff]);
          expect(Array.from(result['cutGains'])).toEqual(incomplete ? [NaN, NaN] : [0, 0]);
          if (!incomplete)
            for (const name of outputs) expect(result[name]).toEqual(unflagged[name]);
        }
      } finally {
        compiled.destroy();
        if (operation instanceof GPUSkaterRegionsOperation) operation.destroy();
      }
    }
  });

for (const cause of ['overflow', 'candidateOverflow'] as const)
  it(`final3 SKATER operations suppress ${cause} before missing-label traversal and recover`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: SkaterRegionsOperationProps = {
      rowCount: 2,
      edgeCount: 2,
      maximumRegionCount: 2,
      weights: {
        offsets: 'offsets',
        neighbors: 'neighbors',
        weights: 'weights',
        status: {[cause]: 'loss'}
      },
      values: 'values',
      parameters: 'parameters',
      treeEdgeFlags: 'treeEdgeFlags',
      componentLabels: 'componentLabels',
      output: {
        labels: 'labels',
        regionCount: 'regionCount',
        cutEdges: 'cutEdges',
        cutGains: 'cutGains'
      }
    };
    for (const Backend of [GPUSkaterRegionsOperation, CPUSkaterRegionsOperation]) {
      for (const flag of [0, 1, 0]) {
        const result = await runOperationGraph(device, {
          inputs: {
            offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2)},
            neighbors: {format: 'uint32', values: Uint32Array.of(1, 0)},
            weights: {format: 'float32', values: Float32Array.of(1, 1)},
            values: {format: 'float32', values: Float32Array.of(0, 2)},
            parameters: {format: 'uint32', values: Uint32Array.of(2, 1)},
            treeEdgeFlags: {format: 'uint32', values: Uint32Array.of(flag ? 0 : 1, 0)},
            componentLabels: {
              format: 'uint32',
              values: flag ? Uint32Array.of(0xffffffff, 0xffffffff) : Uint32Array.of(0, 0)
            },
            loss: {format: 'uint32', values: Uint32Array.of(flag)}
          },
          operations: [new Backend(props)],
          outputs: ['labels', 'regionCount', 'cutEdges', 'cutGains']
        });
        expect(Array.from(result['labels'])).toEqual(flag ? [0xffffffff, 0xffffffff] : [0, 1]);
        expect(Array.from(result['regionCount'])).toEqual([flag ? 0 : 2]);
        expect(Array.from(result['cutEdges'])).toEqual([flag ? 0xffffffff : 0]);
        if (flag) expect(Number.isNaN(result['cutGains'][0])).toBe(true);
        else {
          expect(Number.isFinite(result['cutGains'][0])).toBe(true);
          expect(result['cutGains'][0]).toBeCloseTo(2, 5);
        }
      }
    }
  });

for (const scene of [
  'ordinary',
  'size-floor',
  'no-cuts',
  'zero-gain-tie',
  'invalid-tree',
  'incomplete-tree'
] as const)
  it(`SKATER CPU/GPU compare every output: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: SkaterRegionsOperationProps = {
      rowCount: 8,
      edgeCount: 12,
      columnCount: 2,
      maximumRegionCount: 5,
      weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      treeEdgeFlags: 'treeEdgeFlags',
      componentLabels: 'componentLabels',
      values: 'values',
      parameters: 'parameters',
      treeStatus: 'treeStatus',
      output: {
        labels: 'labels',
        regionCount: 'regionCount',
        cutEdges: 'cutEdges',
        cutGains: 'cutGains'
      }
    };
    const treeStatus = new Uint32Array(GPU_MINIMUM_SPANNING_TREE_STATUS_LAYOUT.length);
    if (scene === 'invalid-tree')
      treeStatus[GPU_MINIMUM_SPANNING_TREE_STATUS_LAYOUT.invalidCount] = 1;
    if (scene === 'incomplete-tree')
      treeStatus[GPU_MINIMUM_SPANNING_TREE_STATUS_LAYOUT.candidateOverflow] = 9;
    const inputs: Record<string, OperationGraphInput> = {
      offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 3, 5, 6, 7, 9, 11, 12)},
      neighbors: {format: 'uint32', values: Uint32Array.of(1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 7, 6)},
      weights: {format: 'float32', values: new Float32Array(12).fill(1)},
      treeEdgeFlags: {format: 'uint32', values: Uint32Array.of(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0)},
      componentLabels: {format: 'uint32', values: Uint32Array.of(0, 0, 0, 0, 4, 4, 4, 4)},
      values: {
        format: 'float32',
        values:
          scene === 'zero-gain-tie'
            ? new Float32Array(16)
            : Float32Array.of(0, 0, 0, 0, 10, 2, 10, 2, 1, -1, 1, -1, 5, 3, 5, 3)
      },
      parameters: {
        format: 'uint32',
        values: getGPUSkaterRegionsParameterValues({
          regionCount: scene === 'no-cuts' ? 1 : scene === 'ordinary' ? 4 : 5,
          minimumRegionSize: scene === 'size-floor' ? 2 : 1
        })
      },
      treeStatus: {format: 'uint32', values: treeStatus}
    };
    const outputs = Object.values(props.output) as string[];
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUSkaterRegionsOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUSkaterRegionsOperation(props)],
      outputs
    });
    for (const output of outputs) {
      expect(gpu[output].length).toBe(cpu[output].length);
      for (let slot = 0; slot < cpu[output].length; slot++) {
        const expected = cpu[output][slot];
        if (Number.isNaN(expected)) expect(Number.isNaN(gpu[output][slot])).toBe(true);
        else if (output !== 'cutGains' || expected === 0)
          expect(gpu[output][slot] === expected, `${output}[${slot}]`).toBe(true);
        // Gains are positive; relative tolerance 2e-5 cannot let a small result pass as zero.
        else expect(Math.abs(gpu[output][slot] - expected) / expected).toBeLessThan(2e-5);
      }
    }
  });

for (const scene of ['valid', 'attainable-cap', 'rowCount', 'transform', 'symmetric'] as const)
  it(`SKATER review metadata regression: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const information = {
      rowCount: scene === 'rowCount' ? 3 : 2,
      transform: scene === 'transform' ? ('invalid' as 'raw') : ('raw' as const),
      symmetric: scene === 'symmetric' ? ('invalid' as 'unknown') : true
    };
    const props: SkaterRegionsOperationProps = {
      id: 'review-skater',
      rowCount: 2,
      edgeCount: 2,
      maximumRegionCount: scene === 'attainable-cap' ? 8 : 2,
      weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights', information},
      treeEdgeFlags: 'treeEdgeFlags',
      componentLabels: 'componentLabels',
      values: 'values',
      parameters: 'parameters',
      output: {
        labels: 'labels',
        regionCount: 'regionCount',
        cutEdges: 'cutEdges',
        cutGains: 'cutGains'
      }
    };
    const inputs: Record<string, OperationGraphInput> = {
      offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2)},
      neighbors: {format: 'uint32', values: Uint32Array.of(1, 0)},
      weights: {format: 'float32', values: Float32Array.of(1, 1)},
      treeEdgeFlags: {format: 'uint32', values: Uint32Array.of(1, 0)},
      componentLabels: {format: 'uint32', values: Uint32Array.of(0, 0)},
      values: {format: 'float32', values: Float32Array.of(0, 2)},
      parameters: {
        format: 'uint32',
        values: getGPUSkaterRegionsParameterValues({regionCount: 2, minimumRegionSize: 1})
      }
    };
    for (const createOperation of [
      () => new GPUSkaterRegionsOperation(props),
      () => new CPUSkaterRegionsOperation(props)
    ]) {
      const run = async () =>
        runOperationGraph(device, {
          inputs,
          operations: [createOperation()],
          outputs: ['labels', 'regionCount', 'cutEdges', 'cutGains']
        });
      if (scene !== 'valid' && scene !== 'attainable-cap')
        await expect(run()).rejects.toThrow(
          new RegExp(`^review-skater weights\\.information\\.${scene}`)
        );
      else {
        const result = await run();
        expect(Array.from(result['labels'])).toEqual([0, 1]);
        expect(Array.from(result['regionCount'])).toEqual([2]);
        expect(Array.from(result['cutEdges'])).toEqual(
          scene === 'attainable-cap' ? [0, ...new Array(6).fill(0xffffffff)] : [0]
        );
        // SSD of [0,2] is 2, and each singleton has zero SSD. Relative tolerance 2e-5.
        expect(result['cutGains'].length).toBe(scene === 'attainable-cap' ? 7 : 1);
        expect(Math.abs(result['cutGains'][0] - 2) / 2).toBeLessThan(2e-5);
        expect(
          Array.from(result['cutGains'])
            .slice(1)
            .every(value => value === 0)
        ).toBe(true);
      }
    }
  });
