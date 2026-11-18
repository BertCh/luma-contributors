// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUPopulationSupportAlignmentOperation} from '../../../src/gpu-spatial-analysis/landscape-ecology/cpu-population-support-alignment-operation';
import {
  GPUPopulationSupportAlignmentOperation,
  type PopulationSupportAlignmentOperationProps
} from '../../../src/gpu-spatial-analysis/landscape-ecology/gpu-population-support-alignment-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {GPUColumnQuantilesOperation} from '../../../src/dataframe-analysis/column-classification/gpu-column-quantiles-operation';
import {getGPUColumnQuantilesParameterValues} from '../../../src/dataframe-analysis/column-classification/column-quantiles-parameters';

const CASES = [
  {
    name: 'ordinary reorder',
    sources: [3, 1, 2],
    targets: [2, 3, 1],
    codes: [0, 0, 0],
    status: [1, 0, 0, 0]
  },
  {
    name: 'source duplicates',
    sources: [1, 1, 2],
    targets: [1, 2],
    codes: [2, 0],
    status: [0, 0, 1, 0]
  },
  {
    name: 'target duplicates and excluded source',
    sources: [1, 2, 3],
    targets: [1, 1, 2],
    codes: [2, 2, 0],
    status: [0, 0, 2, 1]
  },
  {
    name: 'missing precedes target duplicates',
    sources: [1, 2, 3],
    targets: [4, 4, 2],
    codes: [1, 1, 0],
    status: [0, 2, 0, 2]
  },
  {
    name: 'full uint32 range',
    sources: [0xffffffff, 0, 0xfffffffe],
    targets: [0, 0xfffffffe, 0xffffffff],
    codes: [0, 0, 0],
    status: [1, 0, 0, 0]
  }
];

for (const scene of CASES) {
  it(`population support CPU/GPU operations preserve every output: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: PopulationSupportAlignmentOperationProps = {
      populations: 'input',
      sourceIds: 'sources',
      targetIds: 'targets',
      targetCount: scene.targets.length,
      output: {
        populations: 'populations',
        alignmentCode: 'alignmentCode',
        matchedMask: 'matchedMask',
        status: 'status'
      }
    };
    const inputs = {
      input: {format: 'float32x2' as const, values: Float32Array.of(6, 4, NaN, Infinity, -1, 0)},
      sources: {format: 'uint32' as const, values: Uint32Array.from(scene.sources)},
      targets: {format: 'uint32' as const, values: Uint32Array.from(scene.targets)}
    };
    const outputs = Object.values(props.output);
    const gpu = await runOperationGraph(device, {
      inputs,
      outputs,
      operations: [new GPUPopulationSupportAlignmentOperation(props)]
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      outputs,
      operations: [new CPUPopulationSupportAlignmentOperation(props)]
    });
    for (const name of outputs) expect(Array.from(gpu[name]), name).toEqual(Array.from(cpu[name]));
    expect(Array.from(cpu['alignmentCode'])).toEqual(scene.codes);
    expect(Array.from(cpu['matchedMask'])).toEqual(scene.codes.map(code => Number(code === 0)));
    expect(Array.from(cpu['status'])).toEqual(scene.status);
  });
}

it('routes a CPU alignment through a GPU alignment in one mixed program across partial groups', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const count = 257;
  const sources = Uint32Array.from({length: count}, (_, row) => row);
  const targets = sources.slice().reverse();
  const populations = Float32Array.from({length: 2 * count}, (_, component) => component / 2);
  const result = await runOperationGraph(device, {
    inputs: {
      input: {format: 'float32x2', values: populations},
      sources: {format: 'uint32', values: sources},
      targets: {format: 'uint32', values: targets}
    },
    operations: [
      new CPUPopulationSupportAlignmentOperation({
        id: 'cpu-reorder',
        populations: 'input',
        sourceIds: 'sources',
        targetIds: 'targets',
        targetCount: count,
        output: {
          populations: 'aligned',
          alignmentCode: 'cpuAlignmentCode',
          matchedMask: 'cpuMatchedMask',
          status: 'cpuStatus'
        }
      }),
      new GPUPopulationSupportAlignmentOperation({
        id: 'gpu-restore',
        populations: 'aligned',
        sourceIds: 'targets',
        targetIds: 'sources',
        targetCount: count,
        output: {
          populations: 'restored',
          alignmentCode: 'gpuAlignmentCode',
          matchedMask: 'gpuMatchedMask',
          status: 'gpuStatus'
        }
      })
    ],
    outputs: [
      'aligned',
      'restored',
      'cpuAlignmentCode',
      'gpuAlignmentCode',
      'cpuMatchedMask',
      'gpuMatchedMask',
      'cpuStatus',
      'gpuStatus'
    ]
  });
  expect(Array.from(result['restored'])).toEqual(Array.from(populations));
  expect(Array.from(result['aligned'])).toEqual(
    Array.from(targets).flatMap(row => [populations[2 * row], populations[2 * row + 1]])
  );
  for (const prefix of ['cpu', 'gpu']) {
    expect(Array.from(result[`${prefix}AlignmentCode`])).toEqual(Array(count).fill(0));
    expect(Array.from(result[`${prefix}MatchedMask`])).toEqual(Array(count).fill(1));
    expect(Array.from(result[`${prefix}Status`])).toEqual([1, 0, 0, 0]);
  }
});

for (const AlignmentOperation of [
  GPUPopulationSupportAlignmentOperation,
  CPUPopulationSupportAlignmentOperation
]) {
  it.each([
    {targets: [7, 8], selected: [1, 0], count: 1, median: 10},
    {targets: [7, 9], selected: [1, 1], count: 2, median: 54.5},
    {targets: [7, 7], selected: [0, 0], count: 0, median: NaN}
  ])(`${AlignmentOperation.name} feeds matched rows to GPU quantiles: $targets`, async scene => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const result = await runOperationGraph(device, {
      inputs: {
        populations: {format: 'float32x2', values: Float32Array.of(1, 2, 3, 4)},
        sources: {format: 'uint32', values: Uint32Array.of(7, 9)},
        targets: {format: 'uint32', values: Uint32Array.from(scene.targets)},
        values: {format: 'float32', values: Float32Array.of(10, 99)},
        parameters: {
          format: 'float32',
          values: getGPUColumnQuantilesParameterValues({quantiles: [0.5]})
        }
      },
      operations: [
        new AlignmentOperation({
          populations: 'populations',
          sourceIds: 'sources',
          targetIds: 'targets',
          targetCount: 2,
          output: {
            populations: 'aligned',
            alignmentCode: 'codes',
            matchedMask: 'matched',
            status: 'status'
          }
        }),
        new GPUColumnQuantilesOperation({
          values: 'values',
          mask: 'matched',
          parameters: 'parameters',
          rowCount: 2,
          quantileCount: 1,
          output: {quantiles: 'median', validCount: 'count', outputMask: 'selected'}
        })
      ],
      outputs: ['median', 'count', 'selected']
    });
    expect(Array.from(result['selected'])).toEqual(scene.selected);
    expect(Array.from(result['count'])).toEqual([scene.count]);
    expect(Array.from(result['median'])).toEqual([scene.median]);
  });
}
