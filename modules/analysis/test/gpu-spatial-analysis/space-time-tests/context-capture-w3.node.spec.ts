// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUNearRepeatKnox} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-near-repeat-knox';
import {GPUMantelTest} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-mantel-test';
import {
  GPUMantelTestOperation,
  type MantelTestOperationProps
} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-mantel-test-operation';
import {CPUMantelTestOperation} from '../../../src/gpu-spatial-analysis/space-time-tests/cpu-mantel-test-operation';
import {getGPUSpaceTimeParameterValues} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-parameters';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';

it('W2-14 near-repeat already captures edges, pairs, provenance, status and nested temporal metadata', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `near-repeat-${serial++}`, format, length);
  const props = {
    pairs: {
      offsets: view('uint32', 3),
      neighbors: view('uint32', 1),
      weights: view('float32', 1),
      distances: view('float32', 1),
      status: {overflow: view('uint32', 1)},
      information: {
        rowCount: 2,
        transform: 'raw' as const,
        symmetric: false,
        hasSelfWeights: false,
        provenance: {source: 'neighbor-search' as const, radius: 5}
      }
    },
    times: view('float32', 2),
    parameters: view('uint32', 4),
    maximumPermutations: 3,
    distanceEdges: [0, 5],
    timeEdges: [0, 5],
    temporalContext: {
      unit: 'second' as const,
      epoch: 0n,
      encoding: 'float32-relative' as const,
      axis: {kind: 'regular' as const, step: {calendarUnit: 'day' as const, interval: 1}}
    },
    output: {counts: view('uint32', 4), summary: view('float32', 9)}
  };
  try {
    const contributor = new GPUNearRepeatKnox(props);
    const originalLoss = props.pairs.status.overflow;
    props.pairs.status.overflow = view('uint32', 1);
    props.pairs.information.provenance.radius = 99;
    props.distanceEdges[1] = 99;
    props.timeEdges[1] = 99;
    props.maximumPermutations = 99;
    props.temporalContext.axis.step.interval = 99;
    expect(contributor.props.pairs.status?.overflow).toBe(originalLoss);
    expect(contributor.props.pairs.information?.provenance?.radius).toBe(5);
    expect(contributor.props.distanceEdges).toEqual([0, 5]);
    expect(contributor.props.timeEdges).toEqual([0, 5]);
    expect(contributor.props.maximumPermutations).toBe(3);
    expect(contributor.temporalContext?.axis).toEqual({
      kind: 'regular',
      step: {calendarUnit: 'day', interval: 1}
    });
    expect(contributor.getCommandNodes(graph).length).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});

it('W2-14 Mantel forms capture temporal context and output names while CPU data remains live', async () => {
  const props: MantelTestOperationProps = {
    rowCount: 3,
    pairs: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights', distances: 'distances'},
    times: 'times',
    parameters: 'parameters',
    maximumPermutations: 3,
    temporalContext: {
      unit: 'second',
      epoch: 0n,
      encoding: 'float32-relative',
      axis: {kind: 'regular', step: {calendarUnit: 'day', interval: 1}}
    },
    output: {statistics: 'statistics', summary: 'summary', permutationCount: 'used'}
  };
  const operation = new GPUMantelTestOperation(props);
  const cpu = new CPUMantelTestOperation(props);
  props.temporalContext!.epoch = 99n;
  if (
    props.temporalContext!.axis.kind === 'regular' &&
    typeof props.temporalContext!.axis.step === 'object'
  )
    props.temporalContext!.axis.step.interval = 99;
  props.output.summary = 'changed';
  const contributor = constructNetworkOperationContributor(operation, {
    offsets: {format: 'uint32', length: 4},
    neighbors: {format: 'uint32', length: 3},
    weights: {format: 'float32', length: 3},
    distances: {format: 'float32', length: 3},
    times: {format: 'float32', length: 3},
    parameters: {format: 'uint32', length: 4}
  }) as GPUMantelTest;
  expect(contributor.props.temporalContext).toMatchObject({epoch: 0n, axis: {step: {interval: 1}}});
  const values = {
    offsets: Uint32Array.of(0, 2, 3, 3),
    neighbors: Uint32Array.of(1, 2, 2),
    weights: Float32Array.of(1, 1, 1),
    distances: Float32Array.of(1, 2, 4),
    times: Float32Array.of(0, 1, 3),
    parameters: getGPUSpaceTimeParameterValues({seed: 0, permutationCount: 3})
  };
  const vector = (words: Float32Array | Uint32Array) => ({
    format: words instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
    length: words.length,
    chunks: [{values: words, length: words.length}],
    pack: () => words
  });
  const inputs = Object.fromEntries(
    Object.entries(values).map(([name, words]) => [name, vector(words)])
  );
  // Closed-form centered cross product is +/-1, square sums 14/3 and 2.
  const firstStatistics = (await cpu.execute({inputs}))['statistics'];
  if (!(firstStatistics instanceof Float32Array))
    throw new Error('expected packed Float32Array statistics');
  expect(firstStatistics[0]).toBe(Math.fround(Math.sqrt(3 / 28)));
  values.times.set([0, 3, 1]);
  const replayStatistics = (await cpu.execute({inputs}))['statistics'];
  if (!(replayStatistics instanceof Float32Array))
    throw new Error('expected packed Float32Array replay statistics');
  expect(replayStatistics[0]).toBe(Math.fround(-Math.sqrt(3 / 28)));
  expect(cpu.outputs).toHaveProperty('summary');
});
