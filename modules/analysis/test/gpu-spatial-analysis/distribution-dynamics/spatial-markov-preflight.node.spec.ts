// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialMarkov} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-spatial-markov';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('VAL-08 rejects periodLag before graph reservations and permits corrected same-ID reuse', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `preflight-${serial++}`, format, length);
  const props = {
    id: 'retry',
    values: view('float32', 24),
    classes: view('uint32', 24),
    classCount: 2,
    weights: {
      offsets: view('uint32', 9),
      neighbors: view('uint32', 6),
      weights: view('float32', 6)
    },
    periodCount: 3,
    maximumLagClassCount: 2,
    lagParameters: view('float32', 11),
    output: {
      counts: view('uint32', 8),
      lagBreaks: view('float32', 3),
      lagClassCount: view('uint32', 1)
    }
  };
  expect(() => new GPUSpatialMarkov({...props, periodLag: -1})).toThrow(/periodLag/);
  expect(() => new GPUSpatialMarkov({...props, periodLag: 1}).getCommandNodes(graph)).not.toThrow();
  device.destroy();
});

it('B7 rejects optional strided lag outputs before graph.add reserves resources', () => {
  for (const field of ['lagClasses', 'lagValues'] as const) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `strided-input-${serial++}`, format, length);
    const props = {
      id: 'strided',
      values: view('float32', 15),
      classes: view('uint32', 15),
      classCount: 2,
      periodCount: 3,
      maximumLagClassCount: 2,
      lagParameters: view('float32', 11),
      weights: {
        offsets: view('uint32', 6),
        neighbors: view('uint32', 8),
        weights: view('float32', 8)
      },
      output: {
        counts: view('uint32', 8),
        lagBreaks: view('float32', 3),
        lagClassCount: view('uint32', 1)
      }
    };
    const badOutput =
      field === 'lagClasses'
        ? {
            ...props.output,
            lagClasses: {...view('uint32', 30), length: 15, stride: 2, byteStride: 8}
          }
        : {
            ...props.output,
            lagValues: {...view('float32', 30), length: 15, stride: 2, byteStride: 8}
          };
    expect(() => graph.add(new GPUSpatialMarkov({...props, output: badOutput}))).toThrow();
    const corrected =
      field === 'lagClasses'
        ? {...props.output, lagClasses: view('uint32', 15)}
        : {...props.output, lagValues: view('float32', 15)};
    expect(() => graph.add(new GPUSpatialMarkov({...props, output: corrected}))).not.toThrow();
    device.destroy();
  }
});
