// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUIsobands} from '../../../src/raster-analysis/isolines/gpu-isobands';
import {GPUIsolines} from '../../../src/raster-analysis/isolines/gpu-isolines';
import {GPU_ISOLINES_PARAMETER_LENGTH} from '../../../src/raster-analysis/isolines/isolines-parameters';
import {GPURasterReclassify} from '../../../src/raster-analysis/raster-algebra/gpu-raster-reclassify';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

type MaskedContributor = {
  props: {mask?: GraphDataView<'uint32'>};
};

/**
 * Checks contract rule 7 on one contributor: `mask` is the input row filter, the contributor
 * keeps the view it was given, and a short mask is rejected with a message that starts with the
 * contributor ID and names `mask`.
 */
function expectInputMaskContract(
  id: string,
  mask: GraphDataView<'uint32'>,
  shortMask: GraphDataView<'uint32'>,
  create: (overrides: {id?: string; mask?: GraphDataView<'uint32'>}) => MaskedContributor
): void {
  expect(create({mask}).props.mask).toBe(mask);
  expect(() => create({mask: shortMask})).toThrow(new RegExp(`^${id} mask`));
}

it('isolines, isobands and reclassify take the contract mask', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32' | 'float32x4' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const grid = makeGPURasterMetadata({width: 4, height: 3});
  const mask = view('uint32', 12);
  const shortMask = view('uint32', 11);
  const values = view('float32', 12);

  expectInputMaskContract(
    'isolines',
    mask,
    shortMask,
    overrides =>
      new GPUIsolines({
        grid,
        values,
        levels: view('float32', 3),
        parameters: view('float32', GPU_ISOLINES_PARAMETER_LENGTH),
        output: {
          segments: view('float32x4', 40),
          segmentLevels: view('uint32', 40),
          status: {overflow: view('uint32', 1), count: view('uint32', 1)}
        },
        ...overrides
      })
  );
  expectInputMaskContract(
    'isobands',
    mask,
    shortMask,
    overrides =>
      new GPUIsobands({
        grid,
        values,
        breaks: view('float32', 3),
        parameters: view('float32', 12),
        output: {bandClasses: view('uint32', 12)},
        ...overrides
      })
  );
  expectInputMaskContract(
    'raster-reclassify',
    mask,
    shortMask,
    overrides =>
      new GPURasterReclassify({
        values,
        breaks: view('float32', 4),
        parameters: view('float32', 8),
        output: {classes: view('uint32', 12)},
        ...overrides
      })
  );
  device.destroy();
});
