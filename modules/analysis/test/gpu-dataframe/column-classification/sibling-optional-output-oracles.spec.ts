// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUClassBreaks} from '../../../src/dataframe-analysis/column-classification/gpu-class-breaks';
import {getGPUClassBreaksParameterValues} from '../../../src/dataframe-analysis/column-classification/class-breaks-parameters';
import {GPUColumnQuantiles} from '../../../src/dataframe-analysis/column-classification/gpu-column-quantiles';
import {getGPUColumnQuantilesParameterValues} from '../../../src/dataframe-analysis/column-classification/column-quantiles-parameters';
import {GPUScalarScale} from '../../../src/dataframe-analysis/column-classification/gpu-scalar-scale';
import {prepareGPUScalarScaleControls} from '../../../src/dataframe-analysis/column-classification/scalar-scale-parameters';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// Hand oracle: valid sorted sample is [0, 2, 4, 6]. R-7 quantiles at 0, 1/2, 1
// are [0, 3, 6]. Percentile interval [1/4, 3/4] selects ranks 1..2, values [2, 4].
// Three equal intervals have edges [0, 2, 4, 6] and lower-edge counts [1, 1, 2].
// Linear mapping 10 + 2*x yields [10, 14, 18, 22]; missing/masked rows map to -9.
// All arithmetic on this fixture is exactly representable in binary32.
function createFixture(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  function view<Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    values: Float32Array | Uint32Array
  ) {
    const buffer = device.createBuffer({
      data: values,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, values.length)};
  }
  const values = view('values', 'float32', Float32Array.of(0, 2, 4, 6, NaN, 100)).view;
  const mask = view('mask', 'uint32', Uint32Array.of(1, 1, 1, 1, 1, 0)).view;
  return {graph, buffers, view, values, mask};
}

it.each([
  false,
  true
])('class-breaks optional counts=%s preserve literal edges and count', async includeCounts => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(device);
  const {graph, view, values, mask} = fixture;
  const breaks = view('breaks', 'float32', new Float32Array(5).fill(99));
  const classCount = view('class-count', 'uint32', Uint32Array.of(99));
  const counts = view('counts', 'uint32', new Uint32Array(4).fill(99));
  graph.add(
    new GPUClassBreaks({
      values,
      mask,
      parameters: view(
        'parameters',
        'float32',
        getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 3}, 4)
      ).view,
      maximumClassCount: 4,
      methods: ['equal-interval'],
      output: {
        breaks: breaks.view,
        classCount: classCount.view,
        classCounts: includeCounts ? counts.view : undefined
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const edges = await readFloat32(breaks.buffer, 5);
    expect(edges.slice(0, 4)).toEqual([0, 2, 4, 6]);
    expect(Number.isNaN(edges[4])).toBe(true);
    expect(await readUint32(classCount.buffer, 1)).toEqual([3]);
    expect(await readUint32(counts.buffer, 4)).toEqual(
      includeCounts ? [1, 1, 2, 0] : [99, 99, 99, 99]
    );
  } finally {
    compiled.destroy();
    for (const buffer of fixture.buffers) buffer.destroy();
  }
});

it.each([
  {mask: false, bounds: false},
  {mask: true, bounds: false},
  {mask: false, bounds: true},
  {mask: true, bounds: true}
])('column-quantiles independent optional filter outputs: %j', async subset => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(device);
  const {graph, view, values, mask} = fixture;
  const quantiles = view('quantiles', 'float32', new Float32Array(3).fill(99));
  const count = view('count', 'uint32', Uint32Array.of(99));
  const filterMask = view('filter-mask', 'uint32', new Uint32Array(6).fill(99));
  const bounds = view('bounds', 'float32', new Float32Array(2).fill(99));
  graph.add(
    new GPUColumnQuantiles({
      values,
      mask,
      parameters: view(
        'parameters',
        'float32',
        getGPUColumnQuantilesParameterValues({quantiles: [0, 0.5, 1], filterRange: [0.25, 0.75]})
      ).view,
      quantileCount: 3,
      output: {
        quantiles: quantiles.view,
        validCount: count.view,
        outputMask: subset.mask ? filterMask.view : undefined,
        filterBounds: subset.bounds ? bounds.view : undefined
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readFloat32(quantiles.buffer, 3)).toEqual([0, 3, 6]);
    expect(await readUint32(count.buffer, 1)).toEqual([4]);
    expect(await readUint32(filterMask.buffer, 6)).toEqual(
      subset.mask ? [0, 1, 1, 0, 0, 0] : [99, 99, 99, 99, 99, 99]
    );
    expect(await readFloat32(bounds.buffer, 2)).toEqual(subset.bounds ? [2, 4] : [99, 99]);
  } finally {
    compiled.destroy();
    for (const buffer of fixture.buffers) buffer.destroy();
  }
});

it.each([
  false,
  true
])('scalar-scale optional validity=%s preserves literal mapped values', async includeValidity => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(device);
  const {graph, view, values, mask} = fixture;
  const controls = prepareGPUScalarScaleControls({
    scale: 'linear',
    domain: [0, 8],
    range: [10, 26],
    nullValue: -9
  });
  const mapped = view('mapped', 'float32', new Float32Array(6).fill(99));
  const validity = view('validity', 'uint32', new Uint32Array(6).fill(99));
  graph.add(
    new GPUScalarScale({
      values,
      mask,
      domain: view('domain', 'float32', controls.domain).view,
      range: view('range', 'float32', controls.range).view,
      parameters: view('parameters', 'float32', controls.parameters).view,
      maximumDomainCount: 2,
      output: {values: mapped.view, validity: includeValidity ? validity.view : undefined}
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readFloat32(mapped.buffer, 6)).toEqual([10, 14, 18, 22, -9, -9]);
    expect(await readUint32(validity.buffer, 6)).toEqual(
      includeValidity ? [1, 1, 1, 1, 0, 0] : [99, 99, 99, 99, 99, 99]
    );
  } finally {
    compiled.destroy();
    for (const buffer of fixture.buffers) buffer.destroy();
  }
});
