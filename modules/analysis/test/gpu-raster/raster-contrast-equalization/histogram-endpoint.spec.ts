// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPURasterContrastEqualization,
  getGPURasterContrastEqualizationParameterValues
} from '../../../src/raster-analysis/raster-contrast-equalization/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  getGPUCommandNodeKernelSources,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('WGS-08 sends the 2^127 upper endpoint to bin 255 before CDF mapping', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers = [
    createInputBuffer(device, Float32Array.of(2 ** 127)),
    createInputBuffer(
      device,
      getGPURasterContrastEqualizationParameterValues({inputDomain: [0, 2 ** 127]})
    ),
    createOutputBuffer(device, 1),
    createOutputBuffer(device, 1),
    createOutputBuffer(device, 1),
    createOutputBuffer(device, 1)
  ];
  const grid = {
    width: 1,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'point' as const
  };
  const input = {
    id: 'source',
    format: 'float32' as const,
    storage: {
      kind: 'buffer' as const,
      values: importGraphBuffer(graph, 'input', buffers[0], 'float32', 1)
    }
  };
  const parameters = importGraphBuffer(graph, 'parameters', buffers[1], 'float32', 8);
  const output = {
    values: importGraphBuffer(graph, 'values', buffers[2], 'float32', 1),
    validity: importGraphBuffer(graph, 'validity', buffers[3], 'uint32', 1),
    status: {approximation: importGraphBuffer(graph, 'approximation', buffers[5], 'uint32', 1)}
  };
  const nodes = new GPURasterContrastEqualization({
    grid,
    input,
    parameters,
    binCount: 256,
    mode: 'match-histograms',
    reference: {grid, input},
    output
  }).getCommandNodes(graph);
  // Execute the production getBin declaration in isolation: a CDF mapping can conceal binning errors.
  const source = getGPUCommandNodeKernelSources(
    nodes.filter(node => node.id.endsWith('-source-histogram'))
  )[0].source;
  const declarations = source.slice(
    source.indexOf('fn getScaledQuotient'),
    source.indexOf('fn reflectCell')
  );
  const bins = importGraphBuffer(graph, 'bins', buffers[4], 'uint32', 1);
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'bin-probe',
      operation: 'HistogramEndpointProbe',
      invocationCount: 1,
      declarations,
      bindings: [
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        {name: 'bins', view: bins, type: 'u32', access: 'read_write'}
      ],
      body: 'bins[binsOffset+index]=getBin(parameters[parametersOffset+1u],0u);'
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(buffers[4], 1)).toEqual([255]);
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
