// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, GPUMask, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, test} from 'vitest';
import {
  captureGraphCommandNodes,
  getGraphViewChunks,
  importGraphBuffer,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs,
  validateCompactOutput
} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {type GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {GPUPairwisePointDistance} from '@luma.gl/graph/geospatial';
import {createNullWebGPUDevice} from './gpu-contributor-test-utils';
import * as analysisCore from '../../src/analysis-core/index';
import {validateNetworkOutputsDisjointFromInputs} from '../../src/gpu-network/network-view-utils';
import {validateInterpolationOutputsDisjointFromInputs} from '../../src/gpu-spatial-analysis/spatial-interpolation/interpolation-view-validation';
import {validateIsolinesAliasing} from '../../src/raster-analysis/isolines/isolines-utils';
import {validateRasterAlgebraAliasing} from '../../src/raster-analysis/raster-algebra/raster-algebra-utils';
import {GPURasterArithmetic} from '../../src/raster-analysis/raster-algebra/gpu-raster-arithmetic';
import {GPURasterSampling} from '../../src/raster-analysis/raster-sampling/gpu-raster-sampling';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '../../src/raster-analysis/raster-grid/index';

test.each([
  ['isolines', validateIsolinesAliasing],
  ['raster algebra', validateRasterAlgebraAliasing]
] as const)('%s rejects physical aliases for inputs and writable outputs', (_family, validate) => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const first = device.createBuffer({byteLength: 16, usage: Buffer.STORAGE});
  const second = device.createBuffer({byteLength: 16, usage: Buffer.STORAGE});
  try {
    const input = importGraphBuffer(graph, 'first-read', first, 'float32', 1);
    const alias = importGraphBuffer(graph, 'first-write', first, 'float32', 1);
    const distinct = importGraphBuffer(graph, 'second', second, 'float32', 1);
    expect.soft(() => validate('physical', [alias], [input])).toThrow(/share buffers/);
    expect.soft(() => validate('physical', [input, alias], [])).toThrow(/share buffers/);
    expect(() => validate('physical', [distinct], [input])).not.toThrow();
    expect(() => validate('physical', [input, distinct], [])).not.toThrow();
    const parameters = createTransientView(graph, 'parameters', 'float32', 8);
    expect
      .soft(
        () => new GPURasterArithmetic({cellCount: 1, a: input, parameters, output: {values: alias}})
      )
      .toThrow(/share buffers/);
    expect(
      () =>
        new GPURasterArithmetic({cellCount: 1, a: input, parameters, output: {values: distinct}})
    ).not.toThrow();
  } finally {
    first.destroy();
    second.destroy();
    device.destroy();
  }
});

test('sampling rejects two writable ports imported from one physical word', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const buffer = device.createBuffer({byteLength: 4, usage: Buffer.STORAGE});
  try {
    const values = importGraphBuffer(graph, 'values', buffer, 'float32', 1);
    const validity = importGraphBuffer(graph, 'validity', buffer, 'uint32', 1);
    const props = {
      grid: makeGPURasterMetadata({width: 1, height: 1}),
      values: createTransientView(graph, 'input', 'float32', 1),
      positions: createTransientView(graph, 'positions', 'float32x2', 1),
      parameters: createTransientView(graph, 'parameters', 'float32', 12)
    };
    expect(
      () => new GPURasterSampling({...props, output: {values, outputValidity: validity}})
    ).toThrow(/output.values.*output.outputValidity/);
    expect(
      () =>
        new GPURasterSampling({
          ...props,
          output: {
            values,
            outputValidity: createTransientView(graph, 'distinct-validity', 'uint32', 1)
          }
        })
    ).not.toThrow();
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

test.each([
  'bare',
  'band validity'
] as const)('arithmetic diagnostics use public prop paths: %s', kind => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const values = createTransientView(graph, 'values', 'float32', 1);
    const validity = graph.createDataView(values.buffer, {format: 'uint32', length: 1});
    const grid = makeGPURasterMetadata({width: 1, height: 1});
    const a =
      kind === 'bare'
        ? values
        : makeGPURasterBand(createTransientView(graph, 'band-values', 'float32', 1), grid, {
            validity
          });
    expect(
      () =>
        new GPURasterArithmetic({
          grid,
          cellCount: 1,
          a,
          parameters: createTransientView(graph, 'parameters', 'float32', 8),
          output: {values}
        })
    ).toThrow(`output.values conflicts with ${kind === 'bare' ? 'a' : 'a.validity'}`);
  } finally {
    device.destroy();
  }
});

test.each([
  ['network', validateNetworkOutputsDisjointFromInputs],
  ['interpolation', validateInterpolationOutputsDisjointFromInputs],
  ['isolines', validateIsolinesAliasing],
  ['raster algebra', validateRasterAlgebraAliasing]
] as const)('%s buffer diagnostics identify both indexed ports', (_family, validate) => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const view = createTransientView(graph, 'shared', 'uint32', 1);
    expect(() => validate('pair-check', [view], [view])).toThrow(/outputs\[0\].*inputs\[0\]/);
  } finally {
    device.destroy();
  }
});

test('analysis-core publishes the packed-view authoring validator', () => {
  expect(analysisCore).toHaveProperty('validateSinglePackedView');
});

test('buffer conflicts identify both named ports and imported physical storage', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const buffer = device.createBuffer({byteLength: 64, usage: Buffer.STORAGE});
  try {
    const values = importGraphBuffer(graph, 'values', buffer, 'uint32', 2);
    const classes = importGraphBuffer(graph, 'classes', buffer, 'uint32', 2);
    expect(() =>
      validateGraphOutputsDisjointFromInputs('named-check', {'output.classes': classes}, {values})
    ).toThrow(
      'named-check outputs must not share buffers with inputs: output.classes conflicts with values'
    );
    expect(() =>
      validateGraphOutputsDisjointFromInputs('indexed-check', [classes], [undefined, values])
    ).toThrow(
      'indexed-check outputs must not share buffers with inputs: outputs[0] conflicts with inputs[1]'
    );
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

test('captureGraphCommandNodes records addToGraph-style nodes without scheduling them', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'capture'});
  const input = createTransientView(graph, 'input', 'uint32', 16);
  const output = createTransientView(graph, 'output', 'uint32', 16);
  const originalAddComputePass = graph.addComputePass;

  const nodes = captureGraphCommandNodes(graph, () => {
    graph.add(new GPUMask({id: 'captured-mask', inputs: [input], output, operation: 'not'}));
  });

  expect(nodes.map(node => [node.type, node.id])).toEqual([['compute', 'captured-mask']]);
  expect(graph.addComputePass, 'graph methods are restored').toBe(originalAddComputePass);
  expect(Object.hasOwn(graph, 'addComputePass')).toBe(false);
  device.destroy();
});

test('captureGraphCommandNodes bridges the geospatial addToGraph-style passes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'capture-geospatial'});
  const left = createTransientView(graph, 'left', 'float32x2', 4);
  const right = createTransientView(graph, 'right', 'float32x2', 4);
  const output = createTransientView(graph, 'distance', 'float32', 4);
  const originalAddComputePass = graph.addComputePass;

  // The geospatial classes schedule through `addToGraph`, which captureGraphCommandNodes bridges.
  const nodes = captureGraphCommandNodes(graph, () =>
    new GPUPairwisePointDistance({id: 'distance', left, right, output}).addToGraph(graph)
  );

  expect(nodes.length).toBeGreaterThan(0);
  expect(nodes.every(node => node.id.startsWith('distance'))).toBe(true);
  expect(graph.addComputePass, 'graph methods are restored').toBe(originalAddComputePass);
  device.destroy();
});

test('a contributor built on captureGraphCommandNodes composes through graph.add', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'contributor-composition'});
  const input = createTransientView(graph, 'input', 'uint32', 8);
  const output = createTransientView(graph, 'output', 'uint32', 8);
  const scheduled: string[] = [];
  const addComputePass = graph.addComputePass.bind(graph);
  graph.addComputePass = node => {
    scheduled.push(node.id);
    addComputePass(node);
  };

  class InvertContributor implements GPUCommandNodeProducer {
    readonly id = 'invert';
    getCommandNodes<Parameters>(target: GPUCommandGraph<Parameters>) {
      return captureGraphCommandNodes(target, () => {
        target.add(new GPUMask({id: `${this.id}-mask`, inputs: [input], output, operation: 'not'}));
      });
    }
  }
  graph.add(new InvertContributor());

  expect(scheduled).toEqual(['invert-mask']);
  device.destroy();
});

test('contributor validation helpers enforce graph membership and compact output layout', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'validation'});
  const otherGraph = new GPUCommandGraph(device, {id: 'other'});
  const buffer = device.createBuffer({byteLength: 64, usage: Buffer.STORAGE | Buffer.COPY_SRC});
  const ids = importGraphBuffer(graph, 'ids', buffer, 'uint32');
  const count = createTransientView(graph, 'count', 'uint32', 1);
  const overflow = createTransientView(graph, 'overflow', 'uint32', 1);
  const foreign = createTransientView(otherGraph, 'foreign', 'uint32', 1);

  expect(ids.length, 'length defaults to whole rows').toBe(16);
  expect(getGraphViewChunks(ids)).toEqual([ids]);
  expect(() =>
    validateGraphViewsBelongToGraph('contributor', graph, [ids, undefined])
  ).not.toThrow();
  expect(() => validateGraphViewsBelongToGraph('contributor', graph, [foreign])).toThrow(
    /target graph/
  );
  expect(() =>
    validateCompactOutput('contributor', {ids, status: {count, overflow}})
  ).not.toThrow();
  expect(() =>
    validateCompactOutput('contributor', {
      ids,
      status: {count: createTransientView(graph, 'empty-count', 'uint32', 0), overflow}
    })
  ).toThrow(/output\.status\.count must contain one uint32 row/);
  buffer.destroy();
  device.destroy();
});
