// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPURasterNDWI} from '../../src/raster-analysis/gpu-raster-spectral-indices';
import type {GPURasterBufferBand} from '@luma.gl/graph/gpu-raster';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../utils/gpu-contributor-test-utils';
import {
  createSpectralIndex,
  SPECTRAL_CASES,
  SPECTRAL_REFERENCE
} from './raster-spectral-indices-test-utils';

function createScene() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const bands: Record<string, GPURasterBufferBand> = {};
  for (const name of [
    'green',
    'nearInfrared',
    'shortwaveInfrared',
    'shortwaveInfrared2',
    'red',
    'blue'
  ]) {
    bands[name] = {
      id: name,
      format: 'float32',
      storage: {kind: 'buffer', values: createTransientView(graph, name, 'float32', 12)},
      validity: createTransientView(graph, `${name}-mask`, 'uint32', 12),
      scale: 0.0000275,
      offset: -0.2,
      noDataValue: 0
    };
  }
  const props = {
    id: 'spectral',
    width: 4,
    height: 3,
    output: createTransientView(graph, 'output', 'float32', 12),
    outputValidity: createTransientView(graph, 'output-validity', 'uint32', 12)
  };
  return {device, graph, bands, props};
}

for (const [index, passes] of [
  ['ndwi-mcfeeters', 1],
  ['ndwi-gao', 1],
  ['nbr', 1],
  ['evi', 6],
  ['savi', 3]
] as const) {
  it(`${index} composes bounded band-math kernels with all masks and source calibration`, () => {
    const {device, graph, bands, props} = createScene();
    const addPass = vi.spyOn(graph, 'addComputePass');
    try {
      createSpectralIndex(index, {...props, epsilon: 0.001, clamp: [-1, 1]}, bands).addToGraph(
        graph
      );
      expect(addPass).toHaveBeenCalledTimes(passes);
      const nodes = addPass.mock.calls.map(([node]) => node);
      const sources = getParsedKernelSources(nodes);
      expect(sources.size).toBe(passes);
      for (const [id, source] of sources) {
        expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(6);
      }
      expect(nodes.at(-1)?.id).toBe('spectral');
      expect(nodes.at(-1)?.resources).toContainEqual({
        buffer: props.outputValidity,
        usage: 'storage-write'
      });
    } finally {
      addPass.mockRestore();
      device.destroy();
    }
  });
}

it('spectral recipes reject output aliasing every original source and mask', () => {
  const {device, graph, bands, props} = createScene();
  try {
    for (const source of ['nearInfrared', 'red', 'blue']) {
      const sourceBand = bands[source];
      const aliasedOutput = graph.createDataView(sourceBand.storage.values.buffer, {
        format: 'float32',
        length: 12
      });
      expect(() => createSpectralIndex('evi', {...props, output: aliasedOutput}, bands)).toThrow(
        /separate buffers/
      );
      expect(() =>
        createSpectralIndex('evi', {...props, outputValidity: sourceBand.validity!}, bands)
      ).toThrow(/separate buffers/);
    }
  } finally {
    device.destroy();
  }
});

it('spectral recipes reject foreign resources, malformed layouts, and unstable policies', () => {
  const {device, graph, bands, props} = createScene();
  const foreign = new GPUCommandGraph(device);
  try {
    expect(() => createSpectralIndex('savi', {...props, width: 0}, bands)).toThrow(
      /positive integers/
    );
    expect(() =>
      createSpectralIndex(
        'evi',
        {...props, output: createTransientView(graph, 'short', 'float32', 11)},
        bands
      )
    ).toThrow(/one sample per pixel/);
    expect(() =>
      createSpectralIndex('evi', props, {
        ...bands,
        blue: {
          ...bands['blue'],
          storage: {
            kind: 'buffer',
            values: createTransientView(foreign, 'foreign-blue', 'float32', 12)
          },
          format: 'float32'
        }
      })
    ).toThrow(/same graph/);
    for (const epsilon of [-1, NaN, Infinity]) {
      expect(() => createSpectralIndex('evi', {...props, epsilon}, bands)).toThrow(/epsilon/);
    }
    for (const soilAdjustment of [-1, NaN, Infinity, 1.01]) {
      expect(() => createSpectralIndex('savi', props, bands, soilAdjustment)).toThrow(
        /soilAdjustment/
      );
    }
    expect(
      () =>
        new GPURasterNDWI({
          ...props,
          method: 'ambiguous',
          nearInfrared: bands['nearInfrared']
        } as never)
    ).toThrow(/explicit/);
    expect(() => createSpectralIndex('savi', {...props, clamp: [1, -1]}, bands)).toThrow(/clamp/);
  } finally {
    device.destroy();
  }
});

it('the spectral oracle pins independent library formulas and records validity adaptations', () => {
  expect(SPECTRAL_REFERENCE.versions.spyndex).toBe('0.12.0');
  expect(SPECTRAL_REFERENCE.versions.numpy).toBe('2.2.6');
  expect(SPECTRAL_REFERENCE.reference.catalog).toContain('online=False');
  expect(SPECTRAL_REFERENCE.reference.gaoMapping).toContain('not spectral-response equivalence');
  expect(new Set(SPECTRAL_CASES.map(scene => scene.index)).size).toBe(5);
  for (const scene of SPECTRAL_CASES) {
    expect(scene.expected.values).toHaveLength(scene.width * scene.height);
    expect(scene.expected.validity).toEqual(
      scene.expected.values.map(value => Number(value !== null))
    );
  }
});
