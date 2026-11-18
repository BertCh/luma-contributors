// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {Buffer} from '@luma.gl/core';
import {expect, it, vi} from 'vitest';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {GPURasterNDWI} from '../../src/raster-analysis/gpu-raster-spectral-indices';
import {GPURasterNBR} from '../../src/raster-analysis/gpu-raster-spectral-indices';
import * as spectralExports from '../../src/raster-analysis/gpu-raster-spectral-indices';
import * as spectralFamilyExports from '../../src/raster-analysis/spectral-indices/index';
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

it('keeps spectral producers in separate files without a contributor-contract allowance', () => {
  const root = fileURLToPath(new URL('../../../../', import.meta.url));
  const result = spawnSync(
    process.execPath,
    [
      'scripts/check-contributor-contract.mjs',
      '--all',
      '--rule',
      'multiple-classes',
      '--paths',
      'modules/analysis/src/raster-analysis/gpu-raster-spectral-indices.ts',
      'modules/analysis/src/raster-analysis/spectral-indices'
    ],
    {cwd: root, encoding: 'utf8', timeout: 30_000}
  );
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(0);
  expect(result.stdout).toMatch(/^multiple-classes\s+0\s+0$/m);
}, 30_000);

it('preserves the four public spectral exports through the established source barrel', () => {
  expect(Object.keys(spectralExports).sort()).toEqual([
    'GPURasterEVI',
    'GPURasterNBR',
    'GPURasterNDWI',
    'GPURasterSAVI'
  ]);
  expect(spectralExports).toEqual(spectralFamilyExports);
});

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

it('G3 spectral construction retains validated output and band descriptors after mutation', () => {
  const {device, graph, bands, props} = createScene();
  const nearInfrared = {...bands['nearInfrared']};
  const input = {...props, nearInfrared, shortwaveInfrared2: bands['shortwaveInfrared2']};
  const output = props.output;
  const contributor = new GPURasterNBR(input);
  const source = nearInfrared.storage.values;
  input.output = graph.createDataView(source.buffer, {format: 'float32', length: 12});
  nearInfrared.storage = {...nearInfrared.storage, values: output} as typeof nearInfrared.storage;
  try {
    const nodes = contributor.getCommandNodes(graph);
    expect(
      nodes
        .at(-1)
        ?.resources?.some(
          resource =>
            'buffer' in resource && resource.buffer === output && resource.usage === 'storage-write'
        )
    ).toBe(true);
    expect(
      nodes.some(node =>
        node.resources?.some(
          resource =>
            'buffer' in resource && resource.buffer === source && resource.usage === 'storage-read'
        )
      )
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

it('G3 spectral construction reads inherited band-storage getters once', () => {
  const {device, graph, bands, props} = createScene();
  const source = bands['nearInfrared'].storage.values;
  let reads = 0;
  const storage = Object.create({
    kind: 'buffer',
    get values() {
      reads++;
      return reads === 1 ? source : props.output;
    }
  });
  const band = {...bands['nearInfrared'], storage};
  try {
    const contributor = new GPURasterNBR({
      ...props,
      nearInfrared: band,
      shortwaveInfrared2: bands['shortwaveInfrared2']
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(reads).toBe(1);
    expect(
      nodes.some(node =>
        node.resources?.some(
          resource =>
            'buffer' in resource && resource.buffer === source && resource.usage === 'storage-read'
        )
      )
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

it('F5 rejects spectral output aliases through two physical imports and permits shared read-only bands', () => {
  const {device, graph, bands, props} = createScene();
  const buffer = device.createBuffer({byteLength: 48, usage: Buffer.STORAGE});
  const views = ['nir-import', 'output-import'].map(id =>
    graph.createDataView(graph.importBuffer({id, byteLength: 48, usage: Buffer.STORAGE}, buffer), {
      format: 'float32',
      length: 12
    })
  );
  const nearInfrared: GPURasterBufferBand = {
    ...bands['nearInfrared'],
    format: 'float32',
    storage: {kind: 'buffer', values: views[0]}
  };
  try {
    expect(() =>
      createSpectralIndex('nbr', {...props, output: views[1]}, {...bands, nearInfrared})
    ).toThrow(/share buffers/);
    expect(() =>
      graph.add(
        createSpectralIndex('nbr', props, {
          ...bands,
          nearInfrared,
          shortwaveInfrared2: {...nearInfrared, storage: {kind: 'buffer', values: views[1]}}
        })
      )
    ).not.toThrow();
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

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
      graph.add(createSpectralIndex(index, {...props, epsilon: 0.001, clamp: [-1, 1]}, bands));
      expect(addPass).toHaveBeenCalledTimes(passes);
      const nodes = addPass.mock.calls.map(([node]) => node);
      const sources = getParsedKernelSources(nodes);
      expect(sources.size).toBe(passes);
      for (const {id, source} of sources.configurations) {
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
