// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/// <reference types="vite/client" />

import {expect, test} from 'vitest';
import {EMBEDDED_DATASET_LOADERS} from '../../src/computation-lenses/embedded-dataset-loaders.js';
import '../../src/computation-lenses/scenarios/index';
import {getLensScenario} from '../../src/computation-lenses/index';

const sourceFiles = import.meta.glob('../../src/computation-lenses/data/*.json', {
  query: '?raw',
  import: 'default'
}) as Record<string, () => Promise<string>>;

test('every embedded dataset loads as a Vite browser module', async () => {
  expect(Object.keys(EMBEDDED_DATASET_LOADERS).length).toBeGreaterThan(0);
  expect(Object.keys(sourceFiles).length).toBe(Object.keys(EMBEDDED_DATASET_LOADERS).length);
  for (const [filename, loadDataset] of Object.entries(EMBEDDED_DATASET_LOADERS)) {
    const readSource = sourceFiles[`../../src/computation-lenses/data/${filename}`];
    expect(readSource, filename).toBeDefined();
    const actual = (await loadDataset()).default;
    expect(actual, filename).toEqual(JSON.parse(await readSource()));
  }
});

test('network registry callers load their synchronous fixtures in a browser', async () => {
  for (const id of ['gpu-network.flow-aggregation', 'gpu-network.network-reachability']) {
    const scenario = getLensScenario(id);
    expect(scenario, id).toBeDefined();
    const loaded = await scenario!.load();
    expect(typeof loaded.run, id).toBe('function');
  }
});
