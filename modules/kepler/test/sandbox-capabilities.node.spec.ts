// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {dirname, resolve} from 'node:path';
import {describe, expect, test} from 'vitest';
import {
  filterSandboxCapabilities,
  SANDBOX_CAPABILITY_INVENTORY,
  SANDBOX_CAPABILITY_STATUS_LABELS
} from '../../../examples/deck/kepler-compute/src/sandbox-capabilities';

const require = createRequire(import.meta.url);

describe('Kepler sandbox coverage inventory', () => {
  test('covers every layer in the pinned upstream registry', () => {
    // Read the pinned source instead of loading the complete browser dependency graph in Node.
    const packagePath = resolve(
      dirname(require.resolve('@kepler.gl/constants')),
      '../../package.json'
    );
    const metadata = JSON.parse(readFileSync(packagePath, 'utf8'));
    expect(metadata.version).toBe(SANDBOX_CAPABILITY_INVENTORY.keplerVersion);
    const source = readFileSync(resolve(dirname(packagePath), 'dist/esm/layers.js'), 'utf8');
    const declaration = source.match(/export var LAYER_TYPES = keyMirror\(\{([^]*?)\}\)/);
    expect(declaration, 'Audit the registry parser when upgrading Kepler').not.toBeNull();
    const layerIds = [...declaration![1].matchAll(/['"]?([\w]+)['"]?: null/g)]
      .map(match => `layer.${match[1]}`)
      .sort();
    expect(layerIds.length).toBeGreaterThan(0);
    expect(
      SANDBOX_CAPABILITY_INVENTORY.capabilities
        .filter(capability => capability.group === 'Layers')
        .map(capability => capability.id)
        .sort()
    ).toEqual(layerIds);
  });

  test('every capability has a unique identity, explicit gap and acceptance contract', () => {
    const capabilities = SANDBOX_CAPABILITY_INVENTORY.capabilities;
    expect(new Set(capabilities.map(capability => capability.id)).size).toBe(capabilities.length);
    for (const capability of capabilities) {
      expect(SANDBOX_CAPABILITY_STATUS_LABELS[capability.status]).toBeTruthy();
      expect(capability.foundation).not.toHaveLength(0);
      expect(capability.gap).not.toHaveLength(0);
      expect(capability.acceptance).not.toHaveLength(0);
    }
  });

  test('combines text, workflow and integration stage filters', () => {
    expect(
      filterSandboxCapabilities('  RASTER  ', 'Layers', 'partial').map(item => item.id)
    ).toContain('layer.rasterTile');
    expect(filterSandboxCapabilities('raster', 'Sharing', 'adapter')).toEqual([]);
    expect(filterSandboxCapabilities('bounded cartogram').map(item => item.id)).toContain(
      'cartogram'
    );
    expect(filterSandboxCapabilities('does-not-exist')).toEqual([]);
    expect(filterSandboxCapabilities('')).toHaveLength(
      SANDBOX_CAPABILITY_INVENTORY.capabilities.length
    );
  });
});
