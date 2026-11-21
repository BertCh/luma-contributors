// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {describe, expect, test} from 'vitest';
import {build, transform} from 'esbuild';
import {createPublicSurfaceCensus, listPublicEntries} from './public-surface-census.mjs';
import {findNonPublicImports} from '../../src/computation-lenses/lens-example';
import {createPublicSurfaceSnapshot, repositoryRoot} from './public-surface-census.mjs';
import {resolve} from 'node:path';
import {
  createDefaultViewRegistry,
  getDefaultViewPacks
} from '../../src/computation-views/registry/default-registry';

// Public dataset binding/run options were added for caller-owned scenario inputs. Retain the
// historical snapshot while pinning this reviewed additive contract with explicit type names.
const RUN_OPTION_TYPES = [
  'LensDatasetResolver',
  'LensScenarioInputBinding',
  'LensScenarioRunOptions'
];

describe('lens public surface', () => {
  test('all five public pack entries expose their registration factory', async () => {
    const packs = [
      [
        'gpu-dataframe',
        () => import('@luma-contributors/lenses/gpu-dataframe/computation-pack/index')
      ],
      ['gpu-network', () => import('@luma-contributors/lenses/gpu-network/computation-pack/index')],
      ['gpu-raster', () => import('@luma-contributors/lenses/gpu-raster/computation-pack/index')],
      [
        'gpu-spatial-analysis',
        () => import('@luma-contributors/lenses/gpu-spatial-analysis/computation-pack/index')
      ],
      ['gpu-terrain', () => import('@luma-contributors/lenses/gpu-terrain/computation-pack/index')]
    ] as const;
    for (const [id, load] of packs) {
      const entry = await load();
      expect(Object.keys(entry), id).toEqual(['createViewPack']);
      const defaultRegistration = getDefaultViewPacks().find(pack => pack.id === id);
      expect(defaultRegistration).toBeDefined();
      const registration = (
        entry as unknown as {createViewPack(): {id: string; views: unknown[]}}
      ).createViewPack();
      expect(registration.id).toBe(id);
      expect(Array.isArray(registration.views)).toBe(true);
      expect(createDefaultViewRegistry().getPack(id)?.id).toBe(id);
      expect(getDefaultViewPacks().find(pack => pack.id === id)).toBe(defaultRegistration);
    }
  });
  test('retains historical reader/host names and the documented pack registration factories', () => {
    const allowed: Record<string, string[]> = JSON.parse(
      readFileSync(new URL('./public-surface-retention.json', import.meta.url), 'utf8')
    );
    allowed['./computation-lenses'] = [
      ...new Set([...allowed['./computation-lenses'], ...RUN_OPTION_TYPES])
    ].sort();
    // gate:E1 adds exactly one reviewed registration factory to each public pack entry.
    for (const family of [
      'gpu-dataframe',
      'gpu-network',
      'gpu-raster',
      'gpu-spatial-analysis',
      'gpu-terrain'
    ]) {
      allowed[`./${family}/computation-pack/index`] = ['createViewPack'];
    }
    const unsupported = Object.fromEntries(
      Object.entries(createPublicSurfaceCensus())
        .map(([entry, value]) => [
          entry,
          value.names.filter(name => !allowed[entry]?.includes(name))
        ])
        .filter(([, names]) => names.length)
    );
    expect(unsupported).toEqual({});
    for (const [entry, value] of Object.entries(createPublicSurfaceCensus())) {
      expect(value.names, entry).toEqual(allowed[entry] ?? []);
    }
  });
  test('pins string targets, removal of the data wildcard and runtime/type distinctions', () => {
    const manifest = JSON.parse(
      readFileSync(resolve(repositoryRoot, 'modules/lenses/package.json'), 'utf8')
    );
    const original = createPublicSurfaceSnapshot(manifest);
    const removed = structuredClone(manifest);
    delete removed.exports['./computation-lenses/data/*'];
    expect.soft(createPublicSurfaceSnapshot(removed)).not.toEqual(original);
    const added = structuredClone(manifest);
    added.exports['./extra-registry'] = './dist/computation-lenses/registry.js';
    expect.soft(createPublicSurfaceSnapshot(added)).not.toEqual(original);
    const path = resolve(repositoryRoot, 'modules/lenses/src/computation-views/index.ts');
    const changed = readFileSync(path, 'utf8').replace(
      /\bmountLadderLens\b/,
      'type mountLadderLens'
    );
    expect(createPublicSurfaceSnapshot(manifest, {[path]: changed})).not.toEqual(original);
  });
  test('public named exports link after isolated browser TypeScript transforms', async () => {
    for (const [entry, path] of Object.entries(listPublicEntries())) {
      if (entry === './computation-lenses/scenarios/index') continue;
      await expect(
        build({
          entryPoints: [path],
          bundle: true,
          write: false,
          format: 'esm',
          platform: 'browser',
          packages: 'external',
          logLevel: 'silent',
          // Vite transforms each TS file independently before the browser links JavaScript.
          // Loading transformed JS here prevents esbuild's TS linker from hiding type-export bugs.
          plugins: [
            {
              name: 'isolated-browser-typescript',
              setup(builder) {
                builder.onLoad({filter: /\.ts$/}, async arguments_ => ({
                  contents: (
                    await transform(readFileSync(arguments_.path, 'utf8'), {
                      loader: 'ts',
                      format: 'esm',
                      target: 'es2022'
                    })
                  ).code,
                  loader: 'js'
                }));
              }
            }
          ]
        }),
        entry
      ).resolves.toMatchObject({warnings: []});
    }
  });
  test('local permutation scenario uses published analysis entries', () => {
    const source = readFileSync(
      new URL(
        '../../src/computation-lenses/scenarios/gpu-spatial-analysis/local-permutation-test.lens.ts',
        import.meta.url
      ),
      'utf8'
    );
    expect(
      findNonPublicImports(source).filter(name => name.startsWith('@luma-contributors/analysis'))
    ).toEqual([]);
  });
  test('public entry barrels have explicit named exports', () => {
    for (const [entry, path] of Object.entries(listPublicEntries())) {
      expect(readFileSync(path, 'utf8'), entry).not.toMatch(/export\s+(?:type\s+)?\*/);
    }
  });
  test('pins sorted value and type names for every concrete entry', () => {
    const actual = createPublicSurfaceSnapshot();
    const expected = JSON.parse(
      readFileSync(new URL('./public-surface.json', import.meta.url), 'utf8')
    );
    expected.entries['./computation-lenses'].types = [
      ...new Set([...expected.entries['./computation-lenses'].types, ...RUN_OPTION_TYPES])
    ].sort();
    expect(actual).toEqual(expected);
  });
});
