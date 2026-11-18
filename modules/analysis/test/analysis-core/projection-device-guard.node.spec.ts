// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {runInNewContext} from 'node:vm';
import typeScript from 'typescript';
import {expect, it} from 'vitest';

function transpile(file: string) {
  return typeScript.transpileModule(readFileSync(file, 'utf8'), {
    compilerOptions: {
      module: typeScript.ModuleKind.CommonJS,
      target: typeScript.ScriptTarget.ES2022
    }
  }).outputText;
}

it('final1 F7 fails every projection GPU case when the browser device getter returns undefined', async () => {
  // Evaluate only device-acquisition paths: GPU modules are inert stubs and no GPU runs here.
  const guardModule = {exports: {} as {requireTestDevice?: (device: undefined) => void}};
  runInNewContext(
    transpile(fileURLToPath(new URL('../../src/testing/require-test-device.ts', import.meta.url))),
    {
      exports: guardModule.exports,
      window: {},
      document: {},
      require: () => ({TestRunner: {getCurrentTest: () => undefined}})
    }
  );
  const callbacks = new Map<string, () => Promise<void>>();
  const file = fileURLToPath(new URL('../gpu-project/gpu-project.spec.ts', import.meta.url));
  runInNewContext(transpile(file), {
    exports: {},
    require: (specifier: string) => {
      if (specifier === 'vitest')
        return {
          expect,
          it: (name: string, callback: () => Promise<void>) => callbacks.set(name, callback)
        };
      if (specifier === '@luma.gl/core/test-utils')
        return {getWebGPUTestDevice: async () => undefined};
      if (specifier.endsWith('require-test-device')) return guardModule.exports;
      return {};
    }
  });
  const gpuCases = [...callbacks].filter(([name]) => name.startsWith('GPUProjection'));
  expect(gpuCases).toHaveLength(18);
  const target = callbacks.get(
    'GPUProjection writes origin-relative f32 positions and honors nonzero view offsets'
  );
  expect(target).toBeDefined();
  for (const [name, callback] of gpuCases)
    await expect(callback(), name).rejects.toThrow('No WebGPU test device');
});

it('final2 G7 records explicit skips for five unsupported-device projection cases', async () => {
  const callbacks = new Map<
    string,
    (context: {skip: (reason: string) => never}) => Promise<void>
  >();
  runInNewContext(
    transpile(fileURLToPath(new URL('../gpu-project/gpu-project.spec.ts', import.meta.url))),
    {
      exports: {},
      require: (specifier: string) => {
        if (specifier === 'vitest')
          return {
            expect,
            it: (
              name: string,
              callback: (context: {skip: (reason: string) => never}) => Promise<void>
            ) => callbacks.set(name, callback)
          };
        if (specifier === '@luma.gl/core/test-utils')
          return {getWebGPUTestDevice: async () => ({info: {fallback: true}})};
        if (specifier.endsWith('require-test-device')) return {requireTestDevice: () => {}};
        return {};
      }
    }
  );
  const names = [
    'GPUProjection subtracts raw binary64 origins before converting local offsets to f32',
    'GPUProjection evaluates and returns absolute double-single positions with validity',
    'GPUProjection accepts inclusive binary64 patch endpoints after Float32 normalization',
    'GPUProjection updates exact global bounds without rebuilding its graph',
    'GPUProjection retains tolerant raw-binary64 assignment at internal patch seams'
  ];
  for (const name of names) {
    const recordedSkips: string[] = [];
    const pending = new Error('recorded skip');
    const callback = callbacks.get(name)!;
    await expect(
      callback({
        skip: reason => {
          recordedSkips.push(reason);
          throw pending;
        }
      }),
      name
    ).rejects.toBe(pending);
    expect(recordedSkips, name).toHaveLength(1);
  }
});

it('final2 G7 rejects the lens selection case when no browser device exists', async () => {
  const callbacks: (() => Promise<void>)[] = [];
  const guardModule = {exports: {} as {requireTestDevice?: (device: undefined) => void}};
  runInNewContext(
    transpile(fileURLToPath(new URL('../../src/testing/require-test-device.ts', import.meta.url))),
    {
      exports: guardModule.exports,
      window: {},
      document: {},
      require: () => ({TestRunner: {getCurrentTest: () => undefined}})
    }
  );
  const file = fileURLToPath(
    new URL(
      '../../../lenses/test/gpu-spatial-analysis/computation-pack/gpu-crossfilter-selection-backend.spec.ts',
      import.meta.url
    )
  );
  runInNewContext(transpile(file), {
    exports: {},
    require: (specifier: string) => {
      if (specifier === 'vitest')
        return {
          expect,
          test: (name: string, callback: () => Promise<void>) => {
            if (
              name ===
              'GPU range selection filters rows, transfers CSR counts and reads packed nonzero counts'
            )
              callbacks.push(callback);
          },
          vi: {}
        };
      if (specifier === '@luma.gl/core/test-utils')
        return {getWebGPUTestDevice: async () => undefined};
      if (
        specifier.endsWith('require-test-device') ||
        specifier === '@luma-contributors/analysis/testing'
      )
        return guardModule.exports;
      return {};
    }
  });
  expect(callbacks).toHaveLength(1);
  await expect(callbacks[0]()).rejects.toThrow('No WebGPU test device');
});
