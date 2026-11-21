// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {join} from 'node:path';
import {runInNewContext} from 'node:vm';
import typescript from 'typescript';
import {expect, it} from 'vitest';
import {readPackedBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  REPOSITORY_ROOT,
  getFirstRunFences,
  getPairedFirstRunFences,
  readDocPage
} from './doc-program';

function readFixture(name: string) {
  return readFileSync(join(REPOSITORY_ROOT, 'modules/analysis/test/analysis-core', name), 'utf8');
}

async function executeFixture(source: string, dependencies: Record<string, unknown>) {
  const compiled = typescript.transpileModule(source, {
    compilerOptions: {
      module: typescript.ModuleKind.CommonJS,
      target: typescript.ScriptTarget.ES2022
    }
  }).outputText;
  const exported: Record<string, unknown> = {};
  const displayed: unknown[][] = [];
  await runInNewContext(`(async () => {${compiled}\n})()`, {
    exports: exported,
    require: (specifier: string) => {
      if (!(specifier in dependencies)) throw new Error(`Unexpected fixture import: ${specifier}`);
      return dependencies[specifier];
    },
    console: {log: (...values: unknown[]) => displayed.push(values)},
    navigator: {gpu: {}},
    process,
    Float32Array,
    Uint32Array
  });
  return {exported, displayed};
}

async function checkDensityReads(overflow = 0, positionalOverflow = false) {
  const allocations: {id: string; data: Float32Array | Uint32Array; destroyed: boolean}[] = [];
  const readIds: string[] = [];
  const device = {
    createBuffer({id, data}: {id: string; data: Float32Array | Uint32Array}) {
      const allocation = {id, data, destroyed: false};
      allocations.push(allocation);
      return {
        byteLength: data.byteLength,
        usage: 0,
        readAsync: async () => {
          readIds.push(id);
          return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        },
        destroy: () => {
          allocation.destroyed = true;
        }
      };
    },
    createCommandEncoder: () => ({finish: () => ({})}),
    submit: () => {},
    destroy: () => {}
  };
  class Graph {
    importBuffer(_descriptor: unknown, buffer: unknown) {
      return buffer;
    }
    createDataView(handle: unknown) {
      return handle;
    }
    add() {}
    async compileAsync() {
      return {
        encode() {
          // Hand-binned fixture: both southern points are in the east cell; northern
          // points occupy one cell each. Simulate GPU writes to test only host read routing.
          allocations.find(allocation => allocation.id === 'values')!.data.set([0, 2, 1, 1]);
          allocations.find(allocation => allocation.id === 'overflow')!.data.set([overflow]);
        },
        destroy() {}
      };
    }
  }
  const literal = readFixture('point-density-first-run.test-utils.ts');
  let withUnrelatedInput = literal
    .replace(
      /    const (?:valuesHandle|valuesAllocation) =/,
      "    createBuffer('unrelated-weights', Float32Array.of(1, 1, 1, 1));\n$&"
    )
    .replace('0.25, 0.25, 0.75, 0.75', '1.25, 0.25, 1.75, 0.75');
  expect(withUnrelatedInput).not.toBe(literal);
  if (positionalOverflow) {
    const mutated = withUnrelatedInput.replace(
      'overflowAllocation.buffer.readAsync()',
      'buffers[3].readAsync()'
    );
    expect(mutated).not.toBe(withUnrelatedInput);
    withUnrelatedInput = mutated;
  }
  const execution = executeFixture(withUnrelatedInput, {
    '@luma.gl/core': {
      Buffer: {STORAGE: 0, COPY_SRC: 0, COPY_DST: 0},
      luma: {createDevice: async () => device}
    },
    '@luma.gl/core/webgpu': {webgpuAdapter: {}},
    '@luma.gl/graph': {GPUCommandGraph: Graph},
    '@luma-contributors/analysis/gpu-spatial-analysis': {GPUPointDensity: class {}}
  });
  if (overflow) await expect(execution).rejects.toThrow('Point density output is incomplete');
  else expect((await execution).displayed).toEqual([[[0, 2, 1, 1], 0]]);
  expect(readIds).toEqual(['values', 'overflow']);
  expect(allocations.find(allocation => allocation.id === 'unrelated-weights')).toBeDefined();
  expect(allocations.every(allocation => allocation.destroyed)).toBe(true);
}

it('FST-11 named density reads survive an unrelated allocation before the outputs', async () => {
  await checkDensityReads();
});

it('B1 rejects a positional overflow read even when the first density cell is zero', async () => {
  await expect(checkDensityReads(0, true)).rejects.toThrow();
});

it('B1 nonzero overflow rejects instead of displaying an incomplete density result', async () => {
  await checkDensityReads(1);
});

it('B1 rejects a positional overflow read that hides nonzero overflow', async () => {
  await expect(checkDensityReads(1, true)).rejects.toThrow();
});

it('FST-14 executes the current published east-rising terrain fence', () => {
  const terrainFence = getFirstRunFences().find(
    fence => fence.heading === 'First-run terrain derivatives'
  )!;
  const source = readFixture('family-first-run-terrain.test-utils.ts');
  const literal = source.split('// BEGIN LITERAL FENCES\n')[1]?.split('\n// END LITERAL FENCES')[0];
  expect(literal?.endsWith(terrainFence.code)).toBe(true);
});

const FAMILY_NAMES = ['terrain', 'distance', 'time', 'time-control', 'pointcloud'];

it.each(FAMILY_NAMES)('FST-12 %s preserves the literal published lifecycle', name => {
  const source = readFixture(`family-first-run-${name}.test-utils.ts`);
  const lifecycle = getFirstRunFences()[0].code;
  expect(source.split('// BEGIN LITERAL FENCES\n')[1]?.startsWith(`${lifecycle}\n\n`)).toBe(true);
  const index =
    name === 'time-control' ? 2 : ['terrain', 'distance', 'time', 'pointcloud'].indexOf(name);
  const expected = getPairedFirstRunFences()[index].code;
  const literal = source.split('// BEGIN LITERAL FENCES\n')[1]?.split('\n// END LITERAL FENCES')[0];
  expect(literal).toBe(
    name === 'time-control'
      ? expected
          .replace(
            'start: (t + 1000) - t, end: (t + 1000) - t',
            'start: (t + 1001) - t, end: (t + 1999) - t'
          )
          .replace('// [1], 1, 1, 0', '// [], 0, 0, 0')
      : expected
  );
});

type FirstRunContext = {
  float32: (id: string, data: Float32Array) => {read: () => Promise<Float32Array>};
};
type ConstructFirstRun = (context: FirstRunContext) => () => Promise<Float32Array>;
type BorrowedFirstRun = (
  device: unknown,
  id: string,
  construct: ConstructFirstRun
) => Promise<Float32Array>;
type OwnedFirstRun = (id: string, construct: ConstructFirstRun) => Promise<Float32Array>;

function createLifecycleProbe(failure?: string, ownsDevice = false) {
  const buffers: {destroyed: number}[] = [];
  const compilations: {destroyed: number}[] = [];
  const encoders: {released: boolean; destroyed: number}[] = [];
  let deviceDestroyed = 0;
  let deviceAcquisitions = 0;
  const fail = (phase: string) => {
    if (failure === phase) throw new Error(`fixture ${phase} failure`);
  };
  const device = {
    createBuffer({id, data}: {id: string; data: Float32Array | Uint32Array}) {
      if (deviceDestroyed) throw new Error('Borrowed device was destroyed');
      const record = {destroyed: 0};
      buffers.push(record);
      return {
        id,
        byteLength: data.byteLength,
        usage: 0,
        readAsync: async (offset = 0, byteLength = data.byteLength) => {
          await Promise.resolve();
          if (record.destroyed) throw new Error('Readback buffer was destroyed');
          fail('readback');
          return new Uint8Array(data.buffer, data.byteOffset + offset, byteLength);
        },
        destroy: () => {
          record.destroyed++;
        }
      };
    },
    createCommandEncoder() {
      const record = {released: false, destroyed: 0};
      encoders.push(record);
      return {
        finish() {
          fail('finish');
          record.released = true;
          return {};
        },
        destroy() {
          record.destroyed++;
          record.released = true;
        }
      };
    },
    submit() {
      fail('submit');
    },
    destroy() {
      deviceDestroyed++;
    }
  };
  class Graph {
    constructor(receivedDevice: unknown) {
      expect(receivedDevice).toBe(device);
    }
    importBuffer(_descriptor: unknown, buffer: unknown) {
      return buffer;
    }
    createDataView(handle: unknown) {
      return handle;
    }
    async compileAsync() {
      fail('compile');
      const record = {destroyed: 0};
      compilations.push(record);
      return {
        encode() {
          fail('encode');
        },
        destroy() {
          record.destroyed++;
        }
      };
    }
  }
  const construct: ConstructFirstRun = ({float32}) => {
    const output = float32('result', Float32Array.of(7));
    fail('construct');
    return () => output.read();
  };
  return {
    device,
    construct,
    buffers,
    compilations,
    encoders,
    getDeviceDestroyed: () => deviceDestroyed,
    getDeviceAcquisitions: () => deviceAcquisitions,
    dependencies: {
      '@luma.gl/core': {
        Buffer: {STORAGE: 0, COPY_SRC: 0, COPY_DST: 0},
        luma: {
          createDevice: async () => {
            deviceAcquisitions++;
            if (!ownsDevice) throw new Error('Borrowed helper must not acquire a device');
            return device;
          }
        }
      },
      '@luma.gl/core/webgpu': {webgpuAdapter: {}},
      '@luma.gl/graph': {GPUCommandGraph: Graph},
      '@luma-contributors/analysis/analysis-core': {readPackedBuffer}
    }
  };
}

async function loadFixtureLifecycle(
  name: string,
  probe: ReturnType<typeof createLifecycleProbe>,
  mutate?: (source: string) => string
) {
  const source = readFixture(`family-first-run-${name}.test-utils.ts`);
  // Execute the fixture's lifecycle definitions unchanged, before its domain example/imports.
  const domainImport = source.indexOf('\nimport ', source.indexOf('export async function'));
  expect(domainImport).toBeGreaterThan(0);
  const lifecycle = source.slice(0, domainImport);
  const executable = mutate ? mutate(lifecycle) : lifecycle;
  if (mutate) expect(executable).not.toBe(lifecycle);
  return (await executeFixture(executable, probe.dependencies)).exported;
}

it.each(FAMILY_NAMES)('B2 %s rejects device acquisition inside the borrowed helper', async name => {
  const probe = createLifecycleProbe();
  const exported = await loadFixtureLifecycle(name, probe, source =>
    source.replace(
      '  const buffers: Buffer[] = [];',
      "  device = await luma.createDevice({type: 'webgpu', adapters: [webgpuAdapter]});\n  const buffers: Buffer[] = [];"
    )
  );
  const run = exported['runFirstComputationOnDevice'] as BorrowedFirstRun;
  await expect(run(probe.device, 'unexpected-acquisition', probe.construct)).rejects.toThrow(
    'Borrowed helper must not acquire a device'
  );
});

it.each(FAMILY_NAMES)('S1 %s rejects readback after premature resource destruction', async name => {
  const probe = createLifecycleProbe();
  const exported = await loadFixtureLifecycle(name, probe, source =>
    source.replace('return await readResult();', 'return readResult();')
  );
  const run = exported['runFirstComputationOnDevice'] as BorrowedFirstRun;
  await expect(run(probe.device, 'premature-cleanup', probe.construct)).rejects.toThrow(
    'Readback buffer was destroyed'
  );
});

it.each(
  FAMILY_NAMES
)('FST-12 %s cleans borrowed resources and supports a second readback', async name => {
  const probe = createLifecycleProbe();
  const exported = await loadFixtureLifecycle(name, probe);
  const run = exported['runFirstComputationOnDevice'] as BorrowedFirstRun;
  expect(typeof run).toBe('function');
  for (const id of ['first', 'second']) {
    expect(Array.from(await run(probe.device, id, probe.construct))).toEqual([7]);
    expect(probe.getDeviceDestroyed()).toBe(0);
    expect(probe.getDeviceAcquisitions()).toBe(0);
    expect(probe.buffers.every(buffer => buffer.destroyed === 1)).toBe(true);
    expect(probe.compilations.every(compiled => compiled.destroyed === 1)).toBe(true);
  }
  expect(probe.buffers).toHaveLength(2);
  expect(probe.encoders.every(encoder => encoder.released && encoder.destroyed === 0)).toBe(true);
});

it.each(
  FAMILY_NAMES
)('FST-12 %s cleans up every failure phase without destroying a borrowed device', async name => {
  for (const phase of ['construct', 'compile', 'encode', 'finish', 'submit', 'readback']) {
    const probe = createLifecycleProbe(phase);
    const exported = await loadFixtureLifecycle(name, probe);
    const run = exported['runFirstComputationOnDevice'] as BorrowedFirstRun;
    expect(typeof run).toBe('function');
    await expect(run(probe.device, phase, probe.construct)).rejects.toThrow(
      `fixture ${phase} failure`
    );
    expect(probe.getDeviceDestroyed(), phase).toBe(0);
    expect(probe.getDeviceAcquisitions(), phase).toBe(0);
    expect(probe.buffers).toHaveLength(1);
    expect(
      probe.buffers.every(buffer => buffer.destroyed === 1),
      phase
    ).toBe(true);
    expect(
      probe.compilations.every(compiled => compiled.destroyed === 1),
      phase
    ).toBe(true);
    expect(
      probe.encoders.every(encoder => encoder.released),
      phase
    ).toBe(true);
    if (phase === 'encode' || phase === 'finish') expect(probe.encoders[0].destroyed).toBe(1);
  }
});

it.each(
  FAMILY_NAMES
)('FST-12 %s standalone wrapper destroys its owned device on success and failure', async name => {
  for (const phase of [
    undefined,
    'construct',
    'compile',
    'encode',
    'finish',
    'submit',
    'readback'
  ]) {
    const probe = createLifecycleProbe(phase, true);
    const exported = await loadFixtureLifecycle(name, probe);
    const run = exported['runFirstComputation'] as OwnedFirstRun;
    if (phase)
      await expect(run(phase, probe.construct)).rejects.toThrow(`fixture ${phase} failure`);
    else expect(Array.from(await run('success', probe.construct))).toEqual([7]);
    expect(probe.getDeviceDestroyed(), phase).toBe(1);
    expect(probe.getDeviceAcquisitions(), phase).toBe(1);
    expect(
      probe.buffers.every(buffer => buffer.destroyed === 1),
      phase
    ).toBe(true);
    expect(
      probe.compilations.every(compiled => compiled.destroyed === 1),
      phase
    ).toBe(true);
    expect(
      probe.encoders.every(encoder => encoder.released),
      phase
    ).toBe(true);
  }
});

type FixtureTest = {name: string; action: () => void | Promise<void>};

async function collectFixtureTests(source: string, dependencies: Record<string, unknown>) {
  const tests: FixtureTest[] = [];
  const result = await executeFixture(source, {
    ...dependencies,
    vitest: {
      expect,
      it: (name: string, action: FixtureTest['action']) => tests.push({name, action})
    }
  });
  return {...result, tests};
}

async function loadFamilyAssertions(terrain: {slope: Float32Array; hillshade: Float32Array}) {
  return collectFixtureTests(
    `${readFixture('family-first-run.spec.ts')}\nexport {expectFiniteValues};`,
    {
      '@luma.gl/core': {},
      '@luma.gl/core/webgpu': {},
      '@luma.gl/graph': {},
      '@luma-contributors/analysis/analysis-core': {},
      '@luma-contributors/analysis/gpu-terrain': {},
      '@luma-contributors/analysis/raster-analysis': {},
      '@luma-contributors/analysis/dataframe-analysis': {},
      '@luma-contributors/analysis/pointcloud-analysis': {},
      './family-first-run.test-utils': {
        loadTerrainExample: async () => ({runTerrainLightingControlExample: async () => terrain})
      },
      '../../src/testing/require-test-device': {requireTestDevice: () => {}}
    }
  );
}

it.each([
  ['FST-14 keeps a 45-degree ramp slope', 5, 45, 0],
  ['FST-14 keeps a 45-degree ramp slope', 10, 45, 0],
  ['FST-14 retains a separate flat-gradient', 5, 0, 0.7071067811865475],
  ['FST-14 retains a separate flat-gradient', 10, 0, 0.7071067811865475]
] as const)('S2 %s rejects a %i-cell raster with a correct center', async (pattern, length, slope, shade) => {
  for (const port of ['slope', 'hillshade']) {
    const {tests} = await loadFamilyAssertions({
      slope: new Float32Array(port === 'slope' ? length : 9).fill(slope),
      hillshade: new Float32Array(port === 'hillshade' ? length : 9).fill(shade)
    });
    const test = tests.find(candidate => candidate.name.startsWith(pattern));
    expect(test).toBeDefined();
    await expect(test!.action(), port).rejects.toThrow();
  }
});

it('REVIEW finite-value assertions reject a hole in an ordinary array', async () => {
  const {exported} = await loadFamilyAssertions({
    slope: new Float32Array(9),
    hillshade: new Float32Array(9)
  });
  const check = exported['expectFiniteValues'] as (actual: number[], expected: number[]) => void;
  const sparse = new Array<number>(3);
  sparse[0] = 1;
  sparse[2] = 3;
  expect(() => check(sparse, [1, 2, 3])).toThrow();
});

it.each([
  -1,
  NaN,
  Infinity
])('REVIEW CPU variogram assertions reject residual %s', async residual => {
  // A weighted sum of squared errors is nonnegative. Keep independently correct coefficients
  // while supplying only a bad residual to exercise the actual documentation-test callback.
  const parameters = new Float32Array(12);
  parameters.set([0.2, 1.5, 2.2, residual], 4);
  const {tests} = await collectFixtureTests(readFixture('first-run-docs.node.spec.ts'), {
    'node:fs': {readFileSync},
    'node:path': {join},
    'node:console': {
      Console: class {
        log() {}
      }
    },
    '@luma-contributors/analysis/gpu-network': {},
    '@luma-contributors/analysis/dataframe-analysis': {
      computeVariogramModelFitOnCPU: () => ({parameters, status: Uint32Array.of(0, 0, 0)}),
      fitVariogramModel: () => ({nugget: 0.2}),
      fitVariogramModelWithDiagnostics: () => ({
        model: {nugget: 0.2},
        atBoundary: false,
        logRangeBracket: 0,
        converged: true
      })
    },
    typescript,
    './first-run-readme-links.test-utils': {},
    './doc-program': {REPOSITORY_ROOT, readDocPage}
  });
  const test = tests.find(candidate => candidate.name.includes('CPU variogram walkthrough'));
  expect(test).toBeDefined();
  expect(() => test!.action()).toThrow(
    residual < 0 ? 'to be greater than or equal to 0' : 'expected false to be true'
  );
});
