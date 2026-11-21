// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {join} from 'node:path';
import {expect, it} from 'vitest';
import typescript from 'typescript';
import type {Device} from '@luma.gl/core';
import * as graphCore from '@luma.gl/graph/gpu-core';
import {
  GPUSpatialJoinPrepared,
  type GPUSpatialJoinPreparedProps
} from '../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared';
import {
  GPUParameterBuffer,
  getGPUAdamStepParameterValues,
  GPU_ADAM_STEP_PARAMETER_LENGTH
} from '../../src/analysis-core/index';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';
import {
  ANALYSIS_DOC_PAGES,
  DOCS_ROOT,
  createVirtualProgram,
  getFirstRunFences,
  getPairedFirstRunFences,
  getDeckOwnedDeviceFence,
  getPairedDeckOwnedDeviceFence,
  readDocPage
} from './doc-program';

import {
  extractSnippets,
  getSnippetPath,
  getSnippetDiagnostics,
  getLibraryGlobalNames,
  type DocSnippet
} from './doc-snippet-diagnostics.test-utils';

// Every checked fence must declare its public imports and fragment inputs. The three explicit
// host inputs below have real types. Unknown names remain diagnostics, never `any`. Illustrative
// signatures may opt out with a reason on the preceding Markdown comment. README first-run
// examples are checked with their literal shared lifecycle and both original line maps.

it('checks both TypeScript fence spellings and preserves per-fragment opt-out reasons', () => {
  const snippets = extractSnippets(
    'fixture.md',
    [
      '```ts',
      'const first = 1;',
      '```',
      '',
      '{/* no-typecheck: signature only */}',
      '',
      '```typescript',
      'second(...);',
      '```',
      '',
      '```typescript',
      'const third = 3;',
      '```'
    ].join('\n')
  );
  expect(snippets.map(snippet => [snippet.fenceLine, snippet.code, snippet.optOutReason])).toEqual([
    [1, 'const first = 1;', undefined],
    [7, 'second(...);', 'signature only'],
    [11, 'const third = 3;', undefined]
  ]);
});

it('R6 preserves diagnostics and page identities when flattened snippet names would collide', () => {
  const snippets = [
    ...extractSnippets('a-b.md', '```ts\nconst value: number = "invalid";\n```'),
    ...extractSnippets('a/b.md', '```ts\nconst value: number = 1;\n```')
  ];
  const files = new Map(
    snippets.map(snippet => [getSnippetPath(snippet), `${snippet.code}\nexport {};`])
  );
  const program = createVirtualProgram(files);
  const failures = snippets.flatMap(snippet =>
    program
      .getSemanticDiagnostics(program.getSourceFile(getSnippetPath(snippet)))
      .map(diagnostic => [snippet.page, diagnostic.code])
  );
  expect(failures).toEqual([['a-b.md', 2322]]);
});

it('resolves browser globals used as shorthand fragment inputs', () => {
  const path = join(DOCS_ROOT, '__doc_snippets__', 'shorthand-input.ts');
  const program = createVirtualProgram(new Map([[path, 'const output = {status};\nexport {};']]));
  expect([...getLibraryGlobalNames(program, program.getSourceFile(path))]).toEqual(['status']);
});

it('DOC-110 requires explicit public bindings and typed fragment inputs at the originating line', () => {
  const snippets = [
    ...extractSnippets(
      'valid.md',
      "```ts\nimport {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';\ndeclare const device: import('@luma.gl/core').Device;\nnew GPUParameterBuffer(device, {id: 'parameters', format: 'float32', length: 1, values: Float32Array.of(1)});\n```"
    ),
    ...extractSnippets('misspelled.md', '```ts\nnew GPUParametrBuffer(device, {});\n```'),
    ...extractSnippets(
      'invalid-adam.md',
      "```ts\nimport {getGPUAdamStepParameterValues} from '@luma-contributors/analysis/analysis-core';\ngetGPUAdamStepParameterValues({learningRate: 'fast'});\n```"
    ),
    ...extractSnippets('untyped-input.md', '```ts\nconsole.log(readerInput);\n```'),
    ...extractSnippets(
      'private.md',
      "```ts\nimport {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core/kernels/gpu-contributor-utils';\n```"
    ),
    ...extractSnippets(
      'retired-prop.md',
      "```ts\nimport type {GPURasterBand} from '@luma.gl/graph/gpu-raster';\ndeclare const elevation: import('@luma.gl/graph').GraphDataView<'float32'>;\nconst band: GPURasterBand = {id: 'elevation', format: 'float32', storage: {kind: 'buffer', view: elevation}};\n```"
    )
  ];
  const failures = getSnippetDiagnostics(snippets).map(({page, line, code}) => ({
    page,
    line,
    code
  }));
  expect(failures.filter(failure => failure.page === 'valid.md')).toEqual([]);
  for (const [page, line, code] of [
    ['misspelled.md', 2, 2304],
    ['untyped-input.md', 2, 2304],
    ['private.md', 2, 2307],
    ['invalid-adam.md', 3, 2322],
    ['retired-prop.md', 4, 2353]
  ] as const) {
    expect(failures).toContainEqual({page, line, code});
  }
}, 120_000);

it('analysis doc ```ts blocks type-check against the barrels', () => {
  const lifecycleLines = new Set(
    [...getFirstRunFences(), getDeckOwnedDeviceFence()].map(fence => fence.fenceLine)
  );
  const snippets: DocSnippet[] = [
    ...ANALYSIS_DOC_PAGES.flatMap(page => extractSnippets(page)).filter(
      snippet => snippet.page !== 'README.md' || !lifecycleLines.has(snippet.fenceLine)
    ),
    ...getPairedFirstRunFences(),
    getPairedDeckOwnedDeviceFence()
  ];
  const optOuts = snippets.filter(snippet => snippet.optOutReason !== undefined);
  expect(
    optOuts.filter(snippet => !snippet.optOutReason).map(s => `${s.page}:${s.fenceLine}`),
    'every no-typecheck opt-out gives a reason'
  ).toEqual([]);
  const checked = snippets.filter(snippet => snippet.optOutReason === undefined);

  const failures = getSnippetDiagnostics(checked).map(
    failure => `${failure.page}:${failure.line} ${failure.message}`
  );
  // Count shown in the handoff; a growing opt-out list needs review.
  console.log(
    `doc snippets: ${checked.length} checked, ${optOuts.length} opted out\n` +
      optOuts.map(s => `  ${s.page}:${s.fenceLine} ${s.optOutReason}`).join('\n')
  );
  expect(failures.length, failures.join('\n')).toBe(0);
}, 600_000);

it('FST-12 checks the literal Deck callback with its shared lifecycle and public installed types', () => {
  const unpaired = getSnippetDiagnostics([{page: 'README.md', ...getDeckOwnedDeviceFence()}]);
  for (const name of ['runFirstComputationOnDevice', 'webgpuAdapter']) {
    expect(
      unpaired.some(failure => failure.message.includes(`Cannot find name '${name}'`)),
      name
    ).toBe(true);
  }
  const paired = getPairedDeckOwnedDeviceFence();
  expect(getSnippetDiagnostics([paired])).toEqual([]);
  expect(getPairedFirstRunFences().map(snippet => snippet.heading)).toEqual([
    'First-run terrain derivatives',
    'First-run exact distance field',
    'First-run time-window filter',
    'First-run point-cloud statistical filter'
  ]);
}, 120_000);

it('FST-12 rejects protected device access and invalid Deck callback props at Markdown lines', () => {
  const original = readDocPage('README.md');
  for (const [before, after, code] of [
    [
      "runFirstComputationOnDevice(device, 'deck-analysis', construct)",
      "runFirstComputationOnDevice(new Deck({}).device, 'deck-analysis', construct)",
      2445
    ],
    ['onDeviceInitialized(device) {', 'onDeviceInitialized(device: string) {', 2322],
    ['onDeviceInitialized(device) {', 'onDeviceInitialised(device) {', 2561],
    ['onDeviceInitialized(device) {', 'onDeviceInitialized() {', 2552]
  ] as const) {
    const changed = original.replace(before, after);
    expect(changed).not.toBe(original);
    const failures = getSnippetDiagnostics([getPairedDeckOwnedDeviceFence(changed)]);
    const line =
      changed
        .split('\n')
        .findIndex(text =>
          text.includes(
            code === 2552 ? "runFirstComputationOnDevice(device, 'deck-analysis'" : after
          )
        ) + 1;
    expect(failures).toContainEqual(expect.objectContaining({page: 'README.md', line, code}));
  }
}, 120_000);

it('REVIEW B3 rejects accidental DOM inputs and accepts an explicit declaration', () => {
  for (const input of ['status', 'origin', 'window']) {
    const bad = extractSnippets(
      'dom-input.md',
      ['```ts', `const result = {${input}}; console.log(result.${input});`, '```'].join('\n')
    );
    expect(getSnippetDiagnostics(bad)).toContainEqual(
      expect.objectContaining({
        page: 'dom-input.md',
        line: 2,
        code: 90002,
        message: expect.stringContaining(`'${input}'`)
      })
    );
  }
  const good = extractSnippets(
    'typed-dom-input.md',
    '```ts\ndeclare const status: string;\nconst result = {status}; console.log(result.status, Math.PI, JSON.stringify(result));\n```'
  );
  expect(getSnippetDiagnostics(good)).toEqual([]);
});

it('REVIEW B4 rejects untyped CommonJS bindings with the full import specifier', () => {
  for (const specifier of [
    '@luma-contributors/analysis/gpu-terrain.js',
    '@luma-contributors/analysis/gpu-terrain'
  ]) {
    const snippets = extractSnippets(
      'commonjs.md',
      `\`\`\`ts\nconst {GPUTerrainDerivatves} = require('${specifier}');\nnew GPUTerrainDerivatves({bogus: true});\n\`\`\``
    );
    expect(getSnippetDiagnostics(snippets)).toContainEqual(
      expect.objectContaining({
        page: 'commonjs.md',
        line: 2,
        code: 90003,
        message: expect.stringContaining(specifier)
      })
    );
  }
});

it('DOC-110 checks the owned lifecycle and temporal examples without inferred API bindings', () => {
  const snippets = ['gpu-spatial-analysis/concepts.md', 'gpu-spatial-analysis/temporal-context.md']
    .flatMap(page => extractSnippets(page))
    .filter(snippet => snippet.optOutReason === undefined);
  expect(getSnippetDiagnostics(snippets)).toEqual([]);
}, 120_000);

async function assertDocumentedLifecycle(code: string): Promise<void> {
  const events: string[] = [];
  const currentAtDestruction: number[][] = [];
  let observeCurrent = () => -1;
  const pending: {resolve(): void; reject(error: Error): void}[] = [];
  const buildGraph = () => {
    const index = pending.length;
    const compiled = {
      index,
      encode() {},
      destroy() {
        events.push(`graph ${index}`);
        currentAtDestruction.push([index, observeCurrent()]);
      }
    };
    let resolve!: () => void;
    let reject!: (error: Error) => void;
    const promise = new Promise<typeof compiled>((complete, fail) => {
      resolve = () => complete(compiled);
      reject = fail;
    });
    pending.push({resolve, reject});
    if (index === 0) resolve();
    return {
      graph: {compileAsync: () => promise},
      destroy() {
        events.push(`owner ${index}`);
      }
    };
  };
  const javascript = typescript.transpileModule(code, {
    compilerOptions: {
      target: typescript.ScriptTarget.ES2022,
      module: typescript.ModuleKind.CommonJS
    }
  }).outputText;
  const run = new Function(
    'buildGraph',
    'device',
    'exports',
    `return (async () => {${javascript}\nreturn {rebuild, destroy, getCurrentIndex: () => current.compiled.index};})()`
  );
  const lifecycle = (await run(buildGraph, {commandEncoder: {}, submit() {}}, {})) as {
    rebuild(options: {gridSize: [number, number]}): Promise<void>;
    destroy(): void;
    getCurrentIndex(): number;
  };
  observeCurrent = () => lifecycle.getCurrentIndex();
  const rejected = lifecycle.rebuild({gridSize: [1, 1]});
  pending[1].reject(new Error('compile failed'));
  await expect(rejected).rejects.toThrow('compile failed');
  expect(events).toEqual(['owner 1']);
  const stale = lifecycle.rebuild({gridSize: [2, 2]});
  const replacement = lifecycle.rebuild({gridSize: [3, 3]});
  pending[3].resolve();
  await replacement;
  pending[2].resolve();
  await stale;
  // Independent ownership contract: every graph precedes its owner; failed compiles have no graph.
  expect(events).toEqual(['owner 1', 'graph 0', 'owner 0', 'graph 2', 'owner 2']);
  expect(currentAtDestruction).toEqual([
    [0, 3],
    [2, 3]
  ]);
  const afterTeardown = lifecycle.rebuild({gridSize: [4, 4]});
  lifecycle.destroy();
  pending[4].resolve();
  await afterTeardown;
  expect(events).toEqual([
    'owner 1',
    'graph 0',
    'owner 0',
    'graph 2',
    'owner 2',
    'graph 3',
    'owner 3',
    'graph 4',
    'owner 4'
  ]);
}

it('LIF-10 executes the documented rebuild lifecycle through rejection, supersession, swap and teardown', async () => {
  const snippet = extractSnippets('gpu-spatial-analysis/concepts.md').find(candidate =>
    candidate.code.includes('type GraphOwner')
  )!;
  await assertDocumentedLifecycle(snippet.code);
});

it('round 1 S2 rejects releasing the previous graph before publishing the replacement', async () => {
  const snippet = extractSnippets('gpu-spatial-analysis/concepts.md').find(candidate =>
    candidate.code.includes('type GraphOwner')
  )!;
  const mutated = snippet.code.replace(
    /current = next;[^\n]*\n  release\(previous\);/,
    'release(previous);\n  current = next;'
  );
  expect(mutated).not.toBe(snippet.code);
  await expect(assertDocumentedLifecycle(mutated)).rejects.toThrow();
});

it('round 1 S2 declares the inspected resource once and rejects declaring its producer twice', () => {
  const snippet = extractSnippets('gpu-spatial-analysis/concepts.md').find(candidate =>
    candidate.code.includes('inspect(nodes);')
  )!;
  const execute = (code: string) => {
    const graph = new graphCore.GPUCommandGraph<void>(createNullWebGPUDevice());
    let declarations = 0;
    let inspections = 0;
    const producer: graphCore.GPUCommandNodeProducer<void> = {
      getCommandNodes(target) {
        declarations++;
        graphCore.createTransientView(target, 'inspection-scratch', 'uint32', 1);
        return [];
      }
    };
    const javascript = typescript.transpileModule(code, {
      compilerOptions: {
        target: typescript.ScriptTarget.ES2022,
        module: typescript.ModuleKind.CommonJS
      }
    }).outputText;
    new Function('exports', 'graph', 'producer', 'inspect', javascript)({}, graph, producer, () => {
      inspections++;
    });
    expect(declarations).toBe(1);
    expect(inspections).toBe(1);
  };
  expect(() => execute(snippet.code)).not.toThrow();
  expect(() => execute(snippet.code.replace('graph.add(nodes);', 'graph.add(producer);'))).toThrow(
    'already in use'
  );
});

const PACKER_USAGE_EXAMPLE = `\`\`\`ts
import {GPUParameterBuffer, getGPUAdamStepParameterValues, GPU_ADAM_STEP_PARAMETER_LENGTH} from '@luma-contributors/analysis/analysis-core';
import {getGPURasterReprojectionParameterValues, type GPURasterReprojectionSettings} from '@luma-contributors/analysis/raster-analysis';
declare const settings: GPURasterReprojectionSettings;
getGPUAdamStepParameterValues({learningRate: 0.01});
const parameters = new GPUParameterBuffer(device, {
  id: 'adam', format: 'float32', length: GPU_ADAM_STEP_PARAMETER_LENGTH,
  values: getGPUAdamStepParameterValues({learningRate: 0.01})
});
parameters.write(getGPUAdamStepParameterValues({learningRate: 0.02}));
const words = getGPUAdamStepParameterValues({learningRate: 0.03});
parameters.write(words);
const destination = new Float32Array(20);
getGPURasterReprojectionParameterValues(settings, destination);
getGPURasterReprojectionParameterValues(settings, undefined);
\`\`\``;

it('UPT-09 rejects discarded allocating packers and accepts writes, initialization and destinations', () => {
  const snippets = extractSnippets('packer-usage.md', PACKER_USAGE_EXAMPLE);
  const failures = getSnippetDiagnostics(snippets);
  expect(failures.map(({line, code}) => ({line, code}))).toEqual([
    {line: 5, code: 90004},
    {line: 15, code: 90004}
  ]);
  expect(failures.every(failure => failure.message.includes('discarded'))).toBe(true);
}, 120_000);

it('round 1 B5 accepts aliased destinations while B6 rejects explicit and conditional discards', () => {
  const snippets = extractSnippets(
    'review-packers.md',
    `\`\`\`ts
import {getGPUAdamStepParameterValues as pack, GPUParameterBuffer, GPU_ADAM_STEP_PARAMETER_LENGTH} from '@luma-contributors/analysis/analysis-core';
import {getGPURasterReprojectionParameterValues, type GPURasterReprojectionSettings} from '@luma-contributors/analysis/raster-analysis';
type Words = Float32Array;
declare const destination: Words;
declare const settings: GPURasterReprojectionSettings;
declare const enabled: boolean;
getGPURasterReprojectionParameterValues(settings, destination);
void pack();
enabled && pack();
(pack() as Float32Array);
enabled ? pack() : pack();
const parameters = new GPUParameterBuffer(device, {id: 'adam', format: 'float32', length: GPU_ADAM_STEP_PARAMETER_LENGTH, values: pack()});
enabled && parameters.write(pack());
\`\`\``
  );
  const failures = getSnippetDiagnostics(snippets);
  expect(failures.map(({line, code}) => ({line, code}))).toEqual([
    {line: 9, code: 90004},
    {line: 10, code: 90004},
    {line: 11, code: 90004},
    {line: 12, code: 90004},
    {line: 12, code: 90004}
  ]);
}, 120_000);

it('round 1 B7 executes the positive Adam initialization fence and its subsequent writes', () => {
  // Execute the positive portion of the actual UPT-09 fixture rather than a second example.
  const example = extractSnippets('packer-usage.md', PACKER_USAGE_EXAMPLE)[0].code;
  const start = example.indexOf('const parameters =');
  const code = example.slice(start, example.lastIndexOf('const destination ='));
  const writes: number[][] = [];
  const device = {
    createBuffer: () => ({
      write(values: Float32Array) {
        writes.push(Array.from(values));
      },
      destroy() {}
    })
  } as unknown as Device;
  const javascript = typescript.transpileModule(code, {
    compilerOptions: {target: typescript.ScriptTarget.ES2022}
  }).outputText;
  expect(() =>
    new Function(
      'device',
      'GPUParameterBuffer',
      'getGPUAdamStepParameterValues',
      'GPU_ADAM_STEP_PARAMETER_LENGTH',
      javascript
    )(device, GPUParameterBuffer, getGPUAdamStepParameterValues, GPU_ADAM_STEP_PARAMETER_LENGTH)
  ).not.toThrow();
  expect(writes).toHaveLength(3);
  expect(writes.map(words => words.length)).toEqual([6, 6, 6]);
  expect(writes.map(words => words[0])).toEqual([
    Math.fround(0.01),
    Math.fround(0.02),
    Math.fround(0.03)
  ]);
});

it('round 1 B2 declares the documented prepared polygon on its own graph', () => {
  const snippet = extractSnippets('gpu-spatial-analysis/concepts.md').find(candidate =>
    candidate.code.includes('type Frame =')
  )!;
  const device = createNullWebGPUDevice();
  const importPolygons = (
    graph: graphCore.GPUCommandGraph<{geometryChanged: boolean}>
  ): GPUSpatialJoinPreparedProps<{geometryChanged: boolean}>['geometry'] => ({
    kind: 'polygons',
    positions: graphCore.createTransientView(graph, 'triangle', 'float32x2', 3),
    featureOffsets: graphCore.createTransientView(graph, 'features', 'uint32', 2),
    polygonOffsets: graphCore.createTransientView(graph, 'polygons', 'uint32', 2),
    ringOffsets: graphCore.createTransientView(graph, 'rings', 'uint32', 2)
  });
  const polygons = importPolygons(new graphCore.GPUCommandGraph(device));
  const code =
    snippet.code.split('let compiled:')[0] + '\ngraph.add(prepared); prepared.destroy();';
  const javascript = typescript.transpileModule(code, {
    compilerOptions: {
      target: typescript.ScriptTarget.ES2022,
      module: typescript.ModuleKind.CommonJS
    }
  }).outputText;
  const requirePublic = (specifier: string) =>
    specifier === '@luma.gl/graph/gpu-core' ? graphCore : {GPUSpatialJoinPrepared};
  expect(() =>
    new Function(
      'require',
      'exports',
      'device',
      'polygons',
      'importPolygons',
      `return (() => {${javascript}})();`
    )(requirePublic, {}, device, polygons, importPolygons)
  ).not.toThrow();
});
