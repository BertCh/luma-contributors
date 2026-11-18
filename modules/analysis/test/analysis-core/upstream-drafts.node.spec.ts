// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import typescript from 'typescript';
import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, GPUHistogram} from '@luma.gl/graph/gpu-core';
import * as graphData from '@luma.gl/graph/gpu-data';
import {GPUGraphAlgorithmsBreadthFirstSearch} from '@luma.gl/graph/gpu-graph-algorithms';
import {expect, it, vi} from 'vitest';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';
import {createTopologyFixture} from './upstream-drafts.test-utils';

const REPOSITORY_ROOT = resolve(import.meta.dirname, '../../../..');
const readDraft = (name: string) =>
  readFileSync(resolve(REPOSITORY_ROOT, 'dev-docs/upstream-drafts', name), 'utf8');
const getExamples = (name: string) =>
  [...readDraft(name).matchAll(/```typescript\n([\s\S]*?)```/g)].map(match => match[1]);

function checkTypes(source: string): readonly typescript.Diagnostic[] {
  const filename = resolve(import.meta.dirname, 'upstream-drafts-virtual.ts');
  const configuration = typescript.readConfigFile(
    resolve(REPOSITORY_ROOT, 'tsconfig.json'),
    typescript.sys.readFile
  );
  const {options} = typescript.parseJsonConfigFileContent(
    configuration.config,
    typescript.sys,
    REPOSITORY_ROOT
  );
  options.noEmit = true;
  options.incremental = false;
  const host = typescript.createCompilerHost(options);
  const getSourceFile = host.getSourceFile.bind(host);
  const fileExists = host.fileExists.bind(host);
  host.fileExists = name => name === filename || fileExists(name);
  host.getSourceFile = (name, version, onError, fresh) =>
    name === filename
      ? typescript.createSourceFile(name, source, version, true)
      : getSourceFile(name, version, onError, fresh);
  return typescript.getPreEmitDiagnostics(typescript.createProgram([filename], options, host));
}

it('R1 gives the triangle two hops instead of the published one-hop default', () => {
  const device = createNullWebGPUDevice();
  const fixture = createTopologyFixture(device, [0, 2, 3, 3], [1, 0xffffffff, 2]);
  try {
    const search = new GPUGraphAlgorithmsBreadthFirstSearch({
      topology: fixture.topology,
      seeds: fixture.createVector('seeds', [0]).vector,
      distances: fixture.createVector('distances', [0, 0, 0]).vector,
      predecessors: fixture.createVector('predecessors', [0, 0, 0]).vector
    });
    expect(search.maxDepth).toBe(1);
    expect(readDraft('03-caller-csr-and-masks.md')).toContain('maxDepth: 2');
    expect(readDraft('03-caller-csr-and-masks.md')).toContain('seeds = [0]');
  } finally {
    fixture.destroy();
    device.destroy();
  }
});

it('R2 reproduces accepted malformed views and requires a format-aware import guard', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const handle = graph.createTransientBuffer({id: 'input', byteLength: 32, usage: Buffer.STORAGE});
  const outputHandle = graph.createTransientBuffer({
    id: 'output',
    byteLength: 4,
    usage: Buffer.STORAGE
  });
  const output = graph.createDataView(outputHandle, {format: 'uint32', length: 1});
  try {
    for (const layout of [{byteOffset: 1}, {byteOffset: 31, rowByteLength: 1}]) {
      const input = graph.createDataView(handle, {format: 'float32', length: 1, ...layout});
      expect(() => new GPUHistogram({input, output, domain: [0, 1]})).toThrow(/packed|aligned/);
    }
    const example = getExamples('04-typed-buffer-and-cpu-vector.md').find(source =>
      source.includes('function validateImportLayout')
    );
    expect(example, 'draft must provide the missing format checks').toBeDefined();
    const exports = executeExample(example!, {'@luma.gl/graph/gpu-data': graphData}) as {
      validateImportLayout: (layout: {
        format: string;
        byteOffset: number;
        rowByteLength?: number;
        byteStride?: number;
      }) => unknown;
    };
    expect(() => exports.validateImportLayout({format: 'float32', byteOffset: 1})).toThrow(/align/);
    expect(() =>
      exports.validateImportLayout({format: 'float32', byteOffset: 31, rowByteLength: 1})
    ).toThrow();
    expect(() =>
      exports.validateImportLayout({
        format: 'float32',
        byteOffset: 28,
        rowByteLength: 1,
        byteStride: 4
      })
    ).toThrow(/row width/);
    expect(() => exports.validateImportLayout({format: 'float32', byteOffset: 28})).not.toThrow();
    const exactEnd = graph.createDataView(handle, {format: 'float32', length: 1, byteOffset: 28});
    expect(() => new GPUHistogram({input: exactEnd, output, domain: [0, 1]})).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('R3 conditions helper deletion on sentinel support', () => {
  const helper = readFileSync(
    resolve(
      REPOSITORY_ROOT,
      'modules/analysis/src/analysis-core/kernels/masked-weak-components.ts'
    ),
    'utf8'
  );
  const terrain = readFileSync(
    resolve(
      REPOSITORY_ROOT,
      'modules/analysis/src/gpu-terrain/terrain-decode/gpu-terrain-spike-repair.ts'
    ),
    'utf8'
  );
  expect(helper).toContain('neighbor < NODE_COUNT');
  expect(terrain).toContain("const NO_VERTEX = '0xffffffffu'");
  expect(readDraft('03-caller-csr-and-masks.md')).toContain('Deletion requires sentinel support');
});

it('R4 makes completeness mandatory for each proposed adjacency direction', () => {
  const proposal = getExamples('03-caller-csr-and-masks.md')[1];
  expect(proposal, 'the proposed status-bearing types must be executable').toBeDefined();
  const source = `${proposal}\nconst missing: GPUGraphAlgorithmsAdjacencyView = {} as Omit<GPUGraphAlgorithmsAdjacencyView, 'completeness'>;\nexport {missing};`;
  const diagnostics = checkTypes(source);
  expect(
    diagnostics.map(diagnostic =>
      typescript.flattenDiagnosticMessageText(diagnostic.messageText, '\n')
    )
  ).toEqual([expect.stringContaining("Property 'completeness' is missing")]);
});

it('R5 rejects the example on alpha.16 and accepts the identical ports under the proposed types', () => {
  const [example, proposal] = getExamples('03-caller-csr-and-masks.md');
  expect(checkTypes(example)).toHaveLength(0);
  const unannotated = example.replace(/^\s*\/\/ @ts-expect-error.*\n/gm, '');
  expect(checkTypes(unannotated)).toHaveLength(2);
  const widened = unannotated
    .replace(/import type \{GraphDataView\}[^\n]*\n/, '')
    .replace(
      /import type \{GPUGraphAlgorithmsDegreeProps\}[^\n]*\n/,
      'type GPUGraphAlgorithmsDegreeProps = GPUGraphAlgorithmsViewDegreeProps;\n'
    );
  expect(proposal).toBeDefined();
  expect(
    checkTypes(`${proposal}\n${widened}`).map(diagnostic =>
      typescript.flattenDiagnosticMessageText(diagnostic.messageText, '\n')
    )
  ).toEqual([]);
});

function executeExample(source: string, imports: Record<string, unknown>): Record<string, unknown> {
  const javascript = typescript.transpileModule(source, {
    compilerOptions: {
      module: typescript.ModuleKind.CommonJS,
      target: typescript.ScriptTarget.ES2022
    }
  }).outputText;
  const exports = {};
  new Function('exports', 'require', javascript)(exports, (name: string) => {
    if (!(name in imports)) throw new Error(`Unexpected example import: ${name}`);
    return imports[name];
  });
  return exports;
}

/** Host-only benchmark harness: preserves callback/validation behavior and deliberately caches a snapshot. */
async function runBenchmarkExample(stale: boolean, observations: {invocations: number}) {
  class HostBuffer {
    readonly bytes: Uint8Array;
    constructor(props: {data?: ArrayBufferView; byteLength?: number}) {
      this.bytes = props.data
        ? new Uint8Array(props.data.buffer, props.data.byteOffset, props.data.byteLength).slice()
        : new Uint8Array(props.byteLength ?? 0);
    }
    write(values: ArrayBufferView, byteOffset = 0) {
      this.bytes.set(
        new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
        byteOffset
      );
    }
    async readAsync(byteOffset: number, byteLength: number) {
      return this.bytes.slice(byteOffset, byteOffset + byteLength);
    }
    destroy() {}
  }
  class HostVector {
    readonly buffer: HostBuffer;
    constructor(props: {buffer: HostBuffer}) {
      this.buffer = props.buffer;
    }
    destroy() {}
  }
  class HostOperation {
    constructor(readonly props: {execute?: (context: unknown) => Record<string, Float32Array>}) {}
  }
  class HostGraph {
    readonly operations: HostOperation[] = [];
    addInput() {}
    add(operation: HostOperation) {
      this.operations.push(operation);
    }
    compile() {
      let firstSnapshot: Float32Array | undefined;
      return {
        run: async ({inputs}: {inputs: {input: HostVector}}) => {
          observations.invocations++;
          const values = new Float32Array(inputs.input.buffer.bytes.slice().buffer);
          const result = this.operations[0].props.execute!({inputs: {rows: {chunks: [{values}]}}})[
            'middle'
          ];
          firstSnapshot ??= result.slice();
          const buffer = new HostBuffer({data: stale ? firstSnapshot : result});
          return {result: {data: [{buffer, byteOffset: 0}], destroy() {}}};
        },
        destroy() {}
      };
    }
  }
  // Keep the seven invocations and real callback intact; only reduce benchmark data sizes.
  const source = getExamples('01-packed-cpu-gpu-transfer.md')[0].replace(
    '[1_000, 100_000, 1_000_000, 4_000_000]',
    '[2]'
  );
  const exports = executeExample(source, {
    '@luma.gl/core': {Buffer},
    '@luma.gl/graph/gpu-data': {GPUVector: HostVector},
    '@luma.gl/graph/gpu-core': {
      GPUGraph: HostGraph,
      GPUCPUOperation: HostOperation,
      GPUCopyOperation: HostOperation,
      gpuGraph: {vector: () => ({})}
    }
  });
  const benchmark = exports['benchmarkPackedBoundary'] as (device: unknown) => Promise<unknown>;
  return benchmark({
    createBuffer: (props: {data?: ArrayBufferView; byteLength?: number}) => new HostBuffer(props)
  });
}

it('R6 rejects a cached first snapshot and accepts live snapshots on repeated benchmark runs', async () => {
  const output = vi.spyOn(console, 'table').mockImplementation(() => {});
  try {
    const stale = {invocations: 0};
    await expect(runBenchmarkExample(true, stale)).rejects.toThrow(
      'Identity round trip changed a value'
    );
    expect(stale.invocations).toBe(2);
    const live = {invocations: 0};
    await expect(runBenchmarkExample(false, live)).resolves.toBeDefined();
    expect(live.invocations).toBe(7);
  } finally {
    output.mockRestore();
  }
});
