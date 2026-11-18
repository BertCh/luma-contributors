// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from 'node:buffer';
import {readFileSync, readdirSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL, URL} from 'node:url';
import {runInNewContext} from 'node:vm';
import typescript from 'typescript';
import {expect, test} from 'vitest';
import {getResourceBytesKind} from '../../../../website/src/lens/format';
import {fromRecordedLensFile} from '../../src/computation-lenses/registry';
import {extractLensViewModel} from '../../../../website/src/lens/extract-lens-view-model';

const generator = fileURLToPath(
  new URL('../../../../website/src/lens/generate-view-models.ts', import.meta.url)
);
const lensDirectory = path.dirname(generator);
const recordsDirectory = path.resolve(lensDirectory, '../../../docs/lens-records');
const modelsDirectory = path.join(lensDirectory, 'generated');
const pageDirectory = path.resolve(lensDirectory, '../components/meta/pages/lenses');
const featuredIds = [
  'gpu-spatial-analysis.kernel-density',
  'gpu-spatial-analysis.local-moran',
  'gpu-network.network-service-areas'
];

/** Execute the real writer scripts on an in-memory filesystem; never change lead-owned JSON. */
function runGenerators(useCommittedRecords = false) {
  const files = new Map<string, string>();
  const scriptCalls: string[] = [];
  for (const [index, id] of (useCommittedRecords ? [] : featuredIds).entries()) {
    files.set(
      path.join(recordsDirectory, `${id}.json`),
      JSON.stringify({
        label: index === 0 ? 'é' : 'plain',
        graphs: [
          {form: 'compiled', nodes: [{kind: 'compute'}, {kind: 'copy'}], edges: [], resources: []}
        ]
      })
    );
  }
  if (useCommittedRecords) {
    for (const file of readdirSync(recordsDirectory).filter(file => file.endsWith('.json'))) {
      files.set(
        path.join(recordsDirectory, file),
        readFileSync(path.join(recordsDirectory, file), 'utf8')
      );
    }
  }
  function read(file: string): string {
    const text = files.get(path.resolve(file));
    if (text === undefined) throw new Error(`missing fixture file ${file}`);
    return text;
  }
  const filesystem = {
    mkdirSync() {},
    readdirSync(directory: string) {
      const prefix = `${path.resolve(directory)}/`;
      return [...files.keys()]
        .filter(file => file.startsWith(prefix) && !file.slice(prefix.length).includes('/'))
        .map(file => file.slice(prefix.length));
    },
    readFileSync: read,
    writeFileSync(file: string, text: string) {
      files.set(path.resolve(file), text);
    },
    unlinkSync(file: string) {
      files.delete(path.resolve(file));
    },
    statSync(file: string) {
      return {size: Buffer.byteLength(read(file), 'utf8')};
    }
  };
  function execute(file: string) {
    scriptCalls.push(file);
    const text = readFileSync(file, 'utf8').replaceAll(
      'import.meta.url',
      JSON.stringify(pathToFileURL(file).href)
    );
    const source = typescript.transpileModule(text, {
      compilerOptions: {
        module: typescript.ModuleKind.CommonJS,
        target: typescript.ScriptTarget.ES2022,
        esModuleInterop: true
      }
    }).outputText;
    const errors: string[] = [];
    const runtimeProcess = {
      argv: ['node', file],
      execPath: process.execPath,
      stdout: {write() {}},
      exitCode: 0
    };
    runInNewContext(source, {
      exports: {},
      Buffer,
      URL,
      console: {
        log() {},
        error(message: string) {
          errors.push(message);
        }
      },
      process: runtimeProcess,
      require(specifier: string) {
        switch (specifier) {
          case 'node:fs':
            return filesystem;
          case 'node:path':
            return path;
          case 'node:url':
            return {fileURLToPath};
          case 'node:buffer':
            return {Buffer};
          case 'node:child_process':
            return {
              execFileSync(_executable: string, arguments_: string[]) {
                expect(arguments_[0]).toBe('--experimental-strip-types');
                execute(arguments_[1]);
              }
            };
          case '@luma-contributors/lenses/computation-lenses/registry':
            return {
              fromRecordedLensFile: useCommittedRecords
                ? fromRecordedLensFile
                : (record: unknown) => ({record})
            };
          case './extract-lens-view-model.ts':
            if (useCommittedRecords) return {extractLensViewModel};
            return {
              extractLensViewModel: (id: string, record: {label: string}) => ({
                id,
                title: record.label,
                nodes: [
                  {kind: 'compute', count: 1},
                  {kind: 'copy', count: 1}
                ]
              })
            };
          case './format.ts':
            return {getResourceBytesKind};
          default:
            throw new Error(`unexpected generator dependency ${specifier}`);
        }
      }
    });
    if (runtimeProcess.exitCode !== 0) throw new Error(`generator failed: ${errors.join('\n')}`);
  }
  execute(generator);
  return {files, read, scriptCalls};
}

test('X5 generator counts UTF-8 bytes for a single é in records and view models', () => {
  const output = runGenerators();
  const summary = JSON.parse(output.read(path.join(lensDirectory, 'summary.json')));
  const recordTexts = featuredIds.map(id => output.read(path.join(recordsDirectory, `${id}.json`)));
  const modelTexts = featuredIds.map(id => output.read(path.join(modelsDirectory, `${id}.json`)));
  const recordBytes = recordTexts.reduce(
    (total, text) => total + Buffer.byteLength(text, 'utf8'),
    0
  );
  const modelBytes = modelTexts.reduce((total, text) => total + Buffer.byteLength(text, 'utf8'), 0);
  expect(recordBytes - recordTexts.reduce((total, text) => total + text.length, 0)).toBe(1);
  expect(modelBytes - modelTexts.reduce((total, text) => total + text.length, 0)).toBe(1);
  expect(summary.recordBytes).toBe(recordBytes);
  expect(summary.viewModelBytes).toBe(modelBytes);
});

test('X1 full view-model generation also refreshes lens page histogram, median and featured sizes', () => {
  const output = runGenerators();
  const pageFile = path.join(pageDirectory, 'lenses-data.json');
  expect(output.files.has(pageFile), 'main generator must refresh its page metadata').toBe(true);
  const page = JSON.parse(output.read(pageFile));
  expect(output.scriptCalls).toContain(path.join(pageDirectory, 'generate-data.ts'));
  expect(page.bins.reduce((total: number, bin: {count: number}) => total + bin.count, 0)).toBe(3);
  expect(page.median).toEqual({id: 'gpu-spatial-analysis.kernel-density', kernels: 1});
  for (const entry of page.featured) {
    expect(entry.recordBytes).toBe(
      Buffer.byteLength(output.read(path.join(recordsDirectory, `${entry.id}.json`)), 'utf8')
    );
    expect(entry.viewModelBytes).toBe(
      Buffer.byteLength(output.read(path.join(modelsDirectory, `${entry.id}.json`)), 'utf8')
    );
  }
});

test('X1/X5 real committed records generate matching metadata and UTF-8 totals without disk writes', () => {
  const output = runGenerators(true);
  const ids = readdirSync(recordsDirectory)
    .filter(file => file.endsWith('.json'))
    .sort();
  const summary = JSON.parse(output.read(path.join(lensDirectory, 'summary.json')));
  const page = JSON.parse(output.read(path.join(pageDirectory, 'lenses-data.json')));
  const recordBytes = ids.reduce(
    (total, file) =>
      total + Buffer.byteLength(output.read(path.join(recordsDirectory, file)), 'utf8'),
    0
  );
  const modelBytes = ids.reduce(
    (total, file) =>
      total + Buffer.byteLength(output.read(path.join(modelsDirectory, file)), 'utf8'),
    0
  );
  expect(summary.recordCount).toBe(ids.length);
  expect(summary.recordBytes).toBe(recordBytes);
  expect(summary.viewModelBytes).toBe(modelBytes);
  expect(page.bins.reduce((total: number, bin: {count: number}) => total + bin.count, 0)).toBe(
    ids.length
  );
  const counts = ids
    .map(file => {
      const model = JSON.parse(output.read(path.join(modelsDirectory, file)));
      return {
        id: file.slice(0, -5),
        kernels: model.nodes
          .filter((node: {kind: string}) => node.kind === 'compute')
          .reduce((total: number, node: {count: number}) => total + node.count, 0)
      };
    })
    .sort((left, right) => left.kernels - right.kernels);
  expect(page.median).toEqual(counts[Math.floor(counts.length / 2)]);
  for (const entry of page.featured) {
    expect(entry.recordBytes).toBe(
      Buffer.byteLength(output.read(path.join(recordsDirectory, `${entry.id}.json`)), 'utf8')
    );
    expect(entry.viewModelBytes).toBe(
      Buffer.byteLength(output.read(path.join(modelsDirectory, `${entry.id}.json`)), 'utf8')
    );
  }
});
