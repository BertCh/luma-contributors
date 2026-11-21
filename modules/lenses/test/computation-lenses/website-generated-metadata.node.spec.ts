// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from 'node:buffer';
import {readFileSync, readdirSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL, URL} from 'node:url';
import {runInNewContext} from 'node:vm';
import typescript from 'typescript';
import {expect, test} from 'vitest';
import {createElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {ResourceBars} from '../../../../website/src/components/meta/resource-bars';
import type {LensViewModel} from '../../../../website/src/lens/types';
import {
  getResourceBytesKind,
  formatCount,
  formatResourceBytes,
  formatResourceCount
} from '../../../../website/src/lens/format';
import {fromRecordedLensFile, toRecordedLensFile} from '../../src/computation-lenses/registry';
import {extractLensViewModel} from '../../../../website/src/lens/extract-lens-view-model';
import {
  createLensPageData,
  validateLensModel,
  validateLensFileNames
} from '../../../../website/src/lens/generation-metadata';

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
function runGenerators(
  useCommittedRecords = false,
  options: {
    featuredOnly?: boolean;
    selected?: string[];
    setup?: (files: Map<string, string>) => void;
    allowFailure?: boolean;
    sourceTransform?: (source: string) => string;
    entryPoint?: 'models' | 'page';
    readSource?: (file: string, text: string, count: number) => string;
  } = {}
) {
  const files = new Map<string, string>();
  for (const [index, id] of (useCommittedRecords ? [] : featuredIds).entries()) {
    files.set(
      path.join(recordsDirectory, `${id}.json`),
      JSON.stringify({
        scenarioId: id,
        label: index === 0 ? 'é' : 'plain',
        graphs: [
          {form: 'compiled', nodes: [{kind: 'compute'}, {kind: 'copy'}], edges: [], resources: []}
        ]
      })
    );
  }
  if (useCommittedRecords) {
    for (const file of readdirSync(recordsDirectory).filter(
      file =>
        file.endsWith('.json') && (!options.featuredOnly || featuredIds.includes(file.slice(0, -5)))
    )) {
      files.set(
        path.join(recordsDirectory, file),
        readFileSync(path.join(recordsDirectory, file), 'utf8')
      );
    }
  }
  for (const file of [...files.keys()]) {
    const name = path.basename(file);
    files.set(
      path.join(modelsDirectory, name),
      useCommittedRecords
        ? readFileSync(path.join(modelsDirectory, name), 'utf8')
        : JSON.stringify({id: name.slice(0, -5), title: 'previous', nodes: []})
    );
  }
  files.set(path.join(lensDirectory, 'summary.json'), 'previous summary');
  files.set(path.join(pageDirectory, 'lenses-data.json'), 'previous page metadata');
  options.setup?.(files);
  const before = new Map(files);
  const failures: string[] = [];
  const reads = new Map<string, number>();
  function read(file: string): string {
    const text = files.get(path.resolve(file));
    if (text === undefined) throw new Error(`missing fixture file ${file}`);
    const count = (reads.get(path.resolve(file)) ?? 0) + 1;
    reads.set(path.resolve(file), count);
    return options.readSource?.(path.resolve(file), text, count) ?? text;
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
    const originalText = readFileSync(file, 'utf8');
    const text = (options.sourceTransform?.(originalText) ?? originalText).replaceAll(
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
      argv: ['node', file, ...(file === generator ? (options.selected ?? []) : [])],
      execPath: process.execPath,
      stdout: {write() {}},
      exitCode: 0
    };
    try {
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
            case '@luma-contributors/lenses/computation-lenses/registry':
              return {
                fromRecordedLensFile: useCommittedRecords
                  ? fromRecordedLensFile
                  : (record: {scenarioId: string}) => ({record, scenarioId: record.scenarioId})
              };
            case './extract-lens-view-model.ts':
            case '../../../../lens/extract-lens-view-model.ts':
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
            case './generation-metadata.ts':
            case '../../../../lens/generation-metadata.ts':
              return {createLensPageData, validateLensModel, validateLensFileNames};
            default:
              throw new Error(`unexpected generator dependency ${specifier}`);
          }
        }
      });
    } catch (error) {
      errors.push((error as Error).message);
      runtimeProcess.exitCode = 1;
    }
    if (runtimeProcess.exitCode !== 0) {
      failures.push(...errors);
      if (!options.allowFailure) throw new Error(`generator failed: ${errors.join('\n')}`);
    }
  }
  execute(options.entryPoint === 'page' ? path.join(pageDirectory, 'generate-data.ts') : generator);
  return {files, read, before, failures, reads};
}

test('R1 standalone page generation uses original compute bins for every committed record', () => {
  const output = runGenerators(true, {entryPoint: 'page'});
  const page = JSON.parse(output.read(path.join(pageDirectory, 'lenses-data.json')));
  const counts = readdirSync(recordsDirectory)
    .filter(file => file.endsWith('.json'))
    .map(file => {
      const record = fromRecordedLensFile(
        JSON.parse(output.read(path.join(recordsDirectory, file)))
      ).record;
      const graph = record.graphs.find(graph => graph.form === 'compiled') ?? record.graphs[0];
      return graph.nodes.filter(node => node.kind === 'compute').length;
    });
  const starts = [0, 1, 2, 5, 10, 20, 50, 100, 300];
  expect(page.bins.map((bin: {count: number}) => bin.count)).toEqual(
    starts.map(
      (start, index) =>
        counts.filter(
          count => count >= start && (starts[index + 1] === undefined || count < starts[index + 1])
        ).length
    )
  );
  expect(
    [...output.files.keys()].filter(
      file =>
        file !== path.join(pageDirectory, 'lenses-data.json') &&
        output.files.get(file) !== output.before.get(file)
    )
  ).toEqual([]);
});

test.each([
  'malformed model',
  'swapped record'
])('R3/R4 standalone page writer refuses %s before publication', defect => {
  const output = runGenerators(true, {
    featuredOnly: true,
    entryPoint: 'page',
    allowFailure: true,
    setup(files) {
      if (defect === 'malformed model')
        files.set(path.join(modelsDirectory, `${featuredIds[1]}.json`), '{invalid model');
      else
        files.set(
          path.join(recordsDirectory, `${featuredIds[1]}.json`),
          files.get(path.join(recordsDirectory, `${featuredIds[0]}.json`))!
        );
    }
  });
  expectRejectedWithoutWrites(output);
  if (defect === 'swapped record') {
    expect(output.failures.join('\n')).toContain(featuredIds[0]);
    expect(output.failures.join('\n')).toContain(featuredIds[1]);
  }
});

test('R3 missing featured source rejects full generation before replacing any artifact', () => {
  const output = runGenerators(true, {
    featuredOnly: true,
    allowFailure: true,
    setup(files) {
      files.delete(path.join(recordsDirectory, `${featuredIds[2]}.json`));
    }
  });
  expect(output.failures.join('\n')).toContain(featuredIds[2]);
  expect(
    [...new Set([...output.files.keys(), ...output.before.keys()])].filter(
      file => output.files.get(file) !== output.before.get(file)
    )
  ).toEqual([]);
});

test('R3 malformed retained model rejects selected generation before replacing any artifact', () => {
  const output = runGenerators(true, {
    featuredOnly: true,
    selected: [featuredIds[0]],
    allowFailure: true,
    setup(files) {
      files.set(path.join(modelsDirectory, `${featuredIds[1]}.json`), '{invalid model');
      files.set(
        path.join(recordsDirectory, `${featuredIds[0]}.json`),
        `${files.get(path.join(recordsDirectory, `${featuredIds[0]}.json`))} `
      );
      files.set(
        path.join(modelsDirectory, `${featuredIds[0]}.json`),
        JSON.stringify({id: featuredIds[0], title: 'previous', nodes: []})
      );
    }
  });
  expect(output.failures.length).toBeGreaterThan(0);
  expect(
    [...new Set([...output.files.keys(), ...output.before.keys()])].filter(
      file => output.files.get(file) !== output.before.get(file)
    )
  ).toEqual([]);
});

test('R4 rejects a valid recording under another scenario filename before publication', () => {
  const output = runGenerators(true, {
    featuredOnly: true,
    selected: [featuredIds[1]],
    allowFailure: true,
    setup(files) {
      files.set(
        path.join(recordsDirectory, `${featuredIds[1]}.json`),
        files.get(path.join(recordsDirectory, `${featuredIds[0]}.json`))!
      );
    }
  });
  expect(output.failures.join('\n')).toContain(featuredIds[0]);
  expect(output.failures.join('\n')).toContain(featuredIds[1]);
  expect(
    [...new Set([...output.files.keys(), ...output.before.keys()])].filter(
      file => output.files.get(file) !== output.before.get(file)
    )
  ).toEqual([]);
});

test('R5 rejects one present and one missing requested scenario without publication', () => {
  const missing = 'gpu-spatial-analysis.does-not-exist';
  const output = runGenerators(true, {
    featuredOnly: true,
    selected: [featuredIds[0], missing],
    allowFailure: true
  });
  expect(output.failures.join('\n')).toContain(missing);
  expect(
    [...new Set([...output.files.keys(), ...output.before.keys()])].filter(
      file => output.files.get(file) !== output.before.get(file)
    )
  ).toEqual([]);
});

test('R5 selecting one of several present sources retains the unselected model bytes', () => {
  const output = runGenerators(true, {
    featuredOnly: true,
    selected: [featuredIds[0]],
    setup(files) {
      const file = path.join(modelsDirectory, `${featuredIds[1]}.json`);
      files.set(file, `${files.get(file)} `);
    }
  });
  expect(output.read(path.join(modelsDirectory, `${featuredIds[1]}.json`))).toBe(
    output.before.get(path.join(modelsDirectory, `${featuredIds[1]}.json`))
  );
  expect(output.failures).toEqual([]);
});

test('R6 selected update refreshes aggregate and featured UTF-8 sizes after a one-byte source edit', () => {
  const baseline = runGenerators(true, {featuredOnly: true});
  const baselineSummary = JSON.parse(baseline.read(path.join(lensDirectory, 'summary.json')));
  const output = runGenerators(true, {
    featuredOnly: true,
    selected: [featuredIds[1]],
    setup(files) {
      files.clear();
      for (const [file, text] of baseline.files) files.set(file, text);
      const file = path.join(recordsDirectory, `${featuredIds[1]}.json`);
      files.set(file, `${files.get(file)} `);
    }
  });
  const summary = JSON.parse(output.read(path.join(lensDirectory, 'summary.json')));
  const expectedRecordBytes = featuredIds.reduce(
    (total, id) =>
      total + Buffer.byteLength(output.read(path.join(recordsDirectory, `${id}.json`)), 'utf8'),
    0
  );
  expect(expectedRecordBytes).toBe(baselineSummary.recordBytes + 1);
  expect(summary.recordBytes).toBe(expectedRecordBytes);
  expect(summary.viewModelBytes).toBe(
    featuredIds.reduce(
      (total, id) =>
        total + Buffer.byteLength(output.read(path.join(modelsDirectory, `${id}.json`)), 'utf8'),
      0
    )
  );
  expect(summary.recordCount).toBe(3);
  const page = JSON.parse(output.read(path.join(pageDirectory, 'lenses-data.json')));
  expect(page.featured.find((entry: {id: string}) => entry.id === featuredIds[1]).recordBytes).toBe(
    Buffer.byteLength(output.read(path.join(recordsDirectory, `${featuredIds[1]}.json`)), 'utf8')
  );
  for (const id of [featuredIds[0], featuredIds[2]])
    expect(output.read(path.join(modelsDirectory, `${id}.json`))).toBe(
      baseline.read(path.join(modelsDirectory, `${id}.json`))
    );
});

test.each([
  'missing',
  'empty',
  'mixed'
] as const)('R7 aggregate and displayed resource availability for %s inventories', mode => {
  const output = runGenerators(true, {
    featuredOnly: true,
    setup(files) {
      for (const [index, id] of featuredIds.entries()) {
        const file = path.join(recordsDirectory, `${id}.json`);
        const envelope = fromRecordedLensFile(JSON.parse(files.get(file)!));
        const graphs = [envelope.record.graphs[0]].map(graph => {
          const next = {...graph, nodes: [], edges: []};
          if (mode === 'missing' || (mode === 'mixed' && index === 0)) delete next.resources;
          else
            next.resources =
              mode === 'mixed' && index === 1
                ? [
                    {
                      id: 'fixture-buffer',
                      kind: 'buffer',
                      bytes: 16,
                      ownership: 'imported',
                      bytesKind: 'logical'
                    }
                  ]
                : [];
          return next;
        });
        files.set(
          file,
          JSON.stringify(
            toRecordedLensFile(
              {...envelope.record, graphs, captures: undefined, metrics: undefined},
              envelope
            )
          )
        );
      }
    }
  });
  const summary = JSON.parse(output.read(path.join(lensDirectory, 'summary.json')));
  expect(summary.inventoryKnown).toBe(mode === 'empty');
  expect(summary.resourceCount).toBe(mode === 'mixed' ? 1 : 0);
  expect(summary.resourceBytes).toBe(mode === 'mixed' ? 16 : 0);
  expect(summary.resourceBytesKind).toBe(mode === 'empty' ? 'logical' : 'unavailable');
  const source = typescript.createSourceFile(
    'page.tsx',
    readFileSync(path.join(pageDirectory, 'index.tsx'), 'utf8'),
    typescript.ScriptTarget.Latest,
    true,
    typescript.ScriptKind.TSX
  );
  let statistic: typescript.ObjectLiteralExpression | undefined;
  function visit(node: typescript.Node) {
    if (
      typescript.isObjectLiteralExpression(node) &&
      node.properties.some(
        property =>
          typescript.isPropertyAssignment(property) &&
          property.name.getText(source) === 'value' &&
          property.initializer.getText(source).includes('totals.resourceCount')
      )
    )
      statistic = node;
    typescript.forEachChild(node, visit);
  }
  visit(source);
  if (!statistic) throw new Error('missing resource aggregate statistic');
  const code = typescript.transpileModule(
    `export const statistic = ${statistic.getText(source)};`,
    {compilerOptions: {module: typescript.ModuleKind.CommonJS}}
  ).outputText;
  const exports: {statistic?: unknown} = {};
  runInNewContext(code, {
    exports,
    totals: summary,
    formatCount,
    formatResourceBytes,
    formatResourceCount
  });
  expect(exports.statistic).toEqual(
    mode === 'empty'
      ? {value: '0 resources', label: '0 B'}
      : {value: 'inventory unavailable', label: 'size unavailable'}
  );
});

test.each([
  'texture',
  'opaque plus legacy buffer'
] as const)('review 2 R7 displayed resource inventory handles %s', mode => {
  const output = runGenerators(true, {
    featuredOnly: true,
    setup(files) {
      for (const [index, id] of featuredIds.entries()) {
        const file = path.join(recordsDirectory, `${id}.json`);
        const envelope = fromRecordedLensFile(JSON.parse(files.get(file)!));
        const graph = envelope.record.graphs[0];
        const resources: NonNullable<typeof graph.resources> =
          index === 0
            ? [
                mode === 'texture'
                  ? {
                      id: 'texture',
                      kind: 'texture',
                      ownership: 'imported',
                      bytes: 64,
                      bytesKind: 'estimated'
                    }
                  : {
                      id: 'opaque',
                      kind: 'external-texture',
                      ownership: 'imported',
                      bytes: 0,
                      bytesKind: 'unavailable'
                    }
              ]
            : mode !== 'texture' && index === 1
              ? [{id: 'legacy', kind: 'buffer', ownership: 'imported', bytes: 16}]
              : [];
        files.set(
          file,
          JSON.stringify(
            toRecordedLensFile(
              {
                ...envelope.record,
                graphs: [{...graph, nodes: [], edges: [], resources}],
                captures: undefined,
                metrics: undefined
              },
              envelope
            )
          )
        );
      }
    }
  });
  const summary = JSON.parse(output.read(path.join(lensDirectory, 'summary.json')));
  expect(summary.resourceBytesKind).toBe(mode === 'texture' ? 'estimated' : 'unavailable');
  const source = typescript.createSourceFile(
    'page.tsx',
    readFileSync(path.join(pageDirectory, 'index.tsx'), 'utf8'),
    typescript.ScriptTarget.Latest,
    true,
    typescript.ScriptKind.TSX
  );
  let statistic: typescript.ObjectLiteralExpression | undefined;
  function visit(node: typescript.Node) {
    if (
      typescript.isObjectLiteralExpression(node) &&
      node.properties.some(
        property =>
          typescript.isPropertyAssignment(property) &&
          property.name.getText(source) === 'value' &&
          property.initializer.getText(source).includes('formatResourceCount')
      )
    )
      statistic = node;
    typescript.forEachChild(node, visit);
  }
  visit(source);
  if (!statistic) throw new Error('missing inventory statistic');
  const code = typescript.transpileModule(
    `export const statistic = ${statistic.getText(source)};`,
    {compilerOptions: {module: typescript.ModuleKind.CommonJS}}
  ).outputText;
  const exports: {statistic?: unknown} = {};
  runInNewContext(code, {exports, totals: summary, formatResourceCount, formatResourceBytes});
  expect(exports.statistic).toEqual(
    mode === 'texture'
      ? {value: '1 resource', label: '~64 B'}
      : {value: '2 resources', label: 'size unavailable'}
  );
  const model: LensViewModel = JSON.parse(
    output.files.get(path.join(modelsDirectory, `${featuredIds[0]}.json`))!
  );
  const html = renderToStaticMarkup(createElement(ResourceBars, {model}));
  expect(html).toContain('largest resources');
  expect(html).toContain('1 resource');
  expect(html).not.toContain('1 buffers');
  expect(html).toContain(mode === 'texture' ? '~64 B' : 'size unavailable');
});

function expectRejectedWithoutWrites(output: ReturnType<typeof runGenerators>) {
  expect(output.failures.length).toBeGreaterThan(0);
  // Independent oracle: rejection must leave every pre-existing byte and filename unchanged.
  expect(
    [...new Set([...output.files.keys(), ...output.before.keys()])].filter(
      file => output.files.get(file) !== output.before.get(file)
    )
  ).toEqual([]);
}

test('review 2 R4 standalone page writer rejects a missing nonfeatured model', () => {
  const id = 'gpu-spatial-analysis.neighbor-search';
  const output = runGenerators(true, {
    featuredOnly: true,
    entryPoint: 'page',
    allowFailure: true,
    setup(files) {
      files.set(
        path.join(recordsDirectory, `${id}.json`),
        readFileSync(path.join(recordsDirectory, `${id}.json`), 'utf8')
      );
      files.set(
        path.join(modelsDirectory, `${id}.json`),
        readFileSync(path.join(modelsDirectory, `${id}.json`), 'utf8')
      );
      files.delete(path.join(modelsDirectory, `${id}.json`));
    }
  });
  expect(output.failures.join('\n')).toContain(`${id}.json`);
  expectRejectedWithoutWrites(output);
});

test('review 2 R1 four valid copy-only recordings occupy the zero-compute bin in both writers', () => {
  const output = runGenerators(true, {
    featuredOnly: true,
    setup(files) {
      const neighbor = 'gpu-spatial-analysis.neighbor-search';
      files.set(
        path.join(recordsDirectory, `${neighbor}.json`),
        readFileSync(path.join(recordsDirectory, `${neighbor}.json`), 'utf8')
      );
      for (const id of [...featuredIds, neighbor]) {
        const file = path.join(recordsDirectory, `${id}.json`);
        const envelope = fromRecordedLensFile(JSON.parse(files.get(file)!));
        const graph = envelope.record.graphs[0];
        const copy = graph.nodes.find(node => node.kind === 'copy')!;
        expect(copy.kind).toBe('copy');
        files.set(
          file,
          JSON.stringify(
            toRecordedLensFile(
              {
                ...envelope.record,
                graphs: [
                  {
                    ...graph,
                    nodes: [copy],
                    edges: [],
                    resources: graph.resources?.map(resource => ({
                      ...resource,
                      firstNode: undefined,
                      lastNode: undefined
                    }))
                  }
                ],
                captures: undefined,
                metrics: undefined
              },
              envelope
            )
          )
        );
      }
    }
  });
  const summary = JSON.parse(output.files.get(path.join(lensDirectory, 'summary.json'))!);
  expect({records: summary.recordCount, copies: summary.copies, kernels: summary.kernels}).toEqual({
    records: 4,
    copies: 4,
    kernels: 0
  });
  const expected = [4, 0, 0, 0, 0, 0, 0, 0, 0];
  expect(
    JSON.parse(output.files.get(path.join(pageDirectory, 'lenses-data.json'))!).bins.map(
      (bin: {count: number}) => bin.count
    )
  ).toEqual(expected);
  const page = runGenerators(true, {
    featuredOnly: true,
    entryPoint: 'page',
    setup(files) {
      files.clear();
      for (const [file, text] of output.files) files.set(file, text);
    }
  });
  expect(
    JSON.parse(page.files.get(path.join(pageDirectory, 'lenses-data.json'))!).bins.map(
      (bin: {count: number}) => bin.count
    )
  ).toEqual(expected);
});

test.each([
  false,
  true
])('review 2 R5 model, compute count and bytes use one source read with selected=%s', selected => {
  const target = path.join(recordsDirectory, `${featuredIds[1]}.json`);
  let replacement = '';
  let expectedNodes = 0;
  let expectedKernels = 0;
  let expectedBytes = 0;
  const output = runGenerators(true, {
    featuredOnly: true,
    selected: selected ? [featuredIds[1]] : undefined,
    allowFailure: true,
    setup(files) {
      const original = files.get(target)!;
      const envelope = fromRecordedLensFile(JSON.parse(original));
      const graph = envelope.record.graphs[0];
      expectedNodes = graph.nodes.length;
      expectedKernels = graph.nodes.filter(node => node.kind === 'compute').length;
      expectedBytes = Buffer.byteLength(original, 'utf8');
      replacement = JSON.stringify(
        toRecordedLensFile(
          {
            ...envelope.record,
            graphs: [
              {
                ...graph,
                nodes: [graph.nodes.find(node => node.kind === 'compute')!],
                edges: [],
                resources: graph.resources?.map(resource => ({
                  ...resource,
                  firstNode: undefined,
                  lastNode: undefined
                }))
              }
            ],
            captures: undefined,
            metrics: undefined
          },
          envelope
        )
      );
      expect(fromRecordedLensFile(JSON.parse(replacement)).record.graphs[0].nodes).toHaveLength(1);
    },
    readSource(file, text, count) {
      return file === target && count > 1 ? replacement : text;
    }
  });
  expect(output.reads.get(target)).toBe(1);
  expect(output.failures).toEqual([]);
  const model = JSON.parse(output.files.get(path.join(modelsDirectory, `${featuredIds[1]}.json`))!);
  expect(model.recordedNodeCount).toBe(expectedNodes);
  const page = JSON.parse(output.files.get(path.join(pageDirectory, 'lenses-data.json'))!);
  expect(page.featured.find((entry: {id: string}) => entry.id === featuredIds[1]).recordBytes).toBe(
    expectedBytes
  );
  const summary = JSON.parse(output.files.get(path.join(lensDirectory, 'summary.json'))!);
  const otherKernels = [featuredIds[0], featuredIds[2]].reduce((total, id) => {
    const record = fromRecordedLensFile(
      JSON.parse(output.before.get(path.join(recordsDirectory, `${id}.json`))!)
    ).record;
    return total + record.graphs[0].nodes.filter(node => node.kind === 'compute').length;
  }, 0);
  expect(summary.kernels).toBe(expectedKernels + otherKernels);
});

test.each(
  (['models', 'page'] as const).flatMap(entryPoint =>
    ['ID-only', 'stale graph'].map(defect => ({entryPoint, defect}))
  )
)('review 2 R3 $entryPoint writer rejects $defect retained model content', ({
  entryPoint,
  defect
}) => {
  const output = runGenerators(true, {
    featuredOnly: true,
    entryPoint,
    selected: [featuredIds[0]],
    allowFailure: true,
    setup(files) {
      const file = path.join(modelsDirectory, `${featuredIds[1]}.json`);
      if (defect === 'ID-only') files.set(file, JSON.stringify({id: featuredIds[1]}));
      else {
        const source = path.join(recordsDirectory, `${featuredIds[1]}.json`);
        const envelope = fromRecordedLensFile(JSON.parse(files.get(source)!));
        const graph = envelope.record.graphs[0];
        const node = graph.nodes.find(node => node.kind === 'compute')!;
        files.set(
          source,
          JSON.stringify(
            toRecordedLensFile(
              {
                ...envelope.record,
                graphs: [
                  {
                    ...graph,
                    nodes: [node],
                    edges: [],
                    resources: graph.resources?.map(resource => ({
                      ...resource,
                      firstNode: undefined,
                      lastNode: undefined
                    }))
                  }
                ],
                captures: undefined,
                metrics: undefined
              },
              envelope
            )
          )
        );
      }
    }
  });
  expectRejectedWithoutWrites(output);
  expect(output.failures.join('\n')).toContain(featuredIds[1]);
});

test.each([
  'models',
  'page'
] as const)('review 2 R3 %s writer preserves harmless retained whitespace', entryPoint => {
  const output = runGenerators(true, {
    featuredOnly: true,
    entryPoint,
    selected: [featuredIds[0]],
    setup(files) {
      const file = path.join(modelsDirectory, `${featuredIds[1]}.json`);
      files.set(file, ` \n${files.get(file)} \n`);
    }
  });
  expect(output.failures).toEqual([]);
  expect(output.read(path.join(modelsDirectory, `${featuredIds[1]}.json`))).toBe(
    output.before.get(path.join(modelsDirectory, `${featuredIds[1]}.json`))
  );
});

test.each([
  'models',
  'page'
] as const)('review 2 R3 stale-content oracle detects the previous %s writer', entryPoint => {
  const output = runGenerators(true, {
    featuredOnly: true,
    entryPoint,
    selected: [featuredIds[0]],
    allowFailure: true,
    sourceTransform(source) {
      return source.replace(/^\s*validateLensModel\([^;]*\);/gm, '');
    },
    setup(files) {
      const source = path.join(recordsDirectory, `${featuredIds[1]}.json`);
      const envelope = fromRecordedLensFile(JSON.parse(files.get(source)!));
      const graph = envelope.record.graphs[0];
      const replacement = toRecordedLensFile(
        {
          ...envelope.record,
          graphs: [
            {
              ...graph,
              nodes: [graph.nodes.find(node => node.kind === 'compute')!],
              edges: [],
              resources: graph.resources?.map(resource => ({
                ...resource,
                firstNode: undefined,
                lastNode: undefined
              }))
            }
          ],
          captures: undefined,
          metrics: undefined
        },
        envelope
      );
      expect(fromRecordedLensFile(replacement).record.graphs[0].nodes).toHaveLength(1);
      files.set(source, JSON.stringify(replacement));
    }
  });
  expect(output.failures).toEqual([]);
  expect(() => expectRejectedWithoutWrites(output)).toThrow();
});

test.each([
  false,
  true
])('R9 malformed source preserves all prior artifacts in selected=%s mode', selected => {
  const output = runGenerators(true, {
    featuredOnly: true,
    selected: selected ? [featuredIds[0], featuredIds[1]] : undefined,
    allowFailure: true,
    setup(files) {
      files.set(path.join(recordsDirectory, `${featuredIds[1]}.json`), '{invalid JSON');
    }
  });
  expect(output.failures.join('\n')).toContain(featuredIds[1]);
  expectRejectedWithoutWrites(output);
});

test('R9 successful full generation prunes obsolete JSON and retains other files', () => {
  const output = runGenerators(true, {
    featuredOnly: true,
    setup(files) {
      files.set(path.join(modelsDirectory, 'obsolete.json'), 'old model');
      files.set(path.join(modelsDirectory, 'keep.txt'), 'retained bytes');
    }
  });
  expect(output.files.has(path.join(modelsDirectory, 'obsolete.json'))).toBe(false);
  expect(output.read(path.join(modelsDirectory, 'keep.txt'))).toBe('retained bytes');
  expect(JSON.parse(output.read(path.join(lensDirectory, 'summary.json'))).recordCount).toBe(3);
  expect(
    JSON.parse(output.read(path.join(pageDirectory, 'lenses-data.json'))).featured.map(
      (entry: {id: string}) => entry.id
    )
  ).toEqual(featuredIds);
});

test('R9 successful selected generation retains unselected bytes and refreshes all metadata', () => {
  const output = runGenerators(true, {featuredOnly: true, selected: [featuredIds[0]]});
  for (const id of [featuredIds[1], featuredIds[2]])
    expect(output.read(path.join(modelsDirectory, `${id}.json`))).toBe(
      output.before.get(path.join(modelsDirectory, `${id}.json`))
    );
  expect(JSON.parse(output.read(path.join(lensDirectory, 'summary.json'))).recordCount).toBe(3);
  expect(
    JSON.parse(output.read(path.join(pageDirectory, 'lenses-data.json'))).featured
  ).toHaveLength(3);
});

test('R9 rejection oracle detects a reintroduced eager-deletion implementation', () => {
  const output = runGenerators(true, {
    featuredOnly: true,
    allowFailure: true,
    setup(files) {
      files.set(path.join(recordsDirectory, `${featuredIds[1]}.json`), '{invalid JSON');
    },
    sourceTransform(source) {
      return source.replace(
        'const pendingModels = new Map<string, string>();',
        `const pendingModels = new Map<string, string>();
for (const file of fs.readdirSync(outputDirectory)) if (file.endsWith('.json')) fs.unlinkSync(path.join(outputDirectory, file));`
      );
    }
  });
  expect(() => expectRejectedWithoutWrites(output)).toThrow();
});

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
  expect(output.failures).toEqual([]);
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
      const record = fromRecordedLensFile(
        JSON.parse(output.read(path.join(recordsDirectory, file)))
      ).record;
      const graph = record.graphs.find(graph => graph.form === 'compiled') ?? record.graphs[0];
      return {
        id: file.slice(0, -5),
        kernels: graph.nodes.filter(node => node.kind === 'compute').length
      };
    })
    .sort((left, right) => left.kernels - right.kernels);
  expect(page.median).toEqual(counts[Math.floor(counts.length / 2)]);
  const starts = [0, 1, 2, 5, 10, 20, 50, 100, 300];
  expect(page.bins.map((bin: {count: number}) => bin.count)).toEqual(
    starts.map(
      (start, index) =>
        counts.filter(
          entry =>
            entry.kernels >= start &&
            (starts[index + 1] === undefined || entry.kernels < starts[index + 1])
        ).length
    )
  );
  for (const entry of page.featured) {
    expect(entry.recordBytes).toBe(
      Buffer.byteLength(output.read(path.join(recordsDirectory, `${entry.id}.json`)), 'utf8')
    );
    expect(entry.viewModelBytes).toBe(
      Buffer.byteLength(output.read(path.join(modelsDirectory, `${entry.id}.json`)), 'utf8')
    );
  }
});
