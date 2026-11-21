// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, mkdtempSync, mkdirSync, rmSync, writeFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {tmpdir} from 'node:os';
import {dirname, join, relative, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import typescript from 'typescript';
import {expect, it} from 'vitest';
import {checkDeclarationEdges} from '../../../../scripts/check-packed-consumer.mjs';

const WORKSPACE_DIRECTORY = fileURLToPath(new URL('../../../../', import.meta.url));
const SOURCE_DIRECTORY = join(WORKSPACE_DIRECTORY, 'modules/analysis/src');

// Declarations the shared consumer fixtures use, emitted from current source.
const ROOT_SOURCES = [
  'gpu-spatial-analysis/line-density/gpu-line-density.ts',
  'gpu-spatial-analysis/line-segmentize/line-segmentize-parameters.ts',
  'dataframe-analysis/pair-statistics/cpu-variogram-model-fit-operation.ts'
];

/** Relative specifiers of one emitted declaration: import/export declarations and import types. */
function listRelativeSpecifiers(fileName: string, text: string): string[] {
  const sourceFile = typescript.createSourceFile(fileName, text, typescript.ScriptTarget.ES2022);
  const specifiers: string[] = [];
  const visit = (node: typescript.Node) => {
    if (
      (typescript.isImportDeclaration(node) || typescript.isExportDeclaration(node)) &&
      node.moduleSpecifier &&
      typescript.isStringLiteral(node.moduleSpecifier)
    )
      specifiers.push(node.moduleSpecifier.text);
    else if (
      typescript.isImportTypeNode(node) &&
      typescript.isLiteralTypeNode(node.argument) &&
      typescript.isStringLiteral(node.argument.literal)
    )
      specifiers.push(node.argument.literal.text);
    typescript.forEachChild(node, visit);
  };
  visit(sourceFile);
  return specifiers.filter(specifier => specifier.startsWith('.'));
}

it('the current line-density, line-segmentize and variogram declarations preserve types in native NodeNext consumers', () => {
  const directory = mkdtempSync(join(tmpdir(), 'analysis-native-declarations-'));
  try {
    const configuration = typescript.readConfigFile(
      join(WORKSPACE_DIRECTORY, 'tsconfig.json'),
      typescript.sys.readFile
    );
    const parsed = typescript.parseJsonConfigFileContent(
      configuration.config,
      typescript.sys,
      WORKSPACE_DIRECTORY
    );
    const program = typescript.createProgram(
      ROOT_SOURCES.map(file => join(SOURCE_DIRECTORY, file)),
      {...parsed.options, noEmit: false, declaration: true, emitDeclarationOnly: true}
    );
    // The production append-extension transform, as configured for `afterDeclarations` in the
    // root tsconfig: it rewrites import/export declarations, never synthesized import types.
    const plugin = parsed.options['plugins'] as (typescript.PluginImport & {
      extensions: string[];
    })[];
    const appendExtension = createRequire(import.meta.url)(
      '@vis.gl/ts-plugins/ts-transform-append-extension'
    ).default;
    const transformer = appendExtension(program, plugin[1], {ts: typescript});
    const packageDirectory = join(directory, 'node_modules/@luma-contributors/analysis');
    // Emit the relative declaration closure of the roots, as the build would ship it.
    const pending = ROOT_SOURCES.map(file => join(SOURCE_DIRECTORY, file));
    const emitted = new Set<string>();
    while (pending.length) {
      const sourcePath = pending.pop()!;
      if (emitted.has(sourcePath)) continue;
      emitted.add(sourcePath);
      const source = program.getSourceFile(sourcePath);
      expect(source, sourcePath).toBeDefined();
      let declaration = '';
      const result = program.emit(
        source,
        (_file, text) => {
          declaration = text;
        },
        undefined,
        true,
        {afterDeclarations: [transformer]}
      );
      expect(result.emitSkipped, sourcePath).toBe(false);
      const target = join(
        packageDirectory,
        'dist',
        relative(SOURCE_DIRECTORY, sourcePath).replace(/\.ts$/, '.d.ts')
      );
      mkdirSync(dirname(target), {recursive: true});
      writeFileSync(target, declaration);
      for (const specifier of listRelativeSpecifiers(target, declaration)) {
        const base = resolve(dirname(sourcePath), specifier.replace(/\.js$/, ''));
        const next = [`${base}.ts`, join(base, 'index.ts')].find(candidate =>
          existsSync(candidate)
        );
        // An unresolvable specifier stays unemitted, and the edge census below reports it.
        if (next) pending.push(next);
      }
    }
    const files: Record<string, string> = {
      'package.json': JSON.stringify({
        name: '@luma-contributors/analysis',
        type: 'module',
        exports: {
          './gpu-spatial-analysis': {types: './dist/gpu-spatial-analysis/entry.d.ts'},
          './dataframe-analysis': {types: './dist/dataframe-analysis/entry.d.ts'}
        }
      }),
      // Selective entries keep the closure small; the declarations they re-export are emitted.
      'dist/gpu-spatial-analysis/entry.d.ts': [
        "export {GPU_LINE_DENSITY_PARAMETER_SCHEMA} from './line-density/gpu-line-density.js';",
        "export {GPU_LINE_SEGMENTIZE_PARAMETER_SCHEMA} from './line-segmentize/line-segmentize-parameters.js';"
      ].join('\n'),
      'dist/dataframe-analysis/entry.d.ts':
        "export {computeVariogramModelFitOnCPU} from './pair-statistics/cpu-variogram-model-fit-operation.js';"
    };
    for (const [path, text] of Object.entries(files)) {
      const target = join(packageDirectory, path);
      mkdirSync(dirname(target), {recursive: true});
      writeFileSync(target, text);
    }
    // Every relative edge of the emitted declarations resolves under NodeNext (TOOL-102).
    const census = checkDeclarationEdges(packageDirectory);
    expect(census.declarationCount).toBe(emitted.size + 2);
    const consumers = ['mts', 'cts'].map(extension => {
      const consumer = join(directory, `consumer.${extension}`);
      // The full packed-consumer scripts cover every entry. This selective emitted package
      // covers precisely the three declarations named by this regression, in both formats.
      writeFileSync(
        consumer,
        `
import {GPU_LINE_DENSITY_PARAMETER_SCHEMA, GPU_LINE_SEGMENTIZE_PARAMETER_SCHEMA} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {computeVariogramModelFitOnCPU} from '@luma-contributors/analysis/dataframe-analysis';
type Assert<Value extends true> = Value;
type IsErased<Value> = 0 extends 1 & Value ? true : false;
type DensityRetainsType = Assert<IsErased<typeof GPU_LINE_DENSITY_PARAMETER_SCHEMA> extends false ? true : false>;
type SegmentRetainsType = Assert<IsErased<typeof GPU_LINE_SEGMENTIZE_PARAMETER_SCHEMA> extends false ? true : false>;
type DensityRejectsInventedField = Assert<'notARealField' extends keyof typeof GPU_LINE_DENSITY_PARAMETER_SCHEMA ? false : true>;
type SegmentRejectsInventedField = Assert<'notARealField' extends keyof typeof GPU_LINE_SEGMENTIZE_PARAMETER_SCHEMA ? false : true>;
const wordLength: number = GPU_LINE_DENSITY_PARAMETER_SCHEMA.wordLength;
const format: 'float32' = GPU_LINE_SEGMENTIZE_PARAMETER_SCHEMA.format;
const fit = computeVariogramModelFitOnCPU({distances: [1], semivariances: [1], pairCounts: [1]});
const status: Uint32Array = fit.status;
type FitRetainsType = Assert<IsErased<typeof fit> extends false ? true : false>;
type FitRejectsInventedField = Assert<'notARealField' extends keyof typeof fit ? false : true>;
void [wordLength, format, status];
`
      );
      return consumer;
    });
    const consumer = typescript.createProgram(consumers, {
      module: typescript.ModuleKind.NodeNext,
      moduleResolution: typescript.ModuleResolutionKind.NodeNext,
      target: typescript.ScriptTarget.ES2022,
      skipLibCheck: true,
      types: [],
      strict: true,
      noEmit: true
    });
    const diagnostics = typescript
      .getPreEmitDiagnostics(consumer)
      .map(
        diagnostic =>
          `${diagnostic.file ? relative(directory, diagnostic.file.fileName) : ''} TS${diagnostic.code}: ${typescript.flattenDiagnosticMessageText(diagnostic.messageText, '\n')}`
      );
    expect(diagnostics).toEqual([]);
  } finally {
    rmSync(directory, {recursive: true, force: true});
  }
}, 60_000);
