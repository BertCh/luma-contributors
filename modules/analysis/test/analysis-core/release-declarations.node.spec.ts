// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import typescript from 'typescript';
import {expect, it} from 'vitest';

const WORKSPACE_DIRECTORY = fileURLToPath(new URL('../../../../', import.meta.url));

it('the current line-density declaration preserves schema types in a native NodeNext consumer', () => {
  const directory = mkdtempSync(join(tmpdir(), 'analysis-native-declarations-'));
  try {
    const sourcePath = join(
      WORKSPACE_DIRECTORY,
      'modules/analysis/src/gpu-spatial-analysis/line-density/gpu-line-density.ts'
    );
    const schemaPath = join(
      WORKSPACE_DIRECTORY,
      'modules/analysis/src/analysis-core/contracts/parameter-schema.ts'
    );
    const configuration = typescript.readConfigFile(
      join(WORKSPACE_DIRECTORY, 'tsconfig.json'),
      typescript.sys.readFile
    );
    const parsed = typescript.parseJsonConfigFileContent(
      configuration.config,
      typescript.sys,
      WORKSPACE_DIRECTORY
    );
    const program = typescript.createProgram([sourcePath], {
      ...parsed.options,
      noEmit: false,
      declaration: true,
      emitDeclarationOnly: true
    });
    const declarations = new Map<string, string>();
    for (const path of [sourcePath, schemaPath]) {
      const source = program.getSourceFile(path);
      expect(source, path).toBeDefined();
      const result = program.emit(
        source,
        (file, text) => {
          declarations.set(file, text);
        },
        undefined,
        true
      );
      expect(result.emitSkipped, path).toBe(false);
    }
    const findDeclaration = (suffix: string) => {
      const declaration = [...declarations].find(([path]) => path.endsWith(suffix))?.[1];
      expect(declaration, suffix).toBeDefined();
      return declaration!;
    };
    const files = {
      'node_modules/@luma-contributors/analysis/package.json': JSON.stringify({
        name: '@luma-contributors/analysis',
        type: 'module',
        exports: {
          './gpu-spatial-analysis': {
            types: './dist/gpu-spatial-analysis/line-density/gpu-line-density.d.ts'
          }
        }
      }),
      'node_modules/@luma-contributors/analysis/dist/gpu-spatial-analysis/line-density/gpu-line-density.d.ts':
        findDeclaration('/line-density/gpu-line-density.d.ts'),
      'node_modules/@luma-contributors/analysis/dist/analysis-core/contracts/parameter-schema.d.ts':
        findDeclaration('/contracts/parameter-schema.d.ts'),
      'node_modules/@luma-contributors/analysis/dist/analysis-core/contracts/index.d.ts':
        "export type {GPUSpatialParameterSchema} from './parameter-schema.js';",
      'node_modules/@luma-contributors/analysis/dist/analysis-core/index.d.ts':
        "export type {GPUSpatialParameterSchema} from './contracts/index.js';",
      'consumer.mts': readFileSync(join(WORKSPACE_DIRECTORY, 'scripts/packed-consumer.mts'), 'utf8')
    };
    for (const [path, text] of Object.entries(files)) {
      const target = join(directory, path);
      mkdirSync(dirname(target), {recursive: true});
      writeFileSync(target, text);
    }
    const consumer = typescript.createProgram([join(directory, 'consumer.mts')], {
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
          `TS${diagnostic.code}: ${typescript.flattenDiagnosticMessageText(diagnostic.messageText, '\n')}`
      );
    expect(diagnostics).toEqual([]);
  } finally {
    rmSync(directory, {recursive: true, force: true});
  }
}, 30_000);
