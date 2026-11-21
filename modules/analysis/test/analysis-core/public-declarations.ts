// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {join, resolve, sep} from 'node:path';
import type typescript from 'typescript';
import {
  createPublicSourceInventory,
  createSourceResolvers
} from '../../../../scripts/contributor-contract-source-inventory.mjs';
import {REPOSITORY_ROOT} from './doc-program';

export type PublicDeclaration = {
  entryPoint: string;
  exportName: string;
  declaration: string;
  sourceTarget?: string;
};

/** Compiler-backed docs and vocabulary use the same manifest/origin walk as the CLI checker. */
export function getPublicDeclarations(
  program: typescript.Program,
  repositoryRoot = REPOSITORY_ROOT
): PublicDeclaration[] {
  const sourceRoot = join(repositoryRoot, 'modules/analysis/src');
  const manifest = JSON.parse(
    readFileSync(join(repositoryRoot, 'modules/analysis/package.json'), 'utf8')
  );
  return getPublicDeclarationsFromSources(program, sourceRoot, manifest);
}

export function getPublicDeclarationsFromSources(
  program: typescript.Program,
  sourceRoot: string,
  analysisManifest: {name?: string; exports: Record<string, unknown>}
): PublicDeclaration[] {
  const sourceTrees = new Map(
    program
      .getSourceFiles()
      .filter(source => resolve(source.fileName).startsWith(sourceRoot + sep))
      .map(source => [resolve(source.fileName), source])
  );
  return createPublicSourceInventory({
    analysisManifest,
    sourceRoot,
    sourceTrees,
    ...createSourceResolvers(sourceTrees)
  }).inventory;
}

/** Resolve a canonical origin to its defining compiler declaration, preserving alias identity. */
export function getPublicDeclarationNode(
  program: typescript.Program,
  origin: string
): typescript.Declaration | undefined {
  const separator = origin.lastIndexOf('#');
  const source = program.getSourceFile(origin.slice(0, separator));
  const name = origin.slice(separator + 1);
  for (const statement of source?.statements ?? []) {
    const declarations =
      'declarationList' in statement
        ? (statement as typescript.VariableStatement).declarationList.declarations
        : [statement];
    for (const declaration of declarations) {
      const declarationName = (declaration as typescript.NamedDeclaration).name;
      if (declarationName?.getText() === name) {
        return declaration as typescript.Declaration;
      }
    }
  }
  return undefined;
}
