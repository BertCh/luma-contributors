// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readFileSync} from 'node:fs';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import typescript from 'typescript';

// Shared by the docs specs (A-7, A-22): the analysis doc pages, the entry barrels they import,
// and a TypeScript program over the repo's own compiler options (the root `tsconfig.json` paths
// map `@luma.gl/*` package names to `modules/*/src`) with optional in-memory files.

export const REPOSITORY_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
export const DOCS_ROOT = join(REPOSITORY_ROOT, 'docs/api');

/**
 * The analysis doc pages, relative to `DOCS_ROOT`. Docs describe only this library's entries;
 * luma.gl-owned APIs are linked to the luma.gl docs, not documented here.
 */
export const ANALYSIS_DOC_PAGES = [
  'gpu-spatial-analysis/README.md',
  'gpu-spatial-analysis/concepts.md',
  'gpu-spatial-analysis/temporal-context.md',
  'gpu-spatial-analysis/vecchia-likelihood.md',
  'gpu-spatial-analysis/change-points-and-expectation-based-scan.md',
  'gpu-spatial-analysis/movement-semantics.md',
  'gpu-spatial-analysis/home-range-and-recursion-mob07.md',
  'gpu-spatial-analysis/space-time-density-and-k-function.md',
  'gpu-spatial-analysis/space-time-hawkes.md',
  'gpu-spatial-analysis/sinkhorn-divergence.md',
  'gpu-spatial-analysis/operators.md',
  'gpu-spatial-analysis/operators-weights.md',
  'gpu-spatial-analysis/operators-statistics.md',
  'gpu-spatial-analysis/operators-dynamics.md',
  'gpu-spatial-analysis/operators-accessibility.md',
  'gpu-spatial-analysis/operators-joins.md',
  'gpu-spatial-analysis/operators-validity.md',
  'gpu-spatial-analysis/operators-density.md',
  'gpu-spatial-analysis/operators-cells.md',
  'gpu-spatial-analysis/operators-lines.md',
  'gpu-spatial-analysis/operators-geometry.md',
  'gpu-spatial-analysis/operators-regression.md',
  'gpu-spatial-analysis/spatial-validation-advanced.md',
  'gpu-spatial-analysis/geographical-detector-redcap-entropy.md',
  'gpu-spatial-analysis/colocation-category-dependence-adaptive-density.md',
  'gpu-spatial-analysis/recipes.md',
  'gpu-spatial-analysis-cross-reference.md',
  'gpu-network/README.md',
  'gpu-network/concepts.md',
  'gpu-network/operators.md',
  'gpu-network/flow-models-mob10.md',
  'gpu-network/map-matched-edge-volumes-mob05.md',
  'gpu-network/network-time-geography-mob09.md',
  'gpu-terrain/README.md',
  'gpu-terrain/concepts.md',
  'gpu-terrain/operators.md',
  'gpu-dataframe-analysis.md',
  'geospatial.md',
  'gpu-raster/README.md',
  'gpu-raster/operations.md',
  'gpu-raster/operations-analysis.md',
  'gpu-raster/raster-patches.md',
  'gpu-raster/expansion.md',
  'gpu-raster/fill-nodata.md',
  'gpu-raster/focal-pairs.md',
  'gpu-raster/fractional-categorical-statistics.md',
  'gpu-raster/fractional-zonal-distribution.md',
  'gpu-raster/remote-sensing.md',
  'gpu-raster/raster-change-monitoring.md',
  'gpu-raster/raster-eof-analysis.md',
  'gpu-raster/lagrangian-diagnostics-mob06.md',
  'gpu-raster/raster-time-series.md',
  'gpu-raster/raster-embeddings.md',
  'gpu-raster/spectral-indices.md',
  'gpu-raster/tile-reprojection.md',
  'pointcloud-analysis.md',
  'variable-standardization.md',
  'computation-record.md',
  'gpu-geometry-refinement.md'
] as const;

/** Package entries an analysis page may import, in the order free names are resolved. */
export const ANALYSIS_ENTRIES = [
  '@luma-contributors/analysis/gpu-spatial-analysis',
  '@luma-contributors/analysis/gpu-network',
  '@luma-contributors/analysis/gpu-terrain',
  '@luma-contributors/analysis/raster-analysis',
  '@luma-contributors/analysis/dataframe-analysis',
  '@luma-contributors/analysis/geospatial-analysis',
  '@luma-contributors/analysis/pointcloud-analysis',
  '@luma-contributors/analysis/computation-record',
  '@luma.gl/graph/gpu-raster',
  '@luma.gl/graph/gpu-dataframe',
  '@luma.gl/graph/geospatial',
  '@luma.gl/graph/gpu-tables',
  '@luma.gl/graph/gpu-core',
  '@luma.gl/core'
] as const;

/** Reads the root compiler options (strict, bundler resolution, `@luma.gl/*` paths). */
export function getRepositoryCompilerOptions(): typescript.CompilerOptions {
  const configPath = join(REPOSITORY_ROOT, 'tsconfig.json');
  const config = typescript.readConfigFile(configPath, typescript.sys.readFile);
  const parsed = typescript.parseJsonConfigFileContent(
    config.config,
    typescript.sys,
    REPOSITORY_ROOT
  );
  return {
    ...parsed.options,
    noEmit: true,
    composite: false,
    incremental: false,
    declaration: false,
    // Snippets show isolated steps; an unused local is not an API error.
    noUnusedLocals: false
  };
}

/**
 * Creates a program whose root files are the given in-memory files (absolute path → text). Every
 * other file is read from disk, so package-name imports resolve to the current sources.
 */
export function createVirtualProgram(
  virtualFiles: Map<string, string>,
  options: typescript.CompilerOptions = getRepositoryCompilerOptions(),
  oldProgram?: typescript.Program
): typescript.Program {
  const host = typescript.createCompilerHost(options, true);
  const readFile = host.readFile.bind(host);
  const fileExists = host.fileExists.bind(host);
  const getSourceFile = host.getSourceFile.bind(host);
  host.readFile = fileName => virtualFiles.get(fileName) ?? readFile(fileName);
  host.fileExists = fileName => virtualFiles.has(fileName) || fileExists(fileName);
  host.getSourceFile = (fileName, languageVersion, onError, shouldCreate) => {
    const text = virtualFiles.get(fileName);
    return text === undefined
      ? getSourceFile(fileName, languageVersion, onError, shouldCreate)
      : typescript.createSourceFile(fileName, text, languageVersion, true);
  };
  return typescript.createProgram([...virtualFiles.keys()], options, host, oldProgram);
}

/** Reads one analysis doc page. */
export function readDocPage(page: string): string {
  const path = join(DOCS_ROOT, page);
  if (!existsSync(path)) {
    throw new Error(`missing doc page ${page}`);
  }
  return readFileSync(path, 'utf8');
}

/** Formats a diagnostic as `message` with its line and column in the file. */
export function formatDiagnostic(diagnostic: typescript.Diagnostic): string {
  const message = typescript.flattenDiagnosticMessageText(diagnostic.messageText, '\n');
  if (!diagnostic.file || diagnostic.start === undefined) {
    return message;
  }
  const {line, character} = diagnostic.file.getLineAndCharacterOfPosition(diagnostic.start);
  return `${line + 1}:${character + 1} TS${diagnostic.code} ${message}`;
}
