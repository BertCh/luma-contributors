// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readdirSync, readFileSync} from 'node:fs';
import {dirname, join, relative, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {describe, expect, it} from 'vitest';

// Math correctness review, luma-official-02. AGENTS.md forbids copied luma.gl source. A few kernels
// still hold a private copy of a WGSL function that published luma.gl does not export, because they
// read state an official class built (the cell tables of `GPUGridIndex`, the counts of
// `GPUGridBinning`) and must decide a cell with the same arithmetic. Each such copy is a registered
// workaround (`dev-docs/upstream-workarounds.md`, section 2) and needs a pin: a spec that compares
// the copy with the text of the official kernel, so a luma.gl upgrade that changes one side fails a
// test instead of silently changing results next to cell edges.
//
// This spec is the list of every known copy. It fails when
// - a listed copy has no pin spec, or its pin spec does not name both files,
// - a WGSL function in `modules/analysis/src` is identical to a function in the published
//   `@luma.gl/graph` or `@luma.gl/shaders` kernels and is not listed here,
// - a listed copy no longer exists (remove the row here and in the register).
// Only text is read. Nothing is imported from luma.gl or from the analysis source.

const REPOSITORY_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const THIS_SPEC = 'modules/analysis/test/analysis-core/official-wgsl-copies.node.spec.ts';
const ANALYSIS_SOURCE = 'modules/analysis/src';
const OFFICIAL_ROOTS = ['node_modules/@luma.gl/graph/dist', 'node_modules/@luma.gl/shaders/dist'];
const GRID_INDEX_KERNEL = 'node_modules/@luma.gl/graph/dist/gpu-core/gpu-grid-index-internals.js';
const GRID_BINNING_KERNEL = 'node_modules/@luma.gl/graph/dist/gpu-core/gpu-grid-binning.js';
const GRID_AGGREGATION_KERNEL = 'node_modules/@luma.gl/graph/dist/gpu-core/gpu-grid-aggregation.js';

/**
 * A function shorter than this (whitespace removed) is an idiom, not a copy: the one-line
 * exponent-mask finite test `(bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u` is 87 characters and
 * is written the same way in about thirty local kernels and in several official ones.
 */
const MINIMUM_COPY_LENGTH = 100;

type OfficialWGSLCopy = {
  /** WGSL function name in the local file. */
  functionName: string;
  /** Local file that holds the copy. */
  localFile: string;
  /** Official kernels whose function of the same name the copy must equal, character for character. */
  officialFiles: string[];
  /** The spec that compares the texts: a per-copy spec, or this file. */
  pinSpec: string;
  /** Why the copy exists and what equality protects. */
  reason: string;
};

const OFFICIAL_WGSL_COPIES: OfficialWGSLCopy[] = [
  {
    functionName: 'getCoordinate',
    localFile:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-encounters/encounter-kernels.ts',
    officialFiles: [GRID_INDEX_KERNEL],
    pinSpec:
      'modules/analysis/test/gpu-spatial-analysis/trajectory-encounters/encounter-grid-coordinate-pin.node.spec.ts',
    reason: 'Scans the cell tables of a published GPUGridIndex; a different cell drops pairs.'
  },
  {
    functionName: 'getCoordinate',
    localFile:
      'modules/analysis/src/gpu-spatial-analysis/region-statistics/region-statistics-grid.ts',
    officialFiles: [GRID_INDEX_KERNEL],
    pinSpec:
      'modules/analysis/test/gpu-spatial-analysis/region-statistics/region-statistics-grid-coordinate-pin.node.spec.ts',
    reason: 'Turns a query rectangle into a cell range of a published GPUGridIndex.'
  },
  {
    functionName: 'getCoordinate',
    localFile:
      'modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/gpu-inverse-distance-weighting.ts',
    officialFiles: [GRID_INDEX_KERNEL],
    pinSpec:
      'modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/grid-coordinate-pin.node.spec.ts',
    reason:
      'GRID_COORDINATE_WGSL: every exact gather (inverse distance, kriging, density) reads a published GPUGridIndex.'
  },
  {
    functionName: 'getCoordinate',
    localFile: 'modules/analysis/src/gpu-spatial-analysis/point-density/point-density-kernels.ts',
    officialFiles: [GRID_BINNING_KERNEL, GRID_AGGREGATION_KERNEL],
    pinSpec:
      'modules/analysis/test/gpu-spatial-analysis/point-density/point-density-grid-coordinate-pin.node.spec.ts',
    reason:
      'Keys the weights of a cell whose count comes from a published GPUGridBinning; a different cell splits count and weight.'
  },
  {
    functionName: 'getCoordinate',
    localFile: 'modules/analysis/src/gpu-network/flow-aggregation/flow-aggregation-kernels.ts',
    officialFiles: [GRID_BINNING_KERNEL],
    pinSpec:
      'modules/analysis/test/gpu-network/flow-aggregation/flow-aggregation-grid-coordinate-pin.node.spec.ts',
    reason:
      'Grid zones of a flow must be the cells GPUGridBinning gives the same points; the kernel also needs a per-frame grid size.'
  },
  {
    functionName: 'safeNormal',
    localFile: 'modules/analysis/src/pointcloud-analysis/pointcloud-kernel-nodes.ts',
    officialFiles: ['node_modules/@luma.gl/graph/dist/utils/gpu-spatial-operation.js'],
    pinSpec: THIS_SPEC,
    reason:
      'Carried with the local stand-in for the non-public getSpatialOperationNodes; no shared state, pinned so a luma.gl upgrade shows the drift.'
  },
  {
    functionName: 'encodeOrderedFloat',
    localFile:
      'modules/analysis/src/raster-analysis/raster-zonal-statistics/gpu-raster-zonal-statistics.ts',
    officialFiles: [GRID_AGGREGATION_KERNEL],
    pinSpec: THIS_SPEC,
    reason:
      'Order-preserving float key of the fused zone reduction, the same text as the official min and max aggregation; the local kernel decodes its own keys, pinned so a luma.gl upgrade shows the drift.'
  }
];

type WGSLFunctionText = {
  name: string;
  /** `fn name(...) -> type { ... }` with all whitespace removed. */
  text: string;
  /** The same text without `fn name`, to recognise a renamed copy. */
  signatureAndBody: string;
};

/** Every WGSL `fn name(...) { ... }` in `source`, with whitespace removed. */
function getWGSLFunctionTexts(source: string): WGSLFunctionText[] {
  const functions: WGSLFunctionText[] = [];
  const pattern = /\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/g;
  for (let match = pattern.exec(source); match !== null; match = pattern.exec(source)) {
    const open = source.indexOf('{', match.index);
    let depth = 0;
    for (let index = open; index >= 0 && index < source.length; index++) {
      if (source[index] === '{') {
        depth++;
      } else if (source[index] === '}') {
        depth--;
        if (depth === 0) {
          const text = source.slice(match.index, index + 1).replace(/\s+/g, '');
          functions.push({
            name: match[1],
            text,
            signatureAndBody: text.slice(`fn${match[1]}`.length)
          });
          break;
        }
      }
    }
  }
  return functions;
}

function listFiles(directory: string, namePattern: RegExp): string[] {
  return readdirSync(directory, {withFileTypes: true}).flatMap(entry => {
    const filePath = join(directory, entry.name);
    return entry.isDirectory()
      ? listFiles(filePath, namePattern)
      : namePattern.test(entry.name)
        ? [filePath]
        : [];
  });
}

function readRepositoryFile(repositoryPath: string): string {
  return readFileSync(resolve(REPOSITORY_ROOT, repositoryPath), 'utf8');
}

function getFunctionText(repositoryPath: string, functionName: string): string | undefined {
  return getWGSLFunctionTexts(readRepositoryFile(repositoryPath)).find(
    wgslFunction => wgslFunction.name === functionName
  )?.text;
}

function getCopyKey(localFile: string, functionName: string): string {
  return `${localFile}#${functionName}`;
}

describe('private copies of official WGSL', () => {
  it('lists each copy once', () => {
    const keys = OFFICIAL_WGSL_COPIES.map(copy => getCopyKey(copy.localFile, copy.functionName));
    expect(new Set(keys).size).toBe(keys.length);
  });

  it.each(OFFICIAL_WGSL_COPIES)('$functionName in $localFile still exists and has a pin', copy => {
    expect(
      getFunctionText(copy.localFile, copy.functionName),
      `${copy.localFile} no longer holds fn ${copy.functionName}: remove the row here and in dev-docs/upstream-workarounds.md`
    ).toBeDefined();
    for (const officialFile of copy.officialFiles) {
      expect(
        getFunctionText(officialFile, copy.functionName),
        `${officialFile} no longer holds fn ${copy.functionName}: the official kernel moved, repoint the pin`
      ).toBeDefined();
    }
    if (copy.pinSpec === THIS_SPEC) {
      const local = getFunctionText(copy.localFile, copy.functionName);
      for (const officialFile of copy.officialFiles) {
        expect(local, `${copy.localFile} against ${officialFile}`).toBe(
          getFunctionText(officialFile, copy.functionName)
        );
      }
      return;
    }
    expect(
      existsSync(resolve(REPOSITORY_ROOT, copy.pinSpec)),
      `missing pin spec ${copy.pinSpec}`
    ).toBe(true);
    const pinSpecText = readRepositoryFile(copy.pinSpec);
    expect(pinSpecText, `${copy.pinSpec} must read the local copy`).toContain(copy.localFile);
    for (const officialFile of copy.officialFiles) {
      expect(pinSpecText, `${copy.pinSpec} must read the official kernel`).toContain(officialFile);
    }
    expect(pinSpecText, `${copy.pinSpec} must compare the two texts`).toMatch(/\.toBe\(/);
  });

  it('finds no copy of an official WGSL function that is not listed', () => {
    const officialNamesByText = new Map<string, string>();
    const officialNamesBySignatureAndBody = new Map<string, string>();
    for (const officialRoot of OFFICIAL_ROOTS) {
      const directory = resolve(REPOSITORY_ROOT, officialRoot);
      for (const filePath of listFiles(directory, /\.js$/)) {
        // The two bundles at the root of `dist` repeat every kernel.
        if (dirname(filePath) === directory) {
          continue;
        }
        const origin = relative(REPOSITORY_ROOT, filePath);
        for (const wgslFunction of getWGSLFunctionTexts(readFileSync(filePath, 'utf8'))) {
          if (wgslFunction.text.length < MINIMUM_COPY_LENGTH) {
            continue;
          }
          if (!officialNamesByText.has(wgslFunction.text)) {
            officialNamesByText.set(wgslFunction.text, `${origin} fn ${wgslFunction.name}`);
          }
          if (!officialNamesBySignatureAndBody.has(wgslFunction.signatureAndBody)) {
            officialNamesBySignatureAndBody.set(
              wgslFunction.signatureAndBody,
              `${origin} fn ${wgslFunction.name}`
            );
          }
        }
      }
    }
    // The scan has something to compare with: the published kernels are installed and readable.
    expect(officialNamesByText.size).toBeGreaterThan(1000);

    const listedKeys = new Set(
      OFFICIAL_WGSL_COPIES.map(copy => getCopyKey(copy.localFile, copy.functionName))
    );
    const unlistedCopies: string[] = [];
    const foundKeys = new Set<string>();
    for (const filePath of listFiles(resolve(REPOSITORY_ROOT, ANALYSIS_SOURCE), /\.ts$/)) {
      const localFile = relative(REPOSITORY_ROOT, filePath);
      for (const wgslFunction of getWGSLFunctionTexts(readFileSync(filePath, 'utf8'))) {
        const official =
          officialNamesByText.get(wgslFunction.text) ??
          officialNamesBySignatureAndBody.get(wgslFunction.signatureAndBody);
        // The cell-coordinate function is listed by name too, so a copy that has already drifted
        // from the official text cannot leave the list unnoticed.
        const isCopy =
          (official !== undefined && wgslFunction.text.length >= MINIMUM_COPY_LENGTH) ||
          wgslFunction.name === 'getCoordinate';
        if (!isCopy) {
          continue;
        }
        const key = getCopyKey(localFile, wgslFunction.name);
        foundKeys.add(key);
        if (!listedKeys.has(key)) {
          unlistedCopies.push(`${key} == ${official ?? 'a cell-coordinate function by name'}`);
        }
      }
    }
    expect(
      unlistedCopies,
      'add each to OFFICIAL_WGSL_COPIES with a pin spec, and to dev-docs/upstream-workarounds.md'
    ).toEqual([]);
    // Every listed copy is still an exact copy today (the per-copy pins say which official file).
    expect([...listedKeys].filter(key => !foundKeys.has(key))).toEqual([]);
  });
});
