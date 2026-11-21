// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createHash} from 'node:crypto';
import {writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {
  makeGPURasterMetadata,
  getGPURasterFractionalCoverageParameterValues
} from '@luma-contributors/analysis/raster-analysis';
import {getGPUNeighborSearchParameterValues} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {getGPUEdgeBundlingParameterValues} from '@luma-contributors/analysis/gpu-network';
import {
  collectClassParameters,
  generateDocPages,
  renderClassParameters,
  type ClassParameters
} from './doc-parameters';
import {ANALYSIS_DOC_PAGES, REPOSITORY_ROOT, readDocPage} from './doc-program';

let classes: ClassParameters[] | undefined;
function getClass(name: string) {
  classes ??= collectClassParameters();
  return classes.find(entry => entry.className === name)!;
}

it('REVIEW B2 renders forbidden fields from each narrowed neighbor branch', () => {
  const entry = getClass('GPUNeighborSearch');
  expect(entry.packers).toHaveLength(3);
  for (const packer of entry.packers) {
    const mode = packer.fields.find(field => field.name === 'mode')?.type;
    const rendered = renderClassParameters({...entry, packers: [packer]}, 3);
    expect(rendered).not.toMatch(/^\| `radius`/m);
    if (mode === 'never') {
      expect(rendered).toContain('| `searchRadius` (optional) | `number` |');
      expect(rendered).toContain('| `maximumDistance` (optional) | `number` |');
    } else {
      const forbidden = mode === "'knn'" ? 'searchRadius' : 'maximumDistance';
      expect(rendered).toContain(`| \`${forbidden}\` (optional) | \`never\` |`);
    }
  }
  expect(() =>
    getGPUNeighborSearchParameterValues({
      mode: 'knn',
      bounds: [0, 0, 3, 3],
      searchRadius: 1
    } as never)
  ).toThrow();
  expect(() =>
    getGPUNeighborSearchParameterValues({
      mode: 'radius',
      bounds: [0, 0, 3, 3],
      searchRadius: 1,
      maximumDistance: 2
    } as never)
  ).toThrow();
});

it('REVIEW B3 pins overload and temporal layout lengths from real packer outputs', () => {
  const grid = makeGPURasterMetadata({width: 3, height: 1, cellSize: 1, origin: [0, 9]});
  // Legacy affine has eight words; local layout appends discriminator and absolute Y offset.
  expect(getGPURasterFractionalCoverageParameterValues(grid)).toHaveLength(8);
  expect(getGPURasterFractionalCoverageParameterValues(grid, {localOrigin: [0, 9]})).toHaveLength(
    10
  );
  const coverage = getClass('GPURasterFractionalCoverage');
  expect(coverage.packers.map(packer => packer.length)).toEqual([
    '`GPU_RASTER_FRACTIONAL_COVERAGE_PARAMETER_LENGTH` (8)',
    '`GPU_RASTER_FRACTIONAL_COVERAGE_LOCAL_PARAMETER_LENGTH` (10)'
  ]);
  expect(getGPUNeighborSearchParameterValues({mode: 'knn', bounds: [0, 0, 3, 3]})).toHaveLength(12);
  expect(
    getGPUNeighborSearchParameterValues({
      mode: 'knn',
      bounds: [0, 0, 3, 3],
      timeRadius: 1,
      timeBounds: [0, 2]
    })
  ).toHaveLength(15);
  const rendered = renderClassParameters(getClass('GPUNeighborSearch'), 3);
  expect(rendered).toContain('`GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH` (12; without `timeRadius`)');
  expect(rendered).toContain(
    '`GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH` (15; with `timeRadius` and `timeBounds`)'
  );
});

it('REVIEW B4 renders both possible edge-bundling formats on the union overload', () => {
  expect(getGPUEdgeBundlingParameterValues({}, 'uint32')).toBeInstanceOf(Uint32Array);
  expect(getGPUEdgeBundlingParameterValues({}, 'float32')).toBeInstanceOf(Float32Array);
  const entry = getClass('GPUEdgeBundling');
  expect(entry.packers.map(packer => packer.format)).toEqual([
    'uint32',
    'float32',
    'uint32 | float32'
  ]);
  expect(renderClassParameters({...entry, packers: [entry.packers[2]]}, 3)).toContain(
    '`uint32 | float32` values'
  );
});

it('REVIEW S2 keeps raster metadata separate and documents localOrigin options', () => {
  const entry = getClass('GPURasterFractionalCoverage');
  expect(entry.packers[0].fields).toEqual([]);
  expect(entry.packers[1].fields.map(field => field.name)).toEqual(['localOrigin']);
  for (const packer of entry.packers)
    expect(packer.arguments).toContainEqual(
      expect.objectContaining({name: 'grid', type: 'GPURasterMetadata'})
    );
  const rendered = renderClassParameters(entry, 3);
  expect(rendered).toContain('| `localOrigin` (optional) |');
  for (const field of ['width', 'height', 'affine', 'pixelInterpretation'])
    expect(rendered).not.toContain(`| \`${field}\``);
});

it('REVIEW publication renders corrected real-page tables in memory and stages guarded replacements', () => {
  classes ??= collectClassParameters();
  const originals = new Map(ANALYSIS_DOC_PAGES.map(page => [page, readDocPage(page)]));
  const {pages, undocumented} = generateDocPages(classes, originals);
  expect(undocumented).toEqual([]);
  expect(pages.get('gpu-raster/expansion.md')).toContain(
    '`GPU_RASTER_FRACTIONAL_COVERAGE_LOCAL_PARAMETER_LENGTH` (10)'
  );
  expect(pages.get('gpu-spatial-analysis/operators-weights.md')).toContain(
    '`GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH` (15; with `timeRadius` and `timeBounds`)'
  );
  expect(pages.get('gpu-network/operators.md')).toContain('`uint32 | float32` values');
  const replacements = [...pages]
    .filter(([page, text]) => text !== originals.get(page))
    .map(([page, text]) => ({
      path: `docs/api/${page}`,
      baseSha256: createHash('sha256').update(originals.get(page)!).digest('hex'),
      text
    }));
  writeFileSync(
    join(
      REPOSITORY_ROOT,
      '.typecheck/r19/scratch/core-contract-w5-documentation-contracts/review-generated-pages.json'
    ),
    JSON.stringify(replacements, null, 2) + '\n'
  );
});
