// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import typescript from 'typescript';
import {compileProjectionProgram} from '@luma.gl/graph/gpu-project';
import {generateDocPages} from '../../analysis-core/doc-parameters';
import {getGPURasterReprojectionParameterValues} from '../../../src/raster-analysis/gpu-raster-reprojection';

function readPage(name: string): string {
  return readFileSync(
    new URL(`../../../../../docs/api/gpu-raster/${name}.md`, import.meta.url),
    'utf8'
  );
}

it.each([
  1, 0
])('B1 publishes same-submission zonal columns only for loss zero (loss %s)', async loss => {
  const example = [...readPage('operations-analysis').matchAll(/```ts\n([\s\S]*?)```/g)]
    .map(match => match[1])
    .find(text => text.includes('incompleteSources: [costIterationLimitReached]'))!;
  const publications: unknown[] = [];
  const reads: string[] = [];
  let submitted = false;
  type View = {storage: ReadbackStorage};
  type Selection = {means: View; populationVariances: View; status: {incomplete: View}};
  let selection: Selection;
  class ReadbackStorage {
    words: Float32Array | Uint32Array;
    readonly buffer = {
      readAsync: async () => {
        expect(submitted).toBe(true);
        await Promise.resolve();
        reads.push(this.id);
        // Exercise nonzero byteOffset, as returned by a readback subview.
        const bytes = new Uint8Array(this.words.byteLength + 8);
        bytes.set(new Uint8Array(this.words.buffer), 4);
        return bytes.subarray(4, 4 + this.words.byteLength);
      }
    };
    constructor(
      _device: unknown,
      readonly settings: {id: string; format: string; length: number}
    ) {
      this.words =
        settings.format === 'uint32'
          ? new Uint32Array(settings.length)
          : new Float32Array(settings.length);
    }
    get id(): string {
      return this.settings.id;
    }
    importToGraph(): View {
      return {storage: this};
    }
  }
  const createLegacyView = () => ({
    storage: new ReadbackStorage(undefined, {id: 'legacy', format: 'float32', length: 1})
  });
  const bindings = {
    GPUParameterBuffer: ReadbackStorage,
    GPURasterZonalStatistics: class {
      constructor(readonly props: {output: Selection}) {}
    },
    makeGPURasterBand: (values: unknown) => values,
    grid: {},
    costs: {},
    zones: {},
    polygonOverflow: {},
    costIterationLimitReached: {},
    featureCount: 1,
    // A stale zero must never be sufficient evidence to publish the current frame.
    incompleteReadback: Uint32Array.of(0),
    meanReadback: Float32Array.of(99),
    incomplete: createLegacyView(),
    means: createLegacyView(),
    populationVariances: createLegacyView(),
    publishZoneStatistics: (result: unknown) => publications.push(result),
    graph: {
      add(contributor: {props: {output: Selection}}) {
        selection = contributor.props.output;
      },
      async compileAsync() {
        return {
          encode() {
            selection.status.incomplete.storage.words[0] = loss;
            // Literal fixture: completed zone mean 7, population variance .75.
            selection.means.storage.words[0] = 7;
            selection.populationVariances.storage.words[0] = 0.75;
          }
        };
      }
    },
    device: {
      createCommandEncoder: () => ({finish: () => ({})}),
      submit: () => {
        submitted = true;
      }
    }
  };
  const javascript = typescript.transpileModule(example.replace(/^import .*;\n/gm, ''), {
    compilerOptions: {target: typescript.ScriptTarget.ESNext, module: typescript.ModuleKind.ESNext}
  }).outputText;
  await new Function('bindings', `with (bindings) { return (async () => { ${javascript} })(); }`)(
    bindings
  );
  expect(publications).toEqual(
    loss ? [null] : [{means: Float32Array.of(7), populationVariances: Float32Array.of(0.75)}]
  );
  expect(reads).toEqual(
    loss ? ['zone-incomplete'] : ['zone-incomplete', 'zone-means', 'zone-population-variances']
  );
});

it('B2 generates exactly the complete owned operations and tile pages', () => {
  const {pages} = generateDocPages();
  for (const name of ['operations-analysis', 'tile-reprojection']) {
    expect(pages.get(`gpu-raster/${name}.md`), name).toBe(readPage(name));
  }
}, 600_000);

it('B3 scopes affine-packer CRS checks to the grids supplied without a projection', () => {
  const grid = {
    width: 1,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const mercator = {...grid, coordinateReferenceSystem: {authority: 'EPSG:3857'}};
  const geographic = {...grid, coordinateReferenceSystem: {authority: 'EPSG:4326'}};
  expect(() =>
    getGPURasterReprojectionParameterValues({sourceGrid: mercator, destinationGrid: geographic})
  ).toThrow();
  const inverseProjection = compileProjectionProgram({precision: 'local-f32', operations: []});
  expect(
    getGPURasterReprojectionParameterValues({
      sourceGrid: mercator,
      destinationGrid: geographic,
      inverseProjection
    })
  ).toHaveLength(20);
  expect(
    getGPURasterReprojectionParameterValues({sourceGrid: geographic, destinationGrid: geographic})
  ).toHaveLength(20);
  const page = readPage('tile-reprojection');
  expect(page).not.toContain('per-frame CRS mismatches reject');
  expect(page).toContain('when `inverseProjection` is absent');
  expect(page).toContain('consistent with the compiled contributor');
});

it('S1 identifies the compact path output and its exclusive count bound', () => {
  const page = readPage('operations-analysis');
  expect(page).toContain('`output.path.ids[0..count)`');
  expect(page).toContain('`output.path.status.count`');
  expect(page).toContain('exclusive upper bound');
});

it('S2 explains rank center validity as a policy rather than the quantile comparison', () => {
  const page = readPage('focal-pairs');
  expect(page).not.toContain('rank requires a usable center value to compare');
  expect(page).toContain('preserves center validity as a policy');
  expect(page).toContain('quantile from sorted neighbors');
  expect(page).toContain('optional `changed` output');
});

it('earlier recording note labels the archived serial path implementation', () => {
  const archive = readFileSync(
    new URL('../../../../../docs/lens-records/gpu-raster.cost-distance-path.json', import.meta.url),
    'utf8'
  );
  const hasSerialWalk = [0, 1, 2].some(index => archive.includes(`walk-${index}-walk`));
  // A future recapture may remove this note together with the obsolete serial capture.
  if (hasSerialWalk) {
    expect(readPage('operations-analysis')).toContain('archive records an earlier implementation');
  }
});
