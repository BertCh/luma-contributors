// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/// <reference types="node" />

import {existsSync, readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {describe, expect, expectTypeOf, it} from 'vitest';
import {
  GPU_ANALYSIS_WEBSITE_HELPER_KINDS,
  getGPUAnalysisWebsiteDataTypes
} from '../../../../src/gpu-spatial-analysis/ontology/index';
import {getGPUAnalysisOperatorCatalog} from '../../../../src/gpu-spatial-analysis/ontology/all-entry-points/analysis-operator-catalog';
import {
  GPU_ANALYSIS_HANDLE_OPERATORS,
  GPU_ANALYSIS_OPERATOR_EMPTY_KIND_EXCEPTIONS,
  GPU_ANALYSIS_OPERATOR_KIND_RECORDS,
  GPU_ANALYSIS_OPERATOR_KIND_SUPPLEMENTS
} from '../../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-operator-kind-records';
import {
  findGPUAnalysisOperatorPaths,
  resolveGPUAnalysisOperatorKinds
} from '../../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-operator-kind-supplements';
import {
  GPU_SPATIAL_ANALYSIS_OPERATORS,
  queryGPUSpatialAnalysisOperators
} from '../../../../src/gpu-spatial-analysis/ontology/spatial-analysis-operators';
import {
  GPU_SPATIAL_ANALYSIS_DATA_KINDS,
  type GPUAnalysisOperator,
  type GPUSpatialAnalysisDataKind
} from '../../../../src/gpu-spatial-analysis/ontology/spatial-analysis-ontology';

const KIND_SIDES = ['inputs', 'outputs'] as const;
type KindRecord = Pick<GPUAnalysisOperator, 'exportName' | 'inputs' | 'outputs'>;

it('types known website helper contracts and leaves missing names undefined', () => {
  function getContract(name: string) {
    return GPU_ANALYSIS_WEBSITE_HELPER_KINDS[name];
  }
  expectTypeOf(getContract).returns.toEqualTypeOf<
    | Readonly<{
        input: readonly GPUSpatialAnalysisDataKind[];
        output: readonly GPUSpatialAnalysisDataKind[];
      }>
    | undefined
  >();
  expectTypeOf(GPU_ANALYSIS_WEBSITE_HELPER_KINDS.createGroundSeedCoverageNode.output).toEqualTypeOf<
    readonly ['attributes']
  >();
  expect(getContract('unknown-helper')).toBeUndefined();
  expect(getContract('constructor')).toBeUndefined();
});

it('projects ground seed coverage as a raster input and an attribute mask output', () => {
  // The node reads one packed coverage word per raster cell and writes one validity bit per cell.
  const {input, output} = GPU_ANALYSIS_WEBSITE_HELPER_KINDS.createGroundSeedCoverageNode;
  expect(input).toEqual(['surface']);
  expect(output).toEqual(['attributes']);
  expect(input.flatMap(getGPUAnalysisWebsiteDataTypes)).toEqual(['raster', 'terrain']);
  expect(output.flatMap(getGPUAnalysisWebsiteDataTypes)).toEqual(['table']);
});

it('lists no GPU data ports for the host-only raster encoding plans', () => {
  // The factories consume host settings and return immutable plans; separate packers write GPU
  // parameter values when a plan is used.
  for (const helper of [
    'createGPURasterStreamingAccumulatorEncodingPlan',
    'createGPURasterStandardizedIndexCalibrationEncodingPlan',
    'createGPURasterClimateRunAccumulatorEncodingPlan'
  ] as const) {
    expect(GPU_ANALYSIS_WEBSITE_HELPER_KINDS[helper], helper).toEqual({input: [], output: []});
  }
});

/** Inspect both raw and resolved sides: resolving an empty raw record cannot hide a catalogue gap. */
function findKindCompletenessFailures(operators: readonly KindRecord[]): string[] {
  const failures: string[] = [];
  for (const operator of operators) {
    const exception = Object.hasOwn(
      GPU_ANALYSIS_OPERATOR_EMPTY_KIND_EXCEPTIONS,
      operator.exportName
    )
      ? GPU_ANALYSIS_OPERATOR_EMPTY_KIND_EXCEPTIONS[operator.exportName]
      : undefined;
    const resolved = resolveGPUAnalysisOperatorKinds(operator);
    for (const side of KIND_SIDES) {
      if (operator[side].length === 0 && !exception?.[side]) {
        failures.push(`${operator.exportName} ${side}: unexplained empty raw kinds`);
      }
      if (resolved[`${side}Source`] === 'unknown' && !exception?.[side]) {
        failures.push(`${operator.exportName} ${side}: unknown resolved kinds`);
      }
      for (const kinds of [operator[side], resolved[side]]) {
        if (new Set(kinds).size !== kinds.length) {
          failures.push(`${operator.exportName} ${side}: duplicate kinds`);
        }
        for (const kind of kinds) {
          if (!GPU_SPATIAL_ANALYSIS_DATA_KINDS.includes(kind)) {
            failures.push(`${operator.exportName} ${side}: invalid kind ${kind}`);
          }
        }
      }
    }
  }
  return failures;
}

describe('authoritative operator kind completeness', () => {
  const catalogue = getGPUAnalysisOperatorCatalog();
  const catalogueByName = new Map(catalogue.map(operator => [operator.exportName, operator]));

  it('describes the building tessellation recipe from its boundary and result ports', () => {
    // Both routes read building edges and polygon enclosures. Local mode also reads parameters.
    // The outputs are cells, their building attributes and completeness diagnostics.
    const recipe = catalogueByName.get('addBuildingTessellationRecipe');
    expect(recipe).toBeDefined();
    expect(resolveGPUAnalysisOperatorKinds(recipe!)).toMatchObject({
      inputs: ['lines', 'polygons', 'attributes', 'parameters'],
      outputs: ['polygons', 'attributes', 'summary']
    });
    expect(catalogueByName.get('addBuildingTessellationReferenceRecipe')?.inputs).toEqual([
      'lines',
      'polygons',
      'attributes'
    ]);
  });

  it('describes population support alignment as ID and population columns', () => {
    const recipe = catalogueByName.get('addPopulationSupportAlignmentRecipe');
    expect(recipe?.inputs).toEqual(['attributes']);
    expect(recipe?.status).toEqual(['invalidCount', 'complete']);
  });

  it('fails every unexplained empty or unknown side across all entry points, including helpers', () => {
    const failures = findKindCompletenessFailures(catalogue);
    expect(failures, failures.join('\n')).toEqual([]);
  });

  it('rejects new operators and independently incomplete sides without a baseline allowlist', () => {
    expect(
      findKindCompletenessFailures([{exportName: 'GPUFutureOperator', inputs: [], outputs: []}])
    ).toEqual([
      'GPUFutureOperator inputs: unexplained empty raw kinds',
      'GPUFutureOperator inputs: unknown resolved kinds',
      'GPUFutureOperator outputs: unexplained empty raw kinds',
      'GPUFutureOperator outputs: unknown resolved kinds'
    ]);
    expect(
      findKindCompletenessFailures([
        {exportName: 'GPUFutureHelper', inputs: ['parameters'], outputs: []}
      ])
    ).toEqual([
      'GPUFutureHelper outputs: unexplained empty raw kinds',
      'GPUFutureHelper outputs: unknown resolved kinds'
    ]);
    // A known supplement cannot make an empty raw record pass the catalogue gate.
    expect(
      findKindCompletenessFailures([{exportName: 'GPUKriging', inputs: [], outputs: ['surface']}])
    ).toEqual(['GPUKriging inputs: unexplained empty raw kinds']);
    expect(
      findKindCompletenessFailures([
        {
          exportName: 'GPUFutureOperator',
          inputs: ['points'],
          outputs: ['attributes', 'attributes']
        },
        {
          exportName: 'GPUFutureOperator',
          inputs: ['invalid' as GPUSpatialAnalysisDataKind],
          outputs: ['summary']
        }
      ])
    ).toEqual([
      'GPUFutureOperator outputs: duplicate kinds',
      'GPUFutureOperator outputs: duplicate kinds',
      'GPUFutureOperator inputs: invalid kind invalid',
      'GPUFutureOperator inputs: invalid kind invalid'
    ]);
  });

  // Independent contracts from the named operators' props/outputs, rather than family unions
  // or an equality check against another serialization of the same kind records.
  it.each([
    ['GPUNetworkServiceAreas', ['network', 'attributes', 'parameters'], ['attributes', 'summary']],
    ['GPUNetworkCostMatrix', ['network', 'attributes', 'parameters'], ['attributes']],
    ['GPUNetworkCostTable', ['network', 'parameters'], ['attributes']],
    ['GPUNetworkCostWeights', ['attributes', 'parameters'], ['spatial-weights']],
    ['GPUAdjacencyMatrixOrder', ['attributes'], ['attributes']],
    ['GPUTerrainContours', ['surface', 'parameters'], ['lines']],
    ['GPUSolarPosition', ['points', 'parameters'], ['attributes']],
    ['GPUProfilePeaks', ['attributes', 'parameters'], ['attributes']],
    ['GPURasterSampling', ['surface', 'points', 'parameters'], ['attributes']],
    ['GPUPolygonRasterization', ['polygons', 'parameters'], ['surface']],
    ['GPUIsolines', ['surface', 'parameters'], ['lines', 'attributes']],
    ['GPUIsobands', ['surface', 'parameters'], ['surface', 'lines', 'render-geometry']],
    ['GPUIsobandRings', ['surface', 'parameters'], ['polygons', 'attributes']],
    ['GPUColumnQuantiles', ['attributes', 'parameters'], ['summary', 'attributes']],
    ['GPUVariogramModelFit', ['attributes'], ['summary']],
    ['GPUPointCloudStatisticalFilter', ['attributes', 'parameters'], ['attributes']],
    ['GPUPointCloudHeight', ['points', 'surface'], ['attributes']],
    ['GPUDatumGridTransform', ['points', 'surface'], ['points', 'attributes']],
    ['GPUPointBuffer', ['points', 'parameters'], ['polygons']],
    ['GPUSnapshotArena', ['parameters'], ['summary']]
  ])('publishes the individual %s contract', (name, inputs, outputs) => {
    const operator = catalogueByName.get(name as string)!;
    expect(operator, name as string).toBeDefined();
    expect(operator.inputs).toEqual(inputs);
    expect(operator.outputs).toEqual(outputs);
    const resolved = resolveGPUAnalysisOperatorKinds(operator);
    expect(resolved.inputs).toEqual(inputs);
    expect(resolved.outputs).toEqual(outputs);
  });

  it('does not suggest cost tables or facility assignments as polygon-producing network paths', () => {
    const names = findGPUAnalysisOperatorPaths('network', 'polygons', 1)
      .paths.flat()
      .map(step => step.exportName);
    for (const name of ['GPUNetworkServiceAreas', 'GPUNetworkCostMatrix', 'GPUNetworkCostTable']) {
      expect(names).not.toContain(name);
    }
    expect(names).toContain('GPUNetworkIsochrones');
  });

  it('keeps explicit empty-side exceptions justified, current and limited to their named side', () => {
    for (const [name, exception] of Object.entries(GPU_ANALYSIS_OPERATOR_EMPTY_KIND_EXCEPTIONS)) {
      const operator = catalogueByName.get(name);
      expect(operator, `exception for absent operator ${name}`).toBeDefined();
      if (!operator) continue;
      expect(
        KIND_SIDES.some(side => exception[side]),
        `${name} has an empty-side reason`
      ).toBe(true);
      if (exception.role === 'helper') expect(operator.role, name).toBe('helper');
      for (const side of KIND_SIDES) {
        const reason = exception[side];
        if (!reason) continue;
        expect(reason.length, `${name} ${side} reason`).toBeGreaterThan(30);
        const sourcePath = reason.split(':')[0];
        expect(
          existsSync(fileURLToPath(new URL(`../../../../src/${sourcePath}`, import.meta.url))),
          sourcePath
        ).toBe(true);
        expect(operator[side], `stale ${name} ${side} exception`).toEqual([]);
        expect(
          resolveGPUAnalysisOperatorKinds(operator)[side],
          `${name} ${side} stays empty`
        ).toEqual([]);
        const otherSide = side === 'inputs' ? 'outputs' : 'inputs';
        expect(findKindCompletenessFailures([{...operator, [otherSide]: []}])).toContain(
          `${name} ${otherSide}: unexplained empty raw kinds`
        );
      }
      expect(findKindCompletenessFailures([operator]), name).toEqual([]);
    }
    expect(GPU_ANALYSIS_HANDLE_OPERATORS).toEqual(['GPUSpatialJoinPrepared']);
  });

  it('uses the same frozen records for the raw catalogue, resolver and legacy supplement export', () => {
    expect(GPU_ANALYSIS_OPERATOR_KIND_SUPPLEMENTS).toBe(GPU_ANALYSIS_OPERATOR_KIND_RECORDS);
    expect(Object.isFrozen(GPU_ANALYSIS_OPERATOR_KIND_RECORDS)).toBe(true);
    for (const [name, kinds] of Object.entries(GPU_ANALYSIS_OPERATOR_KIND_RECORDS)) {
      const operator = catalogueByName.get(name);
      expect(operator, `kind record for absent operator ${name}`).toBeDefined();
      expect(operator?.inputs, `${name} raw inputs`).toBe(kinds.inputs);
      expect(operator?.outputs, `${name} raw outputs`).toBe(kinds.outputs);
      const resolved = resolveGPUAnalysisOperatorKinds({exportName: name, inputs: [], outputs: []});
      expect(resolved.inputs, name).toBe(kinds.inputs);
      expect(resolved.outputs, name).toBe(kinds.outputs);
      expect(Object.isFrozen(kinds), name).toBe(true);
      for (const side of KIND_SIDES)
        expect(Object.isFrozen(kinds[side]), `${name} ${side}`).toBe(true);
    }
    expect(findKindCompletenessFailures(GPU_SPATIAL_ANALYSIS_OPERATORS)).toEqual([]);
    expect(
      queryGPUSpatialAnalysisOperators({input: 'points', output: 'surface'}).map(
        operator => operator.exportName
      )
    ).toContain('GPUKriging');
  });

  it('keeps static records free of runtime imports to prevent a catalogue/resolver cycle', () => {
    const source = readFileSync(
      new URL(
        '../../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-operator-kind-records.ts',
        import.meta.url
      ),
      'utf8'
    );
    expect(source).not.toMatch(/^import(?! type )/m);
    expect(source).not.toMatch(/\b(?:import\s*\(|require\s*\(|export\s+\*\s+from)/);
  });
});
