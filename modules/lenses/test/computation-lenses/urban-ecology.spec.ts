// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '@luma-contributors/graph/testing';
import {listLensScenarios} from '../../src/computation-lenses';
import '../../src/computation-lenses/scenarios/gpu-spatial-analysis/urban-ecology.lens';
import reference from '@luma-contributors/graph/test/oracles/fixtures/urban-ecology.json';
import landscapeReference from '@luma-contributors/graph/test/oracles/fixtures/urban-landscape.json';

const elements = [
  'addStreetProfilesRecipe',
  'addBuildingAdjacencyRecipe',
  'addIsovistsRecipe',
  'addVisibilityGraphRecipe',
  'addBuildingTessellationRecipe',
  'addLandscapeMetricsRecipe',
  'addPatchProximityRecipe',
  'addResistanceConnectivityRecipe',
  'addSegregationDecompositionRecipe',
  'addSparseResistanceConnectivityRecipe',
  'addRasterResistanceConnectivityRecipe',
  'addSegregationCounterfactualRecipe',
  'addPopulationSupportAlignmentRecipe'
];
for (const element of elements)
  it(`${element} lens executes and captures numerical outputs`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scenario = listLensScenarios().find(candidate => candidate.element === element)!;
    const loaded = await scenario.load();
    const result = await loaded.run(device);
    expect(result.graphs.length).toBe(1);
    expect(result.graphs[0].nodes.length).toBeGreaterThan(0);
    expect(result.captures?.runId).toBe(result.runId);
    const read = (name: string) => {
      const snapshot = result.captures!.snapshots.find(candidate => candidate.name === name);
      expect(snapshot, name).toBeDefined();
      return Array.from(snapshot!.values);
    };
    const close = (name: string, expected: number[]) => {
      const actual = read(name);
      expect(actual.length).toBe(expected.length);
      expected.forEach((value, index) =>
        Number.isNaN(value)
          ? expect(actual[index]).toBeNaN()
          : expect(actual[index]).toBeCloseTo(value, 4)
      );
    };
    switch (element) {
      case 'addSparseResistanceConnectivityRecipe':
        expect(read('status')[0]).toBe(1);
        expect(read('output-diagnostics')[0]).toBeCloseTo(2 / 3, 4);
        break;
      case 'addRasterResistanceConnectivityRecipe':
        expect(read('status')[0]).toBe(1);
        expect(read('output-diagnostics')[0]).toBeCloseTo(2, 4);
        break;
      case 'addSegregationCounterfactualRecipe':
        expect(read('output-decomposition')[0]).toBeCloseTo(-0.8, 4);
        expect(read('output-decomposition')[3]).toBeCloseTo(0, 5);
        break;
      case 'addPopulationSupportAlignmentRecipe':
        close('output-populations', [6, 4, 2, 8]);
        expect(read('status')).toEqual([1, 0, 0, 0]);
        break;
      case 'addStreetProfilesRecipe':
        expect(read('output-validity')).toEqual([3]);
        close('output-profiles', [2, 3, 6, 9]);
        break;
      case 'addBuildingAdjacencyRecipe': {
        const pairs = read('output-pairs');
        expect(pairs[4]).toBeCloseTo(2, 5);
        expect(pairs[5]).toBeCloseTo(Math.PI * 0.02, 5);
        expect(pairs[6]).toBe(1);
        expect(pairs[8]).toBe(0);
        expect(read('status-complete')).toEqual([1]);
        break;
      }
      case 'addIsovistsRecipe':
        close('output-measures', [...reference.isovist.measures, NaN, NaN, NaN, 0]);
        break;
      case 'addVisibilityGraphRecipe':
        close('output-choice', reference.visibility.choice);
        close('output-metrics', reference.visibility.metrics.flat());
        expect(read('status')).toEqual([1, 2, 1, 0]);
        break;
      case 'addBuildingTessellationRecipe':
        expect(read('status-complete')).toEqual([1, 1]);
        expect(read('status-sampling')).toEqual([8, 8, 0, 0]);
        for (let building = 0; building < 2; building++) {
          const prefix = `output-${building}`;
          const count = read(`${prefix}-status-count`)[0];
          expect(count).toBeGreaterThan(0);
          const features = read(`${prefix}-geometry-featureOffsets`);
          const polygons = read(`${prefix}-geometry-polygonOffsets`);
          const rings = read(`${prefix}-geometry-ringOffsets`);
          const positions = read(`${prefix}-geometry-positions`);
          let area = 0;
          for (let ring = 0; ring < polygons[features[count]]; ring++) {
            for (let vertex = rings[ring]; vertex < rings[ring + 1]; vertex++) {
              const next = vertex + 1 === rings[ring + 1] ? rings[ring] : vertex + 1;
              area +=
                0.5 *
                (positions[2 * vertex] * positions[2 * next + 1] -
                  positions[2 * vertex + 1] * positions[2 * next]);
            }
          }
          expect(Math.abs(area)).toBeCloseTo(reference.tessellation.areas[building], 2);
        }
        break;
      case 'addLandscapeMetricsRecipe':
        close('output-classMetrics', reference.landscape.classMetrics);
        close('output-landscape', reference.landscape.landscape);
        close('output-landscapeAggregation', [landscapeReference.cases[0].landscapeAggregation]);
        close('output-landscapeCohesion', [landscapeReference.cases[0].landscapeCohesion]);
        expect(read('output-classAdjacency')).toEqual(reference.landscape.classAdjacency);
        break;
      case 'addPatchProximityRecipe':
        close('output-metrics', [2, 2, 1, 2, 2, 2, 1, 2]);
        expect(read('output-core')).toEqual([1, 1, 0, 1, 1]);
        break;
      case 'addResistanceConnectivityRecipe':
        close('output-potentials', reference.resistance.potentials);
        close('output-currents', reference.resistance.currents);
        expect(read('status')[0]).toBe(1);
        break;
      case 'addSegregationDecompositionRecipe':
        close('output-indices', reference.decomposition.indices);
        close('output-decomposition', reference.decomposition.decomposition);
        expect(read('output-validity')).toEqual([1, 1, 1, 1]);
        break;
    }
  });
