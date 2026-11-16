// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it, vi} from 'vitest';
import type {CatalogParameterValue} from '../../src/analysis/catalog-binding-types';
import {getCatalogBinding, runCatalogOperation} from '../../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import {evaluateKeplerOperationApplicability} from '../../src/descriptors/applicability';
import {resolveKeplerOperationDescriptor} from '../../src/descriptors/operation-descriptors';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';

function makePoints(count: number, constant = false): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: Array.from({length: count}, (_, index) => ({
      type: 'Feature',
      geometry: {type: 'Point', coordinates: [index % 100, Math.floor(index / 100)]},
      properties: {value: constant ? 7 : index + 1, population: 100, predictor: index + 2}
    }))
  };
}

type RejectionFixture = {
  bindingId: string;
  defect: string;
  inputs: Record<string, CatalogFeatureCollection>;
  parameters: Record<string, CatalogParameterValue>;
  reason: RegExp;
  remedy: RegExp;
  runtimeError: RegExp;
  requirement?: {id: string; input: string};
};

// These are literal runtime expectations, not messages copied out of the checker. Each fixture
// supplies the very same collections and explicit settings to the checker and the CPU oracle.
const REJECTIONS: readonly RejectionFixture[] = [
  {
    bindingId: 'local-moran-statistics',
    defect: 'constant values',
    inputs: {features: makePoints(4, true)},
    parameters: {valueProperty: 'value'},
    reason: /value.*(constant|distinct|differ)/i,
    remedy: /values differ/i,
    runtimeError: /Constant values/,
    requirement: {id: 'varying-values', input: 'features'}
  },
  {
    bindingId: 'global-spatial-statistics',
    defect: 'constant values',
    inputs: {features: makePoints(4, true)},
    parameters: {valueProperty: 'value'},
    reason: /value.*(constant|distinct|differ)/i,
    remedy: /values differ/i,
    runtimeError: /Constant values/,
    requirement: {id: 'varying-values', input: 'features'}
  },
  {
    bindingId: 'hotspot-gi-star',
    defect: 'three features, below the four-feature minimum',
    inputs: {features: makePoints(3)},
    parameters: {valueProperty: 'value'},
    reason: /3.*4|4.*3/,
    remedy: /at least four/i,
    runtimeError: /Too few features/,
    requirement: {id: 'minimum-rows', input: 'features'}
  },
  {
    bindingId: 'empirical-bayes-rates',
    defect: 'only one area',
    inputs: {features: makePoints(1)},
    parameters: {numeratorProperty: 'value', denominatorProperty: 'population'},
    reason: /1.*2|2.*1/,
    remedy: /at least two areas/i,
    runtimeError: /Too few features/,
    requirement: {id: 'minimum-rows', input: 'features'}
  },
  {
    bindingId: 'regression-ols',
    defect: 'explicit predictor column does not exist',
    inputs: {points: makePoints(6)},
    parameters: {responseProperty: 'value', predictorProperties: 'missing_predictor'},
    reason: /missing_predictor/,
    remedy: /column|choose|map/i,
    runtimeError: /Too few complete rows/
  },
  {
    bindingId: 'local-outlier-factor',
    defect: 'three points for ten neighbors',
    inputs: {points: makePoints(3)},
    parameters: {neighborCount: 10},
    reason: /3.*11|11.*3/,
    remedy: /more points|fewer neighbors/i,
    runtimeError: /Too few features/,
    requirement: {id: 'minimum-rows', input: 'points'}
  },
  {
    bindingId: 'hierarchical-density-clustering',
    defect: '3,001 points exceed the declared 3,000-feature capacity',
    inputs: {points: makePoints(3001)},
    parameters: {},
    reason: /3,001.*3,000|3,000.*3,001/,
    remedy: /filter|aggregate/i,
    runtimeError: /Input feature capacity exceeded/,
    requirement: {id: 'maximum-rows', input: 'points'}
  },
  {
    bindingId: 'idw-grid',
    defect: 'line geometry in a point port',
    inputs: {
      points: {
        type: 'FeatureCollection',
        features: [
          {
            geometry: {
              type: 'LineString',
              coordinates: [
                [0, 0],
                [1, 1]
              ]
            },
            properties: {value: 1}
          }
        ]
      }
    },
    parameters: {valueProperty: 'value'},
    reason: /point|line/i,
    remedy: /point|layer/i,
    runtimeError: /Input geometry kind not accepted/
  },
  {
    bindingId: 'density-clustering',
    defect: '20,001 points exceed the declared 20,000-feature capacity',
    inputs: {points: makePoints(20001)},
    parameters: {epsilon: 1, minimumPoints: 4},
    reason: /20,001.*20,000|20,000.*20,001/,
    remedy: /filter|aggregate|GPU/i,
    runtimeError: /Input feature capacity exceeded/,
    requirement: {id: 'maximum-rows', input: 'points'}
  },
  {
    bindingId: 'location-allocation',
    defect: 'two candidate sites for three facilities',
    inputs: {demand: makePoints(4), candidates: makePoints(2)},
    parameters: {facilityCount: 3},
    reason: /2.*3|3.*2/,
    remedy: /as many candidate sites as facilities/i,
    runtimeError: /Too few features/,
    requirement: {id: 'minimum-rows', input: 'candidates'}
  }
];

describe('AX-13 named ten-binding requirements/runtime rejection matrix', () => {
  it('covers ten distinct bindings with independent CPU runtime oracles', () => {
    expect(new Set(REJECTIONS.map(fixture => fixture.bindingId)).size).toBe(10);
    for (const fixture of REJECTIONS) {
      expect(getCatalogBinding(fixture.bindingId)?.run, fixture.bindingId).toBeTypeOf('function');
    }
  });

  for (const fixture of REJECTIONS) {
    it(`${fixture.bindingId}: explains ${fixture.defect} before executing`, () => {
      const binding = getCatalogBinding(fixture.bindingId)!;
      const {descriptor, source} = resolveKeplerOperationDescriptor(binding);
      expect(source).toBe('hand-written');
      const parameters: Record<string, CatalogParameterValue> = {
        ...(binding.parameters.some(parameter => parameter.name === 'coordinateSystem')
          ? {coordinateSystem: 'planar'}
          : {}),
        ...fixture.parameters
      };
      const profiles = Object.entries(fixture.inputs).map(([name, collection]) =>
        profileKeplerFeatureCollection(collection, {
          datasetId: name,
          label: name,
          coordinateSystem: 'planar',
          // Capacity fixtures need exact row/vertex metadata, not full spacing statistics.
          ...(collection.features.length > 100 ? {maximumSampleSize: 8} : {}),
          declarations: {rowSemantics: 'objects'}
        })
      );
      const run = vi.fn(binding.run);
      const runGpu = vi.fn(binding.runGpu);
      const checkedBinding = {...binding, run, runGpu};
      const assessment = evaluateKeplerOperationApplicability(
        checkedBinding,
        descriptor,
        profiles,
        {
          executionEngine: 'cpu',
          inputAssignments: Object.fromEntries(
            Object.keys(fixture.inputs).map(name => [name, name])
          ),
          overrides: parameters,
          fieldBindings: Object.fromEntries(
            Object.entries(parameters).filter(([name]) =>
              descriptor.inputs.some(input => input.fields.some(field => field.parameter === name))
            )
          ) as Record<string, string>
        }
      );
      const report = JSON.stringify({bindingId: binding.id, defect: fixture.defect, assessment});
      expect(run, report).not.toHaveBeenCalled();
      expect(runGpu, report).not.toHaveBeenCalled();
      expect(assessment.status, report).toBe('blocked');
      expect(
        assessment.missing.some(
          entry => fixture.reason.test(entry.what) && fixture.remedy.test(entry.remedy ?? '')
        ),
        report
      ).toBe(true);
      if (fixture.requirement) {
        expect(assessment.requirements, report).toContainEqual(
          expect.objectContaining({
            ...fixture.requirement,
            state: 'failed',
            severity: 'block',
            evidence: 'profile',
            message: expect.stringMatching(fixture.reason),
            remedy: expect.stringMatching(fixture.remedy)
          })
        );
      }
      // Deliberately bypass applicability. The independently invoked runtime must reject the
      // original invalid data, rather than merely trusting the preflight status.
      expect(() => runCatalogOperation(binding, parameters, fixture.inputs), report).toThrow(
        fixture.runtimeError
      );
    });
  }
});
