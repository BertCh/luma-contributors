// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  computePolygonTriangulationOnCPU,
  getPolygonTriangulationIndexCount
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import reference from '../../../analysis/test/oracles/fixtures/dasymetric-redistribution.json';
import {
  getCatalogBinding,
  normalizeCatalogParameters,
  runCatalogOperationAsync
} from '../../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import type {CatalogRasterInput} from '../../src/analysis/catalog-raster-input';
import {
  CURATED_DASYMETRIC_REDISTRIBUTION_BINDING,
  MAXIMUM_DASYMETRIC_PRODUCT,
  MAXIMUM_DASYMETRIC_CELLS,
  prepareCuratedDasymetricInputs
} from '../../src/curated/bindings/curated-dasymetric-redistribution-binding';
import {
  CURATED_DASYMETRIC_REDISTRIBUTION_ANALYSIS,
  DASYMETRIC_SAMPLE_LICENSE,
  createDasymetricRedistributionSample,
  createDasymetricRedistributionTemplate
} from '../../src/curated/samples/index';
import {
  createKeplerCuratedAnalysisSample,
  getKeplerCuratedAnalysisSampleParameters,
  getKeplerCuratedAnalysisSampleMetadata,
  KEPLER_CURATED_ANALYSIS_SAMPLE_IDS
} from '../../src/curated/curated-analysis-samples';
import {
  buildKeplerCuratedAnalysisRequest,
  createKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping,
  validateKeplerCuratedAnalysis
} from '../../src/curated/curated-analysis-mapping';
import {CURATED_DASYMETRIC_REDISTRIBUTION_DESCRIPTOR} from '../../src/descriptors/descriptors-curated-dasymetric-redistribution';
import {getKeplerOperationDescriptor} from '../../src/descriptors/operation-descriptors';
import {validateKeplerOperationDescriptor} from '../../src/descriptors/validate-operation-descriptor';
import {getKeplerWorkbenchCuratedPresets} from '../../src/descriptors/workbench-adapters';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import type {KeplerDatasetProfile} from '../../src/profile/dataset-profile-types';
import {listCuratedWorkflowAnalysisTemplates} from '../../src/workflows/workflow-analysis-template';
import {
  validateWorkflowAnalysisTemplate,
  serializeWorkflowAnalysisTemplate,
  parseWorkflowAnalysisTemplate
} from '../../src/workflows/workflow-analysis-template-parser';
import {
  parseWorkflowDocument,
  serializeWorkflowDocument
} from '../../src/workflows/workflow-document';
import {WORKFLOW_LIMITS} from '../../src/workflows/workflow-types';
import {getWorkflowOperationRegistry} from '../../src/workflows/workflow-operations';
import {DASYMETRIC_GEOMETRY_CASES} from './curated-dasymetric-redistribution-fixtures';

const binding = CURATED_DASYMETRIC_REDISTRIBUTION_BINDING;
const analysis = CURATED_DASYMETRIC_REDISTRIBUTION_ANALYSIS;

describe('dasymetric typed recipe preparation', () => {
  test.each([
    {name: 'collapsedShell', validity: 1},
    {name: 'unbridgeableHole', validity: 0}
  ] as const)('preserves $name and distinguishes empty coverage from triangulation failure', ({
    name,
    validity
  }) => {
    const sample = createDasymetricRedistributionSample();
    const sources = sample.inputs['sources'] as CatalogFeatureCollection;
    sources.features[0].geometry = DASYMETRIC_GEOMETRY_CASES[name];
    const prepared = prepareCuratedDasymetricInputs(sample.inputs, sample.parameters);
    const polygons = prepared.polygons;
    const triangulation = computePolygonTriangulationOnCPU(polygons, {
      indexCapacity: getPolygonTriangulationIndexCount(
        polygons.vertexCount,
        polygons.ringCount,
        polygons.polygonOffsets.length - 1
      )
    });
    expect(prepared.sources).toBe(sources);
    expect(prepared.sources.features[0].geometry).toBe(DASYMETRIC_GEOMETRY_CASES[name]);
    expect(polygons.sourceIndices).toEqual([0, 1, 2]);
    expect(Array.from(triangulation.valid)).toEqual([validity, 1, 1]);
    expect(Array.from(triangulation.triangleCount)).toEqual([0, 2, 2]);
  });

  test.each([
    false,
    true
  ])('retains row identity, loaded density, affine units and renamed totals (second=%s)', second => {
    const sample = createDasymetricRedistributionSample(second);
    const prepared = prepareCuratedDasymetricInputs(sample.inputs, sample.parameters);
    expect(Array.from(prepared.sourceTotals)).toEqual(sample.expected.sourceTotals);
    expect(Array.from(prepared.density)).toEqual(
      second ? [1, 4, 2, 2, 3, 1] : reference.cases[0].ancillary
    );
    expect(Array.from(prepared.targetOwners)).toEqual(reference.targetZones);
    expect(prepared.polygons.sourceIndices).toEqual([0, 1, 2]);
    expect(prepared.sources).toBe(sample.inputs['sources']);
    expect(prepared.targets).toBe(sample.inputs['targets']);
    expect(prepared.ancillary).toBe(sample.inputs['ancillary']);
    expect(prepared.tupleCapacity).toBe(18);
    expect(prepared.pairCapacity).toBe(12);
    expect(prepared.denominator).toBe('zone');
  });

  test('sample expectations equal the committed independent recipe oracle', () => {
    const sample = createDasymetricRedistributionSample();
    expect(sample.expected.sourceTotals).toEqual(reference.sourceTotals);
    for (const denominator of ['zone', 'overlap'] as const) {
      const oracle = reference.cases.find(entry => entry.name === `continuous-${denominator}`)!;
      expect(sample.expected[denominator].targets).toEqual(oracle.targetTotals);
      sample.expected[denominator].allocated.forEach((value, index) =>
        expect(value).toBeCloseTo(oracle.sourceAllocatedTotals[index], 10)
      );
      sample.expected[denominator].residual.forEach((value, index) =>
        expect(value).toBeCloseTo(oracle.sourceUnallocatedTotals[index], 10)
      );
    }
  });

  test('uses caller validity and nodata without changing decoded arrays', () => {
    const sample = createDasymetricRedistributionSample();
    const raster = sample.inputs['ancillary'] as CatalogRasterInput;
    const band = {
      ...raster.band,
      values: new Float64Array([2, -999, 5, -2, NaN, 0]),
      noDataValue: -999,
      validity: new Uint8Array([1, 1, 0, 1, 1, 1])
    };
    const prepared = prepareCuratedDasymetricInputs(
      {...sample.inputs, ancillary: {...raster, band}},
      sample.parameters
    );
    expect(Array.from(prepared.density)).toEqual([2, NaN, NaN, -2, NaN, 0]);
    expect(Array.from(raster.band.values)).toEqual([2, 0, 1, 3, 1, 0]);
  });

  test.each([
    ['coordinatesMatchRaster', false, /coordinatesMatchRaster/],
    ['targetOwnershipConfirmed', false, /targetOwnershipConfirmed/],
    ['targetCellOwners', [], /one target row index/],
    ['targetCellOwners', [0, 0, 1, 1, -2, 2], /targetCellOwners values/],
    ['targetCellOwners', [0, 0, 1, 1, 0.5, 2], /targetCellOwners values/],
    ['targetCellOwners', [0, 0, 1, 1, 4, 2], /targetCellOwners values/],
    ['totalProperty', '', /totalProperty/],
    ['denominator', 'rate', /denominator/],
    ['pairCapacity', 0, /pairCapacity/]
  ] as const)('rejects invalid %s before allocation', (name, value, message) => {
    const sample = createDasymetricRedistributionSample();
    expect(() =>
      prepareCuratedDasymetricInputs(sample.inputs, {...sample.parameters, [name]: value})
    ).toThrow(message);
  });

  test.each([
    'EPSG:4326',
    'EPSG:3857'
  ] as const)('refuses unsupported %s density area semantics', crs => {
    const sample = createDasymetricRedistributionSample();
    const raster = sample.inputs['ancillary'] as CatalogRasterInput;
    expect(() =>
      prepareCuratedDasymetricInputs(
        {
          ...sample.inputs,
          ancillary: {
            ...raster,
            crs,
            cellSizeMode: crs === 'EPSG:4326' ? 'geographic' : 'web-mercator'
          }
        },
        sample.parameters
      )
    ).toThrow(/crs must be planar/);
  });

  test('refuses wrong input kinds, point pixels and absent geometry without dropping rows', () => {
    const sample = createDasymetricRedistributionSample();
    const raster = sample.inputs['ancillary'] as CatalogRasterInput;
    expect(() =>
      prepareCuratedDasymetricInputs({...sample.inputs, sources: raster}, sample.parameters)
    ).toThrow(/sources must contain polygon/);
    expect(() =>
      prepareCuratedDasymetricInputs(
        {...sample.inputs, ancillary: sample.inputs['sources']},
        sample.parameters
      )
    ).toThrow(/decoded raster/);
    expect(() =>
      prepareCuratedDasymetricInputs(
        {
          ...sample.inputs,
          ancillary: {...raster, grid: {...raster.grid, pixelInterpretation: 'point'}}
        },
        sample.parameters
      )
    ).toThrow(/area pixels/);
    expect(() =>
      prepareCuratedDasymetricInputs(
        {
          ...sample.inputs,
          sources: {type: 'FeatureCollection', features: [{id: 'missing', geometry: null}]}
        },
        sample.parameters
      )
    ).toThrow(/no rows may be dropped/);
  });

  test('bounds dense target-tuple work independently of the cell and feature caps', () => {
    const sample = createDasymetricRedistributionSample();
    const raster = sample.inputs['ancillary'] as CatalogRasterInput;
    const sources = sample.inputs['sources'] as CatalogFeatureCollection;
    const targets = sample.inputs['targets'] as CatalogFeatureCollection;
    const cellCount = 4096;
    const inputs = {
      ...sample.inputs,
      sources: {...sources, features: Array.from({length: 256}, () => sources.features[0])},
      ancillary: {
        ...raster,
        grid: {...raster.grid, width: cellCount},
        band: {
          ...raster.band,
          values: new Float32Array(cellCount),
          validity: new Uint8Array(cellCount)
        }
      }
    };
    const parameters = {...sample.parameters, targetCellOwners: new Array(cellCount).fill(-1)};
    expect(256 * cellCount * targets.features.length).toBe(MAXIMUM_DASYMETRIC_PRODUCT);
    expect(() => prepareCuratedDasymetricInputs(inputs, parameters)).not.toThrow();
    expect(() =>
      prepareCuratedDasymetricInputs(
        {...inputs, targets: {...targets, features: [...targets.features, targets.features[0]]}},
        parameters
      )
    ).toThrow(/targets times tupleCapacity/);
  });

  test('retains invalid source totals for the GPU evidence path', () => {
    const sample = createDasymetricRedistributionSample();
    const sources = sample.inputs['sources'] as CatalogFeatureCollection;
    sources.features[0].properties = {residents: '120'};
    sources.features[1].properties = {residents: -80};
    const prepared = prepareCuratedDasymetricInputs(sample.inputs, sample.parameters);
    expect(Array.from(prepared.sourceTotals)).toEqual([NaN, -80, 50]);
    expect(prepared.polygons.sourceIndices).toEqual([0, 1, 2]);
  });

  test('translates large coordinates and the raster origin before float32 packing', () => {
    const sample = createDasymetricRedistributionSample();
    const raster = sample.inputs['ancillary'] as CatalogRasterInput;
    const origin = 2 ** 24;
    const sources: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: [
        {
          id: 'large-origin',
          properties: {residents: 100},
          geometry: {
            type: 'Polygon',
            coordinates: [
              [
                [origin + 0.5, origin],
                [origin + 2.5, origin],
                [origin + 2.5, origin + 1],
                [origin + 0.5, origin + 1],
                [origin + 0.5, origin]
              ]
            ]
          }
        }
      ]
    };
    const prepared = prepareCuratedDasymetricInputs(
      {
        ...sample.inputs,
        sources,
        ancillary: {
          ...raster,
          grid: {...raster.grid, width: 3, affine: [1, 0, origin, 0, 1, origin]},
          band: {
            ...raster.band,
            values: new Float32Array([1, 1, 1]),
            validity: new Uint8Array([1, 1, 1])
          }
        }
      },
      {...sample.parameters, targetCellOwners: [0, 1, 2]}
    );
    expect(prepared.grid.affine).toEqual([1, 0, 0, 0, 1, 0]);
    expect(Array.from(prepared.polygons.positions)).toEqual([0.5, 0, 2.5, 0, 2.5, 1, 0.5, 1]);
    const horizontal = Array.from(prepared.polygons.positions).filter(
      (_, index) => index % 2 === 0
    );
    const minimum = Math.min(...horizontal);
    const maximum = Math.max(...horizontal);
    expect(
      [0, 1, 2].map(
        column => (100 * Math.max(0, Math.min(column + 1, maximum) - Math.max(column, minimum))) / 2
      )
    ).toEqual([25, 50, 25]);
    expect(prepared.ancillary.grid.affine).toEqual([1, 0, origin, 0, 1, origin]);
    expect(prepared.sources).toBe(sources);
  });

  test('rejects density conversion overflow while respecting masked and nodata samples', () => {
    const sample = createDasymetricRedistributionSample();
    const raster = sample.inputs['ancillary'] as CatalogRasterInput;
    const band = {...raster.band, values: new Float64Array([1e40, 0, 1, 3, 1, 0])};
    expect(() =>
      prepareCuratedDasymetricInputs(
        {...sample.inputs, ancillary: {...raster, band}},
        sample.parameters
      )
    ).toThrow(/^curated-dasymetric-redistribution ancillary.band.values.*finite in float32/);
    for (const excluded of [
      {...band, validity: new Uint8Array([0, 1, 1, 1, 1, 1])},
      {...band, noDataValue: 1e40}
    ]) {
      expect(
        prepareCuratedDasymetricInputs(
          {...sample.inputs, ancillary: {...raster, band: excluded}},
          sample.parameters
        ).density[0]
      ).toBeNaN();
      expect(band.values[0]).toBe(1e40);
    }
  });
});

describe('dasymetric shared registration, mapping and serialization', () => {
  test('registers a GPU-only binding with matching descriptor and typed workflow port', () => {
    expect(getCatalogBinding(binding.id)).toBe(binding);
    expect(getKeplerOperationDescriptor(binding.id)).toBe(
      CURATED_DASYMETRIC_REDISTRIBUTION_DESCRIPTOR
    );
    expect(
      validateKeplerOperationDescriptor(CURATED_DASYMETRIC_REDISTRIBUTION_DESCRIPTOR, binding)
    ).toEqual([]);
    expect(validateKeplerCuratedAnalysis(analysis)).toEqual([]);
    expect(binding.engine).toBe('gpu');
    expect(binding.run).toBeUndefined();
    expect(binding.runData).toBeUndefined();
    expect(
      getWorkflowOperationRegistry()
        .get(binding.id)
        ?.inputs.find(input => input.name === 'ancillary')
    ).toMatchObject({accepts: ['raster']});
  });

  test.each([
    false,
    true
  ])('maps dataset identities and renamed total fields (second=%s)', second => {
    const sample = createDasymetricRedistributionSample(second);
    const totalColumn = sample.parameters['totalProperty'] as string;
    const profiles: KeplerDatasetProfile[] = ['sources', 'targets'].map(name =>
      profileKeplerFeatureCollection(sample.inputs[name] as CatalogFeatureCollection, {
        datasetId: `loaded-${name}`,
        label: name,
        coordinateSystem: 'planar',
        declarations: {measurement: {[totalColumn]: 'count'}}
      })
    );
    profiles.push({
      datasetId: 'loaded-ancillary',
      label: 'ancillary',
      kind: 'surface',
      rowCount: 6,
      columns: [],
      geometryColumns: []
    });
    let mapping = createKeplerCuratedAnalysisMapping(analysis, profiles, {
      parameters: sample.parameters
    });
    for (const slot of analysis.slots)
      mapping = updateKeplerCuratedAnalysisMapping(
        mapping,
        {slot: slot.name, datasetId: `loaded-${slot.name}`},
        analysis,
        profiles
      );
    mapping = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: 'sources', role: 'total', columns: [totalColumn]},
      analysis,
      profiles
    );
    const request = buildKeplerCuratedAnalysisRequest(analysis, mapping);
    expect(request.inputDatasetIds).toEqual({
      sources: 'loaded-sources',
      targets: 'loaded-targets',
      ancillary: 'loaded-ancillary'
    });
    expect(request.parameters['totalProperty']).toBe(totalColumn);
    expect(prepareCuratedDasymetricInputs(sample.inputs, request.parameters).sourceTotals).toEqual(
      Float32Array.from(sample.expected.sourceTotals)
    );
  });

  test('registered templates carry the licensed typed sample and result reading', () => {
    const template = listCuratedWorkflowAnalysisTemplates().find(
      candidate => candidate.id === analysis.id
    )!;
    expect(template).toEqual(createDasymetricRedistributionTemplate());
    expect(template.sample).toEqual(DASYMETRIC_SAMPLE_LICENSE);
    expect(validateWorkflowAnalysisTemplate(template)).toEqual({executable: true, issues: []});
    expect(
      parseWorkflowAnalysisTemplate(JSON.stringify(serializeWorkflowAnalysisTemplate(template)))
    ).toEqual(template);
    expect(template.slots.find(slot => slot.name === 'ancillary')).toMatchObject({
      kind: 'surface',
      fields: []
    });
    expect(template.reads.outputs?.map(output => output.source.output)).toEqual([
      'dasymetric-targets',
      'dasymetric-ledger',
      'dasymetric-status'
    ]);
  });

  test('shared discovery preserves emerging hot spots and AX-86 number lists', () => {
    const template = listCuratedWorkflowAnalysisTemplates().find(
      candidate => candidate.id === analysis.id
    )!;
    const inputs = createKeplerCuratedAnalysisSample(template.sample!.id);
    const parameters = getKeplerCuratedAnalysisSampleParameters(analysis.id);
    expect(KEPLER_CURATED_ANALYSIS_SAMPLE_IDS).toContain(template.sample!.id);
    expect(getKeplerCuratedAnalysisSampleMetadata(template.sample!.id)).toEqual(template.sample);
    expect(inputs['ancillary'].type).toBe('raster');
    expect(prepareCuratedDasymetricInputs(inputs, parameters).sourceTotals).toEqual(
      new Float32Array([120, 80, 50])
    );
    expect(getCatalogBinding('curated-emerging-hot-spots')).toBeDefined();
    expect(getKeplerOperationDescriptor('curated-emerging-hot-spots')).toBeDefined();
    expect(
      createKeplerCuratedAnalysisSample('emerging-hot-spots')['points'].features.length
    ).toBeGreaterThan(0);
    const preset = getKeplerWorkbenchCuratedPresets().find(
      candidate => candidate.bindingId === binding.id
    )!;
    expect(preset.parameters['targetCellOwners']).toEqual(parameters['targetCellOwners']);
    expect(preset.executionParameters).toEqual({denominator: 'zone'});
    expect(preset.sampleGeoJSONByInput).toBeUndefined();
  });

  test.each([
    9, 128, 256
  ])('round trips static and promoted ownership for a square grid of side %s', side => {
    const ownership = Array.from({length: side * side}, (_, index) => (index % 4) - 1);
    const sample = createDasymetricRedistributionSample();
    const raster = sample.inputs['ancillary'] as CatalogRasterInput;
    const parameters = normalizeCatalogParameters(binding, {
      ...sample.parameters,
      targetCellOwners: ownership
    });
    expect(MAXIMUM_DASYMETRIC_CELLS).toBe(WORKFLOW_LIMITS.maximumParameterNumbers);
    expect(
      prepareCuratedDasymetricInputs(
        {
          ...sample.inputs,
          ancillary: {
            ...raster,
            grid: {...raster.grid, width: side, height: side},
            band: {
              ...raster.band,
              values: new Float32Array(side * side).fill(1),
              validity: new Uint8Array(side * side).fill(1)
            }
          }
        },
        parameters
      ).targetOwners.length
    ).toBe(side * side);
    const template = structuredClone(createDasymetricRedistributionTemplate());
    const ownerInput = template.exposed.find(
      exposed => exposed.target.parameter === 'targetCellOwners'
    )!;
    template.document.inputs.find(input => input.id === ownerInput.inputId)!.value = ownership;
    expect(validateWorkflowAnalysisTemplate(template)).toEqual({executable: true, issues: []});
    const replay = parseWorkflowAnalysisTemplate(
      JSON.stringify(serializeWorkflowAnalysisTemplate(template))
    );
    expect(replay.document.inputs.find(input => input.id === ownerInput.inputId)!.value).toEqual(
      ownership
    );
    const document = structuredClone(template.document);
    document.steps.find(step => step.id === 'analysis')!.parameters['targetCellOwners'] = {
      from: 'static',
      value: ownership
    };
    document.inputs = document.inputs.filter(input => input.id !== ownerInput.inputId);
    expect(
      parseWorkflowDocument(JSON.parse(JSON.stringify(serializeWorkflowDocument(document))))
    ).toEqual(document);
    const staticTemplate = {
      ...template,
      document,
      exposed: template.exposed.filter(exposed => exposed !== ownerInput)
    };
    expect(
      parseWorkflowAnalysisTemplate(
        JSON.stringify(serializeWorkflowAnalysisTemplate(staticTemplate))
      ).document
    ).toEqual(document);
  });

  test('rejects one ownership entry beyond the shared capacity', () => {
    const ownership = new Array(WORKFLOW_LIMITS.maximumParameterNumbers + 1).fill(0);
    const template = structuredClone(createDasymetricRedistributionTemplate());
    const ownerInput = template.exposed.find(
      exposed => exposed.target.parameter === 'targetCellOwners'
    )!;
    template.document.inputs.find(input => input.id === ownerInput.inputId)!.value = ownership;
    expect(() => serializeWorkflowAnalysisTemplate(template)).toThrow(/maximumArrayLength/);
    expect(() => serializeWorkflowDocument(template.document)).toThrow(
      /maximum parameter number-list length/
    );
    expect(() => normalizeCatalogParameters(binding, {targetCellOwners: ownership})).toThrow();
  });

  test('never claims CPU execution without a WebGPU device', async () => {
    const sample = createDasymetricRedistributionSample();
    await expect(
      runCatalogOperationAsync(binding, sample.parameters, sample.inputs)
    ).rejects.toThrow(/GPU|device|CPU/);
  });
});
