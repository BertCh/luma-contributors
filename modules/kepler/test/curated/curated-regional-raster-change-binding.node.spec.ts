// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {scanSource} from '../../../../examples/deck/spatial-analysis-showcase/scripts/check-wgsl-identifiers.mjs';
import {
  CURATED_REGIONAL_RASTER_CHANGE_BINDING,
  prepareRegionalRasterChange,
  validateRegionalRasterChangeStatus,
  REGIONAL_RASTER_CHANGE_LIMITS
} from '../../src/curated/bindings/curated-regional-raster-change-binding';
import {CURATED_REGIONAL_RASTER_CHANGE_DESCRIPTOR} from '../../src/descriptors/descriptors-curated-regional-raster-change';
import {
  createRegionalRasterChangeSample,
  REGIONAL_RASTER_CHANGE_ANALYSIS_UPDATE,
  REGIONAL_RASTER_CHANGE_SAMPLE_LICENSE
} from '../../src/curated/samples/regional-raster-change';
import {validateKeplerOperationDescriptor} from '../../src/descriptors/validate-operation-descriptor';
import {
  createKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping,
  buildKeplerCuratedAnalysisRequest
} from '../../src/curated/curated-analysis-mapping';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import type {KeplerDatasetProfile} from '../../src/profile/dataset-profile-types';
import {
  createWorkflowOperationRegistry,
  makeWorkflowOperationFromCatalogBinding
} from '../../src/workflows/workflow-operations';
import {getWorkflowSourceOperations} from '../../src/workflows/workflow-source-operations';
import {
  parseWorkflowAnalysisTemplate,
  serializeWorkflowAnalysisTemplate,
  validateWorkflowAnalysisTemplate
} from '../../src/workflows/workflow-analysis-template-parser';
import {createCuratedWorkflowAnalysisTemplate} from '../../src/workflows/workflow-analysis-template';
import {promoteWorkflowParameter} from '../../src/workflows/workflow-edit';

import {getKeplerCuratedAnalysis} from '../../src/curated/curated-analyses';
import {appendCatalogRow, makeCatalogTable} from '../../src/analysis/catalog-geojson';
import {planKeplerResultPresentation} from '../../src/presentation/result-presentation';

const REGIONAL_RASTER_CHANGE_ANALYSIS = {
  ...getKeplerCuratedAnalysis('regional-raster-change')!,
  ...REGIONAL_RASTER_CHANGE_ANALYSIS_UPDATE
};

const parameters = {confirmComparable: true, threshold: 1};

it.each([
  {dataset: 'normalized recipe fixture', means: [2, -0.5], scale: 'diverging'},
  {dataset: 'independent north-up dataset', means: [1 / 3, 0, null], scale: 'sequential'}
])('plans the $scale scale for the $dataset zone means', ({means, scale}) => {
  // The same numerical zone-mean oracles are asserted against GPU results in the headless
  // spec. Missing zones do not add a sign; mixed cell differences can average to one side.
  const regions = makeCatalogTable('regions', 'Regional change', {mean_change: 'float64'});
  for (const mean of means) appendCatalogRow(regions, {mean_change: mean});
  const presentation = planKeplerResultPresentation(
    regions,
    CURATED_REGIONAL_RASTER_CHANGE_DESCRIPTOR.outputs[0]
  );
  expect(presentation?.column).toBe('mean_change');
  expect(presentation?.scale).toBe(scale);
  expect(presentation?.palette).toBe(scale);
  expect(presentation?.missingCount).toBe(means.filter(value => value === null).length);
  if (scale === 'sequential')
    expect(presentation?.notes.some(note => note.includes('lies on one side of 0'))).toBe(true);
});

it('uses legal WGSL identifiers in both generated publication binding declarations', () => {
  const file = new URL(
    '../../src/curated/bindings/curated-regional-raster-change-binding.ts',
    import.meta.url
  );
  const source = readFileSync(file, 'utf8');
  expect(scanSource(source, file.pathname).hits).toEqual([]);
  // Exercise the existing scanner on both publication loops, including names generated
  // outside WGSL template literals. This is source validation, not GPU compilation.
  const regression = source.replaceAll("name: 'destination'", "name: 'target'");
  expect(
    scanSource(regression, file.pathname).hits.filter(
      hit => hit.name === 'target' && hit.kind === 'generated binding'
    )
  ).toHaveLength(2);
});

it('rejects finite cell areas whose worst-case regional sum overflows Float32', () => {
  const sample = createRegionalRasterChangeSample();
  const createInput = (spacing: number) => {
    const grid = {...sample.before.grid, affine: [spacing, 0, 0, 0, spacing, 0] as const};
    return {
      before: {
        ...sample.before,
        grid,
        band: {...sample.before.band, values: new Float32Array([1, 1, 1, 1])}
      },
      after: {
        ...sample.after,
        grid,
        band: {...sample.after.band, values: new Float32Array([2, 2, 2, 2])}
      },
      zones: {
        type: 'FeatureCollection' as const,
        features: [
          {
            id: 'whole-grid',
            geometry: {
              type: 'Polygon',
              coordinates: [
                [
                  [0, 0],
                  [2 * spacing, 0],
                  [2 * spacing, 2 * spacing],
                  [0, 2 * spacing],
                  [0, 0]
                ]
              ]
            }
          }
        ]
      }
    };
  };
  const cellArea = Math.fround(1e19 * 1e19);
  expect(Number.isFinite(cellArea)).toBe(true);
  expect(Number.isFinite(Math.fround(cellArea * 4))).toBe(false);
  expect(() => prepareRegionalRasterChange(createInput(1e19), parameters)).toThrow(
    /curated-regional-raster-change.*maximum aggregate area/
  );
  // Large but representable aggregates remain supported; do not reject based on cell spacing alone.
  const safe = prepareRegionalRasterChange(createInput(1e18), parameters);
  expect(Number.isFinite(Math.fround(safe.cellArea * safe.cellCount))).toBe(true);
  const boundary = prepareRegionalRasterChange(createInput(2 ** 62.5), parameters);
  expect(boundary.cellArea * boundary.cellCount).toBe(REGIONAL_RASTER_CHANGE_LIMITS.aggregateArea);
  // Preserve headroom even for an exact total that is finite but above the declared bound.
  expect(Number.isFinite(Math.fround(Math.fround(7e18 * 7e18) * 4))).toBe(true);
  expect(() => prepareRegionalRasterChange(createInput(7e18), parameters)).toThrow(
    /maximum aggregate area/
  );
});

for (const secondDataset of [false, true]) {
  it(`prepares paired observations, identities and signed row order (${secondDataset})`, () => {
    const input = createRegionalRasterChangeSample(secondDataset);
    const prepared = prepareRegionalRasterChange(input, parameters);
    expect(prepared.cellCount).toBe(secondDataset ? 6 : 4);
    expect(prepared.cellArea).toBe(secondDataset ? 6 : 2);
    expect(Array.from(prepared.slices)).toEqual(
      secondDataset
        ? [10, 12, 20, 18, 30, NaN, 40, 41, NaN, 55, 60, 60]
        : [1, 3, NaN, 4, 5, 4, 2, 2]
    );
    expect(Array.from(prepared.observationMask)).toEqual(
      secondDataset ? [1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1] : [1, 1, 0, 1, 1, 1, 1, 1]
    );
    expect(prepared.polygons.sourceIndices).toEqual(secondDataset ? [0, 1, 2] : [0, 1]);
    expect(prepared.grid.affine).toEqual(secondDataset ? [2, 0, 0, 0, -3, 0] : [1, 0, 0, 0, 2, 0]);
    expect(Array.from(prepared.placement)).toEqual(secondDataset ? [0, -6, 2, -3] : [0, 0, 1, 2]);
    expect(Math.max(...prepared.polygons.positions)).toBeLessThanOrEqual(12);
    expect(input.before.grid.affine[2]).toBe(secondDataset ? 100000000 : 0);
  });

  it(`maps independently named raster datasets through the existing curated mapper (${secondDataset})`, () => {
    const input = createRegionalRasterChangeSample(secondDataset);
    const profiles: KeplerDatasetProfile[] = [
      ...[input.before, input.after].map(raster => ({
        datasetId: raster.id,
        label: raster.band.name!,
        kind: 'surface' as const,
        rowCount: raster.grid.width * raster.grid.height,
        columns: [],
        geometryColumns: []
      })),
      profileKeplerFeatureCollection(input.zones, {
        datasetId: 'districts',
        label: 'Districts',
        coordinateSystem: 'planar'
      })
    ];
    let mapping = createKeplerCuratedAnalysisMapping(REGIONAL_RASTER_CHANGE_ANALYSIS, profiles);
    for (const [slot, datasetId] of [
      ['before', input.before.id],
      ['after', input.after.id],
      ['zones', 'districts']
    ] as const)
      mapping = updateKeplerCuratedAnalysisMapping(
        mapping,
        {slot, datasetId},
        REGIONAL_RASTER_CHANGE_ANALYSIS,
        profiles
      );
    expect(mapping.complete).toBe(true);
    const request = buildKeplerCuratedAnalysisRequest(REGIONAL_RASTER_CHANGE_ANALYSIS, mapping, {
      parameterOverrides: parameters
    });
    expect(request.inputDatasetIds).toEqual({
      before: input.before.id,
      after: input.after.id,
      zones: 'districts'
    });
    expect(request.parameters).toEqual(parameters);
    expect(request.bindingId).toBe(CURATED_REGIONAL_RASTER_CHANGE_BINDING.id);
    expect(() =>
      updateKeplerCuratedAnalysisMapping(
        mapping,
        {slot: 'before', datasetId: 'districts'},
        REGIONAL_RASTER_CHANGE_ANALYSIS,
        profiles
      )
    ).toThrow(/does not fit|surface|raster/);
  });
}

it('rejects incompatible grids, unconfirmed units and unsafe materialization before GPU allocation', () => {
  const input = createRegionalRasterChangeSample();
  const prepare = (overrides: Partial<typeof input> = {}, values = parameters) =>
    prepareRegionalRasterChange({...input, ...overrides}, values);
  expect(() => prepare({}, {...parameters, confirmComparable: false})).toThrow(
    /confirmComparable must be true/
  );
  for (const threshold of [-1, NaN, Infinity, 1e31])
    expect(() => prepare({}, {...parameters, threshold})).toThrow(/threshold/);
  for (const grid of [
    {...input.after.grid, affine: [1, 0, 1, 0, 2, 0] as const},
    {...input.after.grid, pixelInterpretation: 'point' as const}
  ])
    expect(() => prepare({after: {...input.after, grid}})).toThrow(/align exactly/);
  const geographic = {crs: 'EPSG:4326' as const, cellSizeMode: 'geographic' as const};
  expect(() =>
    prepare({before: {...input.before, ...geographic}, after: {...input.after, ...geographic}})
  ).toThrow(/crs must be planar/);
  for (const affine of [
    [1, 1, 0, 0, 2, 0],
    [-1, 0, 0, 0, 2, 0]
  ] as const) {
    const grid = {...input.before.grid, affine};
    expect(() => prepare({before: {...input.before, grid}, after: {...input.after, grid}})).toThrow(
      /axis aligned/
    );
  }
  expect(() => prepare({zones: {type: 'FeatureCollection', features: []}})).toThrow(
    /zones.features.length/
  );
  expect(() =>
    prepare({
      zones: {
        type: 'FeatureCollection',
        features: Array(REGIONAL_RASTER_CHANGE_LIMITS.zones + 1).fill(input.zones.features[0])
      }
    })
  ).toThrow(/zones.features.length/);
  const largeGrid = {...input.before.grid, width: REGIONAL_RASTER_CHANGE_LIMITS.cells + 1};
  expect(() => prepare({before: {...input.before, grid: largeGrid}})).toThrow(/grid cell count/);
  expect(() =>
    prepare({
      zones: {
        type: 'FeatureCollection',
        features: [{geometry: {type: 'Point', coordinates: [0, 0]}}]
      }
    })
  ).toThrow(/zones geometry/);
  const malformed = {...input.before, band: {...input.before.band, validity: new Uint8Array(1)}};
  expect(() => prepare({before: malformed})).toThrow(/band.validity length/);
  expect(() => prepareRegionalRasterChange({}, parameters)).toThrow(/before must be/);
});

it('normalizes finite sentinels, infinities, invalidity and Float32 overflow as missing without mutating inputs', () => {
  const input = createRegionalRasterChangeSample();
  const values = new Float64Array([-9999, Infinity, 1e100, 2]);
  const before = {...input.before, band: {...input.before.band, values}};
  const prepared = prepareRegionalRasterChange({...input, before}, parameters);
  expect(Array.from(prepared.observationMask)).toEqual([0, 1, 0, 1, 0, 1, 1, 1]);
  expect(Array.from(values)).toEqual([-9999, Infinity, 1e100, 2]);
});

it('declares coherent descriptor, bounded output identity and a licence-clean synthetic fixture', () => {
  const binding = CURATED_REGIONAL_RASTER_CHANGE_BINDING;
  expect(
    validateKeplerOperationDescriptor(CURATED_REGIONAL_RASTER_CHANGE_DESCRIPTOR, binding)
  ).toEqual([]);
  expect(binding.engine).toBe('gpu');
  expect(binding.run).toBeUndefined();
  expect(
    binding.outputs.map(output => [output.tableId, output.geometry, output.identityColumns])
  ).toEqual([
    ['regions', 'polygon', ['zone_index', 'source_id']],
    ['differences', 'none', ['cell_index']]
  ]);
  expect(CURATED_REGIONAL_RASTER_CHANGE_DESCRIPTOR.outputs[0].presentation).toEqual({
    column: 'mean_change',
    scale: 'diverging'
  });
  expect(REGIONAL_RASTER_CHANGE_SAMPLE_LICENSE.license).toBe('MIT');
});

it('round-trips the existing AX envelope and validates against the real binding adapter', () => {
  // Validate locally with the real binding without mutating stream X's global registry.
  const template = createCuratedWorkflowAnalysisTemplate({
    ...REGIONAL_RASTER_CHANGE_ANALYSIS,
    exposedParameters: []
  });
  const registry = createWorkflowOperationRegistry([
    ...getWorkflowSourceOperations(),
    makeWorkflowOperationFromCatalogBinding(CURATED_REGIONAL_RASTER_CHANGE_BINDING)
  ]);
  for (const parameter of CURATED_REGIONAL_RASTER_CHANGE_BINDING.parameters) {
    const promoted = promoteWorkflowParameter(
      template.document,
      registry,
      'analysis',
      parameter.name
    );
    template.document = promoted.document;
    template.exposed = [
      ...template.exposed,
      {
        inputId: promoted.inputId,
        target: {stepId: 'analysis', parameter: parameter.name},
        defaultPolicy: {from: 'constant'}
      }
    ];
  }
  template.sample = {id: 'regional-raster-change', ...REGIONAL_RASTER_CHANGE_SAMPLE_LICENSE};
  expect(parseWorkflowAnalysisTemplate(serializeWorkflowAnalysisTemplate(template))).toEqual(
    template
  );
  const validation = validateWorkflowAnalysisTemplate(template, {registry});
  expect(validation, JSON.stringify(validation)).toMatchObject({executable: true});
});

it('rejects rasterization overflow and every named recipe status before publication', () => {
  const complete = {
    rasterOverflow: 0,
    requiredCrossings: 4,
    crossingCapacity: 16,
    stages: [{name: 'change-summary.overflow', value: 0}]
  };
  expect(() => validateRegionalRasterChangeStatus(complete)).not.toThrow();
  expect(() =>
    validateRegionalRasterChangeStatus({...complete, rasterOverflow: 1, requiredCrossings: 17})
  ).toThrow(/crossingCapacity limit 16 exceeded.*required 17/);
  for (const name of [
    'total-area.overflow',
    'valid-area.overflow',
    'changed-area.overflow',
    'change-summary.overflow',
    'change-threshold.invalidCount'
  ])
    expect(() =>
      validateRegionalRasterChangeStatus({...complete, stages: [{name, value: 1}]})
    ).toThrow(name);
});

it('bounds polygon packing before creating large GPU resources', () => {
  const input = createRegionalRasterChangeSample();
  const ring = Array.from({length: REGIONAL_RASTER_CHANGE_LIMITS.vertices + 1}, () => [0, 0]);
  expect(() =>
    prepareRegionalRasterChange(
      {
        ...input,
        zones: {
          type: 'FeatureCollection',
          features: [{geometry: {type: 'Polygon', coordinates: [ring]}}]
        }
      },
      parameters
    )
  ).toThrow(/zones vertexCount/);
});
