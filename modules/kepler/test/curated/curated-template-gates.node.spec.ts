// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync} from 'node:fs';
import {describe, expect, it} from 'vitest';
import type {CatalogParameterValue} from '../../src/analysis/catalog-binding-types';
import type {CatalogDataInput} from '../../src/analysis/catalog-data-input';
import {
  getCatalogBinding,
  normalizeCatalogParameters,
  runCatalogOperationAsync
} from '../../src/analysis/catalog-bindings';
import {KEPLER_CURATED_ANALYSES} from '../../src/curated/curated-analyses';
import {
  createKeplerCuratedAnalysisSample,
  getKeplerCuratedAnalysisSampleParameters,
  KEPLER_CURATED_ANALYSIS_SAMPLE_COLUMNS,
  type KeplerCuratedAnalysisSampleId
} from '../../src/curated/curated-analysis-samples';
import {
  buildKeplerCuratedAnalysisRequest,
  createKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping
} from '../../src/curated/curated-analysis-mapping';
import {createDasymetricRedistributionSample} from '../../src/curated/samples/dasymetric-redistribution';
import {DRIVE_TIME_BANDS_SECOND_SAMPLE} from '../../src/curated/samples/drive-time-bands-data';
import {NETWORK_DISRUPTION_SECOND_SAMPLE} from '../../src/curated/samples/network-disruption';
import {createMappedZonalExposureSample} from '../../src/curated/samples/zonal-exposure';
import {resolveKeplerOperationParameters} from '../../src/descriptors/default-parameters';
import {getKeplerOperationDescriptor} from '../../src/descriptors/operation-descriptors';
import {profileKeplerWorkbenchData} from '../../src/descriptors/workbench-adapters';
import {
  applyKeplerDatasetProfileDeclarations,
  profileKeplerFeatureCollection
} from '../../src/profile/dataset-profile';
import {makeWorkflowData, makeWorkflowRasterData} from '../../src/workflows/workflow-data';
import {createCuratedWorkflowAnalysisTemplate} from '../../src/workflows/workflow-analysis-template';
import {
  parseWorkflowAnalysisTemplate,
  serializeWorkflowAnalysisTemplate,
  validateWorkflowAnalysisTemplate
} from '../../src/workflows/workflow-analysis-template-parser';
import {appendCatalogRow, makeCatalogTable} from '../../src/analysis/catalog-geojson';
import {
  KEPLER_CATEGORICAL_PALETTE,
  planKeplerResultPresentation
} from '../../src/presentation/result-presentation';

/** Explicit execution exclusions: Node proves mapping, never claims a GPU recipe ran. */
const GPU_ONLY_SPECS: Readonly<Record<string, string>> = {
  'emerging-hot-spots': 'curated-emerging-hot-spots-binding.spec.ts',
  'dasymetric-redistribution': 'curated-dasymetric-redistribution-binding.spec.ts',
  'drainage-network': 'curated-drainage-network-binding.spec.ts',
  'surface-preparation': 'curated-surface-preparation-binding.spec.ts',
  'zonal-exposure': 'curated-zonal-exposure-binding.spec.ts',
  'regional-raster-change': 'curated-regional-raster-change-binding.spec.ts',
  'landscape-metrics': 'curated-landscape-metrics-binding.spec.ts',
  'network-disruption': 'curated-network-disruption-binding.spec.ts',
  'fleet-dwell': 'curated-fleet-dwell-binding.spec.ts',
  'track-quality': 'curated-track-quality-binding.spec.ts',
  'cell-flow-topology': 'curated-cell-flow-topology-binding.spec.ts',
  'drive-time-bands': 'curated-drive-time-bands.spec.ts'
};

function getFixtureSettings(analysisId: string, second: boolean) {
  let parameters: Record<string, CatalogParameterValue> = {
    ...getKeplerCuratedAnalysisSampleParameters(analysisId)
  };
  let columns: Readonly<Record<string, string>> | undefined;
  if (second) {
    if (analysisId === 'network-disruption') {
      parameters = {...NETWORK_DISRUPTION_SECOND_SAMPLE.parameters};
      columns = NETWORK_DISRUPTION_SECOND_SAMPLE.columns;
    }
    if (analysisId === 'drive-time-bands') {
      parameters = {
        ...DRIVE_TIME_BANDS_SECOND_SAMPLE.parameters,
        costBreaks: [...DRIVE_TIME_BANDS_SECOND_SAMPLE.parameters.costBreaks]
      };
      columns = DRIVE_TIME_BANDS_SECOND_SAMPLE.columns;
    }
    if (analysisId === 'dasymetric-redistribution') {
      parameters = createDasymetricRedistributionSample(true).parameters;
      columns = {'sources.total': 'inhabitants_2026'};
    }
    if (analysisId === 'zonal-exposure') {
      parameters = createMappedZonalExposureSample().parameters;
      columns = {
        'cells.cell-index': String(parameters['cellIndexProperty']),
        'cells.zone': String(parameters['zoneProperty']),
        'cells.population': String(parameters['populationProperty']),
        'cells.cell-area': String(parameters['cellAreaProperty'])
      };
    }
    if (analysisId === 'cell-flow-topology') {
      parameters = {resolution: 2};
      columns = {'flows.origin': 'from_hex', 'flows.destination': 'to_hex', 'flows.flow': 'tonnes'};
    }
  }
  return {parameters, columns};
}

describe('curated registry executable template gates', () => {
  it('keeps all twelve original wave-one rows and the additional drive-time recipe', () => {
    expect(KEPLER_CURATED_ANALYSES).toHaveLength(28);
    expect(KEPLER_CURATED_ANALYSES.filter(analysis => analysis.wave === 1)).toHaveLength(13);
    expect(
      KEPLER_CURATED_ANALYSES.filter(analysis => analysis.execution.status !== 'available')
    ).toEqual([]);
    expect(
      KEPLER_CURATED_ANALYSES.filter(
        analysis =>
          analysis.execution.status === 'available' &&
          getCatalogBinding(analysis.execution.bindingId)?.engine === 'gpu'
      )
        .map(analysis => analysis.id)
        .sort()
    ).toEqual(Object.keys(GPU_ONLY_SPECS).sort());
    for (const spec of Object.values(GPU_ONLY_SPECS))
      expect(existsSync(new URL(spec, import.meta.url))).toBe(true);
  });

  for (const analysis of KEPLER_CURATED_ANALYSES) {
    it(`${analysis.id}: parses, validates and maps sample plus independently named data`, async () => {
      const template = createCuratedWorkflowAnalysisTemplate(analysis);
      expect(
        parseWorkflowAnalysisTemplate(JSON.stringify(serializeWorkflowAnalysisTemplate(template)))
      ).toEqual(template);
      expect(validateWorkflowAnalysisTemplate(template)).toMatchObject({
        executable: true,
        issues: []
      });
      expect(template.sample).toMatchObject({id: analysis.sampleId, license: 'MIT'});
      if (analysis.execution.status !== 'available' || !analysis.sampleId)
        throw new Error(`${analysis.id}: missing execution/sample`);
      const execution = analysis.execution;
      const binding = getCatalogBinding(execution.bindingId)!;
      const descriptor = getKeplerOperationDescriptor(binding.id)!;
      for (const second of [false, true]) {
        const sample = createKeplerCuratedAnalysisSample(analysis.sampleId, second ? 733 : 0);
        const settings = getFixtureSettings(analysis.id, second);
        const columns =
          settings.columns ??
          KEPLER_CURATED_ANALYSIS_SAMPLE_COLUMNS[
            analysis.sampleId as KeplerCuratedAnalysisSampleId
          ];
        // The second realization has different observations; every mapped column also gets a
        // new name, so a binding that silently falls back to the original sample names fails.
        const rename = (name: string) => (second ? `survey_${name}` : name);
        const inputsByDataset: Record<string, CatalogDataInput> = {};
        const profiles = analysis.slots.map(slot => {
          const input = sample[slot.name];
          const datasetId = `${second ? 'independent-survey' : 'bundled-sample'}/${analysis.id}/${slot.name}`;
          const coordinateSystem =
            settings.parameters['coordinateSystem'] === 'planar' ? 'planar' : 'EPSG:4326';
          let profile;
          if (input.type === 'FeatureCollection') {
            const collection = {
              ...input,
              features: input.features.map((feature, index) => ({
                ...feature,
                id: `${datasetId}/${index}`,
                properties: Object.fromEntries(
                  Object.entries(feature.properties ?? {}).map(([name, value]) => [
                    rename(name),
                    value
                  ])
                )
              }))
            };
            inputsByDataset[datasetId] = collection;
            profile = profileKeplerFeatureCollection(collection, {
              datasetId,
              label: datasetId,
              coordinateSystem,
              maximumSampleSize: Math.max(2, collection.features.length)
            });
          } else if (input.type === 'raster') {
            const raster = {...input, id: datasetId};
            inputsByDataset[datasetId] = raster;
            profile = profileKeplerWorkbenchData(makeWorkflowRasterData(raster, 65536));
          } else {
            const table = {
              ...input.table,
              id: datasetId,
              columns: Object.fromEntries(
                Object.entries(input.table.columns).map(([name, values]) => [rename(name), values])
              ),
              columnTypes: Object.fromEntries(
                Object.entries(input.table.columnTypes).map(([name, type]) => [rename(name), type])
              )
            };
            inputsByDataset[datasetId] = {
              ...input,
              table,
              identityColumn: input.identityColumn ? rename(input.identityColumn) : undefined
            };
            profile = profileKeplerWorkbenchData(makeWorkflowData(table));
          }
          const measurement = Object.fromEntries(
            slot.fields.flatMap(field =>
              (columns[`${slot.name}.${field.role}`]?.split(',') ?? []).flatMap(name =>
                field.measurement?.[0] ? [[rename(name), field.measurement[0]]] : []
              )
            )
          );
          return applyKeplerDatasetProfileDeclarations(profile, {
            measurement,
            ...(slot.rowSemantics?.length === 1 ? {rowSemantics: slot.rowSemantics[0]} : {})
          });
        });
        // Field parameters belong to the mapping, not to sample presets.
        const parameters = Object.fromEntries(
          Object.entries(settings.parameters).filter(
            ([name]) => !Object.values(execution.fieldParameters).includes(name)
          )
        );
        let mapping = createKeplerCuratedAnalysisMapping(analysis, profiles, {parameters});
        for (const [index, slot] of analysis.slots.entries()) {
          mapping = updateKeplerCuratedAnalysisMapping(
            mapping,
            {slot: slot.name, datasetId: profiles[index].datasetId},
            analysis,
            profiles
          );
          for (const field of slot.fields) {
            const names = columns[`${slot.name}.${field.role}`]?.split(',') ?? [];
            mapping = updateKeplerCuratedAnalysisMapping(
              mapping,
              {slot: slot.name, role: field.role, columns: names.map(rename)},
              analysis,
              profiles
            );
          }
        }
        expect(
          mapping.complete,
          `${analysis.id} ${second}: ${JSON.stringify(mapping.problems)}`
        ).toBe(true);
        const request = buildKeplerCuratedAnalysisRequest(analysis, mapping);
        expect(
          Object.values(request.inputDatasetIds).every(identifier =>
            identifier.startsWith(second ? 'independent-survey/' : 'bundled-sample/')
          )
        ).toBe(true);
        for (const [key, parameter] of Object.entries(execution.fieldParameters))
          if (columns[key])
            expect(request.parameters[parameter]).toBe(
              columns[key].split(',').map(rename).join(',')
            );
        const profilesByInput = Object.fromEntries(
          Object.entries(request.inputDatasetIds).map(([name, identifier]) => [
            name,
            profiles.find(profile => profile.datasetId === identifier)!
          ])
        );
        if (binding.parameters.some(parameter => parameter.name === 'coordinateSystem')) {
          const profile = binding.inputs
            .map(input => profilesByInput[input.name])
            .find(profile => profile?.coordinates);
          const coordinateSystem =
            profile?.coordinates?.model === 'planar' ? 'planar' : 'EPSG:4326';
          const overrides = {...request.parameters};
          delete overrides['coordinateSystem'];
          for (const executionEngine of ['cpu', 'gpu'] as const) {
            const defaults = resolveKeplerOperationParameters(
              binding,
              descriptor,
              profilesByInput,
              {overrides, executionEngine}
            );
            expect(
              normalizeCatalogParameters(binding, defaults.parameters)['coordinateSystem'],
              `${analysis.id}: ${second ? 'second' : 'sample'} ${executionEngine} coordinate model`
            ).toBe(coordinateSystem);
          }
        }
        if (binding.engine === 'gpu') continue;
        const resolved = resolveKeplerOperationParameters(binding, descriptor, profilesByInput, {
          overrides: request.parameters,
          executionEngine: 'cpu'
        });
        const inputs = Object.fromEntries(
          Object.entries(request.inputDatasetIds).map(([name, identifier]) => [
            name,
            inputsByDataset[identifier]
          ])
        );
        const result = await runCatalogOperationAsync(binding, resolved.parameters, inputs);
        expect(result.engine).toBe('cpu-reference');
        expect(result.diagnostics.filter(diagnostic => diagnostic.severity === 'error')).toEqual(
          []
        );
        expect(result.tables.reduce((count, table) => count + table.rowCount, 0)).toBeGreaterThan(
          0
        );
        for (const output of binding.outputs) {
          const table = result.tables.find(table => table.id === output.tableId)!;
          expect(table, `${analysis.id}: ${output.tableId}`).toBeDefined();
          for (const identity of output.identityColumns)
            expect(table.columns[identity]).toHaveLength(table.rowCount);
        }
      }
    }, 120_000);
  }

  it('uses the shared AX categorical palette for both drive-time band maps', () => {
    const descriptor = getKeplerOperationDescriptor('curated-drive-time-bands')!;
    for (const output of descriptor.outputs.filter(output => output.presentation)) {
      const table = makeCatalogTable(output.tableId, output.tableId, {band: 'int32'});
      for (const band of [0, 1, 2]) appendCatalogRow(table, {band});
      const plan = planKeplerResultPresentation(table, output)!;
      expect(plan.palette).toBe('categorical');
      expect(plan.legend.map(entry => entry.color)).toEqual(KEPLER_CATEGORICAL_PALETTE.slice(0, 3));
    }
  });
});
