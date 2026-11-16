// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateCapacity, validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {
  applyKeplerDatasetProfileDeclarations,
  countCatalogCatchmentSeedCollisions,
  createKeplerCuratedAnalysisSample,
  getKeplerCuratedAnalysis,
  getKeplerWorkbenchCuratedPresets,
  getCatalogBinding,
  getCatalogDefaultParameters,
  getKeplerOperationDescriptor,
  makeWorkflowDataFromFeatures,
  makeWorkflowData,
  makeWorkflowRasterData,
  profileKeplerWorkbenchData,
  type CatalogDataInput,
  profileKeplerFeatureCollection,
  profileKeplerDatasetNetwork,
  resolveKeplerOperationParameters,
  WORKFLOW_LIMITS,
  type CatalogFeatureCollection,
  type CatalogParameterValue,
  type WorkflowData,
  type WorkflowHost,
  type WorkflowDatasetProfile
} from '@luma-contributors/kepler';
import {
  makeCatalogPointColumns,
  readCatalogFeaturesFromNativeDataset
} from '../analysis-catalog-input';
import {getH3Boundary, prepareH3SourceFromArrow} from '../h3-source';
import {createSandboxSample, SANDBOX_SAMPLES, type SandboxSampleId} from '../sandbox-samples';
import type {WorkbenchStore} from './workbench-store';

const SOURCE_PREFIX = 'ax-sample/';
const SAMPLE_DATA = new Map<string, WorkflowData>();
const CONNECTED_HOSTS = new WeakSet<WorkflowHost>();

export function isWorkbenchAnalysisSample(datasetId: string): boolean {
  return datasetId.startsWith(SOURCE_PREFIX);
}

export function getWorkbenchCuratedSampleId(analysisId: string, slot: string): string {
  return `${SOURCE_PREFIX}curated/${analysisId}/${slot}`;
}

export function getWorkbenchSandboxSampleId(sampleId: SandboxSampleId): string {
  return `${SOURCE_PREFIX}sandbox/${sampleId}/data`;
}

function makeSampleData(
  collection: CatalogDataInput,
  datasetId: string,
  coordinateSystem: 'planar' | 'EPSG:4326'
): WorkflowData {
  if (collection.type !== 'FeatureCollection') {
    const data =
      collection.type === 'raster'
        ? makeWorkflowRasterData({...collection, id: datasetId}, WORKFLOW_LIMITS.maximumSourceRows)
        : makeWorkflowData(
            {...collection.table, id: datasetId},
            {identityColumn: collection.identityColumn}
          );
    data.analysisProfile = profileKeplerWorkbenchData(data);
    return data;
  }
  validateCapacity(
    'WorkbenchAnalysisSample',
    'rowCount',
    collection.features.length,
    0,
    WORKFLOW_LIMITS.maximumSourceRows
  );
  const data = makeWorkflowDataFromFeatures(collection, {id: datasetId, label: datasetId});
  data.table.coordinateSystem = coordinateSystem;
  // Authored samples are small enough to profile every row. Sampled null counts cannot certify
  // requirements such as emerging-hot-spots' complete timestamps.
  data.analysisProfile = {
    ...profileKeplerFeatureCollection(collection, {
      datasetId,
      label: datasetId,
      coordinateSystem
    }),
    geometryColumns: ['geometry']
  };
  return data;
}

export function getWorkbenchCuratedSampleParameters(
  analysisId: string
): Record<string, CatalogParameterValue> {
  const parameters = {
    ...getKeplerWorkbenchCuratedPresets().find(preset => preset.curatedAnalysisId === analysisId)
      ?.parameters
  };
  if (analysisId === 'track-quality' || analysisId === 'fleet-dwell') {
    parameters['timestampEncoding'] = 'unix-millisecond';
  }
  return parameters;
}

/** Use the actual sandbox factory, including exact H3 identities and rows without valid geometry. */
export function getWorkbenchSandboxSampleCollection(
  sampleId: SandboxSampleId
): CatalogFeatureCollection {
  const sample = createSandboxSample(sampleId);
  const fields = sample.table.schema.fields;
  const vectors = fields.map(field => sample.table.getChild(field.name)!);
  if (sample.h3Column) {
    const source = prepareH3SourceFromArrow(sample.table.getChild(sample.h3Column)!, {
      maximumRowCount: WORKFLOW_LIMITS.maximumSourceRows
    });
    return {
      type: 'FeatureCollection',
      features: source.batches.flatMap((batch, batchIndex) =>
        Array.from({length: batch.rowCount}, (_, rowIndex) => {
          const sourceRow = batch.sourceRowOffset + rowIndex;
          const boundary = getH3Boundary(source, batchIndex, rowIndex);
          return {
            type: 'Feature' as const,
            id: sourceRow,
            geometry: boundary.length ? {type: 'Polygon' as const, coordinates: [boundary]} : null,
            properties: Object.fromEntries(
              fields.map((field, columnIndex) => {
                const value: unknown = vectors[columnIndex].get(sourceRow);
                return [field.name, typeof value === 'bigint' ? String(value) : value];
              })
            )
          };
        })
      )
    };
  }
  const grouping = sample.tripOptions?.grouping;
  const geometry = grouping
    ? makeCatalogPointColumns(grouping.longitudeColumn, grouping.latitudeColumn)
    : (sample.geometryColumn ?? makeCatalogPointColumns('longitude', 'latitude'));
  return readCatalogFeaturesFromNativeDataset(
    {
      fields,
      dataContainer: {
        numRows: () => sample.table.numRows,
        valueAt: (rowIndex, columnIndex) => vectors[columnIndex].get(rowIndex)
      }
    },
    geometry
  );
}

/** Populate immutable dataset sources rather than placing large populations in text parameters. */
export function prepareWorkbenchCuratedSample(analysisId: string): void {
  const analysis = getKeplerCuratedAnalysis(analysisId);
  validateRequirement(
    'WorkbenchAnalysisSample',
    Boolean(analysis?.sampleId && analysis.execution.status === 'available'),
    'analysis must have an available sample'
  );
  if (!analysis?.sampleId || analysis.execution.status !== 'available') return;
  if (
    analysis.slots.every(slot =>
      SAMPLE_DATA.has(getWorkbenchCuratedSampleId(analysisId, slot.name))
    )
  )
    return;
  const inputs = createKeplerCuratedAnalysisSample(analysis.sampleId);
  const parameters = getWorkbenchCuratedSampleParameters(analysisId);
  const raster = Object.values(inputs).find(input => input.type === 'raster');
  const coordinateSystem =
    parameters['coordinateSystem'] === 'planar' ||
    (raster?.type === 'raster' && raster.crs !== 'EPSG:4326')
      ? 'planar'
      : 'EPSG:4326';
  const dataBySlot = Object.fromEntries(
    analysis.slots.map(slot => {
      const input = inputs[slot.name];
      validateRequirement('WorkbenchAnalysisSample', Boolean(input), 'sample input must exist');
      const data = makeSampleData(
        input,
        getWorkbenchCuratedSampleId(analysisId, slot.name),
        coordinateSystem
      );
      data.analysisProfile = applyKeplerDatasetProfileDeclarations(data.analysisProfile!, {
        ...(input.type === 'FeatureCollection'
          ? {
              coordinateModel:
                coordinateSystem === 'planar' ? ('planar' as const) : ('geographic' as const)
            }
          : {}),
        ...(slot.rowSemantics?.length === 1 ? {rowSemantics: slot.rowSemantics[0]} : {})
      });
      return [slot.name, data];
    })
  );
  const execution = analysis.execution;
  const binding = getCatalogBinding(execution.bindingId)!;
  const descriptor = getKeplerOperationDescriptor(binding.id)!;
  const profilesByInput = Object.fromEntries(
    analysis.slots.map(slot => [
      execution.inputs[slot.name],
      dataBySlot[slot.name].analysisProfile!
    ])
  );
  const resolved = resolveKeplerOperationParameters(binding, descriptor, profilesByInput, {
    overrides: parameters,
    executionEngine: 'gpu'
  }).parameters;
  for (const slot of analysis.slots) {
    const data = dataBySlot[slot.name];
    if (slot.kind === 'network')
      data.analysisProfile = {
        ...data.analysisProfile!,
        network: profileKeplerDatasetNetwork(
          data.analysisProfile!,
          inputs[slot.name] as CatalogFeatureCollection,
          resolved,
          'gpu',
          binding.id
        )
      };
  }
  if (binding.id === 'recipe-straight-line-catchments') {
    const facilities = dataBySlot['facilities'];
    const observations = dataBySlot['observations'];
    facilities.analysisProfile = {
      ...facilities.analysisProfile!,
      catchmentSeeds: {
        facilityRevision: facilities.analysisProfile!.revision,
        observationsDatasetId: observations.table.id,
        observationsRevision: observations.analysisProfile!.revision,
        coordinateSystem: String(resolved['coordinateSystem']),
        cellSize: resolved['cellSize'] as number,
        collisionCount: countCatalogCatchmentSeedCollisions(
          inputs as Record<string, CatalogFeatureCollection>,
          resolved
        )
      }
    };
  }
  for (const data of Object.values(dataBySlot)) SAMPLE_DATA.set(data.table.id, data);
}

export function getWorkbenchAnalysisSampleData(datasetId: string): WorkflowData | undefined {
  const cached = SAMPLE_DATA.get(datasetId);
  if (cached) return cached;
  const [, family, identity, slot] = datasetId.split('/');
  if (family === 'curated') {
    const analysis = getKeplerCuratedAnalysis(identity);
    if (
      !analysis?.slots.some(candidate => candidate.name === slot) ||
      !analysis.sampleId ||
      analysis.execution.status !== 'available'
    )
      return undefined;
    prepareWorkbenchCuratedSample(identity);
    return SAMPLE_DATA.get(datasetId);
  }
  const sample = SANDBOX_SAMPLES.find(candidate => candidate.id === identity);
  if (family !== 'sandbox' || !sample || slot !== 'data') return undefined;
  const collection = getWorkbenchSandboxSampleCollection(sample.id);
  const data = makeSampleData(collection, datasetId, 'EPSG:4326');
  data.analysisProfile = applyKeplerDatasetProfileDeclarations(data.analysisProfile!, {
    coordinateModel: 'geographic',
    rowSemantics:
      sample.id === 'points-time'
        ? 'events'
        : sample.id === 'numeric-grid'
          ? 'surface-samples'
          : 'objects'
  });
  if (sample.id === 'directed-network') {
    const binding = getCatalogBinding('network-structure')!;
    data.analysisProfile = {
      ...data.analysisProfile,
      network: profileKeplerDatasetNetwork(
        data.analysisProfile,
        collection,
        getCatalogDefaultParameters(binding),
        'gpu',
        binding.id
      )
    };
  }
  SAMPLE_DATA.set(datasetId, data);
  return data;
}

function describeSample(data: WorkflowData): WorkflowDatasetProfile {
  return {
    id: data.table.id,
    label: data.table.label,
    rowCount: data.table.rowCount,
    resident: false,
    fields: Object.entries(data.table.columnTypes).map(([name, type]) => ({name, type})),
    geometryFields: data.table.geometryColumn ? [data.table.geometryColumn] : []
  };
}

/** Extend the existing host adapter; execution still uses source-dataset and the shared runner. */
export function connectWorkbenchAnalysisSampleSources(store: WorkbenchStore): void {
  const host = store.host;
  if (CONNECTED_HOSTS.has(host)) return;
  const listDatasets = host.listDatasets.bind(host);
  const readDataset = host.readDataset.bind(host);
  const getRevision = host.getDatasetRevisionKey.bind(host);
  const getProfile = host.getAnalysisProfile?.bind(host);
  host.listDatasets = () => {
    for (const document of store.getState().workflows)
      for (const step of document.steps) {
        const source = step.parameters['dataset'];
        if (
          step.operationId === 'source-dataset' &&
          source?.from === 'static' &&
          typeof source.value === 'string' &&
          isWorkbenchAnalysisSample(source.value)
        )
          getWorkbenchAnalysisSampleData(source.value);
      }
    return [...listDatasets(), ...[...SAMPLE_DATA.values()].map(describeSample)];
  };
  host.getAnalysisProfile = datasetId =>
    isWorkbenchAnalysisSample(datasetId)
      ? getWorkbenchAnalysisSampleData(datasetId)?.analysisProfile
      : getProfile?.(datasetId);
  host.getDatasetRevisionKey = request =>
    isWorkbenchAnalysisSample(request.datasetId)
      ? `analysis-sample-v1:${request.datasetId}`
      : getRevision(request);
  host.readDataset = async (request, signal) => {
    if (!isWorkbenchAnalysisSample(request.datasetId)) return readDataset(request, signal);
    signal.throwIfAborted();
    const data = getWorkbenchAnalysisSampleData(request.datasetId);
    validateRequirement(
      'WorkbenchAnalysisSample',
      Boolean(data),
      'datasetId must identify an available sample'
    );
    validateRequirement(
      'WorkbenchAnalysisSample',
      request.population === 'all',
      'population must be all'
    );
    validateRequirement(
      'WorkbenchAnalysisSample',
      data!.table.geometryColumn
        ? request.geometry.mode === 'geojson-field' &&
            request.geometry.field === data!.table.geometryColumn
        : request.geometry.mode === 'none',
      'geometry must match the native sample input'
    );
    validateCapacity(
      'WorkbenchAnalysisSample',
      'rowCount',
      data!.table.rowCount,
      0,
      request.maximumRows
    );
    return data!;
  };
  CONNECTED_HOSTS.add(host);
}
