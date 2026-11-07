// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  doGraphDataViewsOverlap,
  validatePackedView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/graph/gpu-dataframe';
import {
  GPUInverseDistanceWeighting,
  getGPUInverseDistanceWeightingParameterValues
} from '@luma-contributors/graph/gpu-spatial-analysis';
import type {GPURasterMetadata} from '@luma-contributors/graph/gpu-raster';
import {
  MAXIMUM_KEPLER_IDW_SOURCE_ROWS,
  parseKeplerIDWSettings,
  type KeplerIDWSettings
} from '../recipes/analysis-recipe';
import type {KeplerAnalysisProvenance} from './analysis-bindings';
import {
  addKeplerLinearKernel as addKernel,
  getKeplerIndexGridSize,
  KEPLER_ANALYSIS_ROW_TILE_SIZE
} from './analysis-scaling';
import {copyPointStatus} from './point-hotspots';

export type KeplerPointIDWOutputs = {
  values?: GraphDataView<'float32'>;
  counts?: GraphDataView<'uint32'>;
  validity?: GraphDataView<'uint32'>;
  status?: {
    validSourceCount?: GraphDataView<'uint32'>;
    validCellCount?: GraphDataView<'uint32'>;
    noDataCellCount?: GraphDataView<'uint32'>;
  };
};

export type KeplerPointIDWProps = KeplerAnalysisProvenance & {
  id?: string;
  settings: KeplerIDWSettings;
  longitude: GraphDataView<'float32'>;
  latitude: GraphDataView<'float32'>;
  values: GraphDataView<'float32'>;
  /** Numeric validity intersected with the declared population. */
  mask?: GraphDataView<'uint32'>;
  coordinateValidityMask?: GraphDataView<'uint32'>;
  outputs?: KeplerPointIDWOutputs;
};

/**
 * Geographic IDW into a bounded north-up raster. Selection changes the contributing samples;
 * no source-row topology or source-row output identity is implied. Values are sampled at cell
 * centers, with spherical distances in meters. Exact hits use the first source row even below
 * the minimum count. Counts include radius-boundary samples whose tapered weight is zero.
 *
 * This is interpolation without variance, confidence intervals or cross-validation. The caller
 * retains borrowed inputs and outputs through submission and destroys parameters only after GPU
 * completion. Omitted outputs and scratch belong to the compiled graph.
 */
export function addKeplerPointIDW<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: KeplerPointIDWProps
) {
  const settings = parseKeplerIDWSettings(props.settings);
  const id = props.id ?? 'kepler-point-idw';
  const rowCount = props.values.length;
  const cellCount = settings.grid.width * settings.grid.height;
  if (
    !props.source.datasetId ||
    !Number.isSafeInteger(props.source.revision) ||
    props.source.revision < 1 ||
    rowCount < 1 ||
    rowCount > MAXIMUM_KEPLER_IDW_SOURCE_ROWS ||
    (props.population !== 'all' && props.population !== 'selected') ||
    (props.population === 'selected'
      ? !props.mask ||
        !Number.isSafeInteger(props.selectionRevision) ||
        (props.selectionRevision ?? 0) < 1
      : props.selectionRevision !== undefined)
  )
    throw new Error('IDW requires a current source and explicit population');
  const inputs = [
    props.values,
    props.longitude,
    props.latitude,
    props.mask,
    props.coordinateValidityMask
  ].filter(view => view !== undefined);
  for (const view of inputs) {
    validatePackedView(
      view,
      view === props.mask || view === props.coordinateValidityMask ? ['uint32'] : ['float32'],
      id
    );
    if (view.length !== rowCount || view.buffer.graph !== graph)
      throw new Error('IDW inputs must match source rows and graph');
  }
  const provided = props.outputs ?? {};
  for (const [view, format, length] of [
    [provided.values, 'float32', cellCount],
    [provided.counts, 'uint32', cellCount],
    [provided.validity, 'uint32', cellCount],
    [provided.status?.validSourceCount, 'uint32', 1],
    [provided.status?.validCellCount, 'uint32', 1],
    [provided.status?.noDataCellCount, 'uint32', 1]
  ] as const) {
    if (!view) continue;
    validatePackedView(view, [format], id);
    if (view.length !== length || view.buffer.graph !== graph)
      throw new Error('IDW outputs must match raster cells and graph');
  }
  const supplied = [
    provided.values,
    provided.counts,
    provided.validity,
    ...Object.values(provided.status ?? {})
  ].filter(view => view !== undefined);
  for (let index = 0; index < supplied.length; index++) {
    if (
      [...inputs, ...supplied.slice(0, index)].some(view =>
        doGraphDataViewsOverlap(view, supplied[index])
      )
    )
      throw new Error('IDW outputs must be disjoint');
  }
  const values =
    provided.values ?? createTransientView(graph, `${id}-values-output`, 'float32', cellCount);
  const counts =
    provided.counts ?? createTransientView(graph, `${id}-counts-output`, 'uint32', cellCount);
  const validity =
    provided.validity ?? createTransientView(graph, `${id}-validity-output`, 'uint32', cellCount);
  const status = {
    validSourceCount:
      provided.status?.validSourceCount ??
      createTransientView(graph, `${id}-source-count-output`, 'uint32', 1),
    validCellCount:
      provided.status?.validCellCount ??
      createTransientView(graph, `${id}-cell-count-output`, 'uint32', 1),
    noDataCellCount:
      provided.status?.noDataCellCount ??
      createTransientView(graph, `${id}-nodata-count-output`, 'uint32', 1)
  };
  const positions = createTransientView(graph, `${id}-positions`, 'float32x2', rowCount);
  const effectiveMask = createTransientView(graph, `${id}-effective-mask`, 'uint32', rowCount);
  const [west, south, east, north] = settings.grid.bounds;
  const grid: GPURasterMetadata = {
    width: settings.grid.width,
    height: settings.grid.height,
    affine: [
      (east - west) / settings.grid.width,
      0,
      west,
      0,
      (south - north) / settings.grid.height,
      north
    ],
    pixelInterpretation: 'area',
    coordinateReferenceSystem: {authority: 'EPSG:4326'}
  };
  const spatialContext = {
    coordinateSpace: settings.coordinateSpace,
    metric: settings.metric,
    units: settings.units
  };
  const parameters = new GPUParameterBuffer(graph.device, {
    id: `${id}-parameters`,
    format: 'float32',
    length: 9,
    values: getGPUInverseDistanceWeightingParameterValues({...settings, spatialContext}, grid)
  });
  try {
    const contributor = new GPUInverseDistanceWeighting({
      id,
      spatialContext,
      positions,
      values: props.values,
      mask: effectiveMask,
      parameters: parameters.importToGraph(graph),
      grid,
      indexGridSize: getKeplerIndexGridSize(rowCount),
      invocationTileSize: KEPLER_ANALYSIS_ROW_TILE_SIZE,
      indexBounds: [-180, -90, 180, 90],
      maximumNeighborCount: settings.neighborCount,
      output: {values, counts}
    });
    addKernel(
      graph,
      `${id}-prepare`,
      rowCount,
      [
        {name: 'longitude', view: props.longitude},
        {name: 'latitude', view: props.latitude},
        {name: 'sourceValues', view: props.values},
        ...(props.mask ? [{name: 'populationMask', view: props.mask}] : []),
        ...(props.coordinateValidityMask
          ? [{name: 'coordinateValidity', view: props.coordinateValidityMask}]
          : []),
        {name: 'positions', view: positions, write: true},
        {name: 'effectiveMask', view: effectiveMask, write: true}
      ],
      `let longitudeValue = longitude[longitudeOffset + index];
let latitudeValue = latitude[latitudeOffset + index];
let included = isFiniteValue(longitudeValue) && isFiniteValue(latitudeValue) && abs(longitudeValue) <= 180.0 && abs(latitudeValue) <= 90.0 && isFiniteValue(sourceValues[sourceValuesOffset + index])
${props.mask ? '&& populationMask[populationMaskOffset + index] != 0u' : ''}
${props.coordinateValidityMask ? '&& coordinateValidity[coordinateValidityOffset + index] != 0u' : ''};
positions[positionsOffset + index * 2u] = select(181.0, longitudeValue, included);
positions[positionsOffset + index * 2u + 1u] = select(181.0, latitudeValue, included);
effectiveMask[effectiveMaskOffset + index] = select(0u, 1u, included);`
    );
    graph.add(contributor);
    // One binding avoids overlapping aligned ranges when caller status scalars share a buffer.
    const compactStatus = createTransientView(graph, `${id}-compact-status`, 'uint32', 3);
    // Exact order-independent integer sums over all rows and cells (atomic adds).
    addKernel(
      graph,
      `${id}-diagnostics-initialize`,
      1,
      [{name: 'compactStatus', view: compactStatus, write: true, atomic: true}],
      `atomicStore(&compactStatus[compactStatusOffset], 0u);
atomicStore(&compactStatus[compactStatusOffset + 1u], 0u);
atomicStore(&compactStatus[compactStatusOffset + 2u], 0u);`
    );
    addKernel(
      graph,
      `${id}-diagnostics-sources`,
      rowCount,
      [
        {name: 'effectiveMask', view: effectiveMask},
        {name: 'compactStatus', view: compactStatus, write: true, atomic: true}
      ],
      `atomicAdd(&compactStatus[compactStatusOffset], effectiveMask[effectiveMaskOffset + index]);`
    );
    addKernel(
      graph,
      `${id}-diagnostics-cells`,
      cellCount,
      [
        {name: 'values', view: values},
        {name: 'validity', view: validity, write: true},
        {name: 'compactStatus', view: compactStatus, write: true, atomic: true}
      ],
      `let valid = select(0u, 1u, isFiniteValue(values[valuesOffset + index]));
validity[validityOffset + index] = valid;
atomicAdd(&compactStatus[compactStatusOffset + 1u], valid);
atomicAdd(&compactStatus[compactStatusOffset + 2u], 1u - valid);`
    );
    for (const [index, output] of Object.values(status).entries()) {
      const input = graph.createDataView(compactStatus.buffer, {
        format: 'uint32',
        byteOffset: index * 4,
        length: 1
      });
      copyPointStatus(graph, `${id}-status-copy-${index}`, input, output);
    }
    return {
      operation: 'idw-points' as const,
      source: Object.freeze({...props.source}),
      population: props.population,
      selectionRevision: props.selectionRevision,
      settings,
      grid,
      spatialContext,
      numericalPolicy: 'float32' as const,
      outputs: {values, counts, validity, status},
      status,
      effectiveMask,
      parameterByteLength: parameters.buffer.byteLength,
      destroyParameters() {
        parameters.destroy();
      }
    };
  } catch (error) {
    parameters.destroy();
    throw error;
  }
}
