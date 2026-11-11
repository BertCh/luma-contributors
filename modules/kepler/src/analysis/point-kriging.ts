// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  doGraphDataViewsOverlap,
  validatePackedView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  GPUKriging,
  getGPUKrigingParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  MAXIMUM_KEPLER_KRIGING_SOURCE_ROWS,
  parseKeplerKrigingSettings,
  type KeplerKrigingSettings
} from '../recipes/analysis-recipe';
import type {KeplerAnalysisProvenance} from './analysis-bindings';
import {
  addKeplerLinearKernel as addKernel,
  getKeplerIndexGridSize,
  KEPLER_ANALYSIS_ROW_TILE_SIZE
} from './analysis-scaling';
import {copyPointStatus} from './point-hotspots';

export type KeplerPointKrigingOutputs = {
  values?: GraphDataView<'float32'>;
  variance?: GraphDataView<'float32'>;
  validity?: GraphDataView<'uint32'>;
  /**
   * Leave-one-out cross-validation per SOURCE row (length = source rows): each retained sample is
   * kriged from its nearest other samples under the same variogram and neighborhood. NaN for
   * excluded rows or insufficient neighborhoods. Computed only when `prediction` is supplied.
   */
  crossValidation?: {
    prediction: GraphDataView<'float32'>;
    variance?: GraphDataView<'float32'>;
  };
  status?: {
    validSourceCount?: GraphDataView<'uint32'>;
    validCellCount?: GraphDataView<'uint32'>;
    noDataCellCount?: GraphDataView<'uint32'>;
    /** Core invalid settings or insufficient distinct-neighbor cells. */
    invalidCount?: GraphDataView<'uint32'>;
    /** Core rejected solve or unreliable variance cells. */
    uncertainCount?: GraphDataView<'uint32'>;
  };
};

export type KeplerPointKrigingProps = KeplerAnalysisProvenance & {
  id?: string;
  settings: KeplerKrigingSettings;
  longitude: GraphDataView<'float32'>;
  latitude: GraphDataView<'float32'>;
  values: GraphDataView<'float32'>;
  /** Numeric validity intersected with the declared population. */
  mask?: GraphDataView<'uint32'>;
  coordinateValidityMask?: GraphDataView<'uint32'>;
  outputs?: KeplerPointKrigingOutputs;
};

/**
 * Geographic local ordinary kriging at north-up raster cell centers, using caller-declared
 * variogram parameters. Distances/range are spherical meters and variance is in squared value
 * units. Variogram parameters are never inferred here: automatic fitting lives in
 * kriging-variogram-fit.ts and produces explicit settings. Optional leave-one-out outputs give
 * cross-validation per source row; prediction intervals derive from the variance output. No
 * measurement-error model is inferred. Selected population changes the samples; outputs belong to grid cells, not source rows.
 *
 * The core retains up to 16 nearest source rows. Duplicate positions within that neighborhood
 * are averaged but occupy slots before coalescing. Exact hits return their retained mean and zero
 * variance even below minimumNeighbors; nugget denotes microscale variation, not observation error.
 * Singular/ill-conditioned solves are rejected, with only tiny negative variance roundoff clamped.
 * Final validity additionally requires finite prediction and finite nonnegative variance; both
 * outputs are NaN otherwise. noDataCellCount includes final arithmetic overflow beyond the core's
 * separate invalidCount/uncertainCount categories. No neighborhood-count raster is fabricated.
 *
 * Borrowed inputs and supplied outputs must survive submission. Destroy the 48-byte parameter
 * buffer only after dependent work completes. Omitted outputs and scratch belong to the graph.
 */
export function addKeplerPointKriging<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: KeplerPointKrigingProps
) {
  const settings = parseKeplerKrigingSettings(props.settings);
  const id = props.id ?? 'kepler-point-kriging';
  const rowCount = props.values.length;
  const cellCount = settings.grid.width * settings.grid.height;
  if (
    !props.source.datasetId ||
    !Number.isSafeInteger(props.source.revision) ||
    props.source.revision < 1 ||
    rowCount < 1 ||
    rowCount > MAXIMUM_KEPLER_KRIGING_SOURCE_ROWS ||
    (props.population !== 'all' && props.population !== 'selected') ||
    (props.population === 'selected'
      ? !props.mask ||
        !Number.isSafeInteger(props.selectionRevision) ||
        (props.selectionRevision ?? 0) < 1
      : props.selectionRevision !== undefined)
  )
    throw new Error('Kriging requires a current source and explicit population');
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
      throw new Error('Kriging inputs must match source rows and graph');
  }
  const provided = props.outputs ?? {};
  for (const [view, format, length] of [
    [provided.values, 'float32', cellCount],
    [provided.variance, 'float32', cellCount],
    [provided.validity, 'uint32', cellCount],
    [provided.crossValidation?.prediction, 'float32', rowCount],
    [provided.crossValidation?.variance, 'float32', rowCount],
    [provided.status?.validSourceCount, 'uint32', 1],
    [provided.status?.validCellCount, 'uint32', 1],
    [provided.status?.noDataCellCount, 'uint32', 1],
    [provided.status?.invalidCount, 'uint32', 1],
    [provided.status?.uncertainCount, 'uint32', 1]
  ] as const) {
    if (!view) continue;
    validatePackedView(view, [format], id);
    if (view.length !== length || view.buffer.graph !== graph)
      throw new Error('Kriging outputs must match raster cells and graph');
  }
  const supplied = [
    provided.values,
    provided.variance,
    provided.validity,
    provided.crossValidation?.prediction,
    provided.crossValidation?.variance,
    ...Object.values(provided.status ?? {})
  ].filter(view => view !== undefined);
  for (let index = 0; index < supplied.length; index++) {
    if (
      [...inputs, ...supplied.slice(0, index)].some(view =>
        doGraphDataViewsOverlap(view, supplied[index])
      )
    )
      throw new Error('Kriging outputs must be disjoint');
  }
  const values =
    provided.values ?? createTransientView(graph, `${id}-values-output`, 'float32', cellCount);
  const variance =
    provided.variance ?? createTransientView(graph, `${id}-variance-output`, 'float32', cellCount);
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
      createTransientView(graph, `${id}-nodata-count-output`, 'uint32', 1),
    invalidCount:
      provided.status?.invalidCount ??
      createTransientView(graph, `${id}-invalid-count-output`, 'uint32', 1),
    uncertainCount:
      provided.status?.uncertainCount ??
      createTransientView(graph, `${id}-uncertain-count-output`, 'uint32', 1)
  };
  const coreValues = createTransientView(graph, `${id}-core-values`, 'float32', cellCount);
  const coreVariance = createTransientView(graph, `${id}-core-variance`, 'float32', cellCount);
  const coreStatus = {
    invalidCount: createTransientView(graph, `${id}-core-invalid-count`, 'uint32', 1),
    uncertainCount: createTransientView(graph, `${id}-core-uncertain-count`, 'uint32', 1)
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
    length: 12,
    values: getGPUKrigingParameterValues(
      {
        spatialContext,
        searchRadius: settings.searchRadiusMeters,
        neighborCount: settings.neighborCount,
        minimumNeighborCount: settings.minimumNeighbors,
        variogram: {...settings.variogram, range: settings.variogram.rangeMeters}
      },
      grid
    )
  });
  try {
    const contributor = new GPUKriging({
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
      output: {
        values: coreValues,
        variance: coreVariance,
        status: coreStatus,
        ...(provided.crossValidation
          ? {
              crossValidationPrediction: provided.crossValidation.prediction,
              crossValidationVariance: provided.crossValidation.variance
            }
          : {})
      }
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
    addKernel(
      graph,
      `${id}-finalize`,
      cellCount,
      [
        {name: 'coreValues', view: coreValues},
        {name: 'coreVariance', view: coreVariance},
        {name: 'values', view: values, write: true},
        {name: 'variance', view: variance, write: true},
        {name: 'validity', view: validity, write: true}
      ],
      `let prediction = coreValues[coreValuesOffset + index];
let predictionVariance = coreVariance[coreVarianceOffset + index];
let valid = isFiniteValue(prediction) && isFiniteValue(predictionVariance) && predictionVariance >= 0.0;
let missing = bitcast<f32>(0x7fc00000u | (index & 0u));
values[valuesOffset + index] = select(missing, prediction, valid);
variance[varianceOffset + index] = select(missing, predictionVariance, valid);
validity[validityOffset + index] = select(0u, 1u, valid);`
    );
    // One binding avoids overlapping aligned ranges when caller status scalars share a buffer.
    const compactStatus = createTransientView(graph, `${id}-compact-status`, 'uint32', 5);
    // Exact order-independent integer sums over all rows and cells (atomic adds).
    addKernel(
      graph,
      `${id}-diagnostics-initialize`,
      1,
      [
        {name: 'invalidCount', view: coreStatus.invalidCount},
        {name: 'uncertainCount', view: coreStatus.uncertainCount},
        {name: 'compactStatus', view: compactStatus, write: true, atomic: true}
      ],
      `atomicStore(&compactStatus[compactStatusOffset], 0u);
atomicStore(&compactStatus[compactStatusOffset + 1u], 0u);
atomicStore(&compactStatus[compactStatusOffset + 2u], 0u);
atomicStore(&compactStatus[compactStatusOffset + 3u], invalidCount[invalidCountOffset]);
atomicStore(&compactStatus[compactStatusOffset + 4u], uncertainCount[uncertainCountOffset]);`
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
        {name: 'validity', view: validity},
        {name: 'compactStatus', view: compactStatus, write: true, atomic: true}
      ],
      `let cellValid = validity[validityOffset + index];
atomicAdd(&compactStatus[compactStatusOffset + 1u], cellValid);
atomicAdd(&compactStatus[compactStatusOffset + 2u], 1u - cellValid);`
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
      operation: 'kriging-points' as const,
      source: Object.freeze({...props.source}),
      population: props.population,
      selectionRevision: props.selectionRevision,
      settings,
      grid,
      spatialContext,
      numericalPolicy: 'float32' as const,
      outputs: {values, variance, validity, status},
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
