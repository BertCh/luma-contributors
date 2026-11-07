// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '../runtime/dataset-registry';
import {addKeplerPointKriging, type KeplerPointKrigingOutputs} from './point-kriging';
import {addKeplerDorlingCartogram, type KeplerDorlingOutputs} from './dorling-cartogram';
import {
  areKeplerPointWeightCoordinatesEqual,
  getKeplerPointWeightCoordinateColumns,
  matchesKeplerPointWeightSpatialContext
} from './point-weight-coordinates';
import {
  createTransientView,
  GraphVectorView,
  GPUMask,
  validatePackedView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {
  getGPUClassBreaksParameterValues,
  GPUParameterBuffer
} from '@luma-contributors/graph/gpu-dataframe';
import {
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  type GPUSpatialWeights,
  type GPUPackedGeometryPort,
  type GPUPolygonGeometryPort,
  type GPUHotSpotAnalysisRecipeProps,
  isGPUChunkedSpatialWeights
} from '@luma-contributors/graph/gpu-spatial-analysis';
import {
  parseKeplerAnalysisRecipe,
  parseKeplerPointWeightsSettings,
  DEFAULT_KEPLER_PERMUTATION_DEGREE,
  MAXIMUM_KEPLER_LOCAL_MORAN_PERMUTATION_ROWS,
  type KeplerAnalysisRecipe,
  type KeplerDatasetReference,
  type KeplerPolygonWeightsSettings,
  type KeplerPointWeightsSettings
} from '../recipes/analysis-recipe';
import {
  addKeplerClassification,
  addKeplerExcessRisk,
  addKeplerHotSpots,
  addKeplerSpatialRate,
  addKeplerStandardization
} from './analysis-bindings';

import {
  addKeplerPointWeights,
  getKeplerPointWeightsCapacity,
  type KeplerPointWeightsBinding,
  type KeplerPointWeightsStatus
} from './point-weights';
import {getKeplerPointWeightsRowCount, getKeplerPointWeightsViews} from './point-weights-chunked';
import {addKeplerPointHotSpots, copyPointStatus} from './point-hotspots';
import {
  addKeplerPointLocalMoran,
  addKeplerPolygonLocalMoran,
  type KeplerPointLocalMoranOutputs
} from './point-local-moran';
import {addKeplerPointBivariateLocalMoran} from './point-bivariate-local-moran';
import {addKeplerPointMultivariateLocalGeary} from './point-multivariate-local-geary';
import {
  addKeplerPointQuantileLISA,
  type KeplerPointQuantileLISAOutputs
} from './point-quantile-lisa';

import {addKeplerPolygonQuantileLISA} from './polygon-quantile-lisa';

import {addKeplerPointIDW, type KeplerPointIDWOutputs} from './point-idw';

import {addKeplerPointDBSCAN, type KeplerPointDBSCANOutputs} from './point-dbscan';

import {addKeplerOLS, type KeplerOLSOutputs} from './ordinary-least-squares';

import {
  addKeplerPolygonWeights,
  validateKeplerPolygonWeightsBinding,
  type KeplerPolygonWeightsBinding,
  type KeplerPolygonWeightsStatus
} from './polygon-weights';

type FloatColumn = GraphDataView<'float32'> | GraphVectorView<'float32'>;
export type KeplerAnalysisBindings = {
  source: KeplerDatasetReference;
  columns: Readonly<Record<string, FloatColumn>>;
  /** Source validity only; applies even to full-population analysis. */
  validityMask?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Selected-population mask, already intersected with input validity. */
  mask?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  selectionRevision?: number;
  /** Coordinate null validity only; point weights never depend on rate or selection masks. */
  coordinateValidityMask?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Source-wide point topology imported into this graph; retained by its owner. */
  pointWeights?: KeplerPointWeightsBinding;
  polygonWeights?: KeplerPolygonWeightsBinding;
  /** Explicit packed source-XY equality tokens for uncached polygon builds. */
  polygons?: GPUPackedGeometryPort<GPUPolygonGeometryPort>;
  geometryMask?: GraphDataView<'uint32'>;
  weights?: {
    id: string;
    revision: number;
    source: KeplerDatasetReference;
    value: GPUSpatialWeights;
  };
  /** Imported caller-owned outputs are required when used outside this graph's lifetime. */
  outputs?: {
    values?: FloatColumn;
    breaks?: GraphDataView<'float32'>;
    classCount?: GraphDataView<'uint32'>;
    classCounts?: GraphDataView<'uint32'>;
    hotSpots?: GPUHotSpotAnalysisRecipeProps['outputs'];
    /** Primary localI falls back to values when omitted here. */
    localMoran?: KeplerPointLocalMoranOutputs;
    /** Primary pseudoPValues falls back to values when omitted here. */
    quantileLisa?: KeplerPointQuantileLISAOutputs;
    pointWeights?: KeplerPointWeightsStatus;
    polygonWeights?: KeplerPolygonWeightsStatus;
    /** Raster cell outputs, distinct from source row columns. */
    idw?: KeplerPointIDWOutputs;
    /** Prediction/variance raster cells, never source-row columns. */
    kriging?: KeplerPointKrigingOutputs;
    dbscan?: KeplerPointDBSCANOutputs;
    /** Primary residuals fall back to values; model vectors are not source-row columns. */
    ols?: KeplerOLSOutputs;
    /** Source-row circle geometry; primary radius falls back to values. */
    dorling?: KeplerDorlingOutputs & {effectiveMask?: GraphDataView<'uint32'>};
  };
};

/** Resolves a saved recipe to actual contributor nodes, checking source/selection/weights revisions. */
export function addKeplerAnalysisRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  recipeValue: KeplerAnalysisRecipe,
  bindings: KeplerAnalysisBindings
) {
  const recipe = parseKeplerAnalysisRecipe(recipeValue);
  if (
    recipe.source.datasetId !== bindings.source.datasetId ||
    recipe.source.revision !== bindings.source.revision
  )
    throw new Error('Analysis binding dataset revision is stale');
  if (
    recipe.population === 'selected' &&
    (!bindings.mask || recipe.selectionRevision !== bindings.selectionRevision)
  )
    throw new Error('Analysis binding selection revision is stale');
  const provenance = {
    source: recipe.source,
    population: recipe.population,
    selectionRevision: recipe.selectionRevision
  };
  // Full-population recipes never consume a viewport/selection mask accidentally.
  const mask = recipe.population === 'selected' ? bindings.mask : bindings.validityMask;
  const operation = recipe.operation;
  const id = recipe.id;
  function column(name: string): FloatColumn {
    const value = bindings.columns[name];
    if (!value || value.format !== 'float32')
      throw new Error(`Analysis requires a float32 column: ${name}`);
    return value;
  }
  function resolvePointWeights(settings: KeplerPointWeightsSettings, rowCount: number) {
    getKeplerPointWeightsCapacity(settings, rowCount);
    const supplied = bindings.pointWeights;
    if (supplied) {
      const current = parseKeplerPointWeightsSettings(supplied.settings);
      if (
        supplied.source.datasetId !== recipe.source.datasetId ||
        supplied.source.revision !== recipe.source.revision ||
        current.kind !== settings.kind ||
        !areKeplerPointWeightCoordinatesEqual(current, settings) ||
        (current.kind === 'knn' && settings.kind === 'knn' && current.k !== settings.k) ||
        (current.kind === 'distance-band' &&
          settings.kind === 'distance-band' &&
          (current.radiusMeters !== settings.radiusMeters ||
            current.neighborCapacity !== settings.neighborCapacity ||
            (current.permutationDegree ?? DEFAULT_KEPLER_PERMUTATION_DEGREE) !==
              (settings.permutationDegree ?? DEFAULT_KEPLER_PERMUTATION_DEGREE)))
      )
        throw new Error('Point weights source or settings are stale');
      const isChunked = isGPUChunkedSpatialWeights(supplied.weights);
      if (
        supplied.coordinateMask.length !== rowCount ||
        getKeplerPointWeightsRowCount(supplied.weights) !== rowCount
      )
        throw new Error('Point weights rows do not match analysis rows');
      const information = supplied.weights.information;
      const context = supplied.weights.spatialContext;
      if (
        information?.rowCount !== rowCount ||
        information.transform !== 'binary' ||
        information.hasSelfWeights !== isChunked ||
        information.provenance?.source !== 'GPUNeighborSearch' ||
        information.provenance.criterion !== settings.kind ||
        (settings.kind === 'knn' && information.provenance.k !== settings.k) ||
        (settings.kind === 'distance-band' &&
          !isGPUChunkedSpatialWeights(supplied.weights) &&
          supplied.weights.neighbors.length !== settings.neighborCapacity) ||
        !matchesKeplerPointWeightSpatialContext(settings, context)
      )
        throw new Error('Point weights metadata does not match binary point settings');
      validatePackedView(supplied.coordinateMask, ['uint32'], 'Point coordinate mask');
      for (const status of Object.values(supplied.status)) {
        validatePackedView(status, ['uint32'], 'Point weights status');
        if (status.length !== 1) throw new Error('Point weights status must be scalar');
      }
      for (const view of [
        supplied.coordinateMask,
        ...getKeplerPointWeightsViews(supplied.weights),
        ...Object.values(supplied.status)
      ])
        if (view && view.buffer.graph !== graph)
          throw new Error('Point weights must be imported into the consumer graph');
      const result = {
        ...supplied,
        settings: current,
        weights: {...supplied.weights, status: supplied.status},
        parameterByteLength: 0,
        destroyParameters() {}
      };
      const status = bindings.outputs?.pointWeights;
      if (status) {
        copyPointStatus(
          graph,
          `${id}-point-overflow-copy`,
          supplied.status.overflow,
          status.overflow
        );
        copyPointStatus(
          graph,
          `${id}-point-required-copy`,
          supplied.status.requiredCount,
          status.requiredCount
        );
        copyPointStatus(
          graph,
          `${id}-point-invalid-copy`,
          supplied.status.invalidCount,
          status.invalidCount
        );
      }
      return result;
    }
    const [firstColumn, secondColumn] = getKeplerPointWeightCoordinateColumns(settings);
    const firstCoordinate = packed(column(firstColumn));
    const secondCoordinate = packed(column(secondColumn));
    if (firstCoordinate.length !== rowCount || secondCoordinate.length !== rowCount)
      throw new Error('Point coordinates do not match analysis rows');
    return addKeplerPointWeights(graph, {
      id: `${id}-point-weights`,
      source: recipe.source,
      ...(settings.coordinateSpace === 'planar'
        ? {settings, x: firstCoordinate, y: secondCoordinate}
        : {settings, longitude: firstCoordinate, latitude: secondCoordinate}),
      coordinateValidityMask:
        bindings.coordinateValidityMask && packed(bindings.coordinateValidityMask),
      status: bindings.outputs?.pointWeights
    });
  }
  function resolvePolygonWeights(settings: KeplerPolygonWeightsSettings, rowCount: number) {
    if (bindings.polygonWeights) {
      const supplied = bindings.polygonWeights;
      validateKeplerPolygonWeightsBinding(graph, supplied, recipe.source, rowCount, settings);
      if (bindings.outputs?.polygonWeights)
        for (const name of [
          'overflow',
          'requiredCount',
          'candidateOverflow',
          'invalidCount'
        ] as const)
          copyPointStatus(
            graph,
            `${id}-polygon-status-${name}`,
            supplied.status[name],
            bindings.outputs.polygonWeights[name]
          );
      return {
        ...supplied,
        settings,
        weights: {...supplied.weights, status: supplied.status},
        parameterByteLength: 0,
        destroyParameters() {}
      };
    }
    if (
      !bindings.polygons ||
      !bindings.geometryMask ||
      bindings.geometryMask.length !== rowCount ||
      bindings.polygons.featureOffsets.length !== rowCount + 1
    )
      throw new Error(
        'Polygon analysis requires source-aligned equality topology and geometry validity'
      );
    return addKeplerPolygonWeights(graph, {
      id: `${id}-polygon-weights`,
      source: recipe.source,
      settings,
      polygons: bindings.polygons,
      geometryMask: bindings.geometryMask,
      status: bindings.outputs?.polygonWeights
    });
  }
  function makePointMask(coordinateMask: GraphDataView<'uint32'>) {
    if (!mask) return coordinateMask;
    const effectiveMask = createTransientView(
      graph,
      `${id}-point-analysis-mask`,
      'uint32',
      coordinateMask.length
    );
    graph.add(
      new GPUMask({
        id: `${id}-point-analysis-validity`,
        inputs: [coordinateMask, packed(mask)],
        output: effectiveMask
      })
    );
    return effectiveMask;
  }
  switch (operation.kind) {
    case 'dorling-cartogram': {
      const {kind, ...settings} = operation;
      const x = packed(column(settings.xColumn));
      const y = packed(column(settings.yColumn));
      const values = packed(column(settings.valueColumn));
      const coordinateValidity =
        bindings.coordinateValidityMask && packed(bindings.coordinateValidityMask);
      const numericValidity = bindings.validityMask && packed(bindings.validityMask);
      let validityMask = numericValidity ?? coordinateValidity;
      if (numericValidity && coordinateValidity) {
        validityMask = createTransientView(graph, `${id}-input-validity`, 'uint32', values.length);
        graph.add(
          new GPUMask({
            id: `${id}-input-validity`,
            inputs: [numericValidity, coordinateValidity],
            output: validityMask
          })
        );
      }
      const result = addKeplerDorlingCartogram(graph, {
        ...provenance,
        id,
        settings,
        x,
        y,
        values,
        validityMask,
        mask: recipe.population === 'selected' && bindings.mask ? packed(bindings.mask) : undefined,
        effectiveMask: bindings.outputs?.dorling?.effectiveMask,
        outputs: {
          ...bindings.outputs?.dorling,
          radii:
            bindings.outputs?.dorling?.radii ??
            (bindings.outputs?.values && packed(bindings.outputs.values))
        }
      });
      return {...result, recipe};
    }
    case 'ols': {
      const {kind, ...settings} = operation;
      const result = addKeplerOLS(graph, {
        ...provenance,
        id,
        settings,
        response: packed(column(operation.responseColumn)),
        predictors: operation.predictorColumns.map(name => packed(column(name))),
        mask: mask && packed(mask),
        outputs: {
          ...bindings.outputs?.ols,
          residuals:
            bindings.outputs?.ols?.residuals ??
            (bindings.outputs?.values && packed(bindings.outputs.values))
        }
      });
      return {...result, recipe};
    }
    case 'dbscan-points': {
      const {kind, ...settings} = operation;
      const result = addKeplerPointDBSCAN(graph, {
        ...provenance,
        id,
        settings,
        longitude: packed(column(operation.longitudeColumn)),
        latitude: packed(column(operation.latitudeColumn)),
        mask: mask && packed(mask),
        coordinateValidityMask:
          bindings.coordinateValidityMask && packed(bindings.coordinateValidityMask),
        outputs: {
          ...bindings.outputs?.dbscan,
          values:
            bindings.outputs?.dbscan?.values ??
            (bindings.outputs?.values && packed(bindings.outputs.values))
        }
      });
      return {...result, recipe};
    }
    case 'kriging-points': {
      const {kind, column: valueColumn, ...settings} = operation;
      const result = addKeplerPointKriging(graph, {
        ...provenance,
        id,
        settings,
        longitude: packed(column(operation.longitudeColumn)),
        latitude: packed(column(operation.latitudeColumn)),
        values: packed(column(valueColumn)),
        mask: mask && packed(mask),
        coordinateValidityMask:
          bindings.coordinateValidityMask && packed(bindings.coordinateValidityMask),
        outputs: {
          ...bindings.outputs?.kriging,
          values:
            bindings.outputs?.kriging?.values ??
            (bindings.outputs?.values && packed(bindings.outputs.values))
        }
      });
      return {...result, recipe};
    }
    case 'idw-points': {
      const {kind, column: valueColumn, ...settings} = operation;
      const result = addKeplerPointIDW(graph, {
        ...provenance,
        id,
        settings,
        longitude: packed(column(operation.longitudeColumn)),
        latitude: packed(column(operation.latitudeColumn)),
        values: packed(column(valueColumn)),
        mask: mask && packed(mask),
        coordinateValidityMask:
          bindings.coordinateValidityMask && packed(bindings.coordinateValidityMask),
        outputs: {
          ...bindings.outputs?.idw,
          values:
            bindings.outputs?.idw?.values ??
            (bindings.outputs?.values && packed(bindings.outputs.values))
        }
      });
      return {...result, recipe};
    }
    case 'quantile-lisa-points': {
      const values = packed(column(operation.column));
      if (values.length < 3 || values.length >= 2 ** 24 || operation.quantileCount >= values.length)
        throw new KeplerAllocationError('capacity', {
          message: 'Point Quantile LISA exceeds row or quantile capacity'
        });
      const pointWeights = resolvePointWeights(operation.weights, values.length);
      try {
        const {overflow, requiredCount, invalidCount} = pointWeights.status;
        if (!requiredCount || !invalidCount)
          throw new Error('Point Quantile LISA requires complete weights diagnostics');
        const result = addKeplerPointQuantileLISA(graph, {
          ...provenance,
          id,
          values,
          mask: mask ? packed(mask) : pointWeights.coordinateMask,
          pointWeights: {...pointWeights, status: {overflow, requiredCount, invalidCount}},
          settings: {
            quantileCount: operation.quantileCount,
            selectedQuantile: operation.selectedQuantile,
            seed: operation.seed,
            permutationCount: operation.permutationCount,
            significanceLevel: operation.significanceLevel,
            inference: operation.inference,
            multipleTesting: operation.multipleTesting,
            islands: operation.islands,
            quantileMethod: operation.quantileMethod
          },
          outputs: {
            ...bindings.outputs?.quantileLisa,
            pseudoPValues:
              bindings.outputs?.quantileLisa?.pseudoPValues ??
              (bindings.outputs?.values && packed(bindings.outputs.values))
          }
        });
        return {
          ...result,
          recipe,
          pointWeights,
          destroyParameters() {
            result.destroyParameters();
            pointWeights.destroyParameters();
          }
        };
      } catch (error) {
        pointWeights.destroyParameters();
        throw error;
      }
    }
    case 'quantile-lisa-polygons': {
      const values = packed(column(operation.column));
      if (values.length < 3 || values.length >= 2 ** 24 || operation.quantileCount >= values.length)
        throw new KeplerAllocationError('capacity', {
          message: 'Polygon Quantile LISA exceeds row or quantile capacity'
        });
      const polygonWeights = resolvePolygonWeights(operation.weights, values.length);
      try {
        const {requiredCount, invalidCount} = polygonWeights.status;
        if (!requiredCount || !invalidCount)
          throw new Error('Polygon Quantile LISA requires complete weights diagnostics');
        const result = addKeplerPolygonQuantileLISA(graph, {
          ...provenance,
          id,
          values,
          mask: mask ? packed(mask) : polygonWeights.geometryMask,
          polygonWeights,
          settings: {
            quantileCount: operation.quantileCount,
            selectedQuantile: operation.selectedQuantile,
            seed: operation.seed,
            permutationCount: operation.permutationCount,
            significanceLevel: operation.significanceLevel,
            inference: operation.inference,
            multipleTesting: operation.multipleTesting,
            islands: operation.islands,
            quantileMethod: operation.quantileMethod
          },
          outputs: {
            ...bindings.outputs?.quantileLisa,
            pseudoPValues:
              bindings.outputs?.quantileLisa?.pseudoPValues ??
              (bindings.outputs?.values && packed(bindings.outputs.values))
          }
        });
        return {
          ...result,
          recipe,
          polygonWeights,
          destroyParameters() {
            result.destroyParameters();
            polygonWeights.destroyParameters();
          }
        };
      } catch (error) {
        polygonWeights.destroyParameters();
        throw error;
      }
    }
    case 'local-moran-points': {
      const values = packed(column(operation.column));
      if (
        values.length < 1 ||
        values.length >= 2 ** 24 ||
        (operation.inference === 'permutation-conditional-two-sided' &&
          values.length > MAXIMUM_KEPLER_LOCAL_MORAN_PERMUTATION_ROWS)
      )
        throw new KeplerAllocationError('capacity', {
          message: 'Point local Moran exceeds row capacity'
        });
      const pointWeights = resolvePointWeights(operation.weights, values.length);
      try {
        const {overflow, requiredCount, invalidCount} = pointWeights.status;
        if (!requiredCount || !invalidCount)
          throw new Error('Point local Moran requires complete weights diagnostics');
        if (operation.secondColumn !== undefined) {
          const secondValues = packed(column(operation.secondColumn));
          const localMoranOutputs = bindings.outputs?.localMoran;
          const bivariate = addKeplerPointBivariateLocalMoran(graph, {
            ...provenance,
            id,
            values,
            secondValues,
            mask: mask ? packed(mask) : pointWeights.coordinateMask,
            pointWeights: {...pointWeights, status: {overflow, requiredCount, invalidCount}},
            settings: {
              islands: operation.islands,
              multipleTesting: operation.multipleTesting,
              significanceLevel: operation.significanceLevel,
              permutationCount: operation.permutationCount!,
              seed: operation.seed!
            },
            outputs: {
              localI:
                localMoranOutputs?.localI ??
                (bindings.outputs?.values && packed(bindings.outputs.values)),
              pValues: localMoranOutputs?.pValues,
              adjustedPValues: localMoranOutputs?.adjustedPValues,
              quadrants: localMoranOutputs?.quadrants,
              status: {inferenceOverflow: localMoranOutputs?.status?.inferenceOverflow}
            }
          });
          return {
            ...bivariate,
            recipe,
            pointWeights,
            destroyParameters() {
              bivariate.destroyParameters();
              pointWeights.destroyParameters();
            }
          };
        }
        const result = addKeplerPointLocalMoran(graph, {
          ...provenance,
          id,
          values,
          mask: mask ? packed(mask) : pointWeights.coordinateMask,
          pointWeights: {...pointWeights, status: {overflow, requiredCount, invalidCount}},
          settings: {
            minimumNeighbors: operation.minimumNeighbors,
            islands: operation.islands,
            multipleTesting: operation.multipleTesting,
            significanceLevel: operation.significanceLevel,
            quadrantGating:
              operation.inference === 'permutation-conditional-two-sided'
                ? 'permutation'
                : 'analytic',
            ...(operation.inference === 'permutation-conditional-two-sided'
              ? {
                  permutationCount: operation.permutationCount,
                  seed: operation.seed
                }
              : {})
          },
          outputs: {
            ...bindings.outputs?.localMoran,
            localI:
              bindings.outputs?.localMoran?.localI ??
              (bindings.outputs?.values && packed(bindings.outputs.values))
          }
        });
        return {
          ...result,
          recipe,
          pointWeights,
          destroyParameters() {
            result.destroyParameters();
            pointWeights.destroyParameters();
          }
        };
      } catch (error) {
        pointWeights.destroyParameters();
        throw error;
      }
    }
    case 'multivariate-local-geary-points': {
      const variables = operation.columns.map(name => packed(column(name)));
      const rowCount = variables[0].length;
      const pointWeights = resolvePointWeights(operation.weights, rowCount);
      try {
        const {overflow, requiredCount, invalidCount} = pointWeights.status;
        if (!requiredCount || !invalidCount)
          throw new Error('Multivariate Local Geary requires complete weights diagnostics');
        const geary = addKeplerPointMultivariateLocalGeary(graph, {
          ...provenance,
          id,
          variables,
          mask: mask ? packed(mask) : pointWeights.coordinateMask,
          pointWeights: {...pointWeights, status: {overflow, requiredCount, invalidCount}},
          settings: {
            permutationCount: operation.permutationCount,
            seed: operation.seed,
            significanceLevel: operation.significanceLevel,
            correction: operation.correction,
            tail: operation.tail
          },
          outputs: {
            statistic: bindings.outputs?.values && packed(bindings.outputs.values),
            pValues: bindings.outputs?.localMoran?.pValues,
            classes: bindings.outputs?.localMoran?.quadrants
          }
        });
        return {
          ...geary,
          recipe,
          pointWeights,
          destroyParameters() {
            pointWeights.destroyParameters();
          }
        };
      } catch (error) {
        pointWeights.destroyParameters();
        throw error;
      }
    }
    case 'local-moran-polygons': {
      const values = packed(column(operation.column));
      if (
        values.length < 1 ||
        values.length >= 2 ** 24 ||
        (operation.inference === 'permutation-conditional-two-sided' &&
          values.length > MAXIMUM_KEPLER_LOCAL_MORAN_PERMUTATION_ROWS)
      )
        throw new KeplerAllocationError('capacity', {
          message: 'Point local Moran exceeds row capacity'
        });
      const polygonWeights = resolvePolygonWeights(operation.weights, values.length);
      try {
        const {requiredCount, invalidCount} = polygonWeights.status;
        if (!requiredCount || !invalidCount)
          throw new Error('Point local Moran requires complete weights diagnostics');
        const result = addKeplerPolygonLocalMoran(graph, {
          ...provenance,
          id,
          values,
          mask: mask ? packed(mask) : polygonWeights.geometryMask,
          polygonWeights,
          settings: {
            minimumNeighbors: operation.minimumNeighbors,
            islands: operation.islands,
            multipleTesting: operation.multipleTesting,
            significanceLevel: operation.significanceLevel,
            quadrantGating:
              operation.inference === 'permutation-conditional-two-sided'
                ? 'permutation'
                : 'analytic',
            ...(operation.inference === 'permutation-conditional-two-sided'
              ? {
                  permutationCount: operation.permutationCount,
                  seed: operation.seed
                }
              : {})
          },
          outputs: {
            ...bindings.outputs?.localMoran,
            localI:
              bindings.outputs?.localMoran?.localI ??
              (bindings.outputs?.values && packed(bindings.outputs.values))
          }
        });
        return {
          ...result,
          recipe,
          polygonWeights,
          destroyParameters() {
            result.destroyParameters();
            polygonWeights.destroyParameters();
          }
        };
      } catch (error) {
        polygonWeights.destroyParameters();
        throw error;
      }
    }
    case 'hot-spots-points': {
      const values = packed(column(operation.column));
      if (values.length < 1 || values.length >= 2 ** 24)
        throw new KeplerAllocationError('capacity', {
          message: 'Point hotspots exceed row capacity'
        });
      const pointWeights = resolvePointWeights(operation.weights, values.length);
      try {
        const effectiveMask = makePointMask(pointWeights.coordinateMask);
        const result = addKeplerPointHotSpots(graph, {
          ...provenance,
          id,
          operation,
          values,
          mask: effectiveMask,
          weights: pointWeights.weights,
          outputs: bindings.outputs?.hotSpots
        });
        return {
          ...result,
          recipe,
          pointWeights,
          effectiveMask,
          destroyParameters() {
            result.destroyParameters();
            pointWeights.destroyParameters();
          }
        };
      } catch (error) {
        pointWeights.destroyParameters();
        throw error;
      }
    }
    case 'hot-spots-polygons': {
      const values = packed(column(operation.column));
      if (values.length < 1 || values.length >= 2 ** 24)
        throw new KeplerAllocationError('capacity', {
          message: 'Point hotspots exceed row capacity'
        });
      const polygonWeights = resolvePolygonWeights(operation.weights, values.length);
      try {
        const effectiveMask = makePointMask(polygonWeights.geometryMask);
        const result = addKeplerPointHotSpots(graph, {
          ...provenance,
          id,
          operation,
          values,
          mask: effectiveMask,
          weights: polygonWeights.weights,
          outputs: bindings.outputs?.hotSpots
        });
        return {
          ...result,
          recipe,
          polygonWeights,
          effectiveMask,
          destroyParameters() {
            result.destroyParameters();
            polygonWeights.destroyParameters();
          }
        };
      } catch (error) {
        polygonWeights.destroyParameters();
        throw error;
      }
    }
    case 'hot-spots-lattice': {
      const values = packed(column(operation.column));
      if (values.length !== operation.width * operation.height)
        throw new Error('Hotspot lattice shape does not match source rows');
      const packedMask = mask && packed(mask);
      const parameters = new GPUParameterBuffer(graph.device, {
        id: `${id}-parameters`,
        format: 'float32',
        length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
      });
      try {
        const result = addKeplerHotSpots(graph, {
          ...provenance,
          analysis: {
            id,
            source: {
              kind: 'lattice',
              values,
              mask: packedMask,
              width: operation.width,
              height: operation.height,
              criterion: operation.criterion,
              radius: operation.radius,
              neighborCapacity: operation.neighborCapacity
            },
            parameters,
            selfWeight: operation.selfWeight,
            minimumNeighbors: operation.minimumNeighbors,
            multipleTesting: operation.multipleTesting,
            outputs: bindings.outputs?.hotSpots
          }
        });
        return {
          ...result,
          recipe,
          destroyParameters() {
            parameters.destroy();
          }
        };
      } catch (error) {
        parameters.destroy();
        throw error;
      }
    }
    case 'standardize': {
      const result = addKeplerStandardization(graph, {
        ...provenance,
        id,
        values: column(operation.column),
        mask,
        method: operation.method,
        variance: operation.variance,
        output: bindings.outputs?.values
      });
      return {...result, recipe, destroyParameters() {}};
    }
    case 'excess-risk': {
      const result = addKeplerExcessRisk(graph, {
        ...provenance,
        id,
        events: packed(column(operation.events)),
        populations: packed(column(operation.populations)),
        mask: mask && packed(mask),
        output: bindings.outputs?.values && packed(bindings.outputs.values)
      });
      return {...result, recipe, destroyParameters() {}};
    }
    case 'spatial-rate-points': {
      const events = packed(column(operation.events));
      const populations = packed(column(operation.populations));
      if (events.length !== populations.length)
        throw new Error('Rate columns must have matching rows');
      const pointWeights = resolvePointWeights(operation.weights, events.length);
      try {
        const effectiveMask = makePointMask(pointWeights.coordinateMask);
        const result = addKeplerSpatialRate(graph, {
          ...provenance,
          id,
          events,
          populations,
          mask: effectiveMask,
          weights: pointWeights.weights,
          weightsSource: pointWeights.source,
          scale: operation.scale,
          islandPolicy: operation.islandPolicy,
          output: bindings.outputs?.values && packed(bindings.outputs.values)
        });
        return {
          ...result,
          recipe,
          pointWeights,
          effectiveMask,
          destroyParameters() {
            pointWeights.destroyParameters();
          }
        };
      } catch (error) {
        pointWeights.destroyParameters();
        throw error;
      }
    }
    case 'spatial-rate-polygons': {
      const events = packed(column(operation.events));
      const populations = packed(column(operation.populations));
      if (events.length !== populations.length)
        throw new Error('Rate columns must have matching rows');
      const polygonWeights = resolvePolygonWeights(operation.weights, events.length);
      try {
        const effectiveMask = makePointMask(polygonWeights.geometryMask);
        const result = addKeplerSpatialRate(graph, {
          ...provenance,
          id,
          events,
          populations,
          mask: effectiveMask,
          weights: polygonWeights.weights,
          weightsSource: polygonWeights.source,
          scale: operation.scale,
          islandPolicy: operation.islandPolicy,
          output: bindings.outputs?.values && packed(bindings.outputs.values)
        });
        return {
          ...result,
          recipe,
          polygonWeights,
          effectiveMask,
          destroyParameters() {
            polygonWeights.destroyParameters();
          }
        };
      } catch (error) {
        polygonWeights.destroyParameters();
        throw error;
      }
    }
    case 'spatial-rate': {
      const weights = bindings.weights;
      if (
        !weights ||
        weights.id !== operation.weightsId ||
        weights.revision !== operation.weightsRevision
      )
        throw new Error('Analysis binding weights revision is stale');
      const result = addKeplerSpatialRate(graph, {
        ...provenance,
        id,
        events: packed(column(operation.events)),
        populations: packed(column(operation.populations)),
        mask: mask && packed(mask),
        weights: weights.value,
        weightsSource: weights.source,
        scale: operation.scale,
        islandPolicy: operation.islandPolicy,
        output: bindings.outputs?.values && packed(bindings.outputs.values)
      });
      return {...result, recipe, destroyParameters() {}};
    }
    case 'classify': {
      const values = packed(column(operation.column));
      const packedMask = mask && packed(mask);
      if (
        operation.method === 'natural-breaks' &&
        operation.naturalBreaksMode === 'exact' &&
        values.length > 1024
      )
        throw new Error('Exact natural breaks supports at most 1024 source rows');
      const parameters = new GPUParameterBuffer(graph.device, {
        id: `${id}-parameters`,
        format: 'float32',
        length: operation.classCount + 9,
        values: getGPUClassBreaksParameterValues(operation, operation.classCount)
      });
      try {
        const output = {
          breaks:
            bindings.outputs?.breaks ??
            createTransientView(graph, `${id}-breaks`, 'float32', operation.classCount + 1),
          classCount:
            bindings.outputs?.classCount ??
            createTransientView(graph, `${id}-class-count`, 'uint32', 1),
          classCounts:
            bindings.outputs?.classCounts ??
            createTransientView(graph, `${id}-class-counts`, 'uint32', operation.classCount)
        };
        const result = addKeplerClassification(graph, {
          ...provenance,
          id,
          values,
          mask: packedMask,
          parameters: parameters.importToGraph(graph),
          maximumClassCount: operation.classCount,
          methods: [operation.method],
          naturalBreaksMode: operation.naturalBreaksMode,
          output
        });
        // Destroy only after the compiled graph and all consumers have finished.
        return {
          ...result,
          recipe,
          destroyParameters() {
            parameters.destroy();
          }
        };
      } catch (error) {
        parameters.destroy();
        throw error;
      }
    }
  }
}

function packed<Format extends 'float32' | 'uint32'>(
  column: GraphDataView<Format> | GraphVectorView<Format>
): GraphDataView<Format> {
  if (column instanceof GraphVectorView) {
    if (column.data.length !== 1)
      throw new Error('This analysis requires explicit packing before execution');
    return column.data[0];
  }
  return column;
}
