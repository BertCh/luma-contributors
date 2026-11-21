import {resolveChoroplethOptions} from './choropleth-options';
// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {resolveRecipeDestinations} from './recipe-utils';

import {addPreparedRecipe} from './prepared-recipe';
import {snapshotRecipeGeometryRows, createRecipeRowAssignments} from './recipe-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateSinglePackedView,
  validateRequirement,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUGroupStatistics} from '../../dataframe-analysis/group-statistics/index';
import type {GPUGroupStatisticsColumnOutput} from '../../dataframe-analysis/group-statistics/index';
import {GPUClassBreaks, GPUColorScale} from '../../dataframe-analysis/column-classification/index';
import type {GPUClassBreaksProps} from '../../dataframe-analysis/column-classification/index';
import {
  resolveGPUSpatialContext,
  resolveGPUSecondarySpatialContext,
  GPU_LINEAR_EDGE_MODELS,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../../analysis-core/contracts/spatial-context';
import type {
  GPUPackedGeometryPort,
  GPUPointGeometryPort,
  GPUMultipartPointGeometryPort,
  GPUPolygonGeometryPort,
  GPURecipeStatusPort,
  GPUStatusScalar
} from '../../analysis-core/contracts/index';
import {GPUPointInPolygonJoin} from '../spatial-join/index';
import {
  getPointInPolygonContainmentContext,
  type GPUPointInPolygonBoundaryRule
} from '../spatial-join/gpu-point-in-polygon-join';
import {GPUZonalStatistics} from '../zonal-statistics/index';
import {GPULonLatPolygonPreparation} from '../lon-lat-polygons/index';
import {
  assertRecipe,
  getOrCreateView,
  RecipeBuilder,
  type GPURecipeOverrides,
  type GPURecipeResult
} from './recipe-utils';

const ID = 'GPUPointsInPolygonsChoroplethRecipe';
/** Planar points and polygons, or longitude/latitude degrees (topological containment only). */
const SUPPORT: GPUSpatialContextSupport = Object.freeze({
  coordinateSpaces: Object.freeze(['planar', 'longitude-latitude'] as const)
});

/** Statistic mapped to the choropleth. `'count'` needs no values. */
export type GPUChoroplethStatistic = 'count' | 'sum' | 'mean' | 'minimum' | 'maximum';

/**
 * Polygon features in the four-level `GPUPolygonGeometryPort` layout (one packed `positions`
 * view), one output row per feature, plus the join options. `sourceIds` are not read: every
 * per-feature output is indexed by feature row.
 */
export type GPUChoroplethPolygons = GPUPackedGeometryPort<GPUPolygonGeometryPort> & {
  /** Maximum bounding-box candidate pairs per encoding. */
  candidateCapacity: number;
  /** BVH leaf slots (power of two). */
  leafCapacity?: number;
  /**
   * Ownership of points exactly on a boundary, see `GPUPointInPolygonJoinProps.boundaryRule`.
   * Defaults to `'half-open'`: a point on an edge shared by two features counts once.
   */
  boundaryRule?: GPUPointInPolygonBoundaryRule;
};

/** Optional class breaks and colors of the per-feature statistic. */
export type GPUChoroplethColorOptions = {
  /** `getGPUClassBreaksParameterValues` view. */
  classBreaksParameters: GraphDataView<'float32'>;
  /** Compile-time class capacity. */
  maximumClassCount: number;
  /** Compiled class-break methods. */
  methods?: GPUClassBreaksProps['methods'];
  /** `getGPUColorScaleParameterValues` view (`quantile` or `threshold` scale for class colors). */
  colorScaleParameters: GraphDataView<'float32'>;
  /** Packed rgba8 palette. */
  palette: GraphDataView<'uint32'>;
  /** Compile-time palette length bound. */
  maximumPaletteCount: number;
};

/** Properties for {@link addPointsInPolygonsChoroplethRecipe}. */
export type GPUPointsInPolygonsChoroplethRecipeProps = GPURecipeOverrides<
  Record<never, never>,
  {
    counts?: GraphDataView<'uint32'>;
    featureValues?: GraphDataView<'float32'>;
    /** Mirrors `result.status`. */
    status?: {
      overflow?: GPUStatusScalar;
      /** One-row count of uncertain point-in-polygon pairs (stage `zonal-statistics` or `point-in-polygon`). */
      uncertainCount?: GPUStatusScalar;
    };
    color?: {
      breaks?: GraphDataView<'float32'>;
      classCount?: GraphDataView<'uint32'>;
      colors?: GraphDataView<'uint32'>;
      classIndices?: GraphDataView<'uint32'>;
    };
  }
> & {
  /** Prefix for every node and transient ID. Defaults to `'choropleth-recipe'`. */
  id?: string;
  /**
   * Aggregation back-end. `'zonal'` (default) joins and reduces inside `GPUZonalStatistics`;
   * `'group'` runs `GPUPointInPolygonJoin` then `GPUGroupStatistics` in dense mode (`keyCount` is
   * the feature count), so its tables already hold one row per feature.
   */
  backend?: 'zonal' | 'group';
  /**
   * Coordinate context of the points and polygons. Omitted means native planar. The mapped
   * statistics (count, sum, mean, minimum, maximum) are topological, so a `longitude-latitude`
   * context is accepted: containment is tested with straight polygon edges in degrees (not
   * geodesics). Set `prepare: true` or prepare with `GPULonLatPolygonPreparation` for
   * polygons crossing the antimeridian or great-circle edges. The recipe maps no area
   * statistic (no density), so no spherical area is needed. The resolved context is returned as
   * `result.spatialContext`.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Context of `polygons` declared separately; must match `spatialContext` in datum, coordinate
   * space, metric, units and planar frame. Omitted means one context describes both inputs.
   * Longitude/latitude needs linear edges unless `prepare` is true: this context feeds preparation,
   * then prepared edges are compared with the primary context resolved to linear edges.
   */
  polygonsSpatialContext?: GPUSpatialContext;
  /** Prepare longitude/latitude polygons on the GPU before containment. Defaults to false. */
  prepare?: boolean;
  /** Maximum prepared vertices; required with `prepare: true`. */
  preparedPositionCapacity?: number;
  /** Per-frame getGPULonLatPolygonPreparationParameterValues view; required with preparation. */
  preparationParameters?: GraphDataView<'float32'>;
  /** Points in the spatial context. Multipart features contribute once to each polygon matched by any part, using polygons.boundaryRule. Both backends use the same feature reduction for multipart inputs. */
  points: GPUPointGeometryPort | GPUMultipartPointGeometryPort;
  /** Per-feature values, including empty multipart features. Required unless statistic is count.
   * Multipart values must be packed float32 with one logical row per feature; offset views are allowed.
   */
  values?: GraphDataView<'float32'>;
  /** Polygon features. */
  polygons: GPUChoroplethPolygons;
  /** Statistic to map. Defaults to `'count'`. */
  statistic?: GPUChoroplethStatistic;
  /** Class breaks and colors; skipped when absent. */
  color?: GPUChoroplethColorOptions;
};

/** Named outputs of {@link addPointsInPolygonsChoroplethRecipe}. */
export type GPUPointsInPolygonsChoroplethRecipeOutputs = Pick<
  GPUPointsInPolygonsChoroplethRecipeViews,
  'counts' | 'featureValues' | 'color'
>;

/** Points-in-polygons outputs and top-level convenience fields; no intermediate ports. */
export type GPUPointsInPolygonsChoroplethRecipeResult = GPURecipeResult<
  GPUPointsInPolygonsChoroplethRecipeOutputs,
  Record<never, never>
> &
  GPUPointsInPolygonsChoroplethRecipeViews;

type GPUPointsInPolygonsChoroplethRecipeViews = {
  /** Resolved coordinate context; prepared polygon edges are linear. */
  spatialContext: GPUSpatialContext;
  /** Number of polygon features, the row count of every per-feature output. */
  featureCount: number;
  /** Points per feature. */
  counts: GraphDataView<'uint32'>;
  /** The mapped statistic per feature: the `uint32` `counts` view for `'count'`, else float32. */
  featureValues: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /** Present when `color` was requested. */
  color?: {
    breaks: GraphDataView<'float32'>;
    classCount: GraphDataView<'uint32'>;
    colors: GraphDataView<'uint32'>;
    classIndices: GraphDataView<'uint32'>;
  };
  /** Status words next to the per-stage `stages`. */
  status: {
    /** One-row overflow flag of the join. */
    overflow: GPUStatusScalar;
    /**
     * One-row count of (point, feature) candidates whose containment could not be proven; those
     * points stay unassigned, so a nonzero value means the statistics may be low.
     */
    uncertainCount: GPUStatusScalar;
  } & GPURecipeStatusPort;
};

/**
 * Points-in-polygons choropleth recipe: points to a per-polygon statistic to class breaks to
 * colors.
 *
 * With `prepare: true`, `GPULonLatPolygonPreparation` precedes containment, preserving feature
 * rows and publishing `polygon-preparation` diagnostics in `status.stages`.
 *
 * Chain: `GPUZonalStatistics` (or `GPUPointInPolygonJoin` -> dense `GPUGroupStatistics`) ->
 * `GPUClassBreaks` -> `GPUColorScale`. Features with no points keep NaN (mean, minimum, maximum)
 * or 0 (count, sum); NaN rows are no-data in the color scale and are skipped by the class breaks.
 * Counts feed the classification as the `uint32` column they are, with no cast.
 *
 * Spatial context: planar by default; `longitude-latitude` is accepted because counts and value
 * statistics are topological (see `spatialContext`). Density is not mapped, so no polygon area is
 * involved.
 *
 * Status: stage `zonal-statistics` (overflow, uncertain count), or `point-in-polygon` (overflow,
 * uncertain count) and `group-statistics` (count, overflow, required count) with `backend: 'group'`.
 */
export function addPointsInPolygonsChoroplethRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUPointsInPolygonsChoroplethRecipeProps
): GPUPointsInPolygonsChoroplethRecipeResult {
  return addPreparedRecipe(graph, props, declarePointsInPolygonsChoroplethRecipe);
}

function declarePointsInPolygonsChoroplethRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUPointsInPolygonsChoroplethRecipeProps
): GPUPointsInPolygonsChoroplethRecipeResult {
  const id = props.id ?? 'choropleth-recipe';
  const destinations = resolveRecipeDestinations(id, props);
  let spatialContext = resolveGPUSpatialContext(id, props.spatialContext, SUPPORT);
  if (!props.prepare) {
    resolveGPUSecondarySpatialContext(
      id,
      'polygons',
      props.polygonsSpatialContext,
      spatialContext,
      GPU_LINEAR_EDGE_MODELS
    );
  }
  if (props.prepare) {
    if (spatialContext.coordinateSpace !== 'longitude-latitude') {
      throw new Error(`${id} prepare needs a longitude-latitude spatialContext`);
    }
    if (props.preparedPositionCapacity === undefined) {
      throw new Error(`${id} prepare needs preparedPositionCapacity`);
    }
    if (!Number.isInteger(props.preparedPositionCapacity) || props.preparedPositionCapacity < 1) {
      throw new Error(`${id} preparedPositionCapacity must be a positive integer`);
    }
    if (!props.preparationParameters) {
      throw new Error(`${id} prepare needs preparationParameters`);
    }
  }

  const {statistic, backend} = resolveChoroplethOptions(id, {
    statistic: props.statistic,
    backend: props.backend,
    boundaryRule: props.polygons.boundaryRule
  });
  const providedOutputs = destinations ?? {};
  validateRequirement(
    id,
    !providedOutputs.color || Boolean(props.color),
    'output.color requires color settings; supply color or omit output.color'
  );
  const polygonInput = props.polygons;
  const polygons = {
    ...snapshotRecipeGeometryRows(polygonInput),
    candidateCapacity: polygonInput.candidateCapacity,
    leafCapacity: polygonInput.leafCapacity,
    boundaryRule: polygonInput.boundaryRule
  };
  const points = snapshotRecipeGeometryRows(props.points);
  const multipart = points.featureOffsets !== undefined;
  const queryCount = points.featureOffsets
    ? points.featureOffsets.length - 1
    : points.positions.length;
  const usesFeatureJoin = multipart || queryCount === 0;
  // Rows are features: feature source IDs never index the outputs.
  let polygonRows = polygons;
  const featureCount = polygons.featureOffsets.length - 1;
  if (usesFeatureJoin) {
    if (props.values) {
      validateSinglePackedView(props.values, ['float32'], `${id} values`);
      validateRequirement(
        id,
        props.values.length === queryCount,
        'values needs one packed row per multipart feature'
      );
    }
    if (providedOutputs.featureValues) {
      validateSinglePackedView(
        providedOutputs.featureValues,
        ['float32'],
        `${id} outputs.featureValues`
      );
      validateRequirement(
        id,
        providedOutputs.featureValues.length === featureCount,
        `outputs.featureValues needs one row per polygon (${featureCount})`
      );
      validateGraphOutputsDisjointFromInputs(
        id,
        [providedOutputs.featureValues],
        [
          providedOutputs.counts,
          providedOutputs.status?.overflow,
          providedOutputs.status?.uncertainCount,
          props.values,
          points.positions,
          points.featureOffsets,
          polygons.positions,
          polygons.featureOffsets,
          polygons.polygonOffsets,
          polygons.ringOffsets
        ]
      );
    }
  }
  assertRecipe(ID, featureCount >= 1, 'needs at least one feature');
  assertRecipe(ID, statistic === 'count' || props.values, `statistic ${statistic} needs values`);
  const builder = new RecipeBuilder(graph);
  const preparationStages: Array<GPURecipeStatusPort['stages'][number]> = [];
  if (props.prepare) {
    const prepared = {
      positions: getOrCreateView(
        graph,
        `${id}-prepare-positions`,
        'float32x2',
        props.preparedPositionCapacity!
      ),
      featureOffsets: getOrCreateView(
        graph,
        `${id}-prepare-features`,
        'uint32',
        polygonRows.featureOffsets.length
      ),
      polygonOffsets: getOrCreateView(
        graph,
        `${id}-prepare-polygons`,
        'uint32',
        2 * polygonRows.polygonOffsets.length - 1
      ),
      ringOffsets: getOrCreateView(
        graph,
        `${id}-prepare-rings`,
        'uint32',
        2 * polygonRows.ringOffsets.length - 1
      ),
      status: {
        overflow: getOrCreateView(graph, `${id}-prepare-overflow`, 'uint32', 1),
        requiredCount: getOrCreateView(graph, `${id}-prepare-required-count`, 'uint32', 1),
        invalidCount: getOrCreateView(graph, `${id}-prepare-invalid-count`, 'uint32', 1),
        approximation: getOrCreateView(graph, `${id}-prepare-approximation`, 'uint32', 1)
      }
    };
    const preparation = builder.add(
      new GPULonLatPolygonPreparation({
        ...polygonRows,
        id: `${id}-prepare`,
        spatialContext: props.polygonsSpatialContext ?? spatialContext,
        parameters: props.preparationParameters!,
        output: prepared
      })
    );
    if (props.polygonsSpatialContext !== undefined) {
      resolveGPUSecondarySpatialContext(
        id,
        'polygons',
        preparation.spatialContext,
        spatialContext.coordinateSpace === 'longitude-latitude'
          ? {...spatialContext, edgeModel: 'linear'}
          : spatialContext,
        GPU_LINEAR_EDGE_MODELS
      );
    }
    polygonRows = {...polygonRows, ...prepared};
    spatialContext =
      props.polygonsSpatialContext === undefined
        ? preparation.spatialContext
        : {...spatialContext, edgeModel: 'linear'};
    preparationStages.push({stage: 'polygon-preparation', status: prepared.status});
  }
  const counts = getOrCreateView(
    graph,
    `${id}-counts`,
    'uint32',
    featureCount,
    providedOutputs.counts
  );
  assertRecipe(
    ID,
    statistic !== 'count' || !providedOutputs.featureValues,
    'count has no featureValues'
  );
  const featureValues: GraphDataView<'float32'> | GraphDataView<'uint32'> =
    statistic === 'count'
      ? counts
      : getOrCreateView(
          graph,
          `${id}-feature-values`,
          'float32',
          featureCount,
          providedOutputs.featureValues
        );
  const overflow = getOrCreateView(
    graph,
    `${id}-overflow`,
    'uint32',
    1,
    providedOutputs.status?.overflow
  );
  const uncertainCount = getOrCreateView(
    graph,
    `${id}-uncertain-count`,
    'uint32',
    1,
    providedOutputs.status?.uncertainCount
  );
  let groupCount: GraphDataView<'uint32'> | undefined;
  let groupRequiredCount: GraphDataView<'uint32'> | undefined;
  let groupOverflow: GraphDataView<'uint32'> | undefined;

  if (usesFeatureJoin) {
    assertRecipe(
      ID,
      !props.values || props.values.length === queryCount,
      'values needs one row per multipart feature'
    );
    const membership =
      statistic === 'count'
        ? undefined
        : getOrCreateView(graph, `${id}-membership`, 'uint32', queryCount * featureCount);
    const assignments = createRecipeRowAssignments(
      graph,
      `${id}-assignments`,
      points.positions,
      multipart ? queryCount : undefined
    );
    builder.add(
      new GPUPointInPolygonJoin({
        id: `${id}-join`,
        points,
        polygons: polygonRows,
        spatialContext: getPointInPolygonContainmentContext(spatialContext),
        candidateCapacity: polygons.candidateCapacity,
        leafCapacity: polygons.leafCapacity,
        boundaryRule: polygons.boundaryRule,
        output: {
          pointFeatureRows: assignments.output,
          featurePolygonMembership: membership,
          ...(multipart ? {polygonMembershipCounts: counts} : {polygonAssignmentCounts: counts}),
          status: {overflow, uncertainCount}
        }
      })
    );
    if (statistic !== 'count' && queryCount === 0) {
      // No observations: initialize every polygon without constructing a nonempty-key reducer.
      builder.add({
        getCommandNodes: targetGraph => [
          createWGSLKernelNode(targetGraph, {
            id: `${id}-empty-values`,
            operation: ID,
            invocationCount: featureCount,
            bindings: [{name: 'values', view: featureValues, type: 'f32', access: 'read_write'}],
            body: `values[valuesOffset + index] = ${statistic === 'sum' ? '0.0' : 'bitcast<f32>(0x7fc00000u | (index & 0u))'};`
          })
        ]
      });
    } else if (statistic !== 'count' && membership) {
      const values = props.values!;
      const keys = getOrCreateView(graph, `${id}-membership-keys`, 'uint32', membership.length);
      const contributions = getOrCreateView(
        graph,
        `${id}-membership-values`,
        'float32',
        membership.length
      );
      builder.add({
        getCommandNodes: targetGraph => [
          createWGSLKernelNode(targetGraph, {
            id: `${id}-membership-contributions`,
            operation: ID,
            bindings: [
              {name: 'membership', view: membership, type: 'u32', access: 'read'},
              {name: 'values', view: values, type: 'f32', access: 'read'},
              {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
              {name: 'contributions', view: contributions, type: 'f32', access: 'read_write'}
            ],
            invocationCount: membership.length,
            body: `let valid = membership[membershipOffset + index] != 0u;
  keys[keysOffset + index] = select(0xffffffffu, index % ${featureCount}u, valid);
  contributions[contributionsOffset + index] = values[valuesOffset + index / ${featureCount}u];`
          })
        ]
      });
      const tableKeys = getOrCreateView(graph, `${id}-group-keys`, 'uint32', featureCount);
      const tableCounts = getOrCreateView(graph, `${id}-finite-counts`, 'uint32', featureCount);
      groupCount = getOrCreateView(graph, `${id}-group-count`, 'uint32', 1);
      groupOverflow = getOrCreateView(graph, `${id}-group-overflow`, 'uint32', 1);
      groupRequiredCount = getOrCreateView(graph, `${id}-group-required-count`, 'uint32', 1);
      const columnOutput = {
        sum: {sumValues: featureValues},
        mean: {means: featureValues},
        minimum: {minimums: featureValues},
        maximum: {maximums: featureValues}
      } as const;
      builder.add(
        new GPUGroupStatistics({
          id: `${id}-multipart-group`,
          keys,
          keyCount: featureCount,
          columns: [
            {
              values: contributions,
              statistics: [statistic],
              output: columnOutput[statistic] as GPUGroupStatisticsColumnOutput
            }
          ],
          output: {
            keys: tableKeys,
            counts: tableCounts,
            status: {count: groupCount, overflow: groupOverflow, requiredCount: groupRequiredCount}
          }
        })
      );
    }
  } else if (backend === 'zonal') {
    const statisticOutput = {
      sum: 'sums',
      mean: 'means',
      minimum: 'minima',
      maximum: 'maxima'
    } as const;
    builder.add(
      new GPUZonalStatistics({
        id: `${id}-zonal`,
        spatialContext: getPointInPolygonContainmentContext(spatialContext),
        features: polygonRows,
        positions: points.positions,
        values: statistic === 'count' ? undefined : props.values,
        output: {
          counts,
          status: {overflow, uncertainCount},
          ...(statistic === 'count' ? {} : {[statisticOutput[statistic]]: featureValues})
        }
      })
    );
  } else {
    const assignments = createRecipeRowAssignments(graph, `${id}-point-features`, points.positions);
    const pointFeatureRows = assignments.values;
    builder.add(
      new GPUPointInPolygonJoin({
        id: `${id}-join`,
        spatialContext: getPointInPolygonContainmentContext(spatialContext),
        points,
        polygons: polygonRows,
        candidateCapacity: polygons.candidateCapacity,
        leafCapacity: polygons.leafCapacity,
        boundaryRule: polygons.boundaryRule,
        output: {pointFeatureRows: assignments.output, status: {overflow, uncertainCount}}
      })
    );
    const groupKeys = getOrCreateView(graph, `${id}-group-keys`, 'uint32', featureCount);
    groupCount = getOrCreateView(graph, `${id}-group-count`, 'uint32', 1);
    groupOverflow = getOrCreateView(graph, `${id}-group-overflow`, 'uint32', 1);
    groupRequiredCount = getOrCreateView(graph, `${id}-group-required-count`, 'uint32', 1);
    const columnOutput = {
      sum: {sumValues: featureValues},
      mean: {means: featureValues},
      minimum: {minimums: featureValues},
      maximum: {maximums: featureValues}
    } as const;
    builder.add(
      new GPUGroupStatistics({
        id: `${id}-group-statistics`,
        keys: pointFeatureRows,
        keyCount: featureCount,
        columns:
          statistic === 'count'
            ? []
            : [
                {
                  values: props.values!,
                  statistics: [statistic],
                  output: columnOutput[statistic] as GPUGroupStatisticsColumnOutput
                }
              ],
        output: {
          keys: groupKeys,
          counts,
          status: {count: groupCount, overflow: groupOverflow, requiredCount: groupRequiredCount}
        }
      })
    );
  }

  const result: GPUPointsInPolygonsChoroplethRecipeResult = {
    contributors: builder.contributors,
    spatialContext,
    featureCount,
    counts,
    featureValues,
    outputs: {counts, featureValues},
    intermediates: {},
    status: {
      overflow,
      uncertainCount,
      stages: usesFeatureJoin
        ? [
            {stage: 'point-in-polygon', status: {overflow, uncertainCount}},
            ...(groupOverflow
              ? [
                  {
                    stage: 'group-statistics',
                    status: {
                      count: groupCount!,
                      overflow: groupOverflow,
                      requiredCount: groupRequiredCount!
                    }
                  }
                ]
              : [])
          ]
        : backend === 'zonal'
          ? [{stage: 'zonal-statistics', status: {overflow, uncertainCount}}]
          : [
              {stage: 'point-in-polygon', status: {overflow, uncertainCount}},
              {
                stage: 'group-statistics',
                status: {
                  count: groupCount!,
                  overflow: groupOverflow!,
                  requiredCount: groupRequiredCount!
                }
              }
            ]
    }
  };
  result.status.stages = [...preparationStages, ...result.status.stages];
  const {color} = props;
  if (color) {
    const providedColorOutputs = providedOutputs.color ?? {};
    const outputs = {
      breaks: getOrCreateView(
        graph,
        `${id}-breaks`,
        'float32',
        color.maximumClassCount + 1,
        providedColorOutputs.breaks
      ),
      classCount: getOrCreateView(
        graph,
        `${id}-class-count`,
        'uint32',
        1,
        providedColorOutputs.classCount
      ),
      colors: getOrCreateView(
        graph,
        `${id}-colors`,
        'uint32',
        featureCount,
        providedColorOutputs.colors
      ),
      classIndices: getOrCreateView(
        graph,
        `${id}-class-indices`,
        'uint32',
        featureCount,
        providedColorOutputs.classIndices
      )
    };
    builder.add(
      new GPUClassBreaks({
        id: `${id}-class-breaks`,
        values: featureValues,
        parameters: color.classBreaksParameters,
        maximumClassCount: color.maximumClassCount,
        methods: color.methods,
        output: {breaks: outputs.breaks, classCount: outputs.classCount}
      })
    );
    builder.add(
      new GPUColorScale({
        id: `${id}-color-scale`,
        values: featureValues,
        integerValues: 'numeric',
        domain: outputs.breaks,
        classCount: outputs.classCount,
        palette: color.palette,
        parameters: color.colorScaleParameters,
        maximumDomainCount: color.maximumClassCount + 1,
        maximumPaletteCount: color.maximumPaletteCount,
        output: {colors: outputs.colors, classIndices: outputs.classIndices}
      })
    );
    result.color = outputs;
  }
  if (result.color) {
    result.outputs['color'] = result.color;
  }
  return result;
}
