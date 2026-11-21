// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RequiredGeometryOutput} from '../group-geometry/required-geometry-output';
import {validateShapeDescriptorMeasurements} from './shape-descriptors-measurement-validation';

import {getEmptyGeometryNodes} from '../group-geometry/empty-geometry-nodes';

import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateFiniteNumber,
  validateGPUStorageBindingByteLength,
  validateRequirement,
  validateCapacity,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUPolygonGeometryFields} from '../../analysis-core/contracts/index';
import {GPUGeometryMeasures} from '../geometry-measures/index';
import {
  createCompactnessNode,
  createConvexityNode,
  createMomentsNode,
  createOrientationNode,
  GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH
} from './shape-descriptors-kernels';
import {
  canUseMonotoneChainConvexity,
  getMonotoneChainConvexityNodes,
  validateMonotoneChainConvexityScratch
} from './convexity-hull';
import {GEOMETRY_STATS_STRIDE} from '../geometry-measures/geometry-measures-kernels';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';
import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges
} from '../../analysis-core/kernels/graph-chunk-dispatch';

/** Logical writes in separate passes may share an aligned prefix, but may never alias. */
function validateShapeDescriptorLogicalOutputs(
  id: string,
  outputs: Readonly<Record<string, GraphDataView | undefined>>
): void {
  const fields = Object.entries(outputs).filter((entry): entry is [string, GraphDataView] =>
    Boolean(entry[1])
  );
  for (let firstIndex = 0; firstIndex < fields.length; firstIndex++) {
    const [firstName, first] = fields[firstIndex];
    if (first.length === 0) continue;
    const firstBuffer = first.buffer.defaultBuffer
      ? getCoreDefaultBuffer(first.buffer.defaultBuffer)
      : first.buffer;
    for (let secondIndex = firstIndex + 1; secondIndex < fields.length; secondIndex++) {
      const [secondName, second] = fields[secondIndex];
      if (second.length === 0) continue;
      const secondBuffer = second.buffer.defaultBuffer
        ? getCoreDefaultBuffer(second.buffer.defaultBuffer)
        : second.buffer;
      validateRequirement(
        id,
        firstBuffer !== secondBuffer ||
          first.byteOffset + first.length * first.rowByteLength <= second.byteOffset ||
          second.byteOffset + second.length * second.rowByteLength <= first.byteOffset,
        `${firstName} and ${secondName} logical byte ranges must not overlap`
      );
    }
  }
}

import {
  SHAPE_MOMENT_TILED_MINIMUM_ROW_COUNT,
  canUseShapeMomentTiles,
  getShapeMomentTileNodes,
  validateShapeMomentTileScratch
} from './shape-moment-tiles';

const OPERATION = 'GPUShapeDescriptors';

export {GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH};

/** Default sliver threshold: Polsby-Popper below this flags a sliver (about a 1:60 rectangle). */
export const GPU_SHAPE_DESCRIPTORS_DEFAULT_SLIVER_THRESHOLD = 0.05;

/** CPU description of the per-frame settings of {@link GPUShapeDescriptors}. */
export type GPUShapeDescriptorsSettings = {
  /**
   * Polsby-Popper compactness below which a polygon is flagged as a sliver. Default
   * {@link GPU_SHAPE_DESCRIPTORS_DEFAULT_SLIVER_THRESHOLD}.
   */
  sliverThreshold?: number;
};

/**
 * Packs {@link GPUShapeDescriptorsSettings} into the 4-element float32 layout
 * `[sliverThreshold, 0, 0, 0]`. Write the result into a `GPUParameterBuffer` between encodings to
 * move the threshold without recompiling.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If the threshold is not a non-negative finite number, or `target` is too short.
 */
export function getGPUShapeDescriptorsParameterValues(
  settings: GPUShapeDescriptorsSettings = {},
  target: Float32Array = new Float32Array(GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH) {
    throw new Error(
      `Shape descriptors target must hold ${GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH} elements`
    );
  }
  const sliverThreshold =
    settings.sliverThreshold ?? GPU_SHAPE_DESCRIPTORS_DEFAULT_SLIVER_THRESHOLD;
  validateFiniteNumber('shape-descriptors', 'sliverThreshold', sliverThreshold, 0);
  validateRequirement(
    'shape-descriptors',
    sliverThreshold >= 0,
    'sliverThreshold must be non-negative'
  );
  target.set([sliverThreshold, 0, 0, 0]);
  return target;
}

/**
 * How {@link GPUShapeDescriptors} computes `convexity`: `'gift-wrapping'` marches a private hull
 * per feature in one invocation (`O(vertices * hull vertices)` each, no extra passes, best for many
 * small features); `'monotone-chain'` builds every feature's hull with `GPUGroupConvexHull`
 * (`O(n log n)` total, balanced for large or skewed features, but about thirty small passes);
 * `'auto'` picks `'monotone-chain'` when features average at least
 * {@link GPU_SHAPE_DESCRIPTORS_MONOTONE_CHAIN_AVERAGE_VERTICES} vertices, or when the caller
 * declares through `maximumVerticesPerFeature` that the largest feature has at least
 * {@link GPU_SHAPE_DESCRIPTORS_MONOTONE_CHAIN_LARGEST_VERTICES} vertices.
 */
export type GPUShapeDescriptorsConvexityMethod = 'auto' | 'gift-wrapping' | 'monotone-chain';

/** Average vertices per feature from which `convexityMethod: 'auto'` uses the monotone chain. */
export const GPU_SHAPE_DESCRIPTORS_MONOTONE_CHAIN_AVERAGE_VERTICES = 256;

/**
 * Declared vertices of the largest feature from which `convexityMethod: 'auto'` uses the monotone
 * chain: gift wrapping marches `O(vertices * hull vertices)` on one invocation, so one feature
 * this size dominates the pass however small the others are.
 */
export const GPU_SHAPE_DESCRIPTORS_MONOTONE_CHAIN_LARGEST_VERTICES = 1024;

/** Per-feature output columns of {@link GPUShapeDescriptors}. Every column is optional. */
export type GPUShapeDescriptorsOutput = RequiredGeometryOutput<{
  /** Polygon area (`|signed area|`), the same column `GPUGeometryMeasures` writes. */
  areas?: GraphDataView<'float32'>;
  /** Perimeter over every ring, the same column `GPUGeometryMeasures` writes as `lengths`. */
  perimeters?: GraphDataView<'float32'>;
  /** Polsby-Popper compactness `4 pi A / P^2`: 1 for a circle, toward 0 for thin or ragged shapes. */
  polsbyPopper?: GraphDataView<'float32'>;
  /** Schwartzberg compactness `P_circle / P = 2 sqrt(pi A) / P`, equal to `sqrt(polsbyPopper)`. */
  schwartzberg?: GraphDataView<'float32'>;
  /**
   * Elongation `1 - sqrt(lambda2 / lambda1)` from the eigenvalues of the second central moments:
   * 0 for a circle or square, toward 1 for a line.
   */
  elongation?: GraphDataView<'float32'>;
  /**
   * Orientation of the major axis in radians in `(-pi / 2, pi / 2]`, counter-clockwise from +x.
   * 0 for shapes with no distinguished axis.
   */
  orientation?: GraphDataView<'float32'>;
  /**
   * Convexity `A / hull area` in `(0, 1]` over the hull of every vertex of the feature (1 for a
   * convex polygon).
   */
  convexity?: GraphDataView<'float32'>;
  /** `1` when the first (exterior) ring is clockwise in the plane (y up), else `0`. */
  clockwise?: GraphDataView<'uint32'>;
  /** `1` when the perimeter is positive and Polsby-Popper is below the threshold or area is 0. */
  sliver?: GraphDataView<'uint32'>;
}>;

/**
 * Properties for {@link GPUShapeDescriptors}.
 *
 * Per-frame (no recompile): the contents of `parameters` (sliver threshold) and every input
 * buffer. Compile-time: view lengths and which outputs are present.
 *
 * Polygons are the four-level layout of {@link GPUPolygonGeometryFields}, one row per feature. Ring
 * role is positional: the first ring of each polygon is a shell and later rings are holes,
 * whatever their winding, so multi-polygons with mixed winding and holed polygons measure
 * correctly. The `clockwise` output reports the first ring of each feature.
 */
export type GPUShapeDescriptorsProps = GPUPolygonGeometryFields & {
  /** Prefix for generated node and transient IDs. Defaults to `'shape-descriptors'`. */
  id?: string;
  /**
   * Planar only. Spatial context of `positions`; omitted means native planar coordinates.
   * Longitude/latitude (or any other non-planar) contexts throw at construction.
   * Selected area/perimeter/compactness/convexity outputs also reject known angular planar
   * frames, including with metric none. Project positions to a planar linear frame first,
   * for example with `createGPULocalFrame` and
   * `GPULocalFrameProjection`, and pass the frame's context so its `frame` identity is kept.
   * Measures both distance and area, so neither local frame is exact: use an
   * azimuthal-equidistant frame for study areas small enough that its area distortion is
   * negligible, or a projected CRS.
   */
  spatialContext?: GPUSpatialContext;
  /** Hull algorithm behind `output.convexity`. Default `'auto'`. */
  convexityMethod?: GPUShapeDescriptorsConvexityMethod;
  /** Auto uses deterministic 256-edge tiles above 4,096 rows when scratch fits the device.
   * Serial retains the compensated per-feature reference. Tiled explicitly requires its scratch.
   * Both routes match shape descriptors within the documented f32 tolerance. */
  momentMethod?: 'auto' | 'serial' | 'tiled';
  /**
   * Compile-time upper bound on the vertices of any single feature (all its rings), known to the
   * caller from its data. With `convexityMethod: 'auto'`, a bound of at least
   * {@link GPU_SHAPE_DESCRIPTORS_MONOTONE_CHAIN_LARGEST_VERTICES} selects the balanced monotone
   * chain even when the average feature is small, so one huge feature among many small ones does
   * not stay on a single-invocation gift-wrapping march. When omitted, `'auto'` decides from the
   * average feature size only, as before. Other convexity methods ignore it.
   */
  maximumVerticesPerFeature?: number;
  /** Per-frame packed float32 view of at least 4 elements, written with {@link getGPUShapeDescriptorsParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /**
   * Per-feature output columns; at least one is required. There is no `status.count`: every
   * column has one row per feature, never a variable number of rows.
   */
  output: GPUShapeDescriptorsOutput;
};

/**
 * Per-polygon shape descriptors: Polsby-Popper and Schwartzberg compactness, convexity,
 * elongation and orientation, plus clockwise and sliver flags (momepy and PySAL `esda`
 * compactness, turf `polygonSmooth` diagnostics, PostGIS `ST_IsPolygonCW`, QGIS "Polygon
 * compactness").
 *
 * Composition: an internal `GPUGeometryMeasures` supplies area, perimeter (including holes) and
 * centroid; a moments node integrates second central moments with Green's theorem about the
 * centroid in Neumaier-compensated f32 (one invocation per feature, deterministic); small nodes
 * derive the descriptors. Convexity runs a private gift-wrapping hull per feature with O(1)
 * memory, so its cost is O(vertices * hull vertices) per feature and grows with the largest
 * feature; `convexityMethod: 'monotone-chain'` (chosen by `'auto'` for large average features, or
 * when `maximumVerticesPerFeature` declares a large feature)
 * uses `GPUGroupConvexHull` instead, which is O(n log n) and balanced across skewed features.
 * Orientation and elongation weight holes negatively, so a polygon with a large hole is
 * described by its area, not its outline. Planar only: a longitude/latitude `spatialContext` throws
 * at construction, so project first (`createGPULocalFrame` with `GPULocalFrameProjection`). The
 * resolved `spatialContext`, including any `frame` identity, is passed to the internal
 * `GPUGeometryMeasures`, so it propagates to the measures instead of a fixed planar descriptor.
 *
 * Degenerate features (zero area or perimeter) report NaN for ratios, 0 for the clockwise flag,
 * and `sliver = 1` when the perimeter is positive.
 *
 * Precision: f32. Moments are relative to the centroid, so far-from-origin coordinates keep their
 * small-scale precision; the descriptors match f64 references to about 1e-4 relative.
 */
export class GPUShapeDescriptors implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUShapeDescriptorsProps;
  /** Resolved spatial context (planar). */
  readonly spatialContext: GPUSpatialContext;
  /** Number of features. */
  readonly featureCount: number;
  /** Whether convexity uses the sort-based monotone chain hull instead of gift wrapping. */
  readonly useMonotoneChainConvexity: boolean;

  constructor(props: GPUShapeDescriptorsProps) {
    this.id = props.id ?? 'shape-descriptors';
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    validateShapeDescriptorMeasurements(this.id, this.spatialContext, props.output);
    const {id} = this;
    validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);

    for (const name of ['ringOffsets', 'polygonOffsets', 'featureOffsets'] as const) {
      validateSinglePackedView(props[name], ['uint32'], `${id} ${name}`);
      if (props[name].length < 1) {
        throw new Error(`${id} ${name} must contain a terminal row`);
      }
    }
    this.featureCount = props.featureOffsets.length - 1;
    validateRequirement(
      id,
      ['auto', 'serial', 'tiled'].includes(props.momentMethod ?? 'auto'),
      "momentMethod must be 'auto', 'serial' or 'tiled'"
    );
    const convexityMethod = props.convexityMethod ?? 'auto';
    if (!['auto', 'gift-wrapping', 'monotone-chain'].includes(convexityMethod)) {
      throw new Error(`${id} convexityMethod must be 'auto', 'gift-wrapping' or 'monotone-chain'`);
    }
    const canChain = canUseMonotoneChainConvexity(props.positions.length, this.featureCount);
    if (convexityMethod === 'monotone-chain' && !canChain) {
      throw new Error(`${id} monotone-chain convexity supports fewer than 2^24 rows and features`);
    }
    if (props.maximumVerticesPerFeature !== undefined) {
      // The bound only steers the 'auto' choice; any positive integer is a valid declaration.
      validateCapacity(
        id,
        'maximumVerticesPerFeature',
        props.maximumVerticesPerFeature,
        1,
        0xffffffff
      );
    }
    const declaresLargeFeature =
      props.maximumVerticesPerFeature !== undefined &&
      props.maximumVerticesPerFeature >= GPU_SHAPE_DESCRIPTORS_MONOTONE_CHAIN_LARGEST_VERTICES;
    this.useMonotoneChainConvexity =
      canChain &&
      (convexityMethod === 'monotone-chain' ||
        (convexityMethod === 'auto' &&
          (declaresLargeFeature ||
            props.positions.length >=
              GPU_SHAPE_DESCRIPTORS_MONOTONE_CHAIN_AVERAGE_VERTICES * this.featureCount)));
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH} float32 values`
      );
    }
    const columns = Object.entries(props.output).filter(([, view]) => view);
    if (columns.length === 0) {
      throw new Error(`${id} requires at least one output column`);
    }
    for (const [name, view] of columns) {
      validateSinglePackedView(
        view as GraphDataView,
        name === 'clockwise' || name === 'sliver' ? ['uint32'] : ['float32'],
        `${id} output.${name}`
      );
      if ((view as GraphDataView).length !== this.featureCount) {
        throw new Error(`${id} output.${name} must hold ${this.featureCount} rows`);
      }
    }
    validateShapeDescriptorLogicalOutputs(id, props.output);
    validateDisjointWritableViews(id, {
      areas: props.output.areas,
      perimeters: props.output.perimeters
    });
    if (props.output.polsbyPopper || props.output.schwartzberg || props.output.sliver) {
      validateDisjointWritableViews(id, {
        areas: props.output.areas,
        perimeters: props.output.perimeters,
        polsbyPopper: props.output.polsbyPopper,
        schwartzberg: props.output.schwartzberg,
        sliver: props.output.sliver
      });
    }
    validateDisjointWritableViews(id, {
      elongation: props.output.elongation,
      orientation: props.output.orientation,
      clockwise: props.output.clockwise
    });
    if (props.output.convexity)
      validateDisjointWritableViews(id, {
        areas: props.output.areas,
        convexity: props.output.convexity
      });
    validateGraphOutputsDisjointFromInputs(
      id,
      columns.map(([, view]) => view as GraphDataView),
      [
        props.positions,
        props.ringOffsets,
        props.featureOffsets,
        props.polygonOffsets,
        props.parameters
      ]
    );
  }

  /**
   * Returns the internal measures nodes, then the moments, compactness, orientation and convexity
   * nodes for the requested outputs.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, featureCount} = this;
    const {output} = props;
    validateGraphChunkBindingRanges(graph, id, {
      positions: props.positions,
      ringOffsets: props.ringOffsets,
      featureOffsets: props.featureOffsets,
      polygonOffsets: props.polygonOffsets,
      parameters: props.parameters,
      ...Object.fromEntries(Object.entries(output).map(([name, view]) => [`output.${name}`, view]))
    });
    if (props.positions.length === 0)
      return getEmptyGeometryNodes(
        graph,
        id,
        OPERATION,
        Object.fromEntries(
          Object.entries(output).map(([name, view]) => [
            name,
            {view, value: ['areas', 'perimeters', 'clockwise', 'sliver'].includes(name) ? 0 : NaN}
          ])
        )
      );
    // The nested geometry measure's feature record is larger than our scalar, centroid and
    // four-word moment planes. Validate it before declaring even those smaller resources.
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'geometry measure feature scratch',
      featureCount * GEOMETRY_STATS_STRIDE * 4,
      'reduce the feature count'
    );
    if (output.convexity && this.useMonotoneChainConvexity) {
      validateMonotoneChainConvexityScratch(graph.device, id, props.positions.length, featureCount);
    }
    const needsMoments = Boolean(output.elongation || output.orientation || output.clockwise);
    const ringCount = props.ringOffsets.length - 1;
    const useTiledMoments =
      needsMoments &&
      ((props.momentMethod ?? 'auto') === 'tiled' ||
        ((props.momentMethod ?? 'auto') === 'auto' &&
          props.positions.length >= SHAPE_MOMENT_TILED_MINIMUM_ROW_COUNT &&
          canUseShapeMomentTiles(graph.device, props.positions.length, ringCount)));
    if (useTiledMoments)
      validateShapeMomentTileScratch(graph.device, id, props.positions.length, ringCount);
    const needsCompactness = Boolean(output.polsbyPopper || output.schwartzberg || output.sliver);
    const needsAreas = needsCompactness || Boolean(output.convexity) || Boolean(output.areas);
    const needsPerimeters = needsCompactness || Boolean(output.perimeters);
    const areas =
      output.areas ??
      (needsAreas ? createTransientView(graph, `${id}-areas`, 'float32', featureCount) : undefined);
    const perimeters =
      output.perimeters ??
      (needsPerimeters
        ? createTransientView(graph, `${id}-perimeters`, 'float32', featureCount)
        : undefined);
    const centroids = needsMoments
      ? createTransientView(graph, `${id}-centroids`, 'float32x2', featureCount)
      : undefined;
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (areas || perimeters || centroids) {
      nodes.push(
        ...new GPUGeometryMeasures({
          spatialContext: this.spatialContext,
          id: `${id}-measures`,
          positions: props.positions,
          geometryType: 'polygons',
          ringOffsets: props.ringOffsets,
          featureOffsets: props.featureOffsets,
          polygonOffsets: props.polygonOffsets,
          output: {lengths: perimeters, areas, centroids}
        }).getCommandNodes(graph)
      );
    }
    const ringInputs = {
      operation: OPERATION,
      positions: props.positions,
      ringOffsets: props.ringOffsets,
      featureOffsets: props.featureOffsets,
      polygonOffsets: props.polygonOffsets,
      featureCount
    };
    if (needsMoments && centroids) {
      const moments = createTransientView(graph, `${id}-moments`, 'float32x4', featureCount);
      nodes.push(
        ...(useTiledMoments
          ? getShapeMomentTileNodes(graph, {
              ...ringInputs,
              id: `${id}-moments`,
              centroids,
              moments
            })
          : [
              createMomentsNode<Parameters>(graph, {
                ...ringInputs,
                id: `${id}-moments`,
                centroids,
                moments
              })
            ]),
        createOrientationNode<Parameters>(graph, {
          id: `${id}-orientation`,
          operation: OPERATION,
          featureCount,
          moments,
          elongation: output.elongation,
          orientation: output.orientation,
          clockwise: output.clockwise
        })
      );
    }
    if (needsCompactness && areas && perimeters) {
      nodes.push(
        createCompactnessNode<Parameters>(graph, {
          id: `${id}-compactness`,
          operation: OPERATION,
          featureCount,
          areas,
          perimeters,
          parameters: props.parameters,
          polsbyPopper: output.polsbyPopper,
          schwartzberg: output.schwartzberg,
          sliver: output.sliver
        })
      );
    }
    if (output.convexity && areas) {
      if (this.useMonotoneChainConvexity) {
        nodes.push(
          ...getMonotoneChainConvexityNodes<Parameters>(graph, {
            ...ringInputs,
            id: `${id}-convexity`,
            areas,
            convexities: output.convexity
          })
        );
      } else {
        nodes.push(
          createConvexityNode<Parameters>(graph, {
            ...ringInputs,
            id: `${id}-convexity`,
            areas,
            convexities: output.convexity
          })
        );
      }
    }
    return nodes;
  }
}
