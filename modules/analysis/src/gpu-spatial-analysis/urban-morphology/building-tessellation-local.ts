// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {captureUrbanLandscapeDestinations} from '../landscape-ecology/urban-landscape-destinations';
import {getSpatialAnalysisPackedKernelNodes} from '../geographic-distribution/spatial-analysis-packed-kernel';
import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validateRequirement,
  validateViewLength,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getRecipeView} from '../recipes/expansion-recipe-utils';
import {
  validateUrbanRecipeBindings,
  validateUrbanStorageByteLength
} from './building-tessellation-validation';
import {addBuildingTessellationSamples} from './building-tessellation-sampling';
import {
  addBuildingTessellationIndex,
  getBuildingTessellationIndexShape
} from './building-tessellation-index';
import {addBuildingTessellationEnclosure} from './building-tessellation-enclosure';
import {
  getPolygonTriangulationIndexCount,
  getPolygonTriangulationNodeCount
} from '../../analysis-core/polygon-triangulation/gpu-polygon-triangulation';
import {getBuildingTessellationCellKernel} from './building-tessellation-cells';

/** Bounded local work replaces graph topology proportional to the building count. Device bindings may lower this. */
export const BUILDING_TESSELLATION_LOCAL_MAXIMUM_BUILDING_COUNT = 1000000;
/** Sample row counts and segment interpolation fractions must remain exact float32 integers. */
export const BUILDING_TESSELLATION_LOCAL_MAXIMUM_SAMPLE_CAPACITY = 16777216;
/** Bounds per-building serial work and the fixed-stride output's number of convex pieces. */
export const BUILDING_TESSELLATION_LOCAL_MAXIMUM_SAMPLES_PER_BUILDING = 32;
/** Bounds private clipping arrays and the candidate-gather table in each building invocation. */
export const BUILDING_TESSELLATION_LOCAL_MAXIMUM_CANDIDATE_CAPACITY = 256;
/** Bounds each convex output piece; overflow is reported independently of candidate overflow. */
export const BUILDING_TESSELLATION_LOCAL_MAXIMUM_CELL_VERTEX_CAPACITY = 128;
/** Bounds enclosure validation, or the one-time triangulation of a simple enclosure. */
export const BUILDING_TESSELLATION_LOCAL_MAXIMUM_ENCLOSURE_VERTEX_COUNT = 32;
/** Bounds point-row visits per search route (grid, then shorter sorted coordinate interval). */
export const BUILDING_TESSELLATION_LOCAL_MAXIMUM_GRID_VISITS = 8192;
/** Eight convex pieces per building leave room for ordinary rectangular boundary sampling in a 128 MiB output binding. */
export const BUILDING_TESSELLATION_LOCAL_DEFAULT_SAMPLES_PER_BUILDING = 8;
/** Bounds the default private candidate table while covering ordinary sparse city neighbourhoods. */
export const BUILDING_TESSELLATION_LOCAL_DEFAULT_CANDIDATE_CAPACITY = 128;
/** Twelve vertices per piece give a 76.8 MB positions binding for 100k buildings at the default own-sample capacity. */
export const BUILDING_TESSELLATION_LOCAL_DEFAULT_CELL_VERTEX_CAPACITY = 12;
/** Bounds false-positive row visits in each indexed query attempt. */
export const BUILDING_TESSELLATION_LOCAL_DEFAULT_GRID_VISITS = 2048;
/** Float32x2 output addressing uses uint32 component indices, so vertex slots must fit half that range. */
const BUILDING_TESSELLATION_LOCAL_MAXIMUM_VERTEX_SLOTS = 0x7fffffff;
/** Conservative allowance for four published sorts, sampling scans and optional bounded triangulation through 2^24 sample rows. */
export const BUILDING_TESSELLATION_LOCAL_COMMAND_NODE_ALLOWANCE = 768;
/** Bounds empty/degenerate ring slots as well as hole-bridging work in the enclosure triangulation. */
export const BUILDING_TESSELLATION_LOCAL_MAXIMUM_ENCLOSURE_RING_COUNT = 32;
/** Per-frame parameter order: sampleSpacing, candidateRadius, vertexTolerance. */
export const GPU_BUILDING_TESSELLATION_LOCAL_PARAMETER_LENGTH = 3;

/** Packs per-frame local tessellation parameters for a caller-owned parameters view. */
export function getGPUBuildingTessellationLocalParameterValues(values: {
  sampleSpacing: number;
  candidateRadius: number;
  vertexTolerance: number;
}): Float32Array {
  return new Float32Array([values.sampleSpacing, values.candidateRadius, values.vertexTolerance]);
}

export type GPUBuildingTessellationLocalStatus = {
  /** Usable building rows (not convex pieces); read buildingStatus for their fixed row locations. */
  count: GraphDataView<'uint32'>;
  /** The requested buildingCount, including rows that could not be certified. */
  requiredCount: GraphDataView<'uint32'>;
  /** One when cellVertexCapacity discarded geometry. Candidate limits do not set this flag. */
  overflow: GraphDataView<'uint32'>;
  /** One when sampling, own-sample, candidate, index-visit or radius certification was insufficient. */
  candidateOverflow: GraphDataView<'uint32'>;
  /** Number of building rows invalidated by bad input or non-finite clipping arithmetic. */
  invalidCount: GraphDataView<'uint32'>;
};

export type GPUBuildingTessellationLocalOutput = {
  /**
   * float32x2, buildingCount * piecesPerBuilding * cellVertexCapacity rows.
   * Piece (building * piecesPerBuilding + part) starts at piece * cellVertexCapacity.
   * piecesPerBuilding is maximumSamplesPerBuilding for convex enclosures, multiplied by the
   * bounded enclosure triangle capacity in simple mode (returned in provenance).
   * Each building's region is the UNION of its convex pieces. Shared internal edges are retained;
   * these are not disjoint components or a dissolved shell/hole ring hierarchy.
   */
  positions: GraphDataView<'float32x2'>;
  /**
   * uint32x4, one record per fixed piece slot:
   * [vertexCount, requiredVertexCount, sourceBuildingId, sourceSampleRow].
   * Unused slots have zero counts and sourceSampleRow 0xffffffff. Failed buildings publish zero
   * vertexCounts for EVERY piece; requiredVertexCount is 0xffffffff when it was not certified.
   */
  cells: GraphDataView<'uint32x4'>;
  /** uint32x4, per building: [validity, overflow, candidateOverflow, invalid]. Each field is zero or one. */
  buildingStatus: GraphDataView<'uint32x4'>;
  /** uint32x4: [stored boundary samples, required samples (saturated uint32), sampling overflow, invalid segments]. */
  samplingStatus: GraphDataView<'uint32x4'>;
  status: GPUBuildingTessellationLocalStatus;
};

export type GPUBuildingTessellationLocalProps = {
  id?: string;
  /** Supplying candidateRadius selects this route; explicit 'reference' selects the legacy recipe. */
  mode?: 'local';
  /** Packed, closed building boundaries; sampling excludes segment end points, as in the reference. */
  segments: GraphDataView<'float32x4'>;
  segmentBuildingIds: GraphDataView<'uint32'>;
  buildingCount: number;
  /** A finite study enclosure without repeated closing vertices. Convex mode requires counterclockwise winding. */
  enclosure: GraphDataView<'float32x2'>;
  enclosureMode?: 'convex' | 'simple';
  /** Simple mode only: shell first, then holes; terminal offsets into enclosure. Either winding is accepted. */
  enclosureRingOffsets?: GraphDataView<'uint32'>;
  spatialContext?: GPUSpatialContext;
  sampleSpacing: number;
  sampleCapacity: number;
  /** Optional per-frame values packed by getGPUBuildingTessellationLocalParameterValues. Otherwise props supply immutable defaults. */
  parameters?: GraphDataView<'float32'>;
  /**
   * Query radius around the midpoint of the building samples' bounding box. All sample sites in
   * this circle are gathered through an adaptive coordinate-rank grid, independently of global
   * coordinate extent. Overfull rank cells retry with the shorter sorted coordinate interval.
   * A cell is published only when the
   * triangle-inequality / twice-cell-radius bound proves every excluded site cannot cut it.
   * Increase this when certification fails; also increase candidateCapacity for dense circles.
   */
  candidateRadius: number;
  maximumSamplesPerBuilding?: number;
  candidateCapacity?: number;
  cellVertexCapacity?: number;
  /** Row-visit budget for each indexed search attempt; retained name for compatibility. */
  maximumGridVisits?: number;
  /**
   * Coordinate tolerance for the float32 radius certificate, not a snapping lattice. The region
   * equals the sampled Voronoi union before the reference overlay's vertexTolerance snapping;
   * compare the two at this coordinate tolerance, allowing reference lattice quantization.
   * Raw vertex counts, ring order and shared internal edges intentionally differ.
   */
  vertexTolerance: number;
  output?: Partial<Omit<GPUBuildingTessellationLocalOutput, 'status'>> & {
    status?: Partial<GPUBuildingTessellationLocalStatus>;
  };
};

/**
 * Bounded, building-parallel sampled Voronoi tessellation for planar coordinates and convex
 * or simple (possibly holed) enclosures. Simple enclosures are triangulated once, and each
 * sample cell is intersected with every bounded enclosure triangle. One clipping dispatch handles
 * every building, regardless of building count.
 * Per-frame: input contents and optional parameters (spacing, radius, tolerance). Compile-time: lengths and capacities.
 * The legacy overlay recipe remains the reference for dissolved shell/hole rings.
 * Per call: at most 1,000,000 buildings, 16,777,216 total samples, 32 total enclosure vertices
 * and 32 enclosure rings. Per building: at most 32 own samples and 256 candidate samples.
 * Per convex piece: at most 128 vertices. Each indexed query attempt visits at most 8,192 rows.
 * Device storage limits may lower these bounds. The named entry and legacy local overload share
 * this implementation and the sample-cell-union output representation.
 * The legacy dispatcher selects this same typed local entry.
 */
export function addLocalBuildingTessellationRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUBuildingTessellationLocalProps
) {
  const hasRemovedOutputs = 'outputs' in props;
  props = snapshotConstructionFields(props, [
    'id',
    'mode',
    'segments',
    'segmentBuildingIds',
    'buildingCount',
    'enclosure',
    'enclosureMode',
    'enclosureRingOffsets',
    'spatialContext',
    'sampleSpacing',
    'sampleCapacity',
    'parameters',
    'candidateRadius',
    'maximumSamplesPerBuilding',
    'candidateCapacity',
    'cellVertexCapacity',
    'maximumGridVisits',
    'vertexTolerance',
    'output'
  ]);

  const id = props.id ?? 'building-tessellation';
  validateRequirement(id, !hasRemovedOutputs, 'outputs was removed; use output');
  props = {
    ...props,
    output: captureUrbanLandscapeDestinations(id, {output: props.output}, [
      'positions',
      'cells',
      'buildingStatus',
      'samplingStatus',
      'status'
    ])
  };
  const maximumSamplesPerBuilding =
    props.maximumSamplesPerBuilding ?? BUILDING_TESSELLATION_LOCAL_DEFAULT_SAMPLES_PER_BUILDING;
  const candidateCapacity =
    props.candidateCapacity ?? BUILDING_TESSELLATION_LOCAL_DEFAULT_CANDIDATE_CAPACITY;
  const cellVertexCapacity =
    props.cellVertexCapacity ?? BUILDING_TESSELLATION_LOCAL_DEFAULT_CELL_VERTEX_CAPACITY;
  const maximumGridVisits =
    props.maximumGridVisits ?? BUILDING_TESSELLATION_LOCAL_DEFAULT_GRID_VISITS;
  const spatialContext = resolveGPUSpatialContext(id, props.spatialContext);
  validateCapacity(
    id,
    'buildingCount',
    props.buildingCount,
    1,
    BUILDING_TESSELLATION_LOCAL_MAXIMUM_BUILDING_COUNT
  );
  validateCapacity(
    id,
    'sampleCapacity',
    props.sampleCapacity,
    1,
    BUILDING_TESSELLATION_LOCAL_MAXIMUM_SAMPLE_CAPACITY
  );
  validateCapacity(
    id,
    'maximumSamplesPerBuilding',
    maximumSamplesPerBuilding,
    1,
    BUILDING_TESSELLATION_LOCAL_MAXIMUM_SAMPLES_PER_BUILDING
  );
  validateCapacity(
    id,
    'candidateCapacity',
    candidateCapacity,
    1,
    BUILDING_TESSELLATION_LOCAL_MAXIMUM_CANDIDATE_CAPACITY
  );
  validateCapacity(
    id,
    'cellVertexCapacity',
    cellVertexCapacity,
    3,
    BUILDING_TESSELLATION_LOCAL_MAXIMUM_CELL_VERTEX_CAPACITY
  );
  validateCapacity(
    id,
    'maximumGridVisits',
    maximumGridVisits,
    1,
    BUILDING_TESSELLATION_LOCAL_MAXIMUM_GRID_VISITS
  );
  validateCapacity(
    id,
    'enclosure.length',
    props.enclosure.length,
    3,
    BUILDING_TESSELLATION_LOCAL_MAXIMUM_ENCLOSURE_VERTEX_COUNT
  );
  validateRequirement(
    id,
    props.enclosureMode === undefined ||
      props.enclosureMode === 'convex' ||
      props.enclosureMode === 'simple',
    "enclosureMode must be 'convex' or 'simple'"
  );
  validateRequirement(
    id,
    !props.enclosureRingOffsets || props.enclosureMode === 'simple',
    "enclosureRingOffsets requires enclosureMode 'simple'"
  );
  if (props.enclosureRingOffsets) {
    validateSinglePackedView(props.enclosureRingOffsets, ['uint32'], `${id} enclosureRingOffsets`);
    validateCapacity(
      id,
      'enclosureRingOffsets.length',
      props.enclosureRingOffsets.length,
      2,
      BUILDING_TESSELLATION_LOCAL_MAXIMUM_ENCLOSURE_RING_COUNT + 1
    );
  }
  const enclosurePieceCount =
    props.enclosureMode === 'simple'
      ? getPolygonTriangulationIndexCount(
          props.enclosure.length,
          (props.enclosureRingOffsets?.length ?? 2) - 1,
          1
        ) / 3
      : 1;
  const piecesPerBuilding = maximumSamplesPerBuilding * enclosurePieceCount;
  for (const name of ['sampleSpacing', 'candidateRadius', 'vertexTolerance'] as const) {
    validateRequirement(
      id,
      Number.isFinite(props[name]) &&
        Math.fround(props[name]) > 0 &&
        Number.isFinite(Math.fround(props[name])),
      `${name} must be a positive finite float32 number`
    );
  }
  validateRequirement(
    id,
    props.segments.length > 0 && props.segmentBuildingIds.length === props.segments.length,
    'segments must have at least 1 row and segmentBuildingIds one row per segment'
  );
  validateSinglePackedView(props.segments, ['float32x4'], `${id} segments`);
  validateSinglePackedView(props.segmentBuildingIds, ['uint32'], `${id} segmentBuildingIds`);
  validateSinglePackedView(props.enclosure, ['float32x2'], `${id} enclosure`);
  if (props.parameters) {
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    validateViewLength(
      id,
      'parameters',
      props.parameters,
      GPU_BUILDING_TESSELLATION_LOCAL_PARAMETER_LENGTH
    );
  }
  validateGraphViewsBelongToGraph(id, graph, [
    props.segments,
    props.segmentBuildingIds,
    props.enclosure,
    props.enclosureRingOffsets,
    props.parameters
  ]);
  validateUrbanRecipeBindings(
    graph,
    id,
    {
      segments: props.segments,
      segmentBuildingIds: props.segmentBuildingIds,
      enclosure: props.enclosure,
      enclosureRingOffsets: props.enclosureRingOffsets,
      parameters: props.parameters
    },
    props.output,
    [
      [
        'output.positions',
        'output.cells',
        'output.buildingStatus',
        'parameters',
        ...(props.enclosureMode === 'simple' ? [] : ['enclosure'])
      ]
    ]
  );
  const pieceCapacity = props.buildingCount * piecesPerBuilding;
  const vertexSlots = pieceCapacity * cellVertexCapacity;
  validateMaximumCount(
    id,
    'buildingCount * piecesPerBuilding * cellVertexCapacity',
    vertexSlots,
    BUILDING_TESSELLATION_LOCAL_MAXIMUM_VERTEX_SLOTS
  );
  // Check every supplied shape before allocating absent outputs; a failed declaration is atomic.
  for (const [name, format, length, view] of [
    ['positions', 'float32x2', vertexSlots, props.output?.positions],
    ['cells', 'uint32x4', pieceCapacity, props.output?.cells],
    ['buildingStatus', 'uint32x4', props.buildingCount, props.output?.buildingStatus],
    ['samplingStatus', 'uint32x4', 1, props.output?.samplingStatus],
    ...Object.entries(props.output?.status ?? {}).map(
      ([name, view]) => [name, 'uint32', 1, view] as const
    )
  ] as const) {
    if (!view) continue;
    validateSinglePackedView(view, [format], `${id} output.${name}`);
    validateViewLength(id, `output.${name}`, view, length);
  }
  const estimatedCommandNodeCount = BUILDING_TESSELLATION_LOCAL_COMMAND_NODE_ALLOWANCE;
  validateCommandNodeBudget(id, estimatedCommandNodeCount);
  // Recipes allocate while declaring nodes: this is their getCommandNodes boundary.
  for (const [name, bytes, hint] of [
    ['segments', props.segments.length * 16, 'reduce segments length'],
    ['segment offsets', (props.segments.length + 1) * 4, 'reduce segments length'],
    ['sample coordinates and owners', props.sampleCapacity * 12, 'reduce sampleCapacity'],
    [
      'sorted coordinate index',
      getBuildingTessellationIndexShape(props.sampleCapacity).byteLength,
      'reduce sampleCapacity'
    ],
    [
      'building index',
      (props.buildingCount + 5 + props.sampleCapacity) * 4,
      'reduce buildingCount or sampleCapacity'
    ],
    [
      'positions',
      vertexSlots * 8,
      'reduce buildingCount, maximumSamplesPerBuilding or cellVertexCapacity'
    ],
    ['cells', pieceCapacity * 16, 'reduce buildingCount or maximumSamplesPerBuilding'],
    ['building status', props.buildingCount * 16, 'reduce buildingCount'],
    [
      'enclosure triangulation scratch',
      getPolygonTriangulationNodeCount(
        props.enclosure.length,
        (props.enclosureRingOffsets?.length ?? 2) - 1,
        1
      ) * 16,
      'reduce enclosure length or ring count'
    ],
    [
      'enclosure triangles',
      (1 + 3 * enclosurePieceCount) * 8,
      'reduce enclosure length or ring count'
    ]
  ] as const)
    validateUrbanStorageByteLength(graph.device, id, name, bytes, hint);
  const provided = props.output;
  const positions = getRecipeView(
    graph,
    `${id}-local-positions`,
    'float32x2',
    vertexSlots,
    provided?.positions
  );
  const cells = getRecipeView(
    graph,
    `${id}-local-cells`,
    'uint32x4',
    pieceCapacity,
    provided?.cells
  );
  const buildingStatus = getRecipeView(
    graph,
    `${id}-local-building-status`,
    'uint32x4',
    props.buildingCount,
    provided?.buildingStatus
  );
  const samplingStatus = getRecipeView(
    graph,
    `${id}-local-sampling-status`,
    'uint32x4',
    1,
    provided?.samplingStatus
  );
  const scalar = (name: keyof GPUBuildingTessellationLocalStatus) =>
    getRecipeView(graph, `${id}-local-${name}`, 'uint32', 1, provided?.status?.[name]);
  const status = {
    count: scalar('count'),
    requiredCount: scalar('requiredCount'),
    overflow: scalar('overflow'),
    candidateOverflow: scalar('candidateOverflow'),
    invalidCount: scalar('invalidCount')
  };
  const parameters =
    props.parameters ??
    createTransientView(
      graph,
      `${id}-local-parameters`,
      'float32',
      GPU_BUILDING_TESSELLATION_LOCAL_PARAMETER_LENGTH
    );
  if (!props.parameters)
    graph.add(
      getSpatialAnalysisPackedKernelNodes(graph, {
        id: `${id}-local-default-parameters`,
        operation: id,
        invocationCount: 1,
        bindings: [{name: 'parameters', view: parameters, type: 'f32', access: 'read_write'}],
        body: `parameters[parametersOffset]=${getWGSLFloatLiteral(props.sampleSpacing)};
parameters[parametersOffset+1u]=${getWGSLFloatLiteral(props.candidateRadius)};
parameters[parametersOffset+2u]=${getWGSLFloatLiteral(props.vertexTolerance)};`
      })
    );
  const sampling = addBuildingTessellationSamples(graph, {
    ...props,
    parameters,
    id: `${id}-local`,
    samplingStatus
  });
  const pointIndex = addBuildingTessellationIndex(graph, {
    id: `${id}-local-index`,
    positions: sampling.samples,
    count: samplingStatus
  });
  const enclosure =
    props.enclosureMode === 'simple'
      ? addBuildingTessellationEnclosure(graph, {
          id: `${id}-local-enclosure`,
          positions: props.enclosure,
          ringOffsets: props.enclosureRingOffsets,
          triangleCapacity: enclosurePieceCount
        })
      : props.enclosure;
  graph.add(
    getBuildingTessellationCellKernel(graph, {
      id: `${id}-local-cells`,
      buildingCount: props.buildingCount,
      maximumSamplesPerBuilding,
      candidateCapacity,
      cellVertexCapacity,
      maximumGridVisits,
      parameters,
      enclosure,
      simpleEnclosure: props.enclosureMode === 'simple',
      enclosurePieceCount,
      pointIndex,
      sampleData: sampling.sampleData,
      buildingIndex: sampling.buildingIndex,
      positions,
      cells,
      buildingStatus
    })
  );
  const flags = ['validity', 'overflow', 'candidateOverflow', 'invalidCount'].map(name =>
    createTransientView(graph, `${id}-local-${name}-rows`, 'uint32', props.buildingCount)
  );
  graph.add(
    getSpatialAnalysisPackedKernelNodes(graph, {
      id: `${id}-local-status-rows`,
      operation: id,
      invocationCount: props.buildingCount,
      bindings: [
        {name: 'records', view: buildingStatus, type: 'u32', access: 'read'},
        ...flags.map(
          (view, index) =>
            ({name: `flag${index}`, view, type: 'u32', access: 'read_write'}) as const
        )
      ],
      body: flags
        .map(
          (_, index) =>
            `flag${index}[flag${index}Offset+index]=records[recordsOffset+4u*index+${index}u];`
        )
        .join('\n')
    })
  );
  const destinations = [
    status.count,
    status.overflow,
    status.candidateOverflow,
    status.invalidCount
  ];
  flags.forEach((input, index) =>
    graph.add(
      new GPUReduction({
        id: `${id}-local-status-${index}`,
        input,
        output: destinations[index],
        operation: index === 0 || index === 3 ? 'sum' : 'max'
      }).getCommandNodes(graph)
    )
  );
  graph.add(
    getSpatialAnalysisPackedKernelNodes(graph, {
      id: `${id}-local-required-count`,
      operation: id,
      invocationCount: 1,
      bindings: [{name: 'required', view: status.requiredCount, type: 'u32', access: 'read_write'}],
      body: `required[requiredOffset]=${props.buildingCount}u;`
    })
  );
  return {
    outputs: {positions, cells, buildingStatus},
    status: {...status, sampling: samplingStatus},
    intermediates: {...sampling, parameters},
    estimatedCommandNodeCount,
    provenance: {
      segments: props.segments,
      segmentBuildingIds: props.segmentBuildingIds,
      enclosure: props.enclosure,
      enclosureRingOffsets: props.enclosureRingOffsets,
      enclosurePieceCount,
      piecesPerBuilding,
      spatialContext,
      sampleSpacing: props.sampleSpacing,
      candidateRadius: props.candidateRadius,
      vertexTolerance: props.vertexTolerance,
      maximumSamplesPerBuilding,
      candidateCapacity,
      cellVertexCapacity,
      maximumGridVisits,
      mode: 'local' as const,
      representation: 'sample-cell-union' as const
    }
  };
}
