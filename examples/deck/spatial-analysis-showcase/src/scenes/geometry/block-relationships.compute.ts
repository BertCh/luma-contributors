// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {PathLayer, PolygonLayer, ScatterplotLayer} from '@deck.gl/layers';
import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {
  addBuildingAdjacencyRecipe,
  addBuildingTessellationRecipe,
  addVisibilityGraphRecipe,
  GPUBufferSurface,
  GPUBoundsFilter,
  GPUGeometryMeasures,
  GPUShapeDescriptors,
  GPUSharedPaths,
  getGPUBoundsFilterParameterValues,
  getGPUShapeDescriptorsParameterValues,
  type GPUPolygonOverlayOutput
} from '@luma-contributors/graph/gpu-spatial-analysis';
import {addKernelPass} from '../../engine/mode-kernels';
import {GraphSwap} from '../../engine/graph-swap';
import {createGraphBufferImporter} from '../../engine/graph-buffers';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {BarChartData} from '../chart-types';
import type {SceneContext, SceneInstance} from '../scene';
import {
  BLOCK_RELATIONSHIP_BUILDINGS,
  BLOCK_RELATIONSHIPS_PROBES,
  buildBlockRelationshipsData,
  type BlockRelationshipsData
} from './block-relationships-data';
import type {BlockRelationshipsOptions} from './block-relationships-options';
import {toDegrees} from '@math.gl/core';

const BUILDING_COUNT = 3;
const SHARED_RUN_CAPACITY = 64;
const SHARED_VERTEX_CAPACITY = 512;
const BUFFER_INTERSECTION_CAPACITY = 512;
const BUFFER_SEGMENT_CAPACITY = 2048;
const BUFFER_RING_CAPACITY = 128;
const BUFFER_VERTEX_CAPACITY = 4096;
const TESS_INTERSECTION_CAPACITY = 512;
const TESS_SEGMENT_CAPACITY = 2048;
const TESS_VERTEX_CAPACITY = 8192;
const TESS_RING_CAPACITY = 512;
const BUILDING_COLORS: [number, number, number, number][] = [
  [215, 133, 72, 215],
  [52, 147, 157, 215],
  [124, 130, 183, 215]
];

type Point = [number, number];
type ReadbackSource = {buffer: Buffer; size: number; kind: 'uint32' | 'float32'};
type ImportView = ReturnType<typeof createGraphBufferImporter<void>>;

type SharedPathRows = {
  positions: Float32Array;
  offsets: Uint32Array;
  leftLineIds: Uint32Array;
  rightLineIds: Uint32Array;
  forward: Uint32Array;
  lineBuildingIds: Uint32Array;
};

type CoreResult = {
  shared: SharedPathRows;
  sharedStatus: {
    writtenRuns: number;
    writtenVertices: number;
    requiredRuns: number;
    requiredVertices: number;
    overflow: number;
    uncertain: number;
    complete: boolean;
  };
  orientation: Float32Array;
  bounds: {ids: Uint32Array; count: number; requiredCount: number; overflow: number};
  buffers: BufferGeometryResult[];
  visibility: VisibilityResult;
};

type BufferGeometryResult = {
  positions: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
  count: number;
  requiredRings: number;
  overflow: number;
  candidateOverflow: number;
  uncertainCount: number;
  actualRings: number;
  boundaryCount: number;
  boundaryRequiredCount: number;
  boundaryOverflow: number;
  boundaryCandidateOverflow: number;
  complete: boolean;
};

type VisibilityResult = {
  edges: Uint32Array;
  count: number;
  requiredCount: number;
  overflow: number;
  invalidCount: number;
  metrics: Float32Array;
  classifications: Uint32Array;
  choice: Float32Array;
  integration: Float32Array;
  normalizedChoice: Float32Array;
  extendedIntegration: Float32Array;
};

type TessellationResult = {
  polygons: {buildingIndex: number; rings: Point[][]}[];
  areas: number[];
  status: string;
  sampleCount: number;
  requiredSamples: number;
  sampleOverflow: number;
  invalidSamples: number;
  complete: Uint32Array;
};

type TessellationVariant = {
  key: string;
  spacing: string;
  sampleCapacity: number;
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  reader: SummaryReader;
  readbackSources: ReadbackSource[];
  dirty: boolean;
  generation: number;
  result: TessellationResult | null;
};

/** GPU bounds, shared edges, boundary shrink and visibility for a three-footprint block inquiry. */
export async function createBlockRelationships(
  ctx: SceneContext<BlockRelationshipsOptions>
): Promise<SceneInstance<BlockRelationshipsOptions>> {
  const swap = new GraphSwap();
  let active: SceneInstance<BlockRelationshipsOptions> | null = null;
  let desiredOptions = {...ctx.options};
  let updateActiveOptions: ((state: Readonly<BlockRelationshipsOptions>) => void) | null = null;
  let generation = 0;
  let destroyed = false;
  const compileKeys: readonly (keyof BlockRelationshipsOptions)[] = [
    'boundsRelation',
    'inset',
    'visibilityDistance',
    'visibilityEdgeCapacity'
  ];
  const rebuild = async () => {
    const requestedGeneration = ++generation;
    const snapshot = {...desiredOptions};
    let variantOptions = {...snapshot};
    active?.destroy();
    active = null;
    updateActiveOptions = null;
    for (const id of ['blockBounds', 'blockAdjacency', 'blockTessellation', 'blockVisibility'])
      ctx.setReadout(id, 'GPU block graph rebuild pending.');
    for (const id of ['blockAdjacencyChart', 'blockVisibilityChart', 'blockTessellationChart'])
      ctx.setChart(id, null);
    ctx.requestLayers();
    const isCurrent = () => !destroyed && !ctx.signal.aborted && generation === requestedGeneration;
    const variantContext: SceneContext<BlockRelationshipsOptions> = {
      ...ctx,
      get options() {
        return variantOptions;
      },
      setReadout: (id, value) => {
        if (isCurrent()) ctx.setReadout(id, value);
      },
      setChart: (id, value) => {
        if (isCurrent()) ctx.setChart(id, value);
      },
      setStatus: message => {
        if (isCurrent()) ctx.setStatus(message);
      },
      setCost: value => {
        if (isCurrent()) ctx.setCost(value);
      },
      requestLayers: () => {
        if (isCurrent()) ctx.requestLayers();
      }
    };
    try {
      await swap.request({
        build: () => createBlockRelationshipsVariant(variantContext),
        commit: next => {
          if (!isCurrent()) {
            next.destroy();
            return;
          }
          active = next;
          updateActiveOptions = state => {
            variantOptions = {...state};
          };
          variantOptions = {...desiredOptions};
          for (const key of Object.keys(desiredOptions) as (keyof BlockRelationshipsOptions)[]) {
            if (snapshot[key] !== desiredOptions[key])
              next.setOption?.(key, desiredOptions[key], desiredOptions);
          }
          ctx.setStatus('');
          ctx.requestLayers();
        },
        discard: next => next.destroy()
      });
    } catch (error) {
      if (isCurrent()) {
        ctx.setStatus(`GPU block graph failed: ${String(error)}`);
        ctx.requestLayers();
      }
    }
  };
  await rebuild();
  return {
    getCompiledGraphs: () => active?.getCompiledGraphs?.() ?? [],
    getLayers: () => active?.getLayers() ?? [],
    encode: (encoder, frame) => active?.encode(encoder, frame),
    setOption(id, value, state) {
      const changed = desiredOptions[id] !== value;
      desiredOptions = {...state};
      if (changed && compileKeys.includes(id)) void rebuild();
      else {
        updateActiveOptions?.(state);
        active?.setOption?.(id, value, state);
      }
    },
    destroy() {
      destroyed = true;
      generation++;
      swap.destroy();
      active?.destroy();
      active = null;
    }
  };
}

async function createBlockRelationshipsVariant(
  ctx: SceneContext<BlockRelationshipsOptions>
): Promise<SceneInstance<BlockRelationshipsOptions>> {
  const dataset = ctx.datasets.get('chicago-buildings');
  const data = buildBlockRelationshipsData(dataset);
  const visibilityEdgeCapacity = Number(ctx.options.visibilityEdgeCapacity);
  const resources = new SpatialAnalysisResources(ctx.device, 'block-relationships-core');
  const graph = new GPUCommandGraph<void>(ctx.device, {id: 'block-relationships-core'});
  const importView = createGraphBufferImporter(graph);
  const inputPositions = resources.createBuffer('source-positions', data.positions);
  const inputRingOffsets = resources.createBuffer('source-ring-offsets', data.ringOffsets);
  const inputPolygonOffsets = resources.createBuffer('source-polygon-offsets', data.polygonOffsets);
  const inputFeatureOffsets = resources.createBuffer('source-feature-offsets', data.featureOffsets);
  const probePositions = resources.createBuffer('observer-probe-positions', data.probePositions);
  const obstacleSegments = resources.createBuffer(
    'observer-obstacle-segments',
    data.obstacleSegments
  );
  const positions = importView(inputPositions, 'float32x2', data.positions.length / 2);
  const ringOffsets = importView(inputRingOffsets, 'uint32', data.ringOffsets.length);
  const polygonOffsets = importView(inputPolygonOffsets, 'uint32', data.polygonOffsets.length);
  const featureOffsets = importView(inputFeatureOffsets, 'uint32', data.featureOffsets.length);
  const measureBounds = resources.createBuffer('source-bounds', BUILDING_COUNT * 16);
  const measureInvalid = resources.createBuffer('measure-invalid-count', 4);
  const boundsView = importView(measureBounds, 'float32x4', BUILDING_COUNT);
  const shapeOrientation = resources.createBuffer('source-orientations', BUILDING_COUNT * 4);
  const orientationView = importView(shapeOrientation, 'float32', BUILDING_COUNT);
  const shapeParameters = resources.createParameterBuffer(
    'shape-parameters',
    'float32',
    4,
    getGPUShapeDescriptorsParameterValues()
  );
  graph.add(
    new GPUGeometryMeasures({
      id: 'source-footprint-measures',
      spatialContext: data.projection.spatialContext,
      geometryType: 'polygons',
      positions,
      ringOffsets,
      polygonOffsets,
      featureOffsets,
      output: {
        bounds: boundsView,
        status: {invalidCount: importView(measureInvalid, 'uint32', 1)}
      }
    })
  );
  graph.add(
    new GPUShapeDescriptors({
      id: 'source-footprint-descriptors',
      spatialContext: data.projection.spatialContext,
      positions,
      ringOffsets,
      polygonOffsets,
      featureOffsets,
      parameters: shapeParameters.importToGraph(graph),
      output: {orientation: orientationView}
    })
  );

  const boundsParameters = resources.createParameterBuffer(
    'query-bounds',
    'float32',
    4,
    makeQueryBounds(ctx.options.queryHalfExtent)
  );
  const boundsIds = resources.createBuffer('bounds-filter-ids', BUILDING_COUNT * 4);
  const boundsCount = resources.createBuffer('bounds-filter-count', 4);
  const boundsRequiredCount = resources.createBuffer('bounds-filter-required-count', 4);
  const boundsOverflow = resources.createBuffer('bounds-filter-overflow', 4);
  graph.add(
    new GPUBoundsFilter({
      id: 'source-footprint-bounds-filter',
      spatialContext: data.projection.spatialContext,
      bounds: boundsView,
      parameters: boundsParameters.importToGraph(graph),
      mode: ctx.options.boundsRelation,
      output: {
        ids: importView(boundsIds, 'uint32', BUILDING_COUNT),
        status: {
          count: importView(boundsCount, 'uint32', 1),
          requiredCount: importView(boundsRequiredCount, 'uint32', 1),
          overflow: importView(boundsOverflow, 'uint32', 1)
        }
      }
    })
  );

  const sharedBuffers = makeSharedPathBuffers(resources, importView);
  graph.add(
    new GPUSharedPaths({
      id: 'source-building-shared-paths',
      spatialContext: data.projection.spatialContext,
      left: {
        kind: 'lines',
        positions,
        lineOffsets: ringOffsets
      },
      right: {
        kind: 'lines',
        positions,
        lineOffsets: ringOffsets
      },
      intersectionCapacity: 256,
      output: sharedBuffers.output
    })
  );

  const bufferParameters = resources.createParameterBuffer(
    'shrink-parameters',
    'float32',
    4,
    Float32Array.of(-Number(ctx.options.inset), 2, 0, 0)
  );
  const bufferParameterView = bufferParameters.importToGraph(graph);
  const bufferOutputs = BLOCK_RELATIONSHIP_BUILDINGS.map((building, buildingIndex) => {
    const buildingGeometry = makeSingleBuildingBufferGeometry(
      resources,
      importView,
      data,
      buildingIndex
    );
    const bufferOutput = makeBufferSurfaceOutput(resources, importView, `shrink-${building.id}`);
    graph.add(
      new GPUBufferSurface({
        id: `synthetic-building-shrink-${building.id}`,
        spatialContext: data.projection.spatialContext,
        geometry: buildingGeometry,
        parameters: bufferParameterView,
        joinStyle: 'mitre',
        vertexTolerance: 0.005,
        capacity: {
          intersections: BUFFER_INTERSECTION_CAPACITY,
          nodedSegments: BUFFER_SEGMENT_CAPACITY
        },
        output: bufferOutput.output
      })
    );
    return bufferOutput;
  });

  const visibility = addVisibilityGraphRecipe(graph, {
    id: 'synthetic-block-probe-visibility',
    positions: importView(probePositions, 'float32x2', data.probePositions.length / 2),
    segments: importView(obstacleSegments, 'float32x4', data.obstacleSegments.length / 4),
    maximumDistance: Number(ctx.options.visibilityDistance),
    edgeCapacity: visibilityEdgeCapacity,
    tolerance: 0.005,
    region: {fillRule: 'nonzero', freeSpace: 'outside'}
  });
  const visibilityBuffers = makeVisibilityBuffers(
    resources,
    importView,
    data.probePositions.length / 2,
    visibilityEdgeCapacity
  );
  addKernelPass(graph, {
    id: 'publish-block-visibility',
    invocationCount: data.probePositions.length / 2,
    bindings: [
      {name: 'edges', view: visibility.outputs.edges, type: 'u32', access: 'read'},
      {name: 'status', view: visibility.status, type: 'u32', access: 'read'},
      {name: 'metrics', view: visibility.outputs.metrics, type: 'f32', access: 'read'},
      {
        name: 'classifications',
        view: visibility.outputs.viewpointClassifications,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'outEdges',
        view: importView(visibilityBuffers.edges, 'uint32x2', visibilityEdgeCapacity),
        type: 'u32',
        access: 'read_write'
      },
      {
        name: 'outStatus',
        view: importView(visibilityBuffers.status, 'uint32x4', 1),
        type: 'u32',
        access: 'read_write'
      },
      {
        name: 'outMetrics',
        view: importView(visibilityBuffers.metrics, 'float32x4', data.probePositions.length / 2),
        type: 'f32',
        access: 'read_write'
      },
      {
        name: 'outClassifications',
        view: importView(
          visibilityBuffers.classifications,
          'uint32',
          data.probePositions.length / 2
        ),
        type: 'u32',
        access: 'read_write'
      }
    ],
    body:
      'if (index == 0u) { for (var row = 0u; row < ' +
      visibilityEdgeCapacity +
      'u; row++) { outEdges[outEdgesOffset + 2u * row] = edges[edgesOffset + 2u * row]; outEdges[outEdgesOffset + 2u * row + 1u] = edges[edgesOffset + 2u * row + 1u]; } for (var word = 0u; word < 4u; word++) { outStatus[outStatusOffset + word] = status[statusOffset + word]; } } for (var component = 0u; component < 4u; component++) { outMetrics[outMetricsOffset + 4u * index + component] = metrics[metricsOffset + 4u * index + component]; } outClassifications[outClassificationsOffset + index] = classifications[classificationsOffset + index];'
  });

  for (const [id, source, destination, format, componentCount] of [
    ['choice', visibility.outputs.choice, visibilityBuffers.choice, 'float32', 1],
    [
      'normalized-choice',
      visibility.outputs.normalizedChoice,
      visibilityBuffers.normalizedChoice,
      'float32',
      1
    ],
    ['integration', visibility.outputs.integration, visibilityBuffers.integration, 'float32x4', 4],
    [
      'extended-integration',
      visibility.outputs.extendedIntegration,
      visibilityBuffers.extendedIntegration,
      'float32x4',
      4
    ]
  ] as const) {
    addKernelPass(graph, {
      id: `copy-block-visibility-${id}`,
      invocationCount: BLOCK_RELATIONSHIPS_PROBES * componentCount,
      bindings: [
        {name: 'source', view: source, type: 'f32', access: 'read'},
        {
          name: 'copiedOutput',
          view: importView(destination, format, BLOCK_RELATIONSHIPS_PROBES),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: 'copiedOutput[copiedOutputOffset + index] = source[sourceOffset + index];'
    });
  }
  const compiled = resources.track(
    await graph.compileAsync().catch(error => {
      resources.destroy();
      throw error;
    })
  );
  const coreGeneration = 0;
  let currentCoreGeneration = coreGeneration;
  let coreDirty = true;
  let coreSummary: CoreResult | null = null;
  let preparedTessellation: {segments: Float32Array; buildingIds: Uint32Array} | null = null;
  let tessellationInputFailure = '';
  let adjacencySummary: ReturnType<typeof decodeAdjacencySummary> | null = null;
  let adjacencyRequested = false;
  let initialCorePublished = false;
  let activeTessellation: TessellationVariant | null = null;
  let tessellationGraphGeneration = 0;
  let destroyed = false;
  let pendingAdjacencyBuild = false;
  let adjacencyFailure = '';
  let tessellationFailure = '';
  const publishBlockStatus = () =>
    ctx.setStatus([adjacencyFailure, tessellationFailure].filter(Boolean).join(' · '));
  const tessellationResources = new SpatialAnalysisResources(
    ctx.device,
    'block-relationships-tessellation-lifecycle'
  );
  const tessellationGraphSwap = tessellationResources.track(new GraphSwap());
  const tessellationCache = new Map<string, TessellationVariant>();
  const coreReadbackSources = [
    ...sharedBuffers.readbackSources,
    {buffer: shapeOrientation, size: BUILDING_COUNT * 4, kind: 'float32'},
    {buffer: boundsIds, size: BUILDING_COUNT * 4, kind: 'uint32'},
    {buffer: boundsCount, size: 4, kind: 'uint32'},
    {buffer: boundsRequiredCount, size: 4, kind: 'uint32'},
    {buffer: boundsOverflow, size: 4, kind: 'uint32'},
    ...bufferOutputs.flatMap(bufferOutput => makeBufferSurfaceReadbackSources(bufferOutput)),
    {buffer: visibilityBuffers.edges, size: visibilityEdgeCapacity * 8, kind: 'uint32'},
    {buffer: visibilityBuffers.status, size: 16, kind: 'uint32'},
    {
      buffer: visibilityBuffers.metrics,
      size: (data.probePositions.length / 2) * 16,
      kind: 'float32'
    },
    {
      buffer: visibilityBuffers.classifications,
      size: (data.probePositions.length / 2) * 4,
      kind: 'uint32'
    },
    {buffer: visibilityBuffers.choice, size: BLOCK_RELATIONSHIPS_PROBES * 4, kind: 'float32'},
    {
      buffer: visibilityBuffers.normalizedChoice,
      size: BLOCK_RELATIONSHIPS_PROBES * 4,
      kind: 'float32'
    },
    {buffer: visibilityBuffers.integration, size: BLOCK_RELATIONSHIPS_PROBES * 16, kind: 'float32'},
    {
      buffer: visibilityBuffers.extendedIntegration,
      size: BLOCK_RELATIONSHIPS_PROBES * 16,
      kind: 'float32'
    }
  ] as ReadbackSource[];
  let coreReader: SummaryReader;
  const requestCoreRead = (generation: number) => {
    coreReader?.release();
    coreReader = new SummaryReader(
      resources,
      `block-relationships-core-${generation}`,
      coreReadbackSources,
      bytes => {
        if (destroyed || generation !== currentCoreGeneration) return;
        coreSummary = decodeCoreSummary(bytes, coreReadbackSources, data, visibilityEdgeCapacity);
        if (!initialCorePublished) {
          publishCoreSummary(ctx, coreSummary);
          initialCorePublished = true;
        } else {
          publishBoundsSummary(ctx, coreSummary);
        }
        if (!preparedTessellation) {
          const tessellationInput = buildTessellationSegments(coreSummary.buffers);
          preparedTessellation = tessellationInput.input;
          tessellationInputFailure = tessellationInput.failure;
        }
        if (!adjacencyRequested) {
          adjacencyRequested = true;
          if (coreSummary.sharedStatus.complete && coreSummary.shared.leftLineIds.length > 0) {
            requestAdjacencyBuild(coreSummary.shared, coreSummary.orientation);
          } else {
            ctx.setReadout(
              'blockAdjacency',
              `Shared-boundary recipe withheld: ${formatCount(coreSummary.sharedStatus.writtenRuns)} / ${formatCount(coreSummary.sharedStatus.requiredRuns)} runs and ${formatCount(coreSummary.sharedStatus.writtenVertices)} / ${formatCount(coreSummary.sharedStatus.requiredVertices)} vertices; overflow ${formatCount(coreSummary.sharedStatus.overflow)}, uncertain ${formatCount(coreSummary.sharedStatus.uncertain)}.`
            );
          }
        }
        if (preparedTessellation && ctx.options.view === 'tessellation' && !activeTessellation) {
          requestTessellationBuild(ctx.options.sampleSpacing, Number(ctx.options.sampleCapacity));
        } else if (!preparedTessellation && !activeTessellation) {
          ctx.setReadout(
            'blockTessellation',
            `Per-building buffer preparation failed: ${tessellationInputFailure || 'the three source footprints did not produce complete rings.'}`
          );
        }
        ctx.requestLayers();
      }
    );
  };
  requestCoreRead(coreGeneration);

  const makeAdjacency = async (sharedRows: SharedPathRows, orientations: Float32Array) => {
    const adjacencyResources = new SpatialAnalysisResources(
      ctx.device,
      'block-relationships-adjacency'
    );
    try {
      const adjacencyGraph = new GPUCommandGraph<void>(ctx.device, {
        id: 'block-relationships-adjacency'
      });
      const adjacencyImporter = createGraphBufferImporter(adjacencyGraph);
      const sharedInput = makeSharedPathInput(adjacencyResources, adjacencyImporter, sharedRows);
      const lineOwnerBuffer = adjacencyResources.createBuffer(
        'adjacency-line-owners',
        data.lineBuildingIds
      );
      const orientationBuffer = adjacencyResources.createBuffer(
        'adjacency-orientations',
        orientations
      );
      const recipe = addBuildingAdjacencyRecipe(adjacencyGraph, {
        id: 'block-building-adjacency',
        sharedPaths: sharedInput,
        lineBuildingIds: adjacencyImporter(lineOwnerBuffer, 'uint32', data.lineBuildingIds.length),
        orientations: adjacencyImporter(orientationBuffer, 'float32', BUILDING_COUNT)
      });
      const outputBuffers = makeAdjacencyReadbackBuffers(
        adjacencyResources,
        adjacencyImporter,
        recipe.outputs.pairs.length,
        recipe.outputs.morphology.length,
        recipe.outputs.components.length
      );
      addKernelPass(adjacencyGraph, {
        id: 'publish-block-adjacency',
        invocationCount: recipe.outputs.pairs.length,
        bindings: [
          {name: 'pairs', view: recipe.outputs.pairs, type: 'f32', access: 'read'},
          {name: 'morphology', view: recipe.outputs.morphology, type: 'f32', access: 'read'},
          {name: 'components', view: recipe.outputs.components, type: 'u32', access: 'read'},
          {name: 'complete', view: recipe.status.complete, type: 'u32', access: 'read'},
          {
            name: 'outPairs',
            view: adjacencyImporter(outputBuffers.pairs, 'float32x4', recipe.outputs.pairs.length),
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'outMorphology',
            view: adjacencyImporter(
              outputBuffers.morphology,
              'float32x4',
              recipe.outputs.morphology.length
            ),
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'outComponents',
            view: adjacencyImporter(
              outputBuffers.components,
              'uint32',
              recipe.outputs.components.length
            ),
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'outComplete',
            view: adjacencyImporter(outputBuffers.complete, 'uint32', 1),
            type: 'u32',
            access: 'read_write'
          }
        ],
        body: `if (index < ${recipe.outputs.pairs.length}u) { for (var component = 0u; component < 4u; component++) { outPairs[outPairsOffset + 4u * index + component] = pairs[pairsOffset + 4u * index + component]; } } if (index < ${recipe.outputs.morphology.length}u) { for (var component = 0u; component < 4u; component++) { outMorphology[outMorphologyOffset + 4u * index + component] = morphology[morphologyOffset + 4u * index + component]; } } if (index < ${recipe.outputs.components.length}u) { outComponents[outComponentsOffset + index] = components[componentsOffset + index]; } if (index == 0u) { outComplete[outCompleteOffset] = complete[completeOffset]; }`
      });
      const graphResult = adjacencyResources.track(await adjacencyGraph.compileAsync());
      if (destroyed) {
        adjacencyResources.destroy();
        return;
      }
      const sources: ReadbackSource[] = [
        {buffer: outputBuffers.pairs, size: recipe.outputs.pairs.length * 16, kind: 'float32'},
        {
          buffer: outputBuffers.morphology,
          size: recipe.outputs.morphology.length * 16,
          kind: 'float32'
        },
        {
          buffer: outputBuffers.components,
          size: recipe.outputs.components.length * 4,
          kind: 'uint32'
        },
        {buffer: outputBuffers.complete, size: 4, kind: 'uint32'}
      ];
      const reader = new SummaryReader(
        adjacencyResources,
        'block-relationships-adjacency',
        sources,
        bytes => {
          if (destroyed || graphResult !== activeAdjacencyGraph) return;
          adjacencySummary = decodeAdjacencySummary(bytes);
          adjacencyFailure = '';
          publishBlockStatus();
          publishAdjacencySummary(ctx, adjacencySummary, data);
          ctx.requestLayers();
        }
      );
      const result = {
        compiled: graphResult,
        resources: adjacencyResources,
        reader,
        sources,
        dirty: true
      };
      activeAdjacencyGraph = graphResult;
      activeAdjacencyResources = result;
      ctx.requestLayers();
    } catch (error) {
      adjacencyResources.destroy();
      throw error;
    }
  };

  let activeAdjacencyGraph: CompiledGPUCommandGraph<void> | null = null;
  let activeAdjacencyResources: {
    compiled: CompiledGPUCommandGraph<void>;
    resources: SpatialAnalysisResources;
    reader: SummaryReader;
    sources: ReadbackSource[];
    dirty: boolean;
  } | null = null;
  const requestAdjacencyBuild = (sharedRows: SharedPathRows, orientations: Float32Array) => {
    if (pendingAdjacencyBuild || destroyed) return;
    pendingAdjacencyBuild = true;
    void makeAdjacency(sharedRows, orientations)
      .catch(error => {
        if (!destroyed) {
          adjacencySummary = null;
          activeAdjacencyGraph = null;
          if (activeAdjacencyResources) {
            activeAdjacencyResources.reader.release();
            activeAdjacencyResources.resources.destroy();
            activeAdjacencyResources = null;
          }
          ctx.setReadout('blockAdjacency', `Adjacency graph failed: ${String(error)}`);
          ctx.setChart('blockAdjacencyChart', null);
          adjacencyFailure = `Adjacency graph failed: ${String(error)}`;
          publishBlockStatus();
          ctx.requestLayers();
        }
      })
      .finally(() => {
        pendingAdjacencyBuild = false;
      });
  };

  const requestTessellationBuild = (spacing: string, sampleCapacity: number) => {
    const input = preparedTessellation;
    if (!input || destroyed) return;
    const key = `${spacing}-${sampleCapacity}`;
    const requestGeneration = ++tessellationGraphGeneration;
    let variantReference: TessellationVariant | null = null;
    activeTessellation = null;
    tessellationFailure = '';
    publishBlockStatus();
    ctx.setReadout(
      'blockTessellation',
      `GPU boundary readback is ready; compiling ${spacing} m samples with a ${sampleCapacity}-site capacity…`
    );
    ctx.setChart('blockTessellationChart', null);
    ctx.requestLayers();
    void tessellationGraphSwap
      .request({
        build: () =>
          buildTessellationVariant(
            ctx,
            data,
            input,
            spacing,
            sampleCapacity,
            requestGeneration,
            () =>
              !destroyed &&
              requestGeneration === tessellationGraphGeneration &&
              ctx.options.sampleSpacing === spacing &&
              Number(ctx.options.sampleCapacity) === sampleCapacity &&
              activeTessellation === variantReference,
            () => {
              tessellationFailure = '';
              publishBlockStatus();
            }
          ),
        commit: variant => {
          if (
            destroyed ||
            requestGeneration !== tessellationGraphGeneration ||
            ctx.options.sampleSpacing !== spacing ||
            Number(ctx.options.sampleCapacity) !== sampleCapacity
          ) {
            if (tessellationCache.get(key) !== variant) {
              variant.reader.release();
              variant.resources.destroy();
            }
            return;
          }
          const existing = tessellationCache.get(key);
          if (existing && existing !== variant) destroyTessellationVariant(existing);
          tessellationCache.set(key, variant);
          activeTessellation = variant;
          variantReference = variant;
          variant.dirty = true;
          ctx.requestLayers();
        },
        discard: variant => {
          if (tessellationCache.get(key) !== variant) {
            variant.reader.release();
            variant.resources.destroy();
          }
        }
      })
      .catch(error => {
        if (!destroyed && requestGeneration === tessellationGraphGeneration) {
          activeTessellation = null;
          ctx.setReadout('blockTessellation', `Tessellation graph failed: ${String(error)}`);
          ctx.setChart('blockTessellationChart', null);
          tessellationFailure = `Tessellation graph failed: ${String(error)}`;
          publishBlockStatus();
          ctx.requestLayers();
        }
      });
  };

  const getLayers = (): Layer[] => {
    const layers: Layer[] = [
      new PolygonLayer({
        id: 'block-source-footprints',
        data: data.sourcePaths.map((path, index) => ({path, index})),
        getPolygon: row => row.path,
        getFillColor: row => BUILDING_COLORS[row.index],
        getLineColor: [55, 68, 69, 220],
        getLineWidth: 1.2,
        lineWidthUnits: 'pixels',
        lineWidthMinPixels: 1,
        filled: true,
        stroked: true,
        pickable: false
      })
    ];
    if (ctx.options.view === 'bounds') {
      const halfExtent = ctx.options.queryHalfExtent;
      const queryRing = [
        data.projection.unproject(-halfExtent, -halfExtent),
        data.projection.unproject(halfExtent, -halfExtent),
        data.projection.unproject(halfExtent, halfExtent),
        data.projection.unproject(-halfExtent, halfExtent),
        data.projection.unproject(-halfExtent, -halfExtent)
      ];
      layers.push(
        new PathLayer({
          id: 'block-bounds-query',
          data: [{path: queryRing}],
          getPath: row => row.path,
          getColor: [202, 114, 70, 255],
          getWidth: 2.2,
          widthUnits: 'pixels',
          widthMinPixels: 2,
          pickable: false
        })
      );
      if (coreSummary && coreSummary.bounds.overflow === 0) {
        const candidateIndices = new Set(
          Array.from(coreSummary.bounds.ids.slice(0, coreSummary.bounds.count))
        );
        layers.push(
          new PolygonLayer({
            id: 'block-bounds-candidates',
            data: data.sourcePaths
              .map((path, index) => ({path, index}))
              .filter(row => candidateIndices.has(row.index)),
            getPolygon: row => row.path,
            getFillColor: [247, 187, 115, 105],
            getLineColor: [198, 109, 58, 255],
            getLineWidth: 3,
            lineWidthUnits: 'pixels',
            lineWidthMinPixels: 2,
            filled: true,
            stroked: true,
            pickable: false
          })
        );
      }
    } else if (ctx.options.view === 'adjacency' && coreSummary) {
      if (coreSummary.sharedStatus.complete && coreSummary.shared.positions.length > 0) {
        layers.push(
          new PathLayer({
            id: 'block-shared-boundaries',
            data: unpackSharedPaths(coreSummary.shared, data),
            getPath: row => row.path,
            getColor: [221, 126, 61, 255],
            getWidth: 5,
            widthUnits: 'pixels',
            widthMinPixels: 4,
            capRounded: true,
            jointRounded: true,
            pickable: false
          })
        );
      }
      if (adjacencySummary?.complete === 1) {
        const componentData = data.sourcePaths.map((path, index) => ({
          path,
          index,
          component: adjacencySummary!.components[index]
        }));
        layers.push(
          new ScatterplotLayer({
            id: 'block-component-centres',
            data: componentData,
            getPosition: row => ringCentroid(data.sourcePaths[row.index]),
            getFillColor: row => BUILDING_COLORS[row.component % BUILDING_COLORS.length],
            getRadius: 5,
            radiusUnits: 'pixels',
            radiusMinPixels: 5,
            stroked: true,
            getLineColor: [255, 251, 242, 255],
            lineWidthMinPixels: 1,
            pickable: false
          })
        );
      }
    } else if (
      ctx.options.view === 'tessellation' &&
      activeTessellation?.result &&
      activeTessellation.spacing === ctx.options.sampleSpacing &&
      activeTessellation.sampleCapacity === Number(ctx.options.sampleCapacity)
    ) {
      const result = activeTessellation.result;
      if (result.status === 'complete') {
        layers.push(
          new PolygonLayer({
            id: 'block-tessellation-cells',
            data: result.polygons,
            getPolygon: row => row.rings,
            getFillColor: row => BUILDING_COLORS[row.buildingIndex],
            getLineColor: [53, 67, 70, 235],
            getLineWidth: 1.2,
            lineWidthUnits: 'pixels',
            lineWidthMinPixels: 1,
            filled: true,
            stroked: true,
            pickable: false
          })
        );
      }
    } else if (ctx.options.view === 'visibility' && coreSummary) {
      const result = coreSummary.visibility;
      const enclosure = Array.from({length: data.enclosure.length / 2}, (_, index) =>
        data.projection.unproject(data.enclosure[index * 2], data.enclosure[index * 2 + 1])
      );
      enclosure.push(enclosure[0]);
      layers.push(
        new PathLayer({
          id: 'block-synthetic-visibility-enclosure',
          data: [{path: enclosure}],
          getPath: row => row.path,
          getColor: [105, 125, 121, 170],
          getWidth: 1.2,
          widthUnits: 'pixels',
          widthMinPixels: 1,
          dashJustified: true,
          pickable: false
        })
      );
      if (result.invalidCount === 0) {
        const points = unpackPoints(data.probePositions, data.projection);
        const lines = Array.from({length: result.count}, (_, index) => ({
          path: [points[result.edges[index * 2]], points[result.edges[index * 2 + 1]]]
        }));
        layers.push(
          new PathLayer({
            id: 'block-probe-visibility-edges',
            data: lines,
            getPath: row => row.path,
            getColor: result.overflow === 0 ? [46, 129, 145, 95] : [190, 112, 67, 135],
            getWidth: 1.2,
            widthUnits: 'pixels',
            widthMinPixels: 1,
            pickable: false
          })
        );
        layers.push(
          new ScatterplotLayer({
            id: 'block-observer-probes',
            data: points.map((position, index) => ({position, index})),
            getPosition: row => row.position,
            getFillColor: [217, 115, 63, 255],
            getRadius: 5,
            radiusUnits: 'pixels',
            radiusMinPixels: 4,
            stroked: true,
            getLineColor: [255, 250, 235, 255],
            lineWidthMinPixels: 1,
            pickable: false
          })
        );
      }
    }
    return layers;
  };

  ctx.setReadout(
    'blockSource',
    `${dataset.count.toLocaleString()} archived source features · selected IDs ${BLOCK_RELATIONSHIP_BUILDINGS.map(building => building.id).join(', ')}.`
  );
  ctx.setReadout('blockBounds', 'GPU bounds filter pending.');
  ctx.setReadout('blockAdjacency', 'GPU shared paths and adjacency are pending.');
  ctx.setReadout('blockTessellation', 'GPU boundary shrink and tessellation are pending.');
  ctx.setReadout('blockVisibility', 'Synthetic observer visibility is pending.');
  ctx.setChart('blockAdjacencyChart', null);
  ctx.setChart('blockVisibilityChart', null);
  ctx.setChart('blockTessellationChart', null);

  return {
    getCompiledGraphs: () => [
      compiled,
      ...(activeAdjacencyResources ? [activeAdjacencyResources.compiled] : []),
      ...(activeTessellation ? [activeTessellation.compiled] : [])
    ],
    setOption(id) {
      if (id === 'queryHalfExtent') {
        boundsParameters.write(makeQueryBounds(ctx.options.queryHalfExtent));
        currentCoreGeneration++;
        coreDirty = true;
        coreSummary = coreSummary
          ? {
              ...coreSummary,
              bounds: {ids: new Uint32Array(0), count: 0, requiredCount: 0, overflow: 0}
            }
          : null;
        ctx.setReadout('blockBounds', 'GPU bounds query updated; awaiting matching filter output.');
        requestCoreRead(currentCoreGeneration);
        ctx.requestLayers();
      } else if (id === 'sampleSpacing') {
        requestTessellationBuild(ctx.options.sampleSpacing, Number(ctx.options.sampleCapacity));
      } else if (id === 'sampleCapacity') {
        requestTessellationBuild(ctx.options.sampleSpacing, Number(ctx.options.sampleCapacity));
      } else if (id === 'view') {
        if (ctx.options.view === 'tessellation') {
          requestTessellationBuild(ctx.options.sampleSpacing, Number(ctx.options.sampleCapacity));
        } else if (ctx.options.view === 'visibility' && coreSummary) {
          publishVisibilitySummary(ctx, coreSummary.visibility, ctx.options.visibilityMetric);
        }
        ctx.requestLayers();
      } else if (id === 'visibilityMetric' && coreSummary) {
        publishVisibilitySummary(ctx, coreSummary.visibility, ctx.options.visibilityMetric);
      }
    },
    encode(commandEncoder) {
      if (destroyed) return;
      if (coreDirty) {
        compiled.encode(commandEncoder, {parameters: undefined});
        coreDirty = false;
        coreReader.request(commandEncoder);
      } else {
        coreReader.flush(commandEncoder);
      }
      if (activeAdjacencyResources?.dirty) {
        activeAdjacencyResources.compiled.encode(commandEncoder, {parameters: undefined});
        activeAdjacencyResources.dirty = false;
        activeAdjacencyResources.reader.request(commandEncoder);
      } else {
        activeAdjacencyResources?.reader.flush(commandEncoder);
      }
      if (activeTessellation?.dirty) {
        activeTessellation.compiled.encode(commandEncoder, {parameters: undefined});
        activeTessellation.dirty = false;
        activeTessellation.reader.request(commandEncoder);
      } else {
        activeTessellation?.reader.flush(commandEncoder);
      }
    },
    getLayers,
    destroy() {
      destroyed = true;
      coreReader.release();
      activeAdjacencyResources?.reader.release();
      activeAdjacencyResources?.resources.destroy();
      tessellationGraphSwap.destroy();
      for (const variant of tessellationCache.values()) destroyTessellationVariant(variant);
      tessellationResources.destroy();
      resources.destroy();
    }
  };
}

// Helpers below keep graph views and application-owned readback buffers distinct.
function makeSharedPathBuffers(resources: SpatialAnalysisResources, importView: ImportView) {
  const positions = resources.createBuffer('shared-path-positions', SHARED_VERTEX_CAPACITY * 8);
  const offsets = resources.createBuffer('shared-path-offsets', (SHARED_RUN_CAPACITY + 1) * 4);
  const leftLineIds = resources.createBuffer('shared-path-left-line-ids', SHARED_RUN_CAPACITY * 4);
  const rightLineIds = resources.createBuffer(
    'shared-path-right-line-ids',
    SHARED_RUN_CAPACITY * 4
  );
  const forward = resources.createBuffer('shared-path-forward', SHARED_RUN_CAPACITY * 4);
  const count = resources.createBuffer('shared-path-count', 4);
  const vertexCount = resources.createBuffer('shared-path-vertex-count', 4);
  const requiredCount = resources.createBuffer('shared-path-required-count', 4);
  const requiredVertexCount = resources.createBuffer('shared-path-required-vertex-count', 4);
  const overflow = resources.createBuffer('shared-path-overflow', 4);
  const uncertainCount = resources.createBuffer('shared-path-uncertain', 4);
  return {
    output: {
      runs: {
        positions: importView(positions, 'float32x2', SHARED_VERTEX_CAPACITY),
        offsets: importView(offsets, 'uint32', SHARED_RUN_CAPACITY + 1),
        leftLineIds: importView(leftLineIds, 'uint32', SHARED_RUN_CAPACITY),
        rightLineIds: importView(rightLineIds, 'uint32', SHARED_RUN_CAPACITY),
        forward: importView(forward, 'uint32', SHARED_RUN_CAPACITY)
      },
      status: {
        count: importView(count, 'uint32', 1),
        vertexCount: importView(vertexCount, 'uint32', 1),
        requiredCount: importView(requiredCount, 'uint32', 1),
        requiredVertexCount: importView(requiredVertexCount, 'uint32', 1),
        overflow: importView(overflow, 'uint32', 1),
        uncertainCount: importView(uncertainCount, 'uint32', 1)
      }
    },
    buffers: {
      positions,
      offsets,
      leftLineIds,
      rightLineIds,
      forward,
      count,
      vertexCount,
      requiredCount,
      requiredVertexCount,
      overflow,
      uncertainCount
    },
    readbackSources: [
      {buffer: count, size: 4, kind: 'uint32'},
      {buffer: vertexCount, size: 4, kind: 'uint32'},
      {buffer: requiredCount, size: 4, kind: 'uint32'},
      {buffer: requiredVertexCount, size: 4, kind: 'uint32'},
      {buffer: overflow, size: 4, kind: 'uint32'},
      {buffer: uncertainCount, size: 4, kind: 'uint32'},
      {buffer: leftLineIds, size: SHARED_RUN_CAPACITY * 4, kind: 'uint32'},
      {buffer: rightLineIds, size: SHARED_RUN_CAPACITY * 4, kind: 'uint32'},
      {buffer: forward, size: SHARED_RUN_CAPACITY * 4, kind: 'uint32'},
      {buffer: offsets, size: (SHARED_RUN_CAPACITY + 1) * 4, kind: 'uint32'},
      {buffer: positions, size: SHARED_VERTEX_CAPACITY * 8, kind: 'float32'}
    ] as ReadbackSource[]
  };
}

function makeSingleBuildingBufferGeometry(
  resources: SpatialAnalysisResources,
  importView: ImportView,
  data: BlockRelationshipsData,
  buildingIndex: number
) {
  const building = BLOCK_RELATIONSHIP_BUILDINGS[buildingIndex];
  const firstPolygon = data.featureOffsets[buildingIndex];
  const lastPolygon = data.featureOffsets[buildingIndex + 1];
  const firstRing = data.polygonOffsets[firstPolygon];
  const lastRing = data.polygonOffsets[lastPolygon];
  const firstVertex = data.ringOffsets[firstRing];
  const lastVertex = data.ringOffsets[lastRing];
  const positionsBuffer = resources.createBuffer(
    `shrink-source-${building.id}-positions`,
    data.positions.slice(firstVertex * 2, lastVertex * 2)
  );
  const ringOffsetsBuffer = resources.createBuffer(
    `shrink-source-${building.id}-ring-offsets`,
    Uint32Array.from(
      data.ringOffsets.slice(firstRing, lastRing + 1),
      offset => offset - firstVertex
    )
  );
  const polygonOffsetsBuffer = resources.createBuffer(
    `shrink-source-${building.id}-polygon-offsets`,
    Uint32Array.from(
      data.polygonOffsets.slice(firstPolygon, lastPolygon + 1),
      offset => offset - firstRing
    )
  );
  const featureOffsetsBuffer = resources.createBuffer(
    `shrink-source-${building.id}-feature-offsets`,
    Uint32Array.of(0, lastPolygon - firstPolygon)
  );
  const sourceIdsBuffer = resources.createBuffer(
    `shrink-source-${building.id}-source-ids`,
    Uint32Array.of(building.id)
  );
  return {
    kind: 'polygons' as const,
    positions: importView(positionsBuffer, 'float32x2', lastVertex - firstVertex),
    ringOffsets: importView(ringOffsetsBuffer, 'uint32', lastRing - firstRing + 1),
    polygonOffsets: importView(polygonOffsetsBuffer, 'uint32', lastPolygon - firstPolygon + 1),
    featureOffsets: importView(featureOffsetsBuffer, 'uint32', 2),
    sourceIds: importView(sourceIdsBuffer, 'uint32', 1)
  };
}

function makeBufferSurfaceOutput(
  resources: SpatialAnalysisResources,
  importView: ImportView,
  prefix: string
) {
  const positions = resources.createBuffer(`${prefix}-positions`, BUFFER_VERTEX_CAPACITY * 8);
  const featureOffsets = resources.createBuffer(
    `${prefix}-feature-offsets`,
    (BUFFER_RING_CAPACITY + 1) * 4
  );
  const polygonOffsets = resources.createBuffer(
    `${prefix}-polygon-offsets`,
    (BUFFER_RING_CAPACITY + 1) * 4
  );
  const ringOffsets = resources.createBuffer(
    `${prefix}-ring-offsets`,
    (BUFFER_RING_CAPACITY + 1) * 4
  );
  const sourceIds = resources.createBuffer(`${prefix}-source-ids`, BUFFER_RING_CAPACITY * 4);
  const makeStatus = (name: string) => ({
    count: resources.createBuffer(`${prefix}-${name}-count`, 4),
    requiredCount: resources.createBuffer(`${prefix}-${name}-required-count`, 4),
    overflow: resources.createBuffer(`${prefix}-${name}-overflow`, 4),
    candidateOverflow: resources.createBuffer(`${prefix}-${name}-candidate-overflow`, 4),
    uncertainCount: resources.createBuffer(`${prefix}-${name}-uncertain`, 4)
  });
  const geometryStatus = makeStatus('geometry');
  const boundaryStatus = makeStatus('boundary');
  const boundaryEndpoints = resources.createBuffer(
    `${prefix}-boundary-endpoints`,
    BUFFER_SEGMENT_CAPACITY * 16
  );
  const boundaryOperands = resources.createBuffer(
    `${prefix}-boundary-operands`,
    BUFFER_SEGMENT_CAPACITY * 4
  );
  const boundarySourceIds = resources.createBuffer(
    `${prefix}-boundary-source-ids`,
    BUFFER_SEGMENT_CAPACITY * 4
  );
  const output: GPUPolygonOverlayOutput = {
    geometry: {
      kind: 'polygons',
      positions: importView(positions, 'float32x2', BUFFER_VERTEX_CAPACITY),
      featureOffsets: importView(featureOffsets, 'uint32', BUFFER_RING_CAPACITY + 1),
      polygonOffsets: importView(polygonOffsets, 'uint32', BUFFER_RING_CAPACITY + 1),
      ringOffsets: importView(ringOffsets, 'uint32', BUFFER_RING_CAPACITY + 1),
      sourceIds: importView(sourceIds, 'uint32', BUFFER_RING_CAPACITY)
    },
    status: {
      count: importView(geometryStatus.count, 'uint32', 1),
      requiredCount: importView(geometryStatus.requiredCount, 'uint32', 1),
      overflow: importView(geometryStatus.overflow, 'uint32', 1),
      candidateOverflow: importView(geometryStatus.candidateOverflow, 'uint32', 1),
      uncertainCount: importView(geometryStatus.uncertainCount, 'uint32', 1)
    },
    boundary: {
      endpoints: importView(boundaryEndpoints, 'float32x4', BUFFER_SEGMENT_CAPACITY),
      operandIds: importView(boundaryOperands, 'uint32', BUFFER_SEGMENT_CAPACITY),
      sourceFeatureIds: importView(boundarySourceIds, 'uint32', BUFFER_SEGMENT_CAPACITY),
      status: {
        count: importView(boundaryStatus.count, 'uint32', 1),
        requiredCount: importView(boundaryStatus.requiredCount, 'uint32', 1),
        overflow: importView(boundaryStatus.overflow, 'uint32', 1),
        candidateOverflow: importView(boundaryStatus.candidateOverflow, 'uint32', 1)
      }
    }
  };
  return {
    output,
    positions,
    featureOffsets,
    polygonOffsets,
    ringOffsets,
    sourceIds,
    count: geometryStatus.count,
    requiredCount: geometryStatus.requiredCount,
    overflow: geometryStatus.overflow,
    candidateOverflow: geometryStatus.candidateOverflow,
    uncertainCount: geometryStatus.uncertainCount,
    boundaryCount: boundaryStatus.count,
    boundaryRequiredCount: boundaryStatus.requiredCount,
    boundaryOverflow: boundaryStatus.overflow,
    boundaryCandidateOverflow: boundaryStatus.candidateOverflow
  };
}

function makeBufferSurfaceReadbackSources(
  bufferOutput: ReturnType<typeof makeBufferSurfaceOutput>
) {
  return [
    {buffer: bufferOutput.positions, size: BUFFER_VERTEX_CAPACITY * 8, kind: 'float32' as const},
    {
      buffer: bufferOutput.featureOffsets,
      size: (BUFFER_RING_CAPACITY + 1) * 4,
      kind: 'uint32' as const
    },
    {
      buffer: bufferOutput.polygonOffsets,
      size: (BUFFER_RING_CAPACITY + 1) * 4,
      kind: 'uint32' as const
    },
    {
      buffer: bufferOutput.ringOffsets,
      size: (BUFFER_RING_CAPACITY + 1) * 4,
      kind: 'uint32' as const
    },
    {buffer: bufferOutput.count, size: 4, kind: 'uint32' as const},
    {buffer: bufferOutput.requiredCount, size: 4, kind: 'uint32' as const},
    {buffer: bufferOutput.overflow, size: 4, kind: 'uint32' as const},
    {buffer: bufferOutput.candidateOverflow, size: 4, kind: 'uint32' as const},
    {buffer: bufferOutput.uncertainCount, size: 4, kind: 'uint32' as const},
    {buffer: bufferOutput.boundaryCount, size: 4, kind: 'uint32' as const},
    {buffer: bufferOutput.boundaryRequiredCount, size: 4, kind: 'uint32' as const},
    {buffer: bufferOutput.boundaryOverflow, size: 4, kind: 'uint32' as const},
    {
      buffer: bufferOutput.boundaryCandidateOverflow,
      size: 4,
      kind: 'uint32' as const
    }
  ] satisfies ReadbackSource[];
}

function makeVisibilityBuffers(
  resources: SpatialAnalysisResources,
  importView: ImportView,
  probeCount: number,
  edgeCapacity: number
) {
  const edges = resources.createBuffer('visibility-edges', edgeCapacity * 8);
  const status = resources.createBuffer('visibility-status', 16);
  const metrics = resources.createBuffer('visibility-metrics', probeCount * 16);
  const classifications = resources.createBuffer('visibility-classifications', probeCount * 4);
  const choice = resources.createBuffer('visibility-choice', probeCount * 4);
  const normalizedChoice = resources.createBuffer('visibility-normalized-choice', probeCount * 4);
  const integration = resources.createBuffer('visibility-integration', probeCount * 16);
  const extendedIntegration = resources.createBuffer(
    'visibility-extended-integration',
    probeCount * 16
  );
  return {
    edges,
    status,
    metrics,
    classifications,
    choice,
    normalizedChoice,
    integration,
    extendedIntegration
  };
}

function makeAdjacencyReadbackBuffers(
  resources: SpatialAnalysisResources,
  importView: ImportView,
  pairLength: number,
  morphologyLength: number,
  componentLength: number
) {
  return {
    pairs: resources.createBuffer('adjacency-pairs', pairLength * 16),
    morphology: resources.createBuffer('adjacency-morphology', morphologyLength * 16),
    components: resources.createBuffer('adjacency-components', componentLength * 4),
    complete: resources.createBuffer('adjacency-complete', 4)
  };
}

function makeSharedPathInput(
  resources: SpatialAnalysisResources,
  importView: ImportView,
  rows: SharedPathRows
) {
  const positionsBuffer = resources.createBuffer('adjacency-shared-positions', rows.positions);
  const offsetsBuffer = resources.createBuffer('adjacency-shared-offsets', rows.offsets);
  const leftBuffer = resources.createBuffer('adjacency-shared-left', rows.leftLineIds);
  const rightBuffer = resources.createBuffer('adjacency-shared-right', rows.rightLineIds);
  const forwardBuffer = resources.createBuffer('adjacency-shared-forward', rows.forward);
  const countBuffer = resources.createBuffer(
    'adjacency-shared-count',
    Uint32Array.of(rows.leftLineIds.length)
  );
  const overflowBuffer = resources.createBuffer('adjacency-shared-overflow', Uint32Array.of(0));
  const uncertainBuffer = resources.createBuffer('adjacency-shared-uncertain', Uint32Array.of(0));
  return {
    runs: {
      positions: importView(positionsBuffer, 'float32x2', rows.positions.length / 2),
      offsets: importView(offsetsBuffer, 'uint32', rows.offsets.length),
      leftLineIds: importView(leftBuffer, 'uint32', rows.leftLineIds.length),
      rightLineIds: importView(rightBuffer, 'uint32', rows.rightLineIds.length),
      forward: importView(forwardBuffer, 'uint32', rows.forward.length)
    },
    status: {
      count: importView(countBuffer, 'uint32', 1),
      overflow: importView(overflowBuffer, 'uint32', 1),
      uncertainCount: importView(uncertainBuffer, 'uint32', 1)
    }
  };
}

function makeQueryBounds(halfExtent: number): Float32Array {
  return getGPUBoundsFilterParameterValues({
    minX: -halfExtent,
    minY: -halfExtent,
    maxX: halfExtent,
    maxY: halfExtent
  });
}

function decodeCoreSummary(
  bytes: ArrayBuffer,
  sources: ReadbackSource[],
  data: BlockRelationshipsData,
  visibilityEdgeCapacity: number
): CoreResult {
  let offset = 0;
  const nextUint = (count: number) => {
    const values = new Uint32Array(bytes.slice(offset, offset + count * 4));
    offset += count * 4;
    return values;
  };
  const nextFloat = (count: number) => {
    const values = new Float32Array(bytes.slice(offset, offset + count * 4));
    offset += count * 4;
    return values;
  };
  const sharedCount = nextUint(1)[0];
  const sharedVertexCount = nextUint(1)[0];
  const sharedRequiredRuns = nextUint(1)[0];
  const sharedRequiredVertices = nextUint(1)[0];
  const sharedOverflow = nextUint(1)[0];
  const sharedUncertain = nextUint(1)[0];
  const rawLeft = nextUint(SHARED_RUN_CAPACITY);
  const rawRight = nextUint(SHARED_RUN_CAPACITY);
  const rawForward = nextUint(SHARED_RUN_CAPACITY);
  const rawOffsets = nextUint(SHARED_RUN_CAPACITY + 1);
  const rawPositions = nextFloat(SHARED_VERTEX_CAPACITY * 2);
  const leftLineIds: number[] = [];
  const rightLineIds: number[] = [];
  const forward: number[] = [];
  const sharedOffsets = [0];
  const sharedPositions: number[] = [];
  let sharedTopologyComplete =
    sharedCount <= SHARED_RUN_CAPACITY &&
    sharedVertexCount <= SHARED_VERTEX_CAPACITY &&
    sharedCount === sharedRequiredRuns &&
    sharedVertexCount === sharedRequiredVertices &&
    rawOffsets[0] === 0 &&
    rawOffsets[sharedCount] === sharedVertexCount;
  for (let run = 0; run < Math.min(sharedCount, SHARED_RUN_CAPACITY); run++) {
    const leftLine = rawLeft[run];
    const rightLine = rawRight[run];
    const start = rawOffsets[run];
    const end = rawOffsets[run + 1];
    if (
      leftLine >= data.lineBuildingIds.length ||
      rightLine >= data.lineBuildingIds.length ||
      start > end ||
      end > sharedVertexCount ||
      end - start < 2
    ) {
      sharedTopologyComplete = false;
      continue;
    }
    for (let vertex = start; vertex < end; vertex++) {
      if (
        !Number.isFinite(rawPositions[vertex * 2]) ||
        !Number.isFinite(rawPositions[vertex * 2 + 1])
      ) {
        sharedTopologyComplete = false;
      }
    }
    if (data.lineBuildingIds[leftLine] === data.lineBuildingIds[rightLine] || leftLine >= rightLine)
      continue;
    leftLineIds.push(leftLine);
    rightLineIds.push(rightLine);
    forward.push(rawForward[run]);
    for (let vertex = start; vertex < end; vertex++)
      sharedPositions.push(rawPositions[vertex * 2], rawPositions[vertex * 2 + 1]);
    sharedOffsets.push(sharedPositions.length / 2);
  }
  const orientation = nextFloat(BUILDING_COUNT);
  const boundsIds = nextUint(BUILDING_COUNT);
  const boundsCount = nextUint(1)[0];
  const boundsRequiredCount = nextUint(1)[0];
  const boundsOverflow = nextUint(1)[0];
  const buffers = Array.from({length: BUILDING_COUNT}, () => {
    const bufferPositions = nextFloat(BUFFER_VERTEX_CAPACITY * 2);
    const bufferFeatureOffsets = nextUint(BUFFER_RING_CAPACITY + 1);
    const bufferPolygonOffsets = nextUint(BUFFER_RING_CAPACITY + 1);
    const bufferRingOffsets = nextUint(BUFFER_RING_CAPACITY + 1);
    const bufferCount = nextUint(1)[0];
    const bufferRequiredCount = nextUint(1)[0];
    const bufferOverflow = nextUint(1)[0];
    const bufferCandidateOverflow = nextUint(1)[0];
    const bufferUncertainCount = nextUint(1)[0];
    const bufferBoundaryCount = nextUint(1)[0];
    const bufferBoundaryRequiredCount = nextUint(1)[0];
    const bufferBoundaryOverflow = nextUint(1)[0];
    const bufferBoundaryCandidateOverflow = nextUint(1)[0];
    const actualRings = countGeometryRings(bufferFeatureOffsets, bufferPolygonOffsets, bufferCount);
    return {
      positions: bufferPositions,
      featureOffsets: bufferFeatureOffsets,
      polygonOffsets: bufferPolygonOffsets,
      ringOffsets: bufferRingOffsets,
      count: bufferCount,
      requiredRings: bufferRequiredCount,
      overflow: bufferOverflow,
      candidateOverflow: bufferCandidateOverflow,
      uncertainCount: bufferUncertainCount,
      actualRings,
      boundaryCount: bufferBoundaryCount,
      boundaryRequiredCount: bufferBoundaryRequiredCount,
      boundaryOverflow: bufferBoundaryOverflow,
      boundaryCandidateOverflow: bufferBoundaryCandidateOverflow,
      complete:
        bufferOverflow === 0 &&
        bufferCandidateOverflow === 0 &&
        bufferUncertainCount === 0 &&
        bufferCount === 1 &&
        bufferRequiredCount <= BUFFER_RING_CAPACITY &&
        bufferBoundaryCount === bufferBoundaryRequiredCount &&
        bufferBoundaryOverflow === 0 &&
        bufferBoundaryCandidateOverflow === 0 &&
        actualRings === bufferRequiredCount &&
        validGeometryOffsets(
          bufferFeatureOffsets,
          bufferPolygonOffsets,
          bufferRingOffsets,
          bufferPositions,
          bufferCount,
          1
        )
    };
  });
  const visibilityEdges = nextUint(visibilityEdgeCapacity * 2);
  const visibilityStatus = nextUint(4);
  const visibilityMetrics = nextFloat(BLOCK_RELATIONSHIPS_PROBES * 4);
  const visibilityClassifications = nextUint(BLOCK_RELATIONSHIPS_PROBES);
  const visibilityChoice = nextFloat(BLOCK_RELATIONSHIPS_PROBES);
  const visibilityNormalizedChoice = nextFloat(BLOCK_RELATIONSHIPS_PROBES);
  const visibilityIntegration = nextFloat(BLOCK_RELATIONSHIPS_PROBES * 4);
  const visibilityExtendedIntegration = nextFloat(BLOCK_RELATIONSHIPS_PROBES * 4);
  if (offset !== bytes.byteLength)
    throw new Error('Block relationship summary buffers did not decode exactly.');
  void sources;
  return {
    shared: {
      positions: Float32Array.from(sharedPositions),
      offsets: Uint32Array.from(sharedOffsets),
      leftLineIds: Uint32Array.from(leftLineIds),
      rightLineIds: Uint32Array.from(rightLineIds),
      forward: Uint32Array.from(forward),
      lineBuildingIds: data.lineBuildingIds
    },
    sharedStatus: {
      requiredRuns: sharedRequiredRuns,
      requiredVertices: sharedRequiredVertices,
      writtenRuns: sharedCount,
      writtenVertices: sharedVertexCount,
      overflow: sharedOverflow,
      uncertain: sharedUncertain,
      complete: sharedTopologyComplete && sharedOverflow === 0 && sharedUncertain === 0
    },
    orientation,
    bounds: {
      ids: boundsIds,
      count: Math.min(boundsCount, BUILDING_COUNT),
      requiredCount: boundsRequiredCount,
      overflow: boundsOverflow
    },
    buffers,
    visibility: {
      edges: visibilityEdges,
      count: Math.min(visibilityStatus[0], visibilityEdgeCapacity),
      requiredCount: visibilityStatus[1],
      overflow: visibilityStatus[2],
      invalidCount: visibilityStatus[3],
      metrics: visibilityMetrics,
      classifications: visibilityClassifications,
      choice: visibilityChoice,
      normalizedChoice: visibilityNormalizedChoice,
      integration: visibilityIntegration,
      extendedIntegration: visibilityExtendedIntegration
    }
  };
}

function countGeometryRings(
  featureOffsets: Uint32Array,
  polygonOffsets: Uint32Array,
  featureCount: number
): number {
  if (featureCount > BUFFER_RING_CAPACITY) return BUFFER_RING_CAPACITY + 1;
  const polygonCount = featureOffsets[featureCount] ?? BUFFER_RING_CAPACITY + 1;
  if (polygonCount > BUFFER_RING_CAPACITY) return BUFFER_RING_CAPACITY + 1;
  return polygonOffsets[polygonCount] ?? BUFFER_RING_CAPACITY + 1;
}

function validGeometryOffsets(
  featureOffsets: Uint32Array,
  polygonOffsets: Uint32Array,
  ringOffsets: Uint32Array,
  positions: Float32Array,
  featureCount: number,
  expectedFeatureCount: number
): boolean {
  if (
    featureCount !== expectedFeatureCount ||
    featureCount > BUFFER_RING_CAPACITY ||
    featureOffsets[0] !== 0 ||
    polygonOffsets[0] !== 0 ||
    ringOffsets[0] !== 0
  ) {
    return false;
  }
  let previousPolygon = 0;
  for (let feature = 0; feature < featureCount; feature++) {
    const nextPolygon = featureOffsets[feature + 1];
    if (nextPolygon < previousPolygon || nextPolygon > BUFFER_RING_CAPACITY) return false;
    previousPolygon = nextPolygon;
  }
  const polygonCount = previousPolygon;
  let previousRing = 0;
  for (let polygon = 0; polygon < polygonCount; polygon++) {
    const nextRing = polygonOffsets[polygon + 1];
    if (nextRing < previousRing || nextRing > BUFFER_RING_CAPACITY) return false;
    previousRing = nextRing;
  }
  let previousVertex = 0;
  for (let ring = 0; ring < previousRing; ring++) {
    const nextVertex = ringOffsets[ring + 1];
    if (nextVertex < previousVertex || nextVertex > BUFFER_VERTEX_CAPACITY) return false;
    for (let vertex = previousVertex; vertex < nextVertex; vertex++) {
      if (!Number.isFinite(positions[vertex * 2]) || !Number.isFinite(positions[vertex * 2 + 1]))
        return false;
    }
    previousVertex = nextVertex;
  }
  return previousVertex > 0;
}

function buildTessellationSegments(buffers: BufferGeometryResult[]): {
  input: {segments: Float32Array; buildingIds: Uint32Array} | null;
  failure: string;
} {
  if (buffers.length !== BLOCK_RELATIONSHIP_BUILDINGS.length) {
    return {input: null, failure: 'the independent buffer outputs are incomplete.'};
  }
  const segments: number[] = [];
  const buildingIds: number[] = [];
  for (let buildingIndex = 0; buildingIndex < buffers.length; buildingIndex++) {
    const buffer = buffers[buildingIndex];
    const building = BLOCK_RELATIONSHIP_BUILDINGS[buildingIndex];
    if (!buffer.complete) {
      return {
        input: null,
        failure: `${building.id} ${building.address}: ${formatBufferSurfaceStatus(buffer)}`
      };
    }
    for (let feature = 0; feature < buffer.count; feature++) {
      const firstPolygon = buffer.featureOffsets[feature];
      const lastPolygon = buffer.featureOffsets[feature + 1];
      for (let polygon = firstPolygon; polygon < lastPolygon; polygon++) {
        for (
          let ring = buffer.polygonOffsets[polygon];
          ring < buffer.polygonOffsets[polygon + 1];
          ring++
        ) {
          const firstVertex = buffer.ringOffsets[ring];
          const lastVertex = buffer.ringOffsets[ring + 1];
          let vertexCount = lastVertex - firstVertex;
          if (vertexCount > 1) {
            const firstX = buffer.positions[firstVertex * 2];
            const firstY = buffer.positions[firstVertex * 2 + 1];
            const lastX = buffer.positions[(lastVertex - 1) * 2];
            const lastY = buffer.positions[(lastVertex - 1) * 2 + 1];
            if (firstX === lastX && firstY === lastY) vertexCount--;
          }
          if (vertexCount < 3) {
            return {
              input: null,
              failure: `${building.id} ${building.address}: polygon ${polygon} ring ${ring} has only ${vertexCount} distinct boundary vertices.`
            };
          }
          for (let vertex = 0; vertex < vertexCount; vertex++) {
            const start = firstVertex + vertex;
            const end = firstVertex + ((vertex + 1) % vertexCount);
            segments.push(
              buffer.positions[start * 2],
              buffer.positions[start * 2 + 1],
              buffer.positions[end * 2],
              buffer.positions[end * 2 + 1]
            );
            buildingIds.push(buildingIndex);
          }
        }
      }
    }
  }
  if (segments.length === 0) {
    return {input: null, failure: 'the complete buffer outputs contained no boundary segments.'};
  }
  return {
    input: {segments: Float32Array.from(segments), buildingIds: Uint32Array.from(buildingIds)},
    failure: ''
  };
}

function formatBufferSurfaceStatus(buffer: BufferGeometryResult): string {
  return `geometry ${formatCount(buffer.count)} / 1 features and ${formatCount(buffer.actualRings)} / ${formatCount(buffer.requiredRings)} rings (overflow ${buffer.overflow}, candidate overflow ${buffer.candidateOverflow}, uncertain ${buffer.uncertainCount}); boundary ${formatCount(buffer.boundaryCount)} / ${formatCount(buffer.boundaryRequiredCount)} rows (overflow ${buffer.boundaryOverflow}, candidate overflow ${buffer.boundaryCandidateOverflow}).`;
}

function publishCoreSummary(
  ctx: SceneContext<BlockRelationshipsOptions>,
  result: CoreResult
): void {
  publishBoundsSummary(ctx, result);
  const shared = result.shared;
  ctx.setReadout(
    'blockAdjacency',
    `GPUSharedPaths ${formatCount(shared.leftLineIds.length)} unique line-pair runs after keeping left line < right line; source orientations ${Array.from(result.orientation, value => `${toDegrees(value).toFixed(1)}°`).join(', ')}. Adjacency recipe pending.`
  );
  const buffers = result.buffers;
  const allBufferOutputsComplete = buffers.every(buffer => buffer.complete);
  const bufferFailures = buffers.flatMap((buffer, buildingIndex) =>
    buffer.complete
      ? []
      : [`${BLOCK_RELATIONSHIP_BUILDINGS[buildingIndex].id}: ${formatBufferSurfaceStatus(buffer)}`]
  );
  const bufferSummary = `GPUBufferSurface ${ctx.options.inset === '0' ? 'zero-distance boundary pass' : `synthetic −${ctx.options.inset} m inset`} runs once per archived footprint: ${formatCount(buffers.reduce((total, buffer) => total + buffer.count, 0))} output features, ${formatCount(buffers.reduce((total, buffer) => total + buffer.actualRings, 0))} / ${formatCount(buffers.reduce((total, buffer) => total + buffer.requiredRings, 0))} rings; boundary ${formatCount(buffers.reduce((total, buffer) => total + buffer.boundaryCount, 0))} / ${formatCount(buffers.reduce((total, buffer) => total + buffer.boundaryRequiredCount, 0))} rows.`;
  ctx.setReadout(
    'blockTessellation',
    allBufferOutputsComplete
      ? `${bufferSummary} All per-building GPU results are complete.`
      : `Per-building buffer preparation failed: ${bufferFailures.join(' ')} ${bufferSummary}`
  );
  const visibility = result.visibility;
  publishVisibilitySummary(ctx, visibility, ctx.options.visibilityMetric);
  ctx.setChart('blockAdjacencyChart', null);
  ctx.setChart('blockTessellationChart', null);
}

function publishBoundsSummary(
  ctx: SceneContext<BlockRelationshipsOptions>,
  result: CoreResult
): void {
  const bounds = result.bounds;
  ctx.setReadout(
    'blockBounds',
    `GPUBoundsFilter ${ctx.options.boundsRelation}: ${formatCount(bounds.count)} / ${BUILDING_COUNT} footprints in query; required ${formatCount(bounds.requiredCount)}, overflow ${bounds.overflow}. This compares GPU-measured axis-aligned bounds, not exact footprint geometry.`
  );
}

function publishVisibilitySummary(
  ctx: SceneContext<BlockRelationshipsOptions>,
  result: VisibilityResult,
  metric: BlockRelationshipsOptions['visibilityMetric']
): void {
  const metricLabels = {
    degree: 'visible neighbors',
    reachability: 'reachable probes',
    'mean-hop-distance': 'mean shortest-path distance (hops)',
    'hop-closeness': 'reachable-hop closeness',
    'relative-asymmetry': 'relative asymmetry',
    'real-relative-asymmetry': 'real relative asymmetry',
    choice: 'betweenness choice',
    'normalized-choice': 'normalized choice',
    'hillier-integration': 'Hillier-Hanson integration',
    pyramid: 'pyramid index',
    teklenburg: 'Teklenburg index',
    penn: 'Penn index',
    'maximum-depth': 'maximum shortest-path depth (hops)'
  } as const;
  const valid = result.invalidCount === 0;
  const rows = valid
    ? Array.from({length: BLOCK_RELATIONSHIPS_PROBES}, (_, index) => {
        const value = getVisibilityMetric(result, metric, index);
        return `P${index + 1} ${Number.isFinite(value) ? value.toFixed(3) : 'undefined'}`;
      }).join(' · ')
    : 'Metric values withheld because one or more probes were blocked or invalid.';
  ctx.setReadout(
    'blockVisibility',
    `Synthetic observers: ${BLOCK_RELATIONSHIPS_PROBES}; ${formatCount(result.count)} / ${formatCount(result.requiredCount)} edges in bounded drawn list (overflow ${formatCount(result.overflow)}); invalid/blocked classifications ${formatCount(result.invalidCount)}. Dense pair visibility drives shortest-path and choice metrics independently of this edge-list capacity. Selected ${metricLabels[metric]} by probe: ${rows}. Hop distances count graph steps, not metres.`
  );
  ctx.setChart('blockVisibilityChart', valid ? makeVisibilityChart(result, metric) : null);
}

function publishAdjacencySummary(
  ctx: SceneContext<BlockRelationshipsOptions>,
  summary: NonNullable<ReturnType<typeof decodeAdjacencySummary>>,
  data: BlockRelationshipsData
): void {
  const sharedRows: string[] = [];
  for (let first = 0; first < BUILDING_COUNT; first++) {
    for (let second = first + 1; second < BUILDING_COUNT; second++) {
      const pairIndex = first * BUILDING_COUNT + second;
      const length = summary.pairs[pairIndex * 4];
      const orientationDifference = summary.pairs[pairIndex * 4 + 1];
      if (!Number.isFinite(length)) continue;
      sharedRows.push(
        `${BLOCK_RELATIONSHIP_BUILDINGS[first].id} ↔ ${BLOCK_RELATIONSHIP_BUILDINGS[second].id}: ${length.toFixed(2)} m mapped shared boundary, axial difference ${Number.isFinite(orientationDifference) ? toDegrees(orientationDifference).toFixed(1) : 'n/a'}°`
      );
    }
  }
  const morphologyRows = Array.from({length: BUILDING_COUNT}, (_, buildingIndex) => {
    const rowOffset = buildingIndex * 4;
    const mappedLength = summary.morphology[rowOffset];
    const cardinalOrientation = summary.morphology[rowOffset + 1];
    const meanNeighborAlignment = summary.morphology[rowOffset + 2];
    let positiveLengthNeighbors = 0;
    for (let otherBuilding = 0; otherBuilding < BUILDING_COUNT; otherBuilding++) {
      if (otherBuilding === buildingIndex) continue;
      const pairIndex = buildingIndex * BUILDING_COUNT + otherBuilding;
      if (summary.pairs[pairIndex * 4 + 2] > 0) positiveLengthNeighbors++;
    }
    return `${BLOCK_RELATIONSHIP_BUILDINGS[buildingIndex].id}: ${positiveLengthNeighbors} positive-length mapped neighbors; ${Number.isFinite(mappedLength) ? mappedLength.toFixed(2) : 'n/a'} m incident mapped boundary; ${Number.isFinite(cardinalOrientation) ? cardinalOrientation.toFixed(1) : 'n/a'}° cardinal axis; ${Number.isFinite(meanNeighborAlignment) ? meanNeighborAlignment.toFixed(1) : 'n/a'}° mean alignment to neighbors`;
  });
  ctx.setReadout(
    'blockAdjacency',
    `GPU adjacency recipe ${summary.complete === 1 ? 'complete' : 'incomplete'}; connected-component labels ${Array.from(summary.components).join(', ')}. Neighbor count means other buildings joined by a positive-length mapped run; pair evidence: ${sharedRows.join('; ')}. Per-building morphology: ${morphologyRows.join('; ')}`
  );
  ctx.setChart(
    'blockAdjacencyChart',
    summary.complete === 1 ? makeAdjacencyChart(summary.pairs) : null
  );
  void data;
}

function decodeAdjacencySummary(bytes: ArrayBuffer) {
  const pairs = new Float32Array(bytes.slice(0, BUILDING_COUNT * BUILDING_COUNT * 16));
  const morphologyOffset = pairs.byteLength;
  const morphology = new Float32Array(
    bytes.slice(morphologyOffset, morphologyOffset + BUILDING_COUNT * 16)
  );
  const componentsOffset = morphologyOffset + morphology.byteLength;
  const components = new Uint32Array(
    bytes.slice(componentsOffset, componentsOffset + BUILDING_COUNT * 4)
  );
  const complete = new Uint32Array(bytes.slice(componentsOffset + components.byteLength))[0] ?? 0;
  return {pairs, morphology, components, complete};
}

async function buildTessellationVariant(
  ctx: SceneContext<BlockRelationshipsOptions>,
  data: BlockRelationshipsData,
  input: {segments: Float32Array; buildingIds: Uint32Array},
  spacing: string,
  sampleCapacity: number,
  generation: number,
  isCurrent: () => boolean,
  onReadbackSuccess: () => void
): Promise<TessellationVariant> {
  const resources = new SpatialAnalysisResources(
    ctx.device,
    `block-relationships-tessellation-${spacing}-${generation}`
  );
  try {
    const graph = new GPUCommandGraph<void>(ctx.device, {
      id: `block-relationships-tessellation-${spacing}-${generation}`
    });
    const importView = createGraphBufferImporter(graph);
    const segmentsBuffer = resources.createBuffer(
      'boundary-samples-source-segments',
      input.segments
    );
    const ownersBuffer = resources.createBuffer(
      'boundary-samples-building-owners',
      input.buildingIds
    );
    const enclosureBuffer = resources.createBuffer('synthetic-convex-enclosure', data.enclosure);
    const recipe = addBuildingTessellationRecipe(graph, {
      id: 'block-boundary-sampled-tessellation',
      segments: importView(segmentsBuffer, 'float32x4', input.segments.length / 4),
      segmentBuildingIds: importView(ownersBuffer, 'uint32', input.buildingIds.length),
      buildingCount: BUILDING_COUNT,
      enclosure: importView(enclosureBuffer, 'float32x2', data.enclosure.length / 2),
      sampleSpacing: Number(spacing),
      sampleCapacity,
      vertexTolerance: 0.005,
      capacity: {
        intersections: TESS_INTERSECTION_CAPACITY,
        nodedSegments: TESS_SEGMENT_CAPACITY,
        vertices: TESS_VERTEX_CAPACITY,
        rings: TESS_RING_CAPACITY
      }
    });
    const summaryBuffers = makeTessellationReadbackBuffers(
      graph,
      resources,
      importView,
      recipe.outputs,
      recipe.status
    );
    const graphResult = resources.track(await graph.compileAsync());
    const sources: ReadbackSource[] = [
      {buffer: summaryBuffers.sampling, size: 16, kind: 'uint32'},
      {buffer: summaryBuffers.complete, size: BUILDING_COUNT * 4, kind: 'uint32'},
      ...summaryBuffers.outputs.flatMap(output => [
        {buffer: output.positions, size: TESS_VERTEX_CAPACITY * 8, kind: 'float32' as const},
        {
          buffer: output.featureOffsets,
          size: (TESS_RING_CAPACITY + 1) * 4,
          kind: 'uint32' as const
        },
        {
          buffer: output.polygonOffsets,
          size: (TESS_RING_CAPACITY + 1) * 4,
          kind: 'uint32' as const
        },
        {buffer: output.ringOffsets, size: (TESS_RING_CAPACITY + 1) * 4, kind: 'uint32' as const},
        {buffer: output.count, size: 4, kind: 'uint32' as const},
        {buffer: output.requiredCount, size: 4, kind: 'uint32' as const},
        {buffer: output.overflow, size: 4, kind: 'uint32' as const},
        {buffer: output.candidateOverflow, size: 4, kind: 'uint32' as const}
      ])
    ];
    const variant: TessellationVariant = {
      key: `${spacing}-${sampleCapacity}`,
      spacing,
      sampleCapacity,
      resources,
      compiled: graphResult,
      reader: new SummaryReader(
        resources,
        `block-relationships-tessellation-${spacing}`,
        sources,
        bytes => {
          if (!isCurrent()) return;
          variant.result = decodeTessellationSummary(bytes, data);
          onReadbackSuccess();
          ctx.setReadout(
            'blockTessellation',
            publishTessellationText(variant.result, spacing, sampleCapacity, ctx.options.inset)
          );
          ctx.setChart(
            'blockTessellationChart',
            variant.result.status === 'complete' ? makeTessellationChart(variant.result) : null
          );
          ctx.requestLayers();
        }
      ),
      readbackSources: sources,
      dirty: true,
      generation,
      result: null
    };
    return variant;
  } catch (error) {
    resources.destroy();
    throw error;
  }
}

function makeTessellationReadbackBuffers(
  graph: GPUCommandGraph<void>,
  resources: SpatialAnalysisResources,
  importView: ImportView,
  outputs: readonly GPUPolygonOverlayOutput[],
  status: {sampling: GraphDataView<'uint32x4'>; complete: GraphDataView<'uint32'>}
) {
  const sampling = resources.createBuffer('tessellation-sampling-status', 16);
  const complete = resources.createBuffer('tessellation-building-complete', BUILDING_COUNT * 4);
  const importedSampling = importView(sampling, 'uint32x4', 1);
  const importedComplete = importView(complete, 'uint32', BUILDING_COUNT);
  const outputBuffers = outputs.map((output, buildingIndex) => {
    const positions = resources.createBuffer(
      `tessellation-${buildingIndex}-positions`,
      TESS_VERTEX_CAPACITY * 8
    );
    const featureOffsets = resources.createBuffer(
      `tessellation-${buildingIndex}-feature-offsets`,
      (TESS_RING_CAPACITY + 1) * 4
    );
    const polygonOffsets = resources.createBuffer(
      `tessellation-${buildingIndex}-polygon-offsets`,
      (TESS_RING_CAPACITY + 1) * 4
    );
    const ringOffsets = resources.createBuffer(
      `tessellation-${buildingIndex}-ring-offsets`,
      (TESS_RING_CAPACITY + 1) * 4
    );
    const count = resources.createBuffer(`tessellation-${buildingIndex}-count`, 4);
    const requiredCount = resources.createBuffer(`tessellation-${buildingIndex}-required`, 4);
    const overflow = resources.createBuffer(`tessellation-${buildingIndex}-overflow`, 4);
    const candidateOverflow = resources.createBuffer(
      `tessellation-${buildingIndex}-candidate-overflow`,
      4
    );
    return {
      positions,
      featureOffsets,
      polygonOffsets,
      ringOffsets,
      count,
      requiredCount,
      overflow,
      candidateOverflow
    };
  });
  addKernelPass(graph, {
    id: 'copy-tessellation-status',
    invocationCount: BUILDING_COUNT,
    bindings: [
      {name: 'sampling', view: status.sampling, type: 'u32', access: 'read'},
      {name: 'complete', view: status.complete, type: 'u32', access: 'read'},
      {name: 'outSampling', view: importedSampling, type: 'u32', access: 'read_write'},
      {name: 'outComplete', view: importedComplete, type: 'u32', access: 'read_write'}
    ],
    body: 'if (index == 0u) { for (var word = 0u; word < 4u; word++) { outSampling[outSamplingOffset + word] = sampling[samplingOffset + word]; } } outComplete[outCompleteOffset + index] = complete[completeOffset + index];'
  });
  outputs.forEach((output, index) => {
    const target = outputBuffers[index];
    const copies: {
      name: string;
      source: GraphDataView<'float32x2' | 'uint32'>;
      destination: Buffer;
      format: 'float32x2' | 'uint32';
      length: number;
    }[] = [
      {
        name: 'positions',
        source: output.geometry.positions,
        destination: target.positions,
        format: 'float32x2',
        length: TESS_VERTEX_CAPACITY
      },
      {
        name: 'feature-offsets',
        source: output.geometry.featureOffsets,
        destination: target.featureOffsets,
        format: 'uint32',
        length: TESS_RING_CAPACITY + 1
      },
      {
        name: 'polygon-offsets',
        source: output.geometry.polygonOffsets,
        destination: target.polygonOffsets,
        format: 'uint32',
        length: TESS_RING_CAPACITY + 1
      },
      {
        name: 'ring-offsets',
        source: output.geometry.ringOffsets,
        destination: target.ringOffsets,
        format: 'uint32',
        length: TESS_RING_CAPACITY + 1
      }
    ];
    for (const copy of copies) {
      const elementWidth = copy.format === 'float32x2' ? 2 : 1;
      const componentType = copy.format === 'float32x2' ? ('f32' as const) : ('u32' as const);
      addKernelPass(graph, {
        id: `copy-tessellation-${index}-${copy.name}`,
        invocationCount: copy.length * elementWidth,
        bindings: [
          {name: 'source', view: copy.source, type: componentType, access: 'read'},
          {
            name: 'copiedOutput',
            view: importView(copy.destination, copy.format, copy.length),
            type: componentType,
            access: 'read_write'
          }
        ],
        body: 'copiedOutput[copiedOutputOffset + index] = source[sourceOffset + index];'
      });
    }
    for (const [name, sourceView, destination] of [
      ['count', output.status.count, target.count],
      ['required', output.status.requiredCount, target.requiredCount],
      ['overflow', output.status.overflow, target.overflow],
      ['candidate-overflow', output.status.candidateOverflow, target.candidateOverflow]
    ] as const) {
      addKernelPass(graph, {
        id: `copy-tessellation-${index}-${name}`,
        invocationCount: 1,
        bindings: [
          {name: 'source', view: sourceView, type: 'u32', access: 'read'},
          {
            name: 'copiedOutput',
            view: importView(destination, 'uint32', 1),
            type: 'u32',
            access: 'read_write'
          }
        ],
        body: 'copiedOutput[copiedOutputOffset] = source[sourceOffset];'
      });
    }
  });
  return {sampling, complete, outputs: outputBuffers};
}

function decodeTessellationSummary(
  bytes: ArrayBuffer,
  data: BlockRelationshipsData
): TessellationResult {
  let offset = 0;
  const uint32 = (length: number) => {
    const values = new Uint32Array(bytes.slice(offset, offset + length * 4));
    offset += length * 4;
    return values;
  };
  const float32 = (length: number) => {
    const values = new Float32Array(bytes.slice(offset, offset + length * 4));
    offset += length * 4;
    return values;
  };
  const sampling = uint32(4);
  const complete = uint32(BUILDING_COUNT);
  const polygons: {buildingIndex: number; rings: Point[][]}[] = [];
  const areas = new Array<number>(BUILDING_COUNT).fill(0);
  let allOutputsComplete =
    sampling[0] === sampling[1] &&
    sampling[2] === 0 &&
    sampling[3] === 0 &&
    complete.every(value => value === 1);
  for (let buildingIndex = 0; buildingIndex < BUILDING_COUNT; buildingIndex++) {
    const positions = float32(TESS_VERTEX_CAPACITY * 2);
    const featureOffsets = uint32(TESS_RING_CAPACITY + 1);
    const polygonOffsets = uint32(TESS_RING_CAPACITY + 1);
    const ringOffsets = uint32(TESS_RING_CAPACITY + 1);
    const count = uint32(1)[0] ?? 0;
    const requiredCount = uint32(1)[0] ?? 0;
    const overflow = uint32(1)[0] ?? 0;
    const candidateOverflow = uint32(1)[0] ?? 0;
    const geometryValid =
      overflow === 0 &&
      candidateOverflow === 0 &&
      isValidTessellationGeometry(
        positions,
        featureOffsets,
        polygonOffsets,
        ringOffsets,
        count,
        requiredCount
      );
    allOutputsComplete = allOutputsComplete && geometryValid;
    if (!geometryValid) continue;
    for (let feature = 0; feature < count; feature++) {
      for (
        let polygon = featureOffsets[feature];
        polygon < featureOffsets[feature + 1];
        polygon++
      ) {
        const rings: Point[][] = [];
        for (let ring = polygonOffsets[polygon]; ring < polygonOffsets[polygon + 1]; ring++) {
          const first = ringOffsets[ring];
          const last = ringOffsets[ring + 1];
          const path = Array.from({length: last - first}, (_, index) =>
            data.projection.unproject(
              positions[(first + index) * 2],
              positions[(first + index) * 2 + 1]
            )
          );
          if (path.length >= 3) {
            rings.push(path);
            let twiceArea = 0;
            for (let vertex = first; vertex + 1 < last; vertex++) {
              twiceArea +=
                positions[vertex * 2] * positions[(vertex + 1) * 2 + 1] -
                positions[(vertex + 1) * 2] * positions[vertex * 2 + 1];
            }
            if (last - first > 2) {
              const firstX = positions[first * 2];
              const firstY = positions[first * 2 + 1];
              const lastX = positions[(last - 1) * 2];
              const lastY = positions[(last - 1) * 2 + 1];
              if (firstX !== lastX || firstY !== lastY)
                twiceArea += lastX * firstY - firstX * lastY;
            }
            areas[buildingIndex] +=
              Math.abs(twiceArea / 2) * (ring === polygonOffsets[polygon] ? 1 : -1);
          }
        }
        if (rings.length > 0) polygons.push({buildingIndex, rings});
      }
    }
  }
  if (offset !== bytes.byteLength)
    throw new Error('Tessellation readback layout did not decode exactly.');
  if (!allOutputsComplete) {
    polygons.length = 0;
    areas.fill(0);
  }
  return {
    polygons,
    areas,
    status: allOutputsComplete ? 'complete' : 'incomplete; cell output is withheld',
    sampleCount: sampling[0],
    requiredSamples: sampling[1],
    sampleOverflow: sampling[2],
    invalidSamples: sampling[3],
    complete
  };
}

function isValidTessellationGeometry(
  positions: Float32Array,
  featureOffsets: Uint32Array,
  polygonOffsets: Uint32Array,
  ringOffsets: Uint32Array,
  featureCount: number,
  requiredRingCount: number
): boolean {
  if (
    featureCount > TESS_RING_CAPACITY ||
    requiredRingCount > TESS_RING_CAPACITY ||
    featureOffsets[0] !== 0 ||
    polygonOffsets[0] !== 0 ||
    ringOffsets[0] !== 0
  ) {
    return false;
  }

  let previousOffset = 0;
  for (let featureOffsetIndex = 1; featureOffsetIndex <= featureCount; featureOffsetIndex++) {
    const nextOffset = featureOffsets[featureOffsetIndex];
    if (nextOffset < previousOffset || nextOffset > TESS_RING_CAPACITY) return false;
    previousOffset = nextOffset;
  }
  const polygonCount = featureOffsets[featureCount];
  if (polygonCount > TESS_RING_CAPACITY) return false;

  previousOffset = 0;
  for (let polygonOffsetIndex = 1; polygonOffsetIndex <= polygonCount; polygonOffsetIndex++) {
    const nextOffset = polygonOffsets[polygonOffsetIndex];
    if (nextOffset < previousOffset || nextOffset > TESS_RING_CAPACITY) return false;
    previousOffset = nextOffset;
  }
  const ringCount = polygonOffsets[polygonCount];
  if (ringCount !== requiredRingCount || ringCount > TESS_RING_CAPACITY) {
    return false;
  }

  previousOffset = 0;
  for (let ringOffsetIndex = 1; ringOffsetIndex <= ringCount; ringOffsetIndex++) {
    const nextOffset = ringOffsets[ringOffsetIndex];
    if (nextOffset < previousOffset || nextOffset > TESS_VERTEX_CAPACITY) return false;
    previousOffset = nextOffset;
  }
  previousOffset = 0;
  for (let ring = 0; ring < ringCount; ring++) {
    const nextOffset = ringOffsets[ring + 1];
    if (
      nextOffset < previousOffset ||
      nextOffset > TESS_VERTEX_CAPACITY ||
      nextOffset - previousOffset < 3
    ) {
      return false;
    }
    for (let vertex = previousOffset; vertex < nextOffset; vertex++) {
      if (!Number.isFinite(positions[vertex * 2]) || !Number.isFinite(positions[vertex * 2 + 1])) {
        return false;
      }
    }
    previousOffset = nextOffset;
  }
  return true;
}

function publishTessellationText(
  result: TessellationResult,
  spacing: string,
  sampleCapacity: number,
  inset: string
): string {
  return `Boundary-sampled Voronoi partition from GPUBufferSurface ${inset === '0' ? 'zero-distance' : `synthetic −${inset} m`} boundaries at ${spacing} m spacing with ${sampleCapacity} site slots: ${formatCount(result.sampleCount)} / ${formatCount(result.requiredSamples)} sites; sample overflow ${result.sampleOverflow}, invalid source segments ${result.invalidSamples}; per-building completion flags ${Array.from(result.complete).join(', ')}; ${result.status}. Areas are shoelace summaries of GPU-readback cell rings.`;
}

function destroyTessellationVariant(variant: TessellationVariant): void {
  variant.reader.release();
  variant.resources.destroy();
}

function getVisibilityMetric(
  result: VisibilityResult,
  metric: BlockRelationshipsOptions['visibilityMetric'],
  probeIndex: number
): number {
  switch (metric) {
    case 'degree':
      return result.metrics[probeIndex * 4];
    case 'reachability':
      return result.metrics[probeIndex * 4 + 1];
    case 'mean-hop-distance':
      return result.integration[probeIndex * 4];
    case 'hop-closeness':
      return result.metrics[probeIndex * 4 + 3];
    case 'relative-asymmetry':
      return result.integration[probeIndex * 4 + 1];
    case 'real-relative-asymmetry':
      return result.integration[probeIndex * 4 + 2];
    case 'choice':
      return result.choice[probeIndex];
    case 'normalized-choice':
      return result.normalizedChoice[probeIndex];
    case 'hillier-integration':
      return result.integration[probeIndex * 4 + 3];
    case 'pyramid':
      return result.extendedIntegration[probeIndex * 4];
    case 'teklenburg':
      return result.extendedIntegration[probeIndex * 4 + 1];
    case 'penn':
      return result.extendedIntegration[probeIndex * 4 + 2];
    case 'maximum-depth':
      return result.extendedIntegration[probeIndex * 4 + 3];
  }
}

function makeVisibilityChart(
  result: VisibilityResult,
  metric: BlockRelationshipsOptions['visibilityMetric']
): BarChartData {
  const metricLabels = {
    degree: 'Visible neighbors',
    reachability: 'Reachable probes',
    'mean-hop-distance': 'Mean shortest-path distance (hops)',
    'hop-closeness': 'Reachable-hop closeness',
    'relative-asymmetry': 'Relative asymmetry',
    'real-relative-asymmetry': 'Real relative asymmetry',
    choice: 'Betweenness choice',
    'normalized-choice': 'Normalized choice',
    'hillier-integration': 'Hillier-Hanson integration',
    pyramid: 'Pyramid index',
    teklenburg: 'Teklenburg index',
    penn: 'Penn index',
    'maximum-depth': 'Maximum shortest-path depth (hops)'
  } as const;
  const labels = Array.from(
    {length: BLOCK_RELATIONSHIPS_PROBES},
    (_, index) => `Probe ${index + 1}`
  );
  const values = Float32Array.from(labels, (_, index) =>
    getVisibilityMetric(result, metric, index)
  );
  return {
    kind: 'bars',
    title: `${metricLabels[metric]} by synthetic probe`,
    labels,
    values,
    color: 4,
    description:
      'GPUVisibilityGraph dense outputs drive this metric; graph hops are not metric distances and no value measures pedestrian access.'
  };
}

function makeAdjacencyChart(pairs: Float32Array): BarChartData {
  const labels: string[] = [];
  const values: number[] = [];
  for (let first = 0; first < BUILDING_COUNT; first++) {
    for (let second = first + 1; second < BUILDING_COUNT; second++) {
      labels.push(
        `${BLOCK_RELATIONSHIP_BUILDINGS[first].id}–${BLOCK_RELATIONSHIP_BUILDINGS[second].id}`
      );
      values.push(Math.max(0, pairs[(first * BUILDING_COUNT + second) * 4] ?? 0));
    }
  }
  return {
    kind: 'bars',
    title: 'Mapped shared-boundary length',
    labels,
    values: Float32Array.from(values),
    color: 2,
    description:
      'GPU shared-run lengths accumulated once per unordered building pair. Mapped contact is not evidence of a physical party wall.'
  };
}

function makeTessellationChart(result: TessellationResult): BarChartData {
  return {
    kind: 'bars',
    title: 'Boundary-sampled cell area',
    labels: BLOCK_RELATIONSHIP_BUILDINGS.map(building => String(building.id)),
    values: Float32Array.from(result.areas),
    color: 3,
    description:
      'Planar shoelace area of GPU-readback tessellation output, grouped by source building owner; values change with boundary sample spacing.'
  };
}

function unpackSharedPaths(
  shared: SharedPathRows,
  data: BlockRelationshipsData
): {path: Point[]}[] {
  const paths: {path: Point[]}[] = [];
  for (let run = 0; run < shared.leftLineIds.length; run++) {
    const first = shared.offsets[run];
    const last = shared.offsets[run + 1];
    paths.push({
      path: Array.from({length: last - first}, (_, index) =>
        data.projection.unproject(
          shared.positions[(first + index) * 2],
          shared.positions[(first + index) * 2 + 1]
        )
      )
    });
  }
  return paths;
}

function unpackPoints(
  positions: Float32Array,
  projection: BlockRelationshipsData['projection']
): Point[] {
  return Array.from({length: positions.length / 2}, (_, index) =>
    projection.unproject(positions[index * 2], positions[index * 2 + 1])
  );
}

function ringCentroid(path: Point[]): Point {
  const count =
    path.length > 1 && path[0][0] === path.at(-1)![0] && path[0][1] === path.at(-1)![1]
      ? path.length - 1
      : path.length;
  return [
    path.slice(0, count).reduce((sum, point) => sum + point[0], 0) / count,
    path.slice(0, count).reduce((sum, point) => sum + point[1], 0) / count
  ];
}
