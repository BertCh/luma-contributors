// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {PathLayer, PolygonLayer} from '@deck.gl/layers';
import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPULineNoding,
  GPUPolygonize,
  GPUSharedPaths,
  type GPUPolygonizeOutput,
  type GPUTopologyPrecisionPolicy
} from '@luma-contributors/graph/gpu-spatial-analysis';
import {
  ALBERS_CONUS_FRAME,
  createAnalysisFrameProjection,
  getAnalysisFrameSpatialContext
} from '../../engine/analysis-frame';
import {GraphSwap} from '../../engine/graph-swap';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {BarChartData} from '../chart-types';
import type {SceneContext, SceneInstance} from '../scene';
import {createGraphImporter, loadPolygonLayout} from './b3-common';
import {
  COUNTY_LINEWORK_FIPS,
  COUNTY_LINEWORK_NAMES,
  buildCountyLineworkInput,
  type CountyLineworkEncoding,
  type CountyLineworkInput
} from './county-linework-data';
import type {CountyLineworkOptions} from './county-linework-options';

// The archived pair has 275 intersecting unordered source-edge pairs, including touches.
const INTERSECTION_CAPACITY = 512;
const NODED_SEGMENT_CAPACITY = 2048;
const POLYGON_RING_CAPACITY = 1024;
const POLYGON_VERTEX_CAPACITY = 4096;
const SHARED_RUN_CAPACITY = 32;
const SHARED_VERTEX_CAPACITY = 512;
const COUNTY_COLORS: Record<number, [number, number, number, number]> = {
  [COUNTY_LINEWORK_FIPS.left]: [221, 133, 79, 255],
  [COUNTY_LINEWORK_FIPS.right]: [60, 155, 174, 255]
};
const LINEWORK_FRAME = createAnalysisFrameProjection(ALBERS_CONUS_FRAME);

type SharedSummary = {
  count: number;
  vertexCount: number;
  requiredCount: number;
  requiredVertexCount: number;
  overflow: number;
  uncertainCount: number;
  forwardCount: number;
  backwardCount: number;
  runs: {path: [number, number][]; forward: boolean; leftLineId: number; rightLineId: number}[];
};

type NodingSummary = {
  count: number;
  requiredCount: number;
  overflow: number;
  candidateOverflow: number;
  uncertainCount: number;
  segments: {
    path: [number, number][];
    sourceFeatureId: number;
    sourceRingId: number;
    sourceEdgeId: number;
    startParameter: number;
    endParameter: number;
  }[];
};

type PolygonizeSummary = {
  count: number;
  requiredCount: number;
  writtenRingCount: number;
  faceAreas: number[];
  overflow: number;
  candidateOverflow: number;
  uncertainCount: number;
  nodedSegmentCount: number;
  nodedRequiredCount: number;
  nodedOverflow: number;
  nodedCandidateOverflow: number;
  nodedUncertainCount: number;
  edgeClasses: Uint32Array;
  diagnostics: Uint32Array;
  diagnosticPaths: {path: [number, number][]; flags: number}[];
  polygons: {rings: [number, number][][]; sourceId: number}[];
};

type CountyLineworkSummary = {
  shared: SharedSummary;
  noding: NodingSummary;
  polygonize: PolygonizeSummary;
};

type BufferReadback = {buffer: Buffer; size: number; kind: 'uint32' | 'float32'};

type Variant = {
  encoding: CountyLineworkEncoding;
  key: string;
  input: CountyLineworkInput;
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  readbackSources: BufferReadback[];
  reader: SummaryReader | null;
  generation: number;
  dirty: boolean;
  summary: CountyLineworkSummary | null;
};

type PolygonizeOutputBuffers = {
  output: GPUPolygonizeOutput;
  readbackSources: BufferReadback[];
};

type GraphImporter = ReturnType<typeof createGraphImporter<void>>;

/** Standalone GPU shared-path, noding and polygonization investigation for a real county pair. */
export async function createCountyLinework(
  ctx: SceneContext<CountyLineworkOptions>
): Promise<SceneInstance<CountyLineworkOptions>> {
  const dataset = ctx.datasets.get('us-counties');
  const layout = loadPolygonLayout(dataset);
  const countyFips = dataset.column<Uint32Array>('fips');
  const countyData = new Map<CountyLineworkEncoding, CountyLineworkInput>();
  const cache = new Map<CountyLineworkEncoding, Variant>();
  const lifecycle = new SpatialAnalysisResources(ctx.device, 'county-linework-lifecycle');
  const graphSwap = lifecycle.track(new GraphSwap());
  let active: Variant | null = null;
  let generation = 0;
  let destroyed = false;

  const getInput = (encoding: CountyLineworkEncoding) => {
    let input = countyData.get(encoding);
    if (!input) {
      input = buildCountyLineworkInput(layout, countyFips, encoding);
      countyData.set(encoding, input);
    }
    return input;
  };

  const clearReadouts = () => {
    ctx.setChart('countyLineworkDirection', null);
    ctx.setChart('countyLineworkDiagnostics', null);
    ctx.setReadout('countyLineworkShared', 'GPUSharedPaths is pending.');
    ctx.setReadout('countyLineworkNoding', 'GPULineNoding is pending.');
    ctx.setReadout('countyLineworkPolygonize', 'GPUPolygonize is pending.');
  };

  const makeReader = (variant: Variant, readerGeneration: number) => {
    variant.reader?.release();
    variant.generation = readerGeneration;
    variant.reader = new SummaryReader(
      variant.resources,
      `county-linework-${variant.key}-${readerGeneration}`,
      variant.readbackSources,
      bytes => {
        if (
          destroyed ||
          active !== variant ||
          variant.generation !== readerGeneration ||
          generation !== readerGeneration ||
          ctx.options.inputEncoding !== variant.encoding
        )
          return;
        const summary = decodeSummary(bytes, variant);
        variant.summary = summary;
        publishSummary(ctx, summary, variant);
        ctx.requestLayers();
      }
    );
  };

  const buildVariant = async (encoding: CountyLineworkEncoding): Promise<Variant> => {
    const cached = cache.get(encoding);
    if (cached) return cached;
    const input = getInput(encoding);
    const key = encoding;
    const resources = new SpatialAnalysisResources(ctx.device, `county-linework-${key}`);
    const buildResources = async (): Promise<Variant> => {
      const graph = new GPUCommandGraph<void>(ctx.device, {id: `county-linework-${key}`});
      const importView = createGraphImporter(graph);
      const positions = resources.createBuffer('linework-positions', input.positions);
      const lineOffsets = resources.createBuffer('linework-offsets', input.lineOffsets);
      const sourceIds = resources.createBuffer('linework-source-fips', input.lineSourceIds);
      const leftPositions = resources.createBuffer('left-county-positions', input.leftPositions);
      const leftOffsets = resources.createBuffer('left-county-offsets', input.leftOffsets);
      const rightPositions = resources.createBuffer('right-county-positions', input.rightPositions);
      const rightOffsets = resources.createBuffer('right-county-offsets', input.rightOffsets);
      const positionCount = input.positions.length / 2;
      const lineCount = input.lineOffsets.length - 1;
      const nodeBuffers = makeNodingOutput(resources, importView, key);
      const sharedBuffers = makeSharedOutput(resources, importView, key);
      const polygonBuffers = makePolygonizeOutput(resources, importView, key);

      graph.add(
        new GPUSharedPaths({
          id: 'shared-county-boundary-runs',
          spatialContext: getAnalysisFrameSpatialContext(ALBERS_CONUS_FRAME),
          left: {
            kind: 'lines',
            positions: importView(
              'left-county-positions',
              leftPositions,
              'float32x2',
              input.leftPositions.length / 2
            ),
            lineOffsets: importView(
              'left-county-offsets',
              leftOffsets,
              'uint32',
              input.leftOffsets.length
            )
          },
          right: {
            kind: 'lines',
            positions: importView(
              'right-county-positions',
              rightPositions,
              'float32x2',
              input.rightPositions.length / 2
            ),
            lineOffsets: importView(
              'right-county-offsets',
              rightOffsets,
              'uint32',
              input.rightOffsets.length
            )
          },
          intersectionCapacity: INTERSECTION_CAPACITY,
          output: sharedBuffers.output
        })
      );
      graph.add(
        new GPULineNoding({
          id: 'county-boundary-line-noding',
          spatialContext: getAnalysisFrameSpatialContext(ALBERS_CONUS_FRAME),
          lines: {
            kind: 'lines',
            positions: importView('linework-positions', positions, 'float32x2', positionCount),
            lineOffsets: importView(
              'linework-offsets',
              lineOffsets,
              'uint32',
              input.lineOffsets.length
            ),
            sourceIds: importView('linework-source-fips', sourceIds, 'uint32', lineCount)
          },
          intersectionCapacity: INTERSECTION_CAPACITY,
          output: nodeBuffers.output
        })
      );
      graph.add(
        new GPUPolygonize({
          id: 'county-boundary-polygonize',
          spatialContext: getAnalysisFrameSpatialContext(ALBERS_CONUS_FRAME),
          lines: {
            kind: 'lines',
            positions: importView('linework-positions', positions, 'float32x2', positionCount),
            lineOffsets: importView(
              'linework-offsets',
              lineOffsets,
              'uint32',
              input.lineOffsets.length
            ),
            sourceIds: importView('linework-source-fips', sourceIds, 'uint32', lineCount)
          },
          intersectionCapacity: INTERSECTION_CAPACITY,
          precision: {
            vertexTolerance: getPolygonizeVertexTolerance(input),
            predicates: 'exact-or-uncertain',
            storage: 'float32'
          } satisfies GPUTopologyPrecisionPolicy,
          output: polygonBuffers.output
        })
      );

      let compiled: CompiledGPUCommandGraph<void>;
      try {
        compiled = resources.track(await graph.compileAsync());
      } catch (error) {
        resources.destroy();
        throw error;
      }
      const variant: Variant = {
        encoding,
        key,
        input,
        resources,
        compiled,
        readbackSources: [
          ...sharedBuffers.readbackSources,
          ...nodeBuffers.readbackSources,
          ...polygonBuffers.readbackSources
        ],
        reader: null,
        generation: -1,
        dirty: true,
        summary: null
      };
      return variant;
    };

    return buildResources();
  };

  const initialEncoding = ctx.options.inputEncoding;
  active = await buildVariant(initialEncoding);
  cache.set(initialEncoding, active);
  makeReader(active, generation);
  ctx.setReadout(
    'countyLineworkSource',
    `${dataset.count.toLocaleString()} archived Census 2022 counties; this bounded pair is ${COUNTY_LINEWORK_NAMES.left} (FIPS ${COUNTY_LINEWORK_FIPS.left}) and ${COUNTY_LINEWORK_NAMES.right} (FIPS ${COUNTY_LINEWORK_FIPS.right}). The bundled GeoArrow rows are ${layout.featureCount.toLocaleString()} counties total. Source pair: ${inputDescription(active.input)}.`
  );
  clearReadouts();

  const requestVariant = (encoding: CountyLineworkEncoding) => {
    const requestGeneration = ++generation;
    clearReadouts();
    ctx.requestLayers();
    void graphSwap
      .request({
        build: () => buildVariant(encoding),
        commit: variant => {
          if (
            destroyed ||
            generation !== requestGeneration ||
            ctx.options.inputEncoding !== encoding
          ) {
            if (cache.get(encoding) !== variant) {
              variant.reader?.release();
              variant.resources.destroy();
            }
            return;
          }
          cache.set(encoding, variant);
          variant.summary = null;
          active = variant;
          ctx.setStatus('');
          variant.dirty = true;
          makeReader(variant, requestGeneration);
          ctx.requestLayers();
        },
        discard: variant => {
          if (cache.get(encoding) !== variant) {
            variant.reader?.release();
            variant.resources.destroy();
          }
        }
      })
      .catch(error => {
        if (!destroyed && generation === requestGeneration) {
          active = null;
          clearReadouts();
          const message = `GPU graph build failed: ${String(error)}`;
          ctx.setStatus(message);
          ctx.setReadout('countyLineworkNoding', message);
          ctx.requestLayers();
        }
      });
  };

  const getLayers = (): Layer[] => {
    const variant = active;
    if (!variant || variant.encoding !== ctx.options.inputEncoding) return [];
    const layers: Layer[] = [
      new PathLayer({
        id: 'county-linework-source-rings',
        data: variant.input.sourcePaths,
        getPath: row => row.path,
        getColor: row =>
          row.countyFips === COUNTY_LINEWORK_FIPS.left
            ? [145, 126, 112, 190]
            : [116, 143, 149, 190],
        getWidth: 1.4,
        widthUnits: 'pixels',
        widthMinPixels: 1,
        pickable: false
      })
    ];
    if (variant.input.omittedSegment) {
      layers.push(
        new PathLayer({
          id: 'county-linework-omitted-edge-marker',
          data: [{path: variant.input.omittedSegment}],
          getPath: row => row.path,
          getColor: [218, 75, 60, 255],
          getWidth: 5,
          widthUnits: 'pixels',
          widthMinPixels: 5,
          pickable: false
        })
      );
    }
    const summary = variant.summary;
    if (!summary) return layers;
    if (ctx.options.display === 'shared-paths' && sharedComplete(summary.shared)) {
      layers.push(
        new PathLayer({
          id: 'county-linework-shared-runs',
          data: summary.shared.runs,
          getPath: row => row.path,
          getColor: row => (row.forward ? [39, 163, 144, 255] : [221, 159, 73, 255]),
          getWidth: 4.5,
          widthUnits: 'pixels',
          widthMinPixels: 3,
          capRounded: true,
          jointRounded: true,
          pickable: false
        })
      );
    } else if (ctx.options.display === 'noded-segments' && nodingComplete(summary.noding)) {
      layers.push(
        new PathLayer({
          id: 'county-linework-noded-segments',
          data: summary.noding.segments,
          getPath: row => row.path,
          getColor: row => COUNTY_COLORS[row.sourceFeatureId] ?? [208, 201, 186, 240],
          getWidth: 2.1,
          widthUnits: 'pixels',
          widthMinPixels: 1.5,
          capRounded: true,
          jointRounded: true,
          pickable: false
        })
      );
    } else if (
      ctx.options.display === 'polygonized-faces' &&
      polygonizeComplete(summary.polygonize)
    ) {
      layers.push(
        new PolygonLayer({
          id: 'county-linework-polygonized-faces',
          data: summary.polygonize.polygons,
          getPolygon: row => row.rings,
          getFillColor: row => {
            const color =
              row.sourceId === 0
                ? [112, 137, 144, 220]
                : (COUNTY_COLORS[row.sourceId] ?? [116, 133, 142, 210]);
            return [color[0], color[1], color[2], 76];
          },
          getLineColor: row =>
            row.sourceId === 0
              ? [112, 137, 144, 245]
              : (COUNTY_COLORS[row.sourceId] ?? [116, 133, 142, 255]),
          getLineWidth: 2,
          lineWidthUnits: 'pixels',
          lineWidthMinPixels: 1,
          filled: true,
          stroked: true,
          pickable: false
        })
      );
      if (summary.polygonize.nodedSegmentCount > 0) {
        layers.push(
          new PathLayer({
            id: 'county-linework-polygonize-diagnostics',
            data: getPolygonizeDiagnosticPaths(summary.polygonize),
            getPath: row => row.path,
            getColor: row => row.color,
            getWidth: 2.8,
            widthUnits: 'pixels',
            widthMinPixels: 2,
            pickable: false
          })
        );
      }
    }
    return layers;
  };

  return {
    getCompiledGraphs: () => Array.from(cache.values(), variant => variant.compiled),
    setOption(id, options) {
      if (id === 'inputEncoding') requestVariant(ctx.options.inputEncoding);
      else if (id === 'display') ctx.requestLayers();
    },
    encode(commandEncoder) {
      if (!active || destroyed || active.encoding !== ctx.options.inputEncoding) return;
      if (active.dirty) {
        active.compiled.encode(commandEncoder, {parameters: undefined});
        active.reader?.request(commandEncoder);
        active.dirty = false;
      } else {
        active.reader?.flush(commandEncoder);
      }
    },
    getLayers,
    destroy() {
      destroyed = true;
      graphSwap.destroy();
      lifecycle.destroy();
      for (const variant of cache.values()) {
        variant.reader?.stop();
        variant.resources.destroy();
      }
      cache.clear();
    }
  };
}

/** Match independently computed f32 endpoints without approaching a distinct source vertex. */
function getPolygonizeVertexTolerance(input: CountyLineworkInput): number {
  let maximumCoordinateMagnitude = 0;
  for (const coordinate of input.positions) {
    maximumCoordinateMagnitude = Math.max(maximumCoordinateMagnitude, Math.abs(coordinate));
  }
  const float32Spacing =
    maximumCoordinateMagnitude === 0
      ? 0
      : 2 ** (Math.floor(Math.log2(maximumCoordinateMagnitude)) - 23);
  const vertexTolerance = Math.max(0.001, float32Spacing * 4);
  const distinctVertices = new Map<string, [number, number]>();
  for (let vertex = 0; vertex < input.positions.length / 2; vertex++) {
    const positionX = input.positions[vertex * 2];
    const positionY = input.positions[vertex * 2 + 1];
    distinctVertices.set(`${positionX},${positionY}`, [positionX, positionY]);
  }
  const vertices = Array.from(distinctVertices.values());
  let minimumDistinctVertexSeparation = Infinity;
  for (let firstVertex = 0; firstVertex < vertices.length; firstVertex++) {
    for (let secondVertex = firstVertex + 1; secondVertex < vertices.length; secondVertex++) {
      minimumDistinctVertexSeparation = Math.min(
        minimumDistinctVertexSeparation,
        Math.hypot(
          vertices[firstVertex][0] - vertices[secondVertex][0],
          vertices[firstVertex][1] - vertices[secondVertex][1]
        )
      );
    }
  }
  // Ensure precision snapping cannot merge any two distinct source vertices.
  if (!(vertexTolerance < minimumDistinctVertexSeparation / 4))
    throw new Error('Polygonize precision tolerance is too large for this source pair.');
  return vertexTolerance;
}

function makeSharedOutput(
  resources: SpatialAnalysisResources,
  importView: GraphImporter,
  key: string
) {
  const runCount = resources.createBuffer(`shared-run-count-${key}`, 4);
  const vertexCount = resources.createBuffer(`shared-vertex-count-${key}`, 4);
  const overflow = resources.createBuffer(`shared-overflow-${key}`, 4);
  const requiredCount = resources.createBuffer(`shared-required-count-${key}`, 4);
  const requiredVertexCount = resources.createBuffer(`shared-required-vertices-${key}`, 4);
  const uncertainCount = resources.createBuffer(`shared-uncertain-${key}`, 4);
  const runPositions = resources.createBuffer(
    `shared-run-positions-${key}`,
    SHARED_VERTEX_CAPACITY * 8
  );
  const runOffsets = resources.createBuffer(
    `shared-run-offsets-${key}`,
    (SHARED_RUN_CAPACITY + 1) * 4
  );
  const leftLineIds = resources.createBuffer(
    `shared-left-line-ids-${key}`,
    SHARED_RUN_CAPACITY * 4
  );
  const rightLineIds = resources.createBuffer(
    `shared-right-line-ids-${key}`,
    SHARED_RUN_CAPACITY * 4
  );
  const forward = resources.createBuffer(`shared-forward-${key}`, SHARED_RUN_CAPACITY * 4);
  return {
    output: {
      runs: {
        leftLineIds: importView(
          `shared-left-line-ids-${key}`,
          leftLineIds,
          'uint32',
          SHARED_RUN_CAPACITY
        ),
        rightLineIds: importView(
          `shared-right-line-ids-${key}`,
          rightLineIds,
          'uint32',
          SHARED_RUN_CAPACITY
        ),
        forward: importView(`shared-forward-${key}`, forward, 'uint32', SHARED_RUN_CAPACITY),
        offsets: importView(
          `shared-run-offsets-${key}`,
          runOffsets,
          'uint32',
          SHARED_RUN_CAPACITY + 1
        ),
        positions: importView(
          `shared-run-positions-${key}`,
          runPositions,
          'float32x2',
          SHARED_VERTEX_CAPACITY
        )
      },
      status: {
        count: importView(`shared-run-count-${key}`, runCount, 'uint32', 1),
        vertexCount: importView(`shared-vertex-count-${key}`, vertexCount, 'uint32', 1),
        overflow: importView(`shared-overflow-${key}`, overflow, 'uint32', 1),
        requiredCount: importView(`shared-required-count-${key}`, requiredCount, 'uint32', 1),
        requiredVertexCount: importView(
          `shared-required-vertices-${key}`,
          requiredVertexCount,
          'uint32',
          1
        ),
        uncertainCount: importView(`shared-uncertain-${key}`, uncertainCount, 'uint32', 1)
      }
    },
    readbackSources: [
      {buffer: runCount, size: 4, kind: 'uint32'},
      {buffer: vertexCount, size: 4, kind: 'uint32'},
      {buffer: overflow, size: 4, kind: 'uint32'},
      {buffer: requiredCount, size: 4, kind: 'uint32'},
      {buffer: requiredVertexCount, size: 4, kind: 'uint32'},
      {buffer: uncertainCount, size: 4, kind: 'uint32'},
      {buffer: leftLineIds, size: SHARED_RUN_CAPACITY * 4, kind: 'uint32'},
      {buffer: rightLineIds, size: SHARED_RUN_CAPACITY * 4, kind: 'uint32'},
      {buffer: forward, size: SHARED_RUN_CAPACITY * 4, kind: 'uint32'},
      {buffer: runOffsets, size: (SHARED_RUN_CAPACITY + 1) * 4, kind: 'uint32'},
      {buffer: runPositions, size: SHARED_VERTEX_CAPACITY * 8, kind: 'float32'}
    ] as BufferReadback[]
  };
}

function makeNodingOutput(
  resources: SpatialAnalysisResources,
  importView: GraphImporter,
  key: string
) {
  const endpoints = resources.createBuffer(`noded-endpoints-${key}`, NODED_SEGMENT_CAPACITY * 16);
  const sourceFeatureIds = resources.createBuffer(
    `noded-source-feature-${key}`,
    NODED_SEGMENT_CAPACITY * 4
  );
  const sourceRingIds = resources.createBuffer(
    `noded-source-ring-${key}`,
    NODED_SEGMENT_CAPACITY * 4
  );
  const sourceEdgeIds = resources.createBuffer(
    `noded-source-edge-${key}`,
    NODED_SEGMENT_CAPACITY * 4
  );
  const sourceStartParameters = resources.createBuffer(
    `noded-start-parameter-${key}`,
    NODED_SEGMENT_CAPACITY * 4
  );
  const sourceEndParameters = resources.createBuffer(
    `noded-end-parameter-${key}`,
    NODED_SEGMENT_CAPACITY * 4
  );
  const count = resources.createBuffer(`noded-count-${key}`, 4);
  const requiredCount = resources.createBuffer(`noded-required-count-${key}`, 4);
  const overflow = resources.createBuffer(`noded-overflow-${key}`, 4);
  const candidateOverflow = resources.createBuffer(`noded-candidate-overflow-${key}`, 4);
  const uncertainCount = resources.createBuffer(`noded-uncertain-${key}`, 4);
  return {
    output: {
      endpoints: importView(
        `noded-endpoints-${key}`,
        endpoints,
        'float32x4',
        NODED_SEGMENT_CAPACITY
      ),
      sourceFeatureIds: importView(
        `noded-source-feature-${key}`,
        sourceFeatureIds,
        'uint32',
        NODED_SEGMENT_CAPACITY
      ),
      sourceRingIds: importView(
        `noded-source-ring-${key}`,
        sourceRingIds,
        'uint32',
        NODED_SEGMENT_CAPACITY
      ),
      sourceEdgeIds: importView(
        `noded-source-edge-${key}`,
        sourceEdgeIds,
        'uint32',
        NODED_SEGMENT_CAPACITY
      ),
      sourceStartParameters: importView(
        `noded-start-parameter-${key}`,
        sourceStartParameters,
        'float32',
        NODED_SEGMENT_CAPACITY
      ),
      sourceEndParameters: importView(
        `noded-end-parameter-${key}`,
        sourceEndParameters,
        'float32',
        NODED_SEGMENT_CAPACITY
      ),
      status: {
        count: importView(`noded-count-${key}`, count, 'uint32', 1),
        requiredCount: importView(`noded-required-count-${key}`, requiredCount, 'uint32', 1),
        overflow: importView(`noded-overflow-${key}`, overflow, 'uint32', 1),
        candidateOverflow: importView(
          `noded-candidate-overflow-${key}`,
          candidateOverflow,
          'uint32',
          1
        ),
        uncertainCount: importView(`noded-uncertain-${key}`, uncertainCount, 'uint32', 1)
      }
    },
    readbackSources: [
      {buffer: count, size: 4, kind: 'uint32'},
      {buffer: requiredCount, size: 4, kind: 'uint32'},
      {buffer: overflow, size: 4, kind: 'uint32'},
      {buffer: candidateOverflow, size: 4, kind: 'uint32'},
      {buffer: uncertainCount, size: 4, kind: 'uint32'},
      {buffer: endpoints, size: NODED_SEGMENT_CAPACITY * 16, kind: 'float32'},
      {buffer: sourceFeatureIds, size: NODED_SEGMENT_CAPACITY * 4, kind: 'uint32'},
      {buffer: sourceRingIds, size: NODED_SEGMENT_CAPACITY * 4, kind: 'uint32'},
      {buffer: sourceEdgeIds, size: NODED_SEGMENT_CAPACITY * 4, kind: 'uint32'},
      {buffer: sourceStartParameters, size: NODED_SEGMENT_CAPACITY * 4, kind: 'float32'},
      {buffer: sourceEndParameters, size: NODED_SEGMENT_CAPACITY * 4, kind: 'float32'}
    ] as BufferReadback[]
  };
}

function makePolygonizeOutput(
  resources: SpatialAnalysisResources,
  importView: GraphImporter,
  key: string
): PolygonizeOutputBuffers {
  const directedCapacity = NODED_SEGMENT_CAPACITY * 2;
  const makeStatus = (name: string) => {
    const count = resources.createBuffer(`${name}-count-${key}`, 4);
    const requiredCount = resources.createBuffer(`${name}-required-${key}`, 4);
    const overflow = resources.createBuffer(`${name}-overflow-${key}`, 4);
    const candidateOverflow = resources.createBuffer(`${name}-candidate-overflow-${key}`, 4);
    const uncertainCount = resources.createBuffer(`${name}-uncertain-${key}`, 4);
    return {
      views: {
        count: importView(`${name}-count-${key}`, count, 'uint32', 1),
        requiredCount: importView(`${name}-required-${key}`, requiredCount, 'uint32', 1),
        overflow: importView(`${name}-overflow-${key}`, overflow, 'uint32', 1),
        candidateOverflow: importView(
          `${name}-candidate-overflow-${key}`,
          candidateOverflow,
          'uint32',
          1
        ),
        uncertainCount: importView(`${name}-uncertain-${key}`, uncertainCount, 'uint32', 1)
      },
      buffers: [count, requiredCount, overflow, candidateOverflow, uncertainCount]
    };
  };
  const finalStatus = makeStatus('polygonize-final');
  const nodedStatus = makeStatus('polygonize-noded');
  const directedStatus = makeStatus('polygonize-directed');
  const nodedEndpoints = resources.createBuffer(
    `polygonize-noded-endpoints-${key}`,
    NODED_SEGMENT_CAPACITY * 16
  );
  const nodedFeatureIds = resources.createBuffer(
    `polygonize-noded-features-${key}`,
    NODED_SEGMENT_CAPACITY * 4
  );
  const nodedRingIds = resources.createBuffer(
    `polygonize-noded-rings-${key}`,
    NODED_SEGMENT_CAPACITY * 4
  );
  const nodedEdgeIds = resources.createBuffer(
    `polygonize-noded-edges-${key}`,
    NODED_SEGMENT_CAPACITY * 4
  );
  const nodedStartParameters = resources.createBuffer(
    `polygonize-noded-starts-${key}`,
    NODED_SEGMENT_CAPACITY * 4
  );
  const nodedEndParameters = resources.createBuffer(
    `polygonize-noded-ends-${key}`,
    NODED_SEGMENT_CAPACITY * 4
  );
  const directedEndpoints = resources.createBuffer(
    `polygonize-directed-endpoints-${key}`,
    directedCapacity * 16
  );
  const directedSourceSegmentIds = resources.createBuffer(
    `polygonize-directed-segment-ids-${key}`,
    directedCapacity * 4
  );
  const directedSourceFeatureIds = resources.createBuffer(
    `polygonize-directed-feature-ids-${key}`,
    directedCapacity * 4
  );
  const directedTwinIds = resources.createBuffer(
    `polygonize-directed-twin-ids-${key}`,
    directedCapacity * 4
  );
  const directedRingIds = resources.createBuffer(
    `polygonize-directed-ring-ids-${key}`,
    directedCapacity * 4
  );
  const directedFlags = resources.createBuffer(
    `polygonize-directed-flags-${key}`,
    directedCapacity * 4
  );
  const positions = resources.createBuffer(
    `polygonize-face-positions-${key}`,
    POLYGON_VERTEX_CAPACITY * 8
  );
  const featureOffsets = resources.createBuffer(
    `polygonize-face-feature-offsets-${key}`,
    (POLYGON_RING_CAPACITY + 1) * 4
  );
  const polygonOffsets = resources.createBuffer(
    `polygonize-face-polygon-offsets-${key}`,
    (POLYGON_RING_CAPACITY + 1) * 4
  );
  const ringOffsets = resources.createBuffer(
    `polygonize-face-ring-offsets-${key}`,
    (POLYGON_RING_CAPACITY + 1) * 4
  );
  const sourceIds = resources.createBuffer(
    `polygonize-face-source-ids-${key}`,
    POLYGON_RING_CAPACITY * 4
  );
  const edgeClasses = resources.createBuffer(
    `polygonize-edge-classes-${key}`,
    NODED_SEGMENT_CAPACITY * 4
  );
  const diagnosticBuffers = [
    resources.createBuffer(`polygonize-closed-ring-count-${key}`, 4),
    resources.createBuffer(`polygonize-open-chain-count-${key}`, 4),
    resources.createBuffer(`polygonize-cut-edge-count-${key}`, 4),
    resources.createBuffer(`polygonize-dangle-count-${key}`, 4),
    resources.createBuffer(`polygonize-unused-edge-count-${key}`, 4)
  ];
  const output: GPUPolygonizeOutput = {
    nodedSegments: {
      endpoints: importView(
        `polygonize-noded-endpoints-${key}`,
        nodedEndpoints,
        'float32x4',
        NODED_SEGMENT_CAPACITY
      ),
      sourceFeatureIds: importView(
        `polygonize-noded-features-${key}`,
        nodedFeatureIds,
        'uint32',
        NODED_SEGMENT_CAPACITY
      ),
      sourceRingIds: importView(
        `polygonize-noded-rings-${key}`,
        nodedRingIds,
        'uint32',
        NODED_SEGMENT_CAPACITY
      ),
      sourceEdgeIds: importView(
        `polygonize-noded-edges-${key}`,
        nodedEdgeIds,
        'uint32',
        NODED_SEGMENT_CAPACITY
      ),
      sourceStartParameters: importView(
        `polygonize-noded-starts-${key}`,
        nodedStartParameters,
        'float32',
        NODED_SEGMENT_CAPACITY
      ),
      sourceEndParameters: importView(
        `polygonize-noded-ends-${key}`,
        nodedEndParameters,
        'float32',
        NODED_SEGMENT_CAPACITY
      ),
      status: nodedStatus.views
    },
    directedEdges: {
      endpoints: importView(
        `polygonize-directed-endpoints-${key}`,
        directedEndpoints,
        'float32x4',
        directedCapacity
      ),
      sourceSegmentIds: importView(
        `polygonize-directed-segment-ids-${key}`,
        directedSourceSegmentIds,
        'uint32',
        directedCapacity
      ),
      sourceFeatureIds: importView(
        `polygonize-directed-feature-ids-${key}`,
        directedSourceFeatureIds,
        'uint32',
        directedCapacity
      ),
      twinIds: importView(
        `polygonize-directed-twin-ids-${key}`,
        directedTwinIds,
        'uint32',
        directedCapacity
      ),
      ringIds: importView(
        `polygonize-directed-ring-ids-${key}`,
        directedRingIds,
        'uint32',
        directedCapacity
      ),
      flags: importView(
        `polygonize-directed-flags-${key}`,
        directedFlags,
        'uint32',
        directedCapacity
      ),
      status: directedStatus.views
    },
    polygons: {
      kind: 'polygons',
      positions: importView(
        `polygonize-face-positions-${key}`,
        positions,
        'float32x2',
        POLYGON_VERTEX_CAPACITY
      ),
      featureOffsets: importView(
        `polygonize-face-feature-offsets-${key}`,
        featureOffsets,
        'uint32',
        POLYGON_RING_CAPACITY + 1
      ),
      polygonOffsets: importView(
        `polygonize-face-polygon-offsets-${key}`,
        polygonOffsets,
        'uint32',
        POLYGON_RING_CAPACITY + 1
      ),
      ringOffsets: importView(
        `polygonize-face-ring-offsets-${key}`,
        ringOffsets,
        'uint32',
        POLYGON_RING_CAPACITY + 1
      ),
      sourceIds: importView(
        `polygonize-face-source-ids-${key}`,
        sourceIds,
        'uint32',
        POLYGON_RING_CAPACITY
      )
    },
    diagnostics: {
      edgeClasses: importView(
        `polygonize-edge-classes-${key}`,
        edgeClasses,
        'uint32',
        NODED_SEGMENT_CAPACITY
      ),
      closedRingCount: importView(
        `polygonize-closed-ring-count-${key}`,
        diagnosticBuffers[0],
        'uint32',
        1
      ),
      openChainCount: importView(
        `polygonize-open-chain-count-${key}`,
        diagnosticBuffers[1],
        'uint32',
        1
      ),
      cutEdgeCount: importView(
        `polygonize-cut-edge-count-${key}`,
        diagnosticBuffers[2],
        'uint32',
        1
      ),
      dangleCount: importView(`polygonize-dangle-count-${key}`, diagnosticBuffers[3], 'uint32', 1),
      unusedEdgeCount: importView(
        `polygonize-unused-edge-count-${key}`,
        diagnosticBuffers[4],
        'uint32',
        1
      )
    },
    status: finalStatus.views
  };
  const finalStatusBuffers = finalStatus.buffers;
  return {
    output,
    readbackSources: [
      ...finalStatusBuffers.map(buffer => ({buffer, size: 4, kind: 'uint32' as const})),
      ...diagnosticBuffers.map(buffer => ({buffer, size: 4, kind: 'uint32' as const})),
      {buffer: edgeClasses, size: NODED_SEGMENT_CAPACITY * 4, kind: 'uint32'},
      ...nodedStatus.buffers.map(buffer => ({buffer, size: 4, kind: 'uint32' as const})),
      {buffer: nodedEndpoints, size: NODED_SEGMENT_CAPACITY * 16, kind: 'float32'},
      {buffer: positions, size: POLYGON_VERTEX_CAPACITY * 8, kind: 'float32'},
      {buffer: featureOffsets, size: (POLYGON_RING_CAPACITY + 1) * 4, kind: 'uint32'},
      {buffer: polygonOffsets, size: (POLYGON_RING_CAPACITY + 1) * 4, kind: 'uint32'},
      {buffer: ringOffsets, size: (POLYGON_RING_CAPACITY + 1) * 4, kind: 'uint32'},
      {buffer: sourceIds, size: POLYGON_RING_CAPACITY * 4, kind: 'uint32'}
    ]
  };
}

function decodeSummary(bytes: ArrayBuffer, variant: Variant): CountyLineworkSummary {
  const wordLength = bytes.byteLength / 4;
  let wordOffset = 0;
  const takeUint = (length: number) => {
    const values = new Uint32Array(bytes, wordOffset * 4, length).slice();
    wordOffset += length;
    return values;
  };
  const takeFloat = (length: number) => {
    const values = new Float32Array(bytes, wordOffset * 4, length).slice();
    wordOffset += length;
    return values;
  };
  const scalar = () => takeUint(1)[0] ?? 0;
  const sharedCount = scalar();
  const sharedVertexCount = scalar();
  const sharedOverflow = scalar();
  const sharedRequiredCount = scalar();
  const sharedRequiredVertexCount = scalar();
  const sharedUncertainCount = scalar();
  const sharedLeftIds = takeUint(SHARED_RUN_CAPACITY);
  const sharedRightIds = takeUint(SHARED_RUN_CAPACITY);
  const sharedForward = takeUint(SHARED_RUN_CAPACITY);
  const sharedOffsets = takeUint(SHARED_RUN_CAPACITY + 1);
  const sharedPositions = takeFloat(SHARED_VERTEX_CAPACITY * 2);
  const sharedRuns = [];
  let forwardCount = 0;
  for (let run = 0; run < Math.min(sharedCount, SHARED_RUN_CAPACITY); run++) {
    const start = sharedOffsets[run];
    const end = Math.min(sharedOffsets[run + 1], sharedVertexCount);
    const path = Array.from({length: Math.max(0, end - start)}, (_, index) =>
      LINEWORK_FRAME.unproject(
        sharedPositions[(start + index) * 2] + variant.input.coordinateOrigin[0],
        sharedPositions[(start + index) * 2 + 1] + variant.input.coordinateOrigin[1]
      )
    );
    const forward = sharedForward[run] === 1;
    if (forward) forwardCount++;
    sharedRuns.push({
      path,
      forward,
      leftLineId: sharedLeftIds[run],
      rightLineId: sharedRightIds[run]
    });
  }
  const nodingCount = scalar();
  const nodingRequiredCount = scalar();
  const nodingOverflow = scalar();
  const nodingCandidateOverflow = scalar();
  const nodingUncertainCount = scalar();
  const nodingEndpoints = takeFloat(NODED_SEGMENT_CAPACITY * 4);
  const nodingFeatureIds = takeUint(NODED_SEGMENT_CAPACITY);
  const nodingRingIds = takeUint(NODED_SEGMENT_CAPACITY);
  const nodingEdgeIds = takeUint(NODED_SEGMENT_CAPACITY);
  const nodingStarts = takeFloat(NODED_SEGMENT_CAPACITY);
  const nodingEnds = takeFloat(NODED_SEGMENT_CAPACITY);
  const nodedSegments = Array.from(
    {length: Math.min(nodingCount, NODED_SEGMENT_CAPACITY)},
    (_, index) => ({
      path: [
        LINEWORK_FRAME.unproject(
          nodingEndpoints[index * 4] + variant.input.coordinateOrigin[0],
          nodingEndpoints[index * 4 + 1] + variant.input.coordinateOrigin[1]
        ),
        LINEWORK_FRAME.unproject(
          nodingEndpoints[index * 4 + 2] + variant.input.coordinateOrigin[0],
          nodingEndpoints[index * 4 + 3] + variant.input.coordinateOrigin[1]
        )
      ] as [number, number][],
      sourceFeatureId: nodingFeatureIds[index],
      sourceRingId: nodingRingIds[index],
      sourceEdgeId: nodingEdgeIds[index],
      startParameter: nodingStarts[index],
      endParameter: nodingEnds[index]
    })
  );
  const polygonCount = scalar();
  const polygonRequiredCount = scalar();
  const polygonOverflow = scalar();
  const polygonCandidateOverflow = scalar();
  const polygonUncertainCount = scalar();
  const closedRingCount = scalar();
  const openChainCount = scalar();
  const cutEdgeCount = scalar();
  const dangleCount = scalar();
  const unusedEdgeCount = scalar();
  const polygonEdgeClasses = takeUint(NODED_SEGMENT_CAPACITY);
  const polygonNodedCount = scalar();
  const polygonNodedRequiredCount = scalar();
  const polygonNodedOverflow = scalar();
  const polygonNodedCandidateOverflow = scalar();
  const polygonNodedUncertainCount = scalar();
  const polygonNodedEndpoints = takeFloat(NODED_SEGMENT_CAPACITY * 4);
  const polygonPositions = takeFloat(POLYGON_VERTEX_CAPACITY * 2);
  const featureOffsets = takeUint(POLYGON_RING_CAPACITY + 1);
  const polygonOffsets = takeUint(POLYGON_RING_CAPACITY + 1);
  const ringOffsets = takeUint(POLYGON_RING_CAPACITY + 1);
  const sourceIds = takeUint(POLYGON_RING_CAPACITY);
  if (wordOffset !== wordLength)
    throw new Error('County linework summary buffer layout was not fully decoded.');
  const boundedPolygonCount = Math.min(polygonCount, POLYGON_RING_CAPACITY);
  const polygonFeatureCount = Math.min(
    featureOffsets[boundedPolygonCount] ?? 0,
    POLYGON_RING_CAPACITY
  );
  const writtenRingCount = Math.min(
    polygonOffsets[polygonFeatureCount] ?? 0,
    POLYGON_RING_CAPACITY
  );
  const polygonFaces = [];
  const faceAreas: number[] = [];
  for (let feature = 0; feature < boundedPolygonCount; feature++) {
    for (let polygon = featureOffsets[feature]; polygon < featureOffsets[feature + 1]; polygon++) {
      const rings: [number, number][][] = [];
      let faceArea = 0;
      for (let ring = polygonOffsets[polygon]; ring < polygonOffsets[polygon + 1]; ring++) {
        const start = ringOffsets[ring];
        const end = Math.min(ringOffsets[ring + 1], POLYGON_VERTEX_CAPACITY);
        let doubledRingArea = 0;
        for (let vertex = start; vertex < end; vertex++) {
          const nextVertex = vertex + 1 < end ? vertex + 1 : start;
          doubledRingArea +=
            polygonPositions[vertex * 2] * polygonPositions[nextVertex * 2 + 1] -
            polygonPositions[nextVertex * 2] * polygonPositions[vertex * 2 + 1];
        }
        faceArea += (ring === polygonOffsets[polygon] ? 1 : -1) * Math.abs(doubledRingArea / 2);
        rings.push(
          Array.from({length: Math.max(0, end - start)}, (_, index) =>
            LINEWORK_FRAME.unproject(
              polygonPositions[(start + index) * 2] + variant.input.coordinateOrigin[0],
              polygonPositions[(start + index) * 2 + 1] + variant.input.coordinateOrigin[1]
            )
          )
        );
      }
      polygonFaces.push({rings, sourceId: sourceIds[feature]});
      faceAreas.push(faceArea);
    }
  }
  const diagnosticPaths = Array.from(
    {length: Math.min(polygonNodedCount, NODED_SEGMENT_CAPACITY)},
    (_, index) => ({
      path: [
        LINEWORK_FRAME.unproject(
          polygonNodedEndpoints[index * 4] + variant.input.coordinateOrigin[0],
          polygonNodedEndpoints[index * 4 + 1] + variant.input.coordinateOrigin[1]
        ),
        LINEWORK_FRAME.unproject(
          polygonNodedEndpoints[index * 4 + 2] + variant.input.coordinateOrigin[0],
          polygonNodedEndpoints[index * 4 + 3] + variant.input.coordinateOrigin[1]
        )
      ] as [number, number][],
      flags: polygonEdgeClasses[index]
    })
  );
  const summary: CountyLineworkSummary = {
    shared: {
      count: sharedCount,
      vertexCount: sharedVertexCount,
      requiredCount: sharedRequiredCount,
      requiredVertexCount: sharedRequiredVertexCount,
      overflow: sharedOverflow,
      uncertainCount: sharedUncertainCount,
      forwardCount,
      backwardCount: Math.max(0, sharedRuns.length - forwardCount),
      runs: sharedRuns
    },
    noding: {
      count: nodingCount,
      requiredCount: nodingRequiredCount,
      overflow: nodingOverflow,
      candidateOverflow: nodingCandidateOverflow,
      uncertainCount: nodingUncertainCount,
      segments: nodedSegments
    },
    polygonize: {
      count: polygonCount,
      requiredCount: polygonRequiredCount,
      writtenRingCount,
      faceAreas,
      overflow: polygonOverflow,
      candidateOverflow: polygonCandidateOverflow,
      uncertainCount: polygonUncertainCount,
      nodedSegmentCount: polygonNodedCount,
      nodedRequiredCount: polygonNodedRequiredCount,
      nodedOverflow: polygonNodedOverflow,
      nodedCandidateOverflow: polygonNodedCandidateOverflow,
      nodedUncertainCount: polygonNodedUncertainCount,
      edgeClasses: polygonEdgeClasses,
      diagnostics: Uint32Array.of(
        closedRingCount,
        openChainCount,
        cutEdgeCount,
        dangleCount,
        unusedEdgeCount
      ),
      diagnosticPaths,
      polygons: polygonFaces
    }
  };
  return summary;
}

function getPolygonizeDiagnosticPaths(summary: PolygonizeSummary) {
  return summary.diagnosticPaths.map(row => ({
    path: row.path,
    color:
      row.flags & 8 ? [218, 75, 60, 255] : row.flags & 4 ? [181, 134, 72, 240] : [70, 122, 137, 150]
  }));
}

function sharedComplete(summary: SharedSummary): boolean {
  return (
    summary.overflow === 0 &&
    summary.uncertainCount === 0 &&
    summary.count === summary.requiredCount &&
    summary.vertexCount === summary.requiredVertexCount
  );
}

function nodingComplete(summary: NodingSummary): boolean {
  return (
    summary.overflow === 0 &&
    summary.candidateOverflow === 0 &&
    summary.uncertainCount === 0 &&
    summary.count === summary.requiredCount
  );
}

function polygonizeComplete(summary: PolygonizeSummary): boolean {
  return (
    summary.overflow === 0 &&
    summary.candidateOverflow === 0 &&
    summary.uncertainCount === 0 &&
    summary.nodedSegmentCount === summary.nodedRequiredCount &&
    summary.nodedOverflow === 0 &&
    summary.nodedCandidateOverflow === 0 &&
    summary.nodedUncertainCount === 0
  );
}

function publishSummary(
  ctx: SceneContext<CountyLineworkOptions>,
  summary: CountyLineworkSummary,
  variant: Variant
): void {
  const shared = summary.shared;
  ctx.setReadout(
    'countyLineworkShared',
    `GPUSharedPaths ${sharedComplete(shared) ? 'complete' : 'incomplete'}; ${formatCount(shared.count)} / ${formatCount(shared.requiredCount)} runs, ${formatCount(shared.vertexCount)} / ${formatCount(shared.requiredVertexCount)} vertices; forward ${formatCount(shared.forwardCount)}, backward ${formatCount(shared.backwardCount)}; uncertain ${formatCount(shared.uncertainCount)}, overflow ${shared.overflow}. ` +
      shared.runs
        .map(
          run =>
            `left line ${run.leftLineId}, right line ${run.rightLineId}, ${run.forward ? 'forward' : 'backward'}, ${formatCount(run.path.length)} vertices`
        )
        .join('; ')
  );
  const noding = summary.noding;
  ctx.setReadout(
    'countyLineworkNoding',
    `GPULineNoding ${nodingComplete(noding) ? 'complete' : 'partial'}; ${formatCount(noding.count)} / ${formatCount(noding.requiredCount)} atomic segments; overflow ${noding.overflow}, candidateOverflow ${noding.candidateOverflow}, uncertain ${formatCount(noding.uncertainCount)}. ` +
      noding.segments
        .slice(0, 6)
        .map(
          segment =>
            `FIPS ${segment.sourceFeatureId}, ring ${segment.sourceRingId}, source edge ${segment.sourceEdgeId}, t ${segment.startParameter.toFixed(3)}–${segment.endParameter.toFixed(3)}`
        )
        .join('; ')
  );
  const polygonize = summary.polygonize;
  ctx.setReadout(
    'countyLineworkPolygonize',
    `GPUPolygonize ${polygonizeComplete(polygonize) ? 'complete' : 'partial'}; ${formatCount(polygonize.count)} bounded faces; face areas ${polygonize.faceAreas.map(area => area.toFixed(3)).join(', ')} m²; ${formatCount(polygonize.writtenRingCount)} bounded output face rings; ${formatCount(polygonize.requiredCount)} total assembly rings required (including unbounded walks); noded input ${formatCount(polygonize.nodedSegmentCount)} / ${formatCount(polygonize.nodedRequiredCount)} segments (overflow ${polygonize.nodedOverflow}, candidateOverflow ${polygonize.nodedCandidateOverflow}, uncertain ${formatCount(polygonize.nodedUncertainCount)}); polygon overflow ${polygonize.overflow}, candidateOverflow ${polygonize.candidateOverflow}, uncertain ${formatCount(polygonize.uncertainCount)}; closed rings ${formatCount(polygonize.diagnostics[0])}, open chains ${formatCount(polygonize.diagnostics[1])}, cut edges ${formatCount(polygonize.diagnostics[2])}, dangles ${formatCount(polygonize.diagnostics[3])}, unused ${formatCount(polygonize.diagnostics[4])}; local-coordinate vertex matching tolerance ${getPolygonizeVertexTolerance(variant.input).toPrecision(4)} m, checked below one quarter of the closest distinct source-vertex spacing. ${variant.input.omittedEdge ? `Scenario omits source edge ${variant.input.omittedEdge.sourceEdgeId} from county FIPS ${variant.input.omittedEdge.countyFips}.` : 'Polygonize nodes the input internally; the separate GPULineNoding graph is for inspecting atomic edges and provenance.'}`
  );
  ctx.setChart('countyLineworkDirection', makeDirectionChart(shared));
  ctx.setChart('countyLineworkDiagnostics', makeDiagnosticChart(polygonize.diagnostics));
}

function makeDirectionChart(shared: SharedSummary): BarChartData {
  return {
    kind: 'bars',
    title: 'Shared boundary runs by direction',
    labels: ['Forward', 'Backward'],
    values: Uint32Array.of(shared.forwardCount, shared.backwardCount),
    color: 3,
    description:
      'GPUSharedPaths direction of the right county ring relative to the left ring. Reversing the right ring changes these counts without changing its geographic boundary.'
  };
}

function makeDiagnosticChart(values: Uint32Array): BarChartData {
  return {
    kind: 'bars',
    title: 'GPU polygonization edge diagnostics',
    labels: ['Closed ring edges', 'Open-chain edges', 'Cut edges', 'Dangles', 'Unused edges'],
    values,
    color: 2,
    description:
      'Actual GPUPolygonize diagnostics; an intentionally omitted outer boundary edge is an input scenario, not a source defect.'
  };
}

function inputDescription(input: CountyLineworkInput): string {
  const leftVertices = input.leftPositions.length / 2;
  const rightVertices = input.rightPositions.length / 2;
  return `${input.leftRingCount + input.rightRingCount} rings, ${leftVertices + rightVertices} line vertices (${leftVertices} left, ${rightVertices} right); local Albers frame rebased at [${input.coordinateOrigin[0].toFixed(1)}, ${input.coordinateOrigin[1].toFixed(1)}] m after global float32 quantization`;
}
