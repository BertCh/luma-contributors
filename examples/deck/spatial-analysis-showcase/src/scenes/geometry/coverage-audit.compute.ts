// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/graph/gpu-core';
import {
  getGPUCoverageValidityParameterValues,
  GPUCoverageDissolve,
  GPUCoverageValidity
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {addKernelPass} from '../../engine/mode-kernels';
import {getAnalysisFrameSpatialContext, ALBERS_CONUS_FRAME} from '../../engine/analysis-frame';
import {GraphSwap} from '../../engine/graph-swap';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {BarChartData} from '../chart-types';
import type {SceneContext} from '../scene';
import {
  buildCoverageAuditData,
  COVERAGE_AUDIT_PERTURBATION_METERS,
  COVERAGE_AUDIT_STATE_FIPS,
  COVERAGE_AUDIT_STATE_NAMES,
  type CoverageAuditData
} from './coverage-audit-data';
import {
  createGraphImporter,
  createStaticPaths,
  copyCountToDrawRecord,
  type GeometryView,
  type PolygonLayout
} from './b3-common';
import {PathOutputLayer} from './b3-layers';
import type {NodingAndCoverageOptions} from './b3-noding-options';

type Options = NodingAndCoverageOptions;
type AuditInput = Options['coverageAuditInput'];
type AuditGroup = Options['coverageAuditGroup'];

const FLAG_COLORS: readonly (readonly [number, number, number, number])[] = [
  [116, 137, 143, 255],
  [237, 177, 74, 255],
  [231, 94, 78, 255],
  [202, 105, 175, 255],
  [133, 106, 208, 255],
  [239, 235, 214, 255],
  [67, 215, 188, 255]
];
const STATE_COLORS: readonly (readonly [number, number, number, number])[] = [
  [65, 174, 188, 255],
  [240, 184, 93, 255],
  [170, 132, 204, 255]
];
const FLAG_BREAKS = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5] as const;
const STATE_BREAKS = [17.5, 36.5] as const;
const FLAG_LABELS = [
  'unmatched edge / T-junction',
  'proper crossing',
  'narrow gap',
  'overlap',
  'uncertain',
  'multiple flags'
] as const;

type ReadbackSource = {buffer: Buffer; size: number};

type AuditSnapshot = {
  valid: boolean;
  invalidCount: number;
  featureInvalidCounts: Uint32Array;
  flags: Uint32Array;
  flagCounts: Uint32Array;
  dissolveReady: boolean;
  dissolveStatus: {
    ringCount: number;
    requiredRings: number;
    overflow: number;
    openSegments: number;
    touchingSegments: number;
    boundarySegments: number;
    vertexCount: number;
    polygonCount: number;
    holeCount: number;
    sourceIds: Uint32Array;
    ringGroups: Uint32Array;
  };
};

type AuditVariant = {
  key: string;
  input: AuditInput;
  group: AuditGroup;
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  parameters: ReturnType<SpatialAnalysisResources['createParameterBuffer']>;
  source: ReturnType<typeof createStaticPaths>;
  edgeCategories: Buffer;
  dissolved: {
    positions: Buffer;
    ringOffsets: Buffer;
    vertexCount: Buffer;
    drawCommands: DrawCommandBuffer;
    ringCategoryValues: Buffer;
  };
  parametersApplied: number;
  dirty: boolean;
  reader: SummaryReader | null;
  snapshot: AuditSnapshot | null;
  generation: number;
  readbackSources: ReadbackSource[];
};

/** GPU coverage validity flags and a validity-gated GPU coverage dissolve for a county subset. */
export async function createCoverageAuditView(
  ctx: SceneContext<Options>,
  getCountyLayout: () => PolygonLayout
): Promise<GeometryView<Options>> {
  const countyDataset = ctx.datasets.get('us-counties');
  const sourceLayout = getCountyLayout();
  const auditData = buildCoverageAuditData(
    sourceLayout,
    countyDataset.column<Uint8Array>('stateFips'),
    countyDataset.column<Uint32Array>('fips'),
    countyDataset.count
  );
  const cache = new Map<string, AuditVariant>();
  const graphSwap = new GraphSwap();
  let active: AuditVariant | null = null;
  let generation = 0;
  let destroyed = false;

  const currentKey = (options: Readonly<Options>) =>
    `${options.coverageAuditInput}-${options.coverageAuditGroup}`;
  const clearReadouts = () => {
    ctx.setChart('coverageAuditFlags', null);
    ctx.setReadout('coverageAuditStatus', 'GPU coverage validity is pending.');
    ctx.setReadout('coverageAuditBits', 'GPU invalid-edge flags are pending.');
    ctx.setReadout('coverageAuditByState', 'GPU feature counts are pending.');
    ctx.setReadout('coverageAuditDissolveStatus', 'Coverage dissolve is pending.');
    ctx.setReadout('coverageAuditDissolveSummary', 'GPU polygon output is pending.');
  };

  const makeReader = (variant: AuditVariant, readerGeneration: number) => {
    variant.reader?.release();
    variant.generation = readerGeneration;
    variant.reader = new SummaryReader(
      variant.resources,
      `coverage-audit-${variant.key}-${readerGeneration}`,
      getReadbackSources(variant),
      bytes => {
        if (
          destroyed ||
          active !== variant ||
          variant.generation !== readerGeneration ||
          generation !== readerGeneration ||
          currentKey(ctx.options) !== variant.key
        )
          return;
        const snapshot = decodeAuditSnapshot(bytes, auditData, variant.group);
        variant.snapshot = snapshot;
        variant.edgeCategories.write(makeFlagCategoryValues(snapshot.flags));
        variant.dissolved.ringCategoryValues.write(
          makeGroupCategoryValues(
            snapshot.dissolveStatus.ringGroups,
            snapshot.dissolveStatus.ringCount,
            variant.group
          )
        );
        if (snapshot.valid) {
          ctx.setReadout(
            'coverageAuditStatus',
            `GPUCoverageValidity: isValid=1; invalid segments ${formatCount(snapshot.invalidCount)}; gapWidth ${ctx.options.coverageAuditGapWidth} m; the selected features have no internal segment defect under this rule.`
          );
        } else {
          ctx.setReadout(
            'coverageAuditStatus',
            `GPUCoverageValidity: isValid=0; invalid segments ${formatCount(snapshot.invalidCount)}; gapWidth ${ctx.options.coverageAuditGapWidth} m; invalid segments are colored, and dissolve output is withheld.`
          );
        }
        ctx.setReadout(
          'coverageAuditBits',
          FLAG_LABELS.map(
            (label, index) => `${label} ${formatCount(snapshot.flagCounts[index])}`
          ).join('; ')
        );
        ctx.setReadout(
          'coverageAuditByState',
          COVERAGE_AUDIT_STATE_FIPS.map((stateFips, index) => {
            const invalidFeatures = auditData.featureStateFips.reduce(
              (count, featureStateFips, featureIndex) =>
                count +
                Number(
                  featureStateFips === stateFips && snapshot.featureInvalidCounts[featureIndex] > 0
                ),
              0
            );
            return `${COVERAGE_AUDIT_STATE_NAMES[stateFips]} (${stateFips}): ${formatCount(auditData.countsByState.get(stateFips) ?? 0)} counties, ${formatCount(invalidFeatures)} invalid`;
          }).join('; ')
        );
        ctx.setChart('coverageAuditFlags', makeFlagChart(snapshot.flagCounts));
        const dissolve = snapshot.dissolveStatus;
        ctx.setReadout(
          'coverageAuditDissolveStatus',
          `GPUCoverageDissolve ${snapshot.dissolveReady ? 'written' : 'withheld'}; labels ${variant.group === 'state' ? 'state FIPS' : 'combined label 0'}; input isValid=${Number(snapshot.valid)}; rings ${formatCount(dissolve.ringCount)}/${formatCount(dissolve.requiredRings)} required; overflow=${dissolve.overflow}; open=${formatCount(dissolve.openSegments)}; touching=${formatCount(dissolve.touchingSegments)}.`
        );
        ctx.setReadout(
          'coverageAuditDissolveSummary',
          snapshot.dissolveReady
            ? `${formatCount(dissolve.polygonCount)} GPU output polygons; ${formatCount(dissolve.ringCount)} rings (${formatCount(dissolve.holeCount)} holes); ${formatCount(dissolve.vertexCount)} output vertices; ${formatCount(dissolve.boundarySegments)} boundary segments; ${variant.group === 'state' ? 'polygon source IDs are state FIPS' : 'all polygons use combined group label 0'}: ${Array.from(dissolve.sourceIds.subarray(0, dissolve.polygonCount)).join(', ')}.`
            : 'No dissolved geometry is drawn or treated as valid until the GPU coverage audit and output capacity checks pass.'
        );
        ctx.requestLayers();
      }
    );
  };

  const buildVariant = async (options: Readonly<Options>): Promise<AuditVariant> => {
    const input = options.coverageAuditInput;
    const group = options.coverageAuditGroup;
    const key = currentKey(options);
    const cached = cache.get(key);
    if (cached) return cached;
    const resources = new SpatialAnalysisResources(ctx.device, `coverage-audit-${key}`);
    const layout = auditData.layout;
    const vertexCount = layout.vertexCount;
    const featureCount = layout.featureCount;
    const ringCount = layout.ringCount;
    const polygonCount = layout.partCount;
    const ringCapacity = vertexCount;
    const positionValues =
      input === 'source' ? auditData.projectedPositions : auditData.perturbedPositions;
    const positionBuffer = resources.createBuffer('positions-albers', positionValues);
    const ringOffsetBuffer = resources.createBuffer('ring-offsets', layout.ringOffsets);
    const polygonOffsetBuffer = resources.createBuffer('polygon-offsets', layout.polygonOffsets);
    const featureOffsetBuffer = resources.createBuffer('feature-offsets', layout.featureOffsets);
    const labels = group === 'state' ? auditData.featureStateFips : new Uint32Array(featureCount);
    const labelBuffer = resources.createBuffer('group-labels', labels);
    const parameterBuffer = resources.createParameterBuffer(
      'gap-width',
      'float32',
      4,
      getGPUCoverageValidityParameterValues({gapWidth: Number(options.coverageAuditGapWidth)})
    );
    const segmentFlags = resources.createBuffer('coverage-segment-flags', vertexCount * 4);
    const featureInvalidCounts = resources.createBuffer(
      'invalid-counts-by-feature',
      featureCount * 4
    );
    const isValid = resources.createBuffer('coverage-is-valid', 4);
    const invalidCount = resources.createBuffer('invalid-segment-count', 4);

    const dissolvedPositions = resources.createBuffer(
      'dissolved-rings',
      (vertexCount + ringCapacity) * 8
    );
    const dissolvedRingOffsets = resources.createBuffer(
      'dissolved-ring-offsets',
      (ringCapacity + 1) * 4
    );
    const ringAreas = resources.createBuffer('dissolved-ring-areas', ringCapacity * 4);
    const ringIsHole = resources.createBuffer('dissolved-ring-holes', ringCapacity * 4);
    const ringShells = resources.createBuffer('dissolved-ring-shells', ringCapacity * 4);
    const ringGroups = resources.createBuffer('dissolved-ring-groups', ringCapacity * 4);
    const dissolvedPolygonPositions = resources.createBuffer(
      'dissolved-polygon-positions',
      (vertexCount + ringCapacity) * 8
    );
    const dissolvedPolygonRingOffsets = resources.createBuffer(
      'dissolved-polygon-ring-offsets',
      (ringCapacity + 1) * 4
    );
    const dissolvedPolygonOffsets = resources.createBuffer(
      'dissolved-polygon-offsets',
      (ringCapacity + 1) * 4
    );
    const dissolvedFeatureOffsets = resources.createBuffer(
      'dissolved-feature-offsets',
      (ringCapacity + 1) * 4
    );
    const dissolvedSourceIds = resources.createBuffer('dissolved-source-fips', ringCapacity * 4);
    const dissolvedRingCount = resources.createBuffer('dissolved-ring-count', 4);
    const dissolvedRequiredRings = resources.createBuffer('dissolved-required-rings', 4);
    const dissolvedOverflow = resources.createBuffer('dissolved-overflow', 4);
    const dissolvedOpenSegments = resources.createBuffer('dissolved-open-segments', 4);
    const dissolvedTouchingSegments = resources.createBuffer('dissolved-touching-segments', 4);
    const boundarySegmentCount = resources.createBuffer('dissolved-boundary-segments', 4);
    const dissolvedVertexCount = resources.createBuffer('dissolved-vertex-count', 4);
    const dissolvedPolygonCount = resources.createBuffer('dissolved-polygon-count', 4);
    const graph = new GPUCommandGraph<void>(ctx.device, {id: `coverage-audit-${key}`});
    const importView = createGraphImporter(graph);
    const positions = importView('positions', positionBuffer, 'float32x2', vertexCount);
    const ringOffsets = importView('ring-offsets', ringOffsetBuffer, 'uint32', ringCount + 1);
    const polygonOffsets = importView(
      'polygon-offsets',
      polygonOffsetBuffer,
      'uint32',
      polygonCount + 1
    );
    const featureOffsets = importView(
      'feature-offsets',
      featureOffsetBuffer,
      'uint32',
      featureCount + 1
    );
    const parameterView = parameterBuffer.importToGraph(graph);
    const flagsView = importView('coverage-segment-flags', segmentFlags, 'uint32', vertexCount);
    const featureInvalidCountView = importView(
      'invalid-counts-by-feature',
      featureInvalidCounts,
      'uint32',
      featureCount
    );
    const isValidView = importView('coverage-is-valid', isValid, 'uint32', 1);
    const invalidCountView = importView('invalid-segment-count', invalidCount, 'uint32', 1);
    const groupView = importView('group-labels', labelBuffer, 'uint32', featureCount);
    const outputRingOffsetsView = importView(
      'dissolved-ring-offsets',
      dissolvedRingOffsets,
      'uint32',
      ringCapacity + 1
    );
    const outputRingCountView = importView('dissolved-ring-count', dissolvedRingCount, 'uint32', 1);
    const outputFeatureOffsetsView = importView(
      'dissolved-feature-offsets',
      dissolvedFeatureOffsets,
      'uint32',
      ringCapacity + 1
    );

    graph.add(
      new GPUCoverageValidity({
        id: 'source-coverage-validity',
        spatialContext: getAnalysisFrameSpatialContext(ALBERS_CONUS_FRAME),
        positions,
        ringOffsets,
        polygonOffsets,
        featureOffsets,
        parameters: parameterView,
        output: {
          segmentFlags: flagsView,
          featureInvalidCounts: featureInvalidCountView,
          isValid: isValidView,
          status: {invalidCount: invalidCountView}
        }
      })
    );

    graph.add(
      new GPUCoverageDissolve({
        id: 'coverage-dissolve',
        spatialContext: getAnalysisFrameSpatialContext(ALBERS_CONUS_FRAME),
        positions,
        ringOffsets,
        polygonOffsets,
        featureOffsets,
        labels: groupView,
        snapGridSize: 0,
        vertexTolerance: 0.001,
        splitTouchingRings: true,
        output: {
          ringOffsets: outputRingOffsetsView,
          positions: importView(
            'dissolved-rings',
            dissolvedPositions,
            'float32x2',
            vertexCount + ringCapacity
          ),
          ringAreas: importView('dissolved-ring-areas', ringAreas, 'float32', ringCapacity),
          ringIsHole: importView('dissolved-ring-holes', ringIsHole, 'uint32', ringCapacity),
          ringShells: importView('dissolved-ring-shells', ringShells, 'uint32', ringCapacity),
          ringGroups: importView('dissolved-ring-groups', ringGroups, 'uint32', ringCapacity),
          polygons: {
            kind: 'polygons',
            positions: importView(
              'dissolved-polygon-positions',
              dissolvedPolygonPositions,
              'float32x2',
              vertexCount + ringCapacity
            ),
            ringOffsets: importView(
              'dissolved-polygon-ring-offsets',
              dissolvedPolygonRingOffsets,
              'uint32',
              ringCapacity + 1
            ),
            polygonOffsets: importView(
              'dissolved-polygon-offsets',
              dissolvedPolygonOffsets,
              'uint32',
              ringCapacity + 1
            ),
            featureOffsets: outputFeatureOffsetsView,
            sourceIds: importView(
              'dissolved-source-fips',
              dissolvedSourceIds,
              'uint32',
              ringCapacity
            )
          },
          status: {
            count: outputRingCountView,
            overflow: importView('dissolved-overflow', dissolvedOverflow, 'uint32', 1),
            requiredCount: importView(
              'dissolved-required-rings',
              dissolvedRequiredRings,
              'uint32',
              1
            ),
            openSegmentCount: importView(
              'dissolved-open-segments',
              dissolvedOpenSegments,
              'uint32',
              1
            ),
            touchingSegmentCount: importView(
              'dissolved-touching-segments',
              dissolvedTouchingSegments,
              'uint32',
              1
            ),
            boundarySegmentCount: importView(
              'dissolved-boundary-segments',
              boundarySegmentCount,
              'uint32',
              1
            )
          }
        }
      })
    );

    addKernelPass(graph, {
      id: 'dissolve-draw-counts',
      invocationCount: 1,
      bindings: [
        {name: 'ringOffsets', view: outputRingOffsetsView, type: 'u32', access: 'read'},
        {name: 'ringCount', view: outputRingCountView, type: 'u32', access: 'read'},
        {
          name: 'vertexCount',
          view: importView('dissolved-vertex-count', dissolvedVertexCount, 'uint32', 1),
          type: 'u32',
          access: 'read_write'
        },
        {name: 'featureOffsets', view: outputFeatureOffsetsView, type: 'u32', access: 'read'},
        {
          name: 'polygonCount',
          view: importView('dissolved-polygon-count', dissolvedPolygonCount, 'uint32', 1),
          type: 'u32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
let ringTotal = min(ringCount[ringCountOffset], ${ringCapacity}u);
vertexCount[vertexCountOffset] = ringOffsets[ringOffsetsOffset + ringTotal];
polygonCount[polygonCountOffset] = featureOffsets[featureOffsetsOffset + ${ringCapacity}u];`
    });

    let compiled: CompiledGPUCommandGraph<void>;
    try {
      compiled = resources.track(await graph.compileAsync());
    } catch (error) {
      resources.destroy();
      throw error;
    }
    const staticSource = createStaticPaths(
      resources,
      'audit-source',
      positionValues,
      layout.ringOffsets
    );
    const edgeCategories = resources.createBuffer(
      'audit-edge-category-values',
      new Float32Array(vertexCount)
    );
    const ringCategoryValues = resources.createBuffer(
      'dissolved-ring-category-values',
      new Float32Array(ringCapacity)
    );
    const drawCommands = new DrawCommandBuffer(ctx.device, {
      id: `dissolved-rings-draw-${key}`,
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    });
    resources.track(drawCommands);
    const variant: AuditVariant = {
      key,
      input,
      group,
      resources,
      compiled,
      parameters: parameterBuffer,
      source: staticSource,
      edgeCategories,
      dissolved: {
        positions: dissolvedPositions,
        ringOffsets: dissolvedRingOffsets,
        vertexCount: dissolvedVertexCount,
        drawCommands,
        ringCategoryValues
      },
      parametersApplied: Number.NaN,
      dirty: true,
      reader: null,
      snapshot: null,
      generation: -1,
      readbackSources: []
    };
    variant.readbackSources = [
      {buffer: segmentFlags, size: vertexCount * 4},
      {buffer: featureInvalidCounts, size: featureCount * 4},
      {buffer: isValid, size: 4},
      {buffer: invalidCount, size: 4},
      {buffer: dissolvedRingCount, size: 4},
      {buffer: dissolvedRequiredRings, size: 4},
      {buffer: dissolvedOverflow, size: 4},
      {buffer: dissolvedOpenSegments, size: 4},
      {buffer: dissolvedTouchingSegments, size: 4},
      {buffer: boundarySegmentCount, size: 4},
      {buffer: dissolvedVertexCount, size: 4},
      {buffer: dissolvedPolygonCount, size: 4},
      {buffer: ringIsHole, size: ringCapacity * 4},
      {buffer: ringGroups, size: ringCapacity * 4},
      {buffer: dissolvedSourceIds, size: ringCapacity * 4}
    ];
    makeReader(variant, generation);
    return variant;
  };

  const requestVariant = (options: Readonly<Options>) => {
    const optionsSnapshot = {...options};
    const requestedKey = currentKey(optionsSnapshot);
    const requestedGeneration = ++generation;
    ctx.setStatus('Compiling the county coverage audit');
    clearReadouts();
    ctx.requestLayers();
    void graphSwap
      .request({
        build: async () => cache.get(requestedKey) ?? buildVariant(optionsSnapshot),
        commit: variant => {
          if (
            destroyed ||
            generation !== requestedGeneration ||
            currentKey(ctx.options) !== requestedKey
          ) {
            if (cache.get(requestedKey) !== variant) {
              variant.reader?.release();
              variant.resources.destroy();
            }
            return;
          }
          cache.delete(requestedKey);
          cache.set(requestedKey, variant);
          variant.snapshot = null;
          active = variant;
          ctx.setStatus('');
          makeReader(variant, requestedGeneration);
          variant.parametersApplied = Number.NaN;
          variant.dirty = true;
          while (cache.size > 4) {
            const oldestKey = cache.keys().next().value;
            if (!oldestKey || oldestKey === requestedKey) break;
            const oldest = cache.get(oldestKey);
            cache.delete(oldestKey);
            oldest?.reader?.release();
            oldest?.resources.destroy();
          }
          ctx.requestLayers();
        },
        discard: variant => {
          if (!cache.has(requestedKey)) {
            variant.reader?.release();
            variant.resources.destroy();
          }
        }
      })
      .catch(error => {
        if (!destroyed && generation === requestedGeneration) {
          active = null;
          clearReadouts();
          const message = `GPU coverage graph build failed: ${String(error)}`;
          ctx.setStatus(message);
          ctx.setReadout('coverageAuditStatus', message);
          ctx.setReadout('coverageAuditDissolveStatus', 'Withheld because construction failed.');
          ctx.requestLayers();
        }
      });
  };

  const invalidateActive = (options: Readonly<Options>) => {
    generation++;
    clearReadouts();
    if (!active || currentKey(options) !== active.key) {
      requestVariant(options);
      return;
    }
    makeReader(active, generation);
    active.parametersApplied = Number.NaN;
    active.snapshot = null;
    active.dirty = true;
    ctx.requestLayers();
  };

  const makeLayers = (): Layer[] => {
    const variant = active;
    const hasCurrentVariant = variant !== null && currentKey(ctx.options) === variant.key;
    const snapshot = hasCurrentVariant ? variant.snapshot : null;
    const layers: Layer[] = [];
    if (variant && hasCurrentVariant) {
      layers.push(
        new PathOutputLayer({
          id: 'coverage-audit-county-segments',
          analysisFrame: ALBERS_CONUS_FRAME,
          positions: variant.source.positions,
          pathOffsets: variant.source.offsets,
          pathOffsetCount: variant.source.offsetCount,
          vertexCount: variant.source.vertexCount,
          drawCommands: variant.source.drawCommands,
          ...(snapshot
            ? {
                values: variant.edgeCategories,
                colorSource: 'vertex-value' as const,
                valueMapping: 'class' as const,
                classBreaks: FLAG_BREAKS,
                classColors: FLAG_COLORS
              }
            : {
                color: [104, 132, 139, 205] as const
              }),
          closed: true,
          widthPixels: 1.1
        })
      );
      if (snapshot?.dissolveReady) {
        const allGroup = variant.group === 'single';
        layers.push(
          new PathOutputLayer({
            id: 'coverage-audit-dissolved-boundaries',
            analysisFrame: ALBERS_CONUS_FRAME,
            positions: variant.dissolved.positions,
            pathOffsets: variant.dissolved.ringOffsets,
            pathOffsetCount: auditData.layout.vertexCount + 1,
            vertexCount: variant.dissolved.vertexCount,
            drawCommands: variant.dissolved.drawCommands,
            closed: true,
            ...(allGroup
              ? {
                  color: [34, 220, 196, 255] as const
                }
              : {
                  values: variant.dissolved.ringCategoryValues,
                  colorSource: 'path-value' as const,
                  valueMapping: 'class' as const,
                  classBreaks: STATE_BREAKS,
                  classColors: STATE_COLORS
                }),
            widthPixels: 2.2
          })
        );
      }
    }
    return layers;
  };

  ctx.setReadout(
    'coverageAuditInput',
    `${formatCount(auditData.sourceFeatureCount)} source counties in the archived Census 2022 boundary file; analysis subset ${formatCount(auditData.layout.featureCount)} counties, ${formatCount(auditData.layout.ringCount)} rings, ${formatCount(auditData.layout.vertexCount)} vertices. ` +
      `Synthetic perturbation: county FIPS ${formatCountyFips(auditData.perturbation.targetCountyFips)} (source feature row ${auditData.perturbation.targetFeatureRow}), shared with county FIPS ${formatCountyFips(auditData.perturbation.partnerCountyFips)}; vertex ${auditData.perturbation.vertexIndex} moves ${COVERAGE_AUDIT_PERTURBATION_METERS} m by (${auditData.perturbation.offsetX.toFixed(2)}, ${auditData.perturbation.offsetY.toFixed(2)}) projected metres. ` +
      COVERAGE_AUDIT_STATE_FIPS.map(
        state =>
          `${COVERAGE_AUDIT_STATE_NAMES[state]} FIPS ${state}: ${auditData.countsByState.get(state) ?? 0}`
      ).join('; ')
  );
  ctx.setReadout('coverageAuditStatus', 'GPU coverage validity is pending.');
  ctx.setReadout('coverageAuditBits', 'GPU invalid-edge flags are pending.');
  ctx.setReadout('coverageAuditByState', 'GPU feature counts are pending.');
  ctx.setReadout('coverageAuditDissolveStatus', 'Coverage dissolve is pending.');
  ctx.setReadout('coverageAuditDissolveSummary', 'GPU polygon output is pending.');
  requestVariant(ctx.options);

  return {
    getCompiledGraphs: () => Array.from(cache.values(), variant => variant.compiled),
    setOption(id, options) {
      if (id === 'coverageAuditInput' || id === 'coverageAuditGroup') requestVariant(options);
      else if (id === 'coverageAuditGapWidth') invalidateActive(options);
    },
    encode(commandEncoder) {
      if (!active || destroyed || currentKey(ctx.options) !== active.key) return;
      const gapWidth = Math.fround(Number(ctx.options.coverageAuditGapWidth));
      if (gapWidth !== active.parametersApplied) {
        active.parametersApplied = gapWidth;
        active.parameters.write(getGPUCoverageValidityParameterValues({gapWidth}));
        active.dirty = true;
      }
      if (active.dirty) {
        active.compiled.encode(commandEncoder, {parameters: undefined});
        copyCountToDrawRecord(
          commandEncoder,
          active.dissolved.vertexCount,
          active.dissolved.drawCommands
        );
        active.reader?.request(commandEncoder);
        active.dirty = false;
      } else {
        active.reader?.flush(commandEncoder);
      }
    },
    getLayers: makeLayers,
    destroy() {
      destroyed = true;
      graphSwap.destroy();
      for (const variant of cache.values()) {
        variant.reader?.stop();
        variant.resources.destroy();
      }
      cache.clear();
    }
  };
}

function getReadbackSources(variant: AuditVariant): ReadbackSource[] {
  return variant.readbackSources;
}

function decodeAuditSnapshot(
  bytes: ArrayBuffer,
  auditData: CoverageAuditData,
  group: AuditGroup
): AuditSnapshot {
  const words = new Uint32Array(bytes);
  let offset = 0;
  const take = (count: number) => {
    const values = words.slice(offset, offset + count);
    offset += count;
    return values;
  };
  const flags = take(auditData.layout.vertexCount);
  const featureInvalidCounts = take(auditData.layout.featureCount);
  const valid = take(1)[0] === 1;
  const invalidCount = take(1)[0];
  const ringCount = take(1)[0];
  const requiredRings = take(1)[0];
  const overflow = take(1)[0];
  const openSegments = take(1)[0];
  const touchingSegments = take(1)[0];
  const boundarySegments = take(1)[0];
  const vertexCount = take(1)[0];
  const polygonCount = take(1)[0];
  const holeFlags = take(auditData.layout.vertexCount);
  const ringGroups = take(auditData.layout.vertexCount);
  const sourceIds = take(auditData.layout.vertexCount);
  const groupMatches =
    group === 'single' ||
    sourceIds
      .subarray(0, polygonCount)
      .every(sourceId => sourceId === 17 || sourceId === 18 || sourceId === 55);
  const flagCounts = countFlagBits(flags);
  return {
    valid,
    invalidCount,
    featureInvalidCounts,
    flags,
    flagCounts,
    dissolveReady:
      valid && overflow === 0 && ringCount === requiredRings && openSegments === 0 && groupMatches,
    dissolveStatus: {
      ringCount,
      requiredRings,
      overflow,
      openSegments,
      touchingSegments,
      boundarySegments,
      vertexCount,
      polygonCount,
      holeCount: holeFlags
        .subarray(0, ringCount)
        .reduce((count, isHole) => count + Number(isHole !== 0), 0),
      sourceIds,
      ringGroups
    }
  };
}

function countFlagBits(flags: Uint32Array): Uint32Array {
  const counts = new Uint32Array(6);
  const flagBits = [1, 2, 4, 8, 16];
  for (const flagsValue of flags) {
    let setCount = 0;
    for (let bit = 0; bit < flagBits.length; bit++) {
      if ((flagsValue & flagBits[bit]) !== 0) {
        counts[bit]++;
        setCount++;
      }
    }
    if (setCount > 1) counts[5]++;
  }
  return counts;
}

function makeFlagCategoryValues(flags: Uint32Array): Float32Array {
  const categories = new Float32Array(flags.length);
  for (let index = 0; index < flags.length; index++) {
    const value = flags[index];
    if (value === 0) categories[index] = 0;
    else if ((value & (value - 1)) !== 0) categories[index] = 6;
    else if ((value & 1) !== 0) categories[index] = 1;
    else if ((value & 2) !== 0) categories[index] = 2;
    else if ((value & 4) !== 0) categories[index] = 3;
    else if ((value & 8) !== 0) categories[index] = 4;
    else categories[index] = 5;
  }
  return categories;
}

function makeGroupCategoryValues(
  ringGroups: Uint32Array,
  ringCount: number,
  group: AuditGroup
): Float32Array {
  const values = new Float32Array(ringGroups.length);
  for (let ring = 0; ring < ringCount; ring++) {
    const sourceId = ringGroups[ring];
    values[ring] = group === 'single' ? 0 : sourceId;
  }
  return values;
}

function makeFlagChart(values: Uint32Array): BarChartData {
  return {
    kind: 'bars',
    title: 'Invalid county-edge segment flags',
    labels: [...FLAG_LABELS],
    values,
    color: 2,
    description:
      'GPUCoverageValidity segment bit counts. Counts can overlap because a segment may carry multiple flag bits; they are not a partition of the invalid total.'
  };
}

function formatCountyFips(value: number): string {
  return String(value).padStart(5, '0');
}
