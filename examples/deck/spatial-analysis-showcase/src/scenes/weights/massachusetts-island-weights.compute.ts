// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GeoJsonLayer, PathLayer} from '@deck.gl/layers';
import type {GeoJSON} from 'geojson';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {
  GPUContiguityWeights,
  GPUFuzzyContiguity,
  GPUIslandAttachment,
  GPUSpatialWeightsValidation
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {createGraphBufferImporter} from '../../engine/graph-buffers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance, TooltipContent} from '../scene';
import {
  createMassachusettsIslandGeometry,
  type MassachusettsIslandGeometry
} from './massachusetts-island-weights.geometry';

const EDGE_CAPACITY = 14 * 13;
const NO_ATTACHMENT = 0xffffffff;

type Scenario = 'exact' | 'fuzzy' | 'bridge';
type BridgeMode = 'reciprocal' | 'outgoing';

export type MassachusettsIslandWeightsOptions = {
  scenario: Scenario;
  toleranceKm: string;
  bridgeMode: BridgeMode;
};

type WeightsBuffers = {
  baselineOffsets: Buffer;
  offsets: Buffer;
  neighbors: Buffer;
  weights: Buffer;
  overflow: Buffer;
  required: Buffer;
  candidateOverflow: Buffer;
  added: Buffer;
  attachment?: Buffer;
  featureStatus?: Buffer;
  validationBefore: Buffer;
  validationAfter: Buffer;
  validationFlagsBefore: Buffer;
  validationFlagsAfter: Buffer;
};

type ActiveGraph = {
  key: string;
  graph: CompiledGPUCommandGraph<void>;
  buffers: WeightsBuffers;
  generation: number;
  reader: SummaryReader;
  dirty: boolean;
};

type ReadResult = {
  baselineOffsets: Uint32Array;
  offsets: Uint32Array;
  neighbors: Uint32Array;
  added: Uint32Array;
  attachment: Uint32Array;
  overflow: number;
  required: number;
  candidateOverflow: number;
  featureStatus: Uint32Array;
  beforeInvalid: number;
  afterInvalid: number;
  beforeFlags: Uint32Array;
  afterFlags: Uint32Array;
  generation: number;
  usable: boolean;
};

function getScenarioKey(options: MassachusettsIslandWeightsOptions): string {
  if (options.scenario === 'exact') return 'exact';
  if (options.scenario === 'fuzzy') return `fuzzy-${Number(options.toleranceKm)}`;
  return `bridge-${options.bridgeMode}`;
}

function createWeightsOutput(
  resources: SpatialAnalysisResources,
  importView: <Format extends 'float32x2' | 'float32' | 'uint32'>(
    buffer: Buffer,
    format: Format,
    length: number
  ) => GraphDataView<Format>,
  name: string,
  featureCount: number,
  includeAttachment: boolean
) {
  const offsets = resources.createBuffer(`${name}-offsets`, new Uint32Array(featureCount + 1));
  const neighbors = resources.createBuffer(`${name}-neighbors`, new Uint32Array(EDGE_CAPACITY));
  const weights = resources.createBuffer(`${name}-weights`, new Float32Array(EDGE_CAPACITY));
  const overflow = resources.createBuffer(`${name}-overflow`, new Uint32Array(1));
  const required = resources.createBuffer(`${name}-required`, new Uint32Array(1));
  const candidateOverflow = resources.createBuffer(
    `${name}-candidate-overflow`,
    new Uint32Array(1)
  );
  const added = resources.createBuffer(`${name}-added`, new Uint32Array(EDGE_CAPACITY));
  const attachment = includeAttachment
    ? resources.createBuffer(
        `${name}-attachment`,
        new Uint32Array(featureCount).fill(NO_ATTACHMENT)
      )
    : undefined;
  const output = {
    offsets: importView(offsets, 'uint32', featureCount + 1),
    neighbors: importView(neighbors, 'uint32', EDGE_CAPACITY),
    weights: importView(weights, 'float32', EDGE_CAPACITY),
    status: {
      overflow: importView(overflow, 'uint32', 1),
      requiredCount: importView(required, 'uint32', 1),
      candidateOverflow: importView(candidateOverflow, 'uint32', 1)
    },
    added: importView(added, 'uint32', EDGE_CAPACITY),
    ...(attachment ? {attachment: importView(attachment, 'uint32', featureCount)} : {})
  };
  return {
    output,
    buffers: {offsets, neighbors, weights, overflow, required, candidateOverflow, added, attachment}
  };
}

function createGraphBundle(
  context: SceneContext<MassachusettsIslandWeightsOptions>,
  resources: SpatialAnalysisResources,
  geometry: MassachusettsIslandGeometry,
  options: MassachusettsIslandWeightsOptions,
  generation: number
) {
  const key = getScenarioKey(options);
  const graph = new GPUCommandGraph<void>(context.device, {id: `ma-island-${key}`});
  const importView = createGraphBufferImporter(graph);
  const polygonPort = {
    kind: 'polygons' as const,
    positions: importView(
      resources.createBuffer(`${key}-polygon-positions`, geometry.positions),
      'float32x2',
      geometry.vertexCount
    ),
    featureOffsets: importView(
      resources.createBuffer(`${key}-feature-offsets`, geometry.featureOffsets),
      'uint32',
      geometry.featureOffsets.length
    ),
    polygonOffsets: importView(
      resources.createBuffer(`${key}-polygon-offsets`, geometry.polygonOffsets),
      'uint32',
      geometry.polygonOffsets.length
    ),
    ringOffsets: importView(
      resources.createBuffer(`${key}-ring-offsets`, geometry.ringOffsets),
      'uint32',
      geometry.ringOffsets.length
    )
  };
  const representativePositions = importView(
    resources.createBuffer(`${key}-representatives`, geometry.representatives),
    'float32x2',
    geometry.featureCount
  );
  const before = createWeightsOutput(
    resources,
    importView,
    `${key}-strict-queen`,
    geometry.featureCount,
    false
  );
  const queen = new GPUContiguityWeights({
    id: `${key}-strict-queen`,
    criterion: 'queen',
    polygons: polygonPort,
    snapTolerance: 0,
    output: before.output
  });
  graph.add(queen);
  const beforeInvalid = resources.createBuffer(`${key}-before-invalid`, new Uint32Array(1));
  const beforeFlags = resources.createBuffer(
    `${key}-before-flags`,
    new Uint32Array(geometry.featureCount)
  );
  graph.add(
    new GPUSpatialWeightsValidation({
      id: `${key}-validate-before`,
      weights: queen.outputWeights,
      output: {
        status: {invalidCount: importView(beforeInvalid, 'uint32', 1)},
        flags: importView(beforeFlags, 'uint32', geometry.featureCount)
      }
    })
  );

  let resultBuffers = before.buffers;
  let resultWeights = queen.outputWeights;
  let featureStatus: Buffer | undefined;
  let attachment: Buffer | undefined;
  let addedView = before.output.added;
  if (options.scenario === 'fuzzy') {
    const final = createWeightsOutput(
      resources,
      importView,
      `${key}-fuzzy`,
      geometry.featureCount,
      false
    );
    featureStatus = resources.createBuffer(
      `${key}-feature-status`,
      new Uint32Array(geometry.featureCount)
    );
    const fuzzyOutput = {
      ...final.output,
      featureStatus: importView(featureStatus, 'uint32', geometry.featureCount)
    };
    const fuzzy = new GPUFuzzyContiguity({
      id: `${key}-fuzzy`,
      polygons: polygonPort,
      tolerance: Number(options.toleranceKm) * 1000,
      output: fuzzyOutput
    });
    graph.add(fuzzy);
    resultWeights = fuzzy.outputWeights;
    resultBuffers = final.buffers;
    addedView = fuzzyOutput.added;
  } else if (options.scenario === 'bridge') {
    const final = createWeightsOutput(
      resources,
      importView,
      `${key}-attachment`,
      geometry.featureCount,
      true
    );
    const attachmentOutput = new GPUIslandAttachment({
      id: `${key}-attachment`,
      weights: queen.outputWeights,
      positions: representativePositions,
      reciprocal: options.bridgeMode === 'reciprocal',
      output: {...final.output, attachment: final.output.attachment!}
    });
    graph.add(attachmentOutput);
    resultWeights = attachmentOutput.outputWeights;
    resultBuffers = final.buffers;
    attachment = final.buffers.attachment;
    addedView = final.output.added;
  }
  const afterInvalid = resources.createBuffer(`${key}-after-invalid`, new Uint32Array(1));
  const afterFlags = resources.createBuffer(
    `${key}-after-flags`,
    new Uint32Array(geometry.featureCount)
  );
  graph.add(
    new GPUSpatialWeightsValidation({
      id: `${key}-validate-after`,
      weights: resultWeights,
      output: {
        status: {invalidCount: importView(afterInvalid, 'uint32', 1)},
        flags: importView(afterFlags, 'uint32', geometry.featureCount)
      }
    })
  );
  return {
    graph,
    key,
    resultBuffers: {
      ...resultBuffers,
      baselineOffsets: before.buffers.offsets,
      added: resultBuffers.added ?? before.buffers.added,
      attachment,
      featureStatus,
      validationBefore: beforeInvalid,
      validationAfter: afterInvalid,
      validationFlagsBefore: beforeFlags,
      validationFlagsAfter: afterFlags
    } satisfies WeightsBuffers,
    generation,
    addedView
  };
}

function parseReadback(
  bytes: ArrayBuffer,
  buffers: WeightsBuffers,
  featureCount: number
): ReadResult {
  let offset = 0;
  const take = (length: number) => {
    const slice = bytes.slice(offset, offset + length);
    offset += length;
    return slice;
  };
  const baselineOffsets = new Uint32Array(take((featureCount + 1) * 4));
  const offsets = new Uint32Array(take((featureCount + 1) * 4));
  const neighbors = new Uint32Array(take(EDGE_CAPACITY * 4));
  const added = new Uint32Array(take(EDGE_CAPACITY * 4));
  const attachment = buffers.attachment
    ? new Uint32Array(take(featureCount * 4))
    : new Uint32Array(featureCount).fill(NO_ATTACHMENT);
  const overflow = new Uint32Array(take(4))[0];
  const required = new Uint32Array(take(4))[0];
  const candidateOverflow = new Uint32Array(take(4))[0];
  const featureStatus = buffers.featureStatus
    ? new Uint32Array(take(featureCount * 4))
    : new Uint32Array(featureCount);
  const beforeInvalid = new Uint32Array(take(4))[0];
  const afterInvalid = new Uint32Array(take(4))[0];
  const beforeFlags = new Uint32Array(take(featureCount * 4));
  const afterFlags = new Uint32Array(take(featureCount * 4));
  const generationReadback = new Uint32Array(take(4))[0];
  const liveCount = Math.min(offsets[featureCount], EDGE_CAPACITY);
  let invalidNeighbor = false;
  for (let row = 0; row < featureCount; row++) {
    const start = Math.min(offsets[row], liveCount);
    const end = Math.min(offsets[row + 1], liveCount);
    for (let slot = start; slot < end; slot++) {
      if (neighbors[slot] >= featureCount) invalidNeighbor = true;
    }
  }
  const usable =
    overflow === 0 &&
    candidateOverflow === 0 &&
    beforeInvalid === 0 &&
    afterInvalid === 0 &&
    !invalidNeighbor &&
    featureStatus.every(status => status === 0) &&
    beforeFlags.every(flags => flags === 0) &&
    afterFlags.every(flags => flags === 0);
  return {
    baselineOffsets,
    offsets,
    neighbors,
    added,
    attachment,
    overflow,
    required,
    candidateOverflow,
    featureStatus,
    beforeInvalid,
    afterInvalid,
    beforeFlags,
    afterFlags,
    generation: generationReadback,
    usable
  };
}

/** Build strict source adjacency, then one explicit fuzzy or nearest-centroid scenario. */
export async function createMassachusettsIslandWeights(
  context: SceneContext<MassachusettsIslandWeightsOptions>
): Promise<SceneInstance<MassachusettsIslandWeightsOptions>> {
  const dataset = context.datasets.get('us-counties');
  const geometry = createMassachusettsIslandGeometry(dataset);
  if (geometry.featureCount !== 14)
    throw new Error(`Expected 14 Massachusetts county features; received ${geometry.featureCount}`);
  if (geometry.featureValidity.some(value => value !== 0))
    throw new Error(
      'Massachusetts polygon rings are not all finite and simple; no source geometry was changed.'
    );
  const resources = new SpatialAnalysisResources(context.device, 'ma-island-weights');
  const generationMarker = resources.createBuffer('generation-marker', Uint32Array.of(0));
  const variants = new Map<string, Promise<ActiveGraph>>();
  let addedPaths: [number, number][][] = [];
  let currentGeneration = 0;
  let active: ActiveGraph | null = null;
  let destroyed = false;
  let currentResult: ReadResult | null = null;

  const publish = (result: ReadResult, key: string) => {
    currentResult = result;
    const liveSlots = Math.min(result.offsets[geometry.featureCount], EDGE_CAPACITY);
    const links: [number, number][][] = [];
    let addedDirected = 0;
    const sourceRow = geometry.fips.findIndex(value => value === 25019);
    const sourceNeighbors: string[] = [];
    for (let row = 0; result.usable && row < geometry.featureCount; row++) {
      const start = Math.min(result.offsets[row], liveSlots);
      const end = Math.min(result.offsets[row + 1], liveSlots);
      for (let slot = start; slot < end; slot++) {
        if (result.added[slot] !== 1) continue;
        addedDirected++;
        const neighbor = result.neighbors[slot];
        if (row < neighbor || key === 'bridge-outgoing') {
          links.push([
            [geometry.representativeLngLat[row * 2], geometry.representativeLngLat[row * 2 + 1]],
            [
              geometry.representativeLngLat[neighbor * 2],
              geometry.representativeLngLat[neighbor * 2 + 1]
            ]
          ]);
        }
        if (row === sourceRow) sourceNeighbors.push(geometry.names[neighbor]);
      }
    }
    // PathLayer may retain an empty array by identity; publish a fresh collection after readback.
    addedPaths = links;
    const islandsBefore = Array.from(
      {length: geometry.featureCount},
      (_, row) => result.baselineOffsets[row + 1] === result.baselineOffsets[row]
    ).filter(Boolean).length;
    const finalIslandCount = Array.from(
      {length: geometry.featureCount},
      (_, row) => result.offsets[row + 1] === result.offsets[row]
    ).filter(Boolean).length;
    const scenario = key.startsWith('fuzzy')
      ? 'fuzzy links'
      : key.startsWith('bridge')
        ? 'authored bridge'
        : 'strict queen';
    const strictUsable =
      result.beforeInvalid === 0 && result.beforeFlags.every(flags => flags === 0);
    context.setReadout('countyCount', geometry.featureCount);
    context.setReadout('vertexCount', geometry.vertexCount);
    context.setReadout('strictIslands', strictUsable ? islandsBefore : 'unavailable');
    context.setReadout(
      'strictEdges',
      strictUsable ? result.baselineOffsets[geometry.featureCount] : 'unavailable'
    );
    context.setReadout('finalEdges', result.usable ? liveSlots : 'unavailable');
    context.setReadout('scenarioEdges', result.usable ? addedDirected : 'unavailable');
    context.setReadout(
      'nantucketLinks',
      result.usable
        ? sourceNeighbors.length
          ? sourceNeighbors.join(', ')
          : 'none added'
        : 'unavailable · invalid/overflow'
    );
    context.setReadout(
      'attachment',
      result.usable &&
        result.attachment[sourceRow] !== NO_ATTACHMENT &&
        result.attachment[sourceRow] < geometry.featureCount
        ? `${geometry.names[result.attachment[sourceRow]]}`
        : result.usable
          ? 'none'
          : 'unavailable · invalid/overflow'
    );
    context.setReadout(
      'attachmentDistance',
      result.usable &&
        result.attachment[sourceRow] !== NO_ATTACHMENT &&
        result.attachment[sourceRow] < geometry.featureCount
        ? `${(
            Math.hypot(
              geometry.representatives[sourceRow * 2] -
                geometry.representatives[result.attachment[sourceRow] * 2],
              geometry.representatives[sourceRow * 2 + 1] -
                geometry.representatives[result.attachment[sourceRow] * 2 + 1]
            ) / 1000
          ).toFixed(1)} km`
        : result.usable
          ? '—'
          : 'unavailable'
    );
    context.setReadout('islandCount', result.usable ? finalIslandCount : 'unavailable');
    context.setReadout(
      'validation',
      `strict ${result.beforeInvalid}/14 · ${scenario} ${result.afterInvalid}/14${result.usable ? '' : ' · graph outputs withheld'}`
    );
    context.setReadout(
      'capacity',
      `${liveSlots}/${EDGE_CAPACITY} directed slots · required ${result.required}${result.overflow ? ' · overflow' : ''}${result.candidateOverflow ? ' · candidate overflow' : ''}`
    );
    context.setReadout(
      'featureStatus',
      `${result.featureStatus.filter(value => value === 0).length} valid · ${result.featureStatus.filter(value => value !== 0).length} rejected`
    );
    context.requestLayers();
  };

  const createVariant = async (
    options: MassachusettsIslandWeightsOptions
  ): Promise<ActiveGraph> => {
    const bundle = createGraphBundle(context, resources, geometry, options, 0);
    const compiled = resources.track(await bundle.graph.compileAsync());
    const sources = [
      {buffer: bundle.resultBuffers.baselineOffsets, size: (geometry.featureCount + 1) * 4},
      {buffer: bundle.resultBuffers.offsets, size: (geometry.featureCount + 1) * 4},
      {buffer: bundle.resultBuffers.neighbors, size: EDGE_CAPACITY * 4},
      {buffer: bundle.resultBuffers.added, size: EDGE_CAPACITY * 4},
      ...(bundle.resultBuffers.attachment
        ? [{buffer: bundle.resultBuffers.attachment, size: geometry.featureCount * 4}]
        : []),
      {buffer: bundle.resultBuffers.overflow, size: 4},
      {buffer: bundle.resultBuffers.required, size: 4},
      {buffer: bundle.resultBuffers.candidateOverflow, size: 4},
      ...(bundle.resultBuffers.featureStatus
        ? [{buffer: bundle.resultBuffers.featureStatus, size: geometry.featureCount * 4}]
        : []),
      {buffer: bundle.resultBuffers.validationBefore, size: 4},
      {buffer: bundle.resultBuffers.validationAfter, size: 4},
      {buffer: bundle.resultBuffers.validationFlagsBefore, size: geometry.featureCount * 4},
      {buffer: bundle.resultBuffers.validationFlagsAfter, size: geometry.featureCount * 4},
      {buffer: generationMarker, size: 4}
    ];
    let variant: ActiveGraph;
    const reader = new SummaryReader(resources, `read-${bundle.key}`, sources, bytes => {
      if (destroyed || !active || active !== variant || variant.generation !== currentGeneration)
        return;
      const result = parseReadback(bytes, bundle.resultBuffers, geometry.featureCount);
      if (result.generation !== variant.generation) return;
      publish(result, bundle.key);
    });
    variant = {
      key: bundle.key,
      graph: compiled,
      buffers: bundle.resultBuffers,
      generation: 0,
      reader,
      dirty: true
    };
    return variant;
  };

  const activate = async (options: MassachusettsIslandWeightsOptions) => {
    const requestGeneration = ++currentGeneration;
    currentResult = null;
    addedPaths = [];
    context.setStatus('Compiling the selected Massachusetts island-weights scenario.');
    context.setReadout('validation', 'pending');
    context.setReadout('capacity', 'pending');
    context.setReadout('nantucketLinks', 'pending');
    context.setReadout('strictIslands', null);
    context.setReadout('islandCount', null);
    context.setReadout('strictEdges', null);
    context.setReadout('finalEdges', null);
    context.setReadout('scenarioEdges', null);
    context.setReadout('attachment', null);
    context.setReadout('attachmentDistance', null);
    context.setReadout('featureStatus', null);
    context.requestLayers();
    try {
      const key = getScenarioKey(options);
      let variantPromise = variants.get(key);
      if (!variantPromise) {
        variantPromise = createVariant(options);
        variants.set(key, variantPromise);
      }
      const next = await variantPromise;
      if (destroyed || requestGeneration !== currentGeneration) {
        return;
      }
      generationMarker.write(Uint32Array.of(requestGeneration));
      next.generation = requestGeneration;
      next.dirty = true;
      active = next;
      context.setStatus('The selected island-weights scenario is ready for GPU validation.');
      context.requestLayers();
    } catch (error) {
      if (requestGeneration === currentGeneration) {
        context.setReadout('validation', `compile failed: ${String(error)}`);
        context.setReadout('capacity', 'Unavailable: scenario graph compilation failed.');
        context.setReadout('strictEdges', null);
        context.setReadout('finalEdges', null);
        context.setReadout('scenarioEdges', null);
        context.setStatus(`Island-weights scenario compilation failed: ${String(error)}`);
        context.requestLayers();
      }
    }
  };

  await activate(context.options);
  const colors = {
    base: [42, 91, 123, 190] as [number, number, number, number],
    island: [221, 128, 60, 230] as [number, number, number, number]
  };
  const sourceRowsByFips = new Map(Array.from(geometry.fips, (fips, row) => [fips, row] as const));
  return {
    getCompiledGraphs: () => (active ? [active.graph] : []),
    setOption: (id, _value, state) => {
      if (
        id === 'scenario' ||
        (id === 'toleranceKm' && state.scenario === 'fuzzy') ||
        (id === 'bridgeMode' && state.scenario === 'bridge')
      ) {
        void activate(state);
      }
    },
    encode: (encoder: CommandEncoder) => {
      if (!active) return;
      if (active.dirty) {
        active.graph.encode(encoder, {parameters: undefined});
        active.dirty = false;
        active.reader.request(encoder);
      }
      active.reader.flush(encoder);
    },
    getLayers: () => {
      const selectedScenario = context.options.scenario;
      return [
        new GeoJsonLayer({
          id: 'ma-island-county-polygons',
          data: geometry.geoJson as unknown as GeoJSON,
          pickable: true,
          stroked: true,
          filled: true,
          lineWidthUnits: 'pixels',
          getLineWidth: 1,
          getLineColor: [255, 255, 255, 180],
          getFillColor: (feature: {properties: Record<string, unknown> | null}) => {
            const row = Number(feature.properties?.['fips'] ?? feature.properties?.['GEOID']);
            const localIndex = sourceRowsByFips.get(row);
            return localIndex !== undefined && geometry.fips[localIndex] === 25019
              ? colors.island
              : colors.base;
          },
          updateTriggers: {getFillColor: [currentResult?.generation, selectedScenario]}
        }),
        new PathLayer({
          id: 'ma-island-added-edges',
          data: addedPaths,
          getPath: (path: [number, number][]) => path,
          getColor: selectedScenario === 'bridge' ? [191, 84, 40, 230] : [0, 137, 123, 230],
          getWidth: selectedScenario === 'bridge' ? 3 : 2,
          widthUnits: 'pixels',
          pickable: false
        })
      ];
    },
    getTooltip: event => {
      const feature = event.info.object as
        | {properties?: Record<string, unknown> | null}
        | undefined;
      const fipsValue = feature?.properties?.['fips'] ?? feature?.properties?.['GEOID'];
      const fips = Number(fipsValue);
      const row = geometry.fips.findIndex(value => value === fips);
      if (row < 0) return null;
      const result = currentResult;
      const degree = result ? result.offsets[row + 1] - result.offsets[row] : null;
      const beforeFlags = result?.beforeFlags[row] ?? null;
      const afterFlags = result?.afterFlags[row] ?? null;
      const gpuFeatureStatus = result?.featureStatus[row] ?? null;
      const tooltip: TooltipContent = {
        title: geometry.names[row],
        subtitle: `Source county FIPS ${geometry.fips[row]}`,
        rows: [
          {
            label: 'Source polygon preflight',
            value: geometry.featureValidity[row] === 0 ? 'finite, simple rings' : 'rejected'
          },
          {
            label: 'Fuzzy input status',
            value:
              !result || !active?.key.startsWith('fuzzy')
                ? 'not applicable'
                : gpuFeatureStatus === 0
                  ? 'valid'
                  : `status ${gpuFeatureStatus}`
          },
          {
            label: 'Strict Queen row flags',
            value: beforeFlags === null ? 'pending' : String(beforeFlags)
          },
          {
            label: 'Scenario row flags',
            value: afterFlags === null ? 'pending' : String(afterFlags)
          },
          {label: 'Scenario neighbor count', value: degree === null ? 'pending' : degree}
        ],
        note: 'Validation bits: offsets 1, neighbor range 2, order/duplicate 4, self-link 8, invalid weight 16.'
      };
      return tooltip;
    },
    destroy: () => {
      destroyed = true;
      active?.reader.stop();
      resources.destroy();
    }
  };
}
