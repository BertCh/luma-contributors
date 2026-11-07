// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUReadbackRing} from '@luma-contributors/graph/gpu-core';
import {
  GPUContinuousSpatialSample,
  GPUSpatialBalancedSample,
  getGPUDotDensityParameterValues,
  GPURandomPointsOnLine,
  GPUPolygonTriangulation
} from '@luma-contributors/graph/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {createGraphBufferImporter} from '../../engine/graph-buffers';
import {addKernelPass} from '../../engine/mode-kernels';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance, ScenePointerEvent, TooltipContent} from '../scene';

export type SamplingFramesOptions = {
  design: 'finite' | 'line' | 'area' | 'iid-line';
  sampleCount: '10' | '20';
  iidCount: '8' | '16' | '32';
  strata: 'category' | 'one';
  unequal: boolean;
  spacing: '100' | '200' | '400';
  waves: '1' | '2' | '3';
  capacity: '1' | '256' | '2048';
  candidateBudget: '1' | '4096' | '65536';
  includeLegacy: boolean;
  seed: number;
  replicate: number;
};

type SampleVariant = {
  graph: CompiledGPUCommandGraph<void>;
  design: SamplingFramesOptions['design'];
  resources: SpatialAnalysisResources;
  reader: SummaryReader;
  output: Buffer;
  count: number;
  origin: readonly [number, number];
  line?: Buffer;
  lineCount: number;
  values?: Buffer;
  parameters: {write: (values: Uint32Array) => void};
  dirty: boolean;
  refreshReader: () => void;
  generation: number;
  summary: string;
  ids: string[];
};

const PARK_ORIGIN: readonly [number, number] = [-87.6, 41.86];
const FACILITY_ORIGIN: readonly [number, number] = [-87.7, 41.86];
const LIBRARY_COLOR = [183, 105, 73, 255] as const;
const FIRE_COLOR = [80, 137, 112, 255] as const;
const PRIMARY_ROLE_COLOR = [80, 137, 112, 255] as const;
const REPLACEMENT_ROLE_COLOR = [151, 163, 129, 245] as const;
const LEGACY_ROLE_COLOR = [183, 105, 73, 255] as const;
const LIBRARY_CATEGORY = 1;
const FIRE_STATION_CATEGORY = 3;

function decodeUint32(bytes: ArrayBuffer, offset: number, length: number): Uint32Array {
  return new Uint32Array(bytes.slice(offset, offset + length * 4));
}

function decodeFloat32(bytes: ArrayBuffer, offset: number, length: number): Float32Array {
  return new Float32Array(bytes.slice(offset, offset + length * 4));
}

function buildParkBoundary(
  dataset: ReturnType<SceneContext<SamplingFramesOptions>['datasets']['get']>
) {
  const features =
    (
      dataset.geojson as {
        features?: Array<{
          properties?: Record<string, unknown>;
          geometry?: {type?: string; coordinates?: unknown};
        }>;
      }
    )?.features ?? [];
  const feature = features.find(item => Number(item.properties?.osmId) === 2914325);
  if (!feature?.geometry || feature.geometry.type !== 'Polygon') {
    throw new Error('Northerly Island OSM Polygon 2914325 is unavailable');
  }
  const rings = feature.geometry.coordinates as number[][][];
  const projection = dataset.getProjection(PARK_ORIGIN);
  const positions: number[] = [];
  const ringOffsets = [0];
  for (const ring of rings) {
    for (const [longitude, latitude] of ring)
      positions.push(...projection.project(longitude, latitude));
    ringOffsets.push(positions.length / 2);
  }
  const outer = positions.slice(0, ringOffsets[1] * 2);
  return {
    feature,
    projection,
    positions: Float32Array.from(positions),
    ringOffsets: Uint32Array.from(ringOffsets),
    outer: Float32Array.from(outer)
  };
}

/** Creates one active, lazily compiled sampling design. All sampled locations and weights stay GPU-authored. */
export async function createSamplingFrames(
  ctx: SceneContext<SamplingFramesOptions>
): Promise<SceneInstance<SamplingFramesOptions>> {
  const facilities = ctx.datasets.get('chicago-facilities');
  const parks = ctx.datasets.get('chicago-parks');
  const park = buildParkBoundary(parks);
  const facilityCoordinates = facilities.column<Float32Array>('position');
  const categories = facilities.column<Uint8Array>('category');
  const projection = facilities.getProjection(FACILITY_ORIGIN);
  const finitePositions: number[] = [];
  const finiteIds: number[] = [];
  const finiteStrata: number[] = [];
  const facilityNames: string[] = [];
  for (let row = 0; row < facilities.count; row++) {
    const category = categories[row];
    if (category !== LIBRARY_CATEGORY && category !== FIRE_STATION_CATEGORY) continue;
    const point = projection.project(
      facilityCoordinates[row * 2],
      facilityCoordinates[row * 2 + 1]
    );
    finitePositions.push(point[0], point[1]);
    finiteIds.push(row + 1);
    finiteStrata.push(category === LIBRARY_CATEGORY ? 0 : 1);
    facilityNames.push(category === LIBRARY_CATEGORY ? 'Library' : 'Fire station');
  }
  const sourceCount = finiteIds.length;
  const finiteBounds: [number, number, number, number] = [
    Math.min(...finitePositions.filter((_, index) => index % 2 === 0)) - 100,
    Math.min(...finitePositions.filter((_, index) => index % 2 === 1)) - 100,
    Math.max(...finitePositions.filter((_, index) => index % 2 === 0)) + 100,
    Math.max(...finitePositions.filter((_, index) => index % 2 === 1)) + 100
  ];
  const outer = park.outer;
  const lineSegments = new Float32Array((outer.length / 2 - 1) * 4);
  for (let segment = 0; segment < outer.length / 2 - 1; segment++) {
    lineSegments.set(outer.subarray(segment * 2, segment * 2 + 4), segment * 4);
  }
  const lineOffsets = Uint32Array.of(0, outer.length / 2);
  const lineLength = (() => {
    let length = 0;
    for (let index = 2; index < outer.length; index += 2) {
      length += Math.hypot(outer[index] - outer[index - 2], outer[index + 1] - outer[index - 1]);
    }
    return length;
  })();
  const resources = new SpatialAnalysisResources(ctx.device, 'sampling-frames');
  let active: SampleVariant | null = null;
  let buildGeneration = 0;
  let buildPending = false;
  const buildingResources = new Map<number, SpatialAnalysisResources>();
  let destroyed = false;
  let summaryText = 'Preparing finite frame';
  let selectedRows: number[] = [];
  let outputReady = false;
  let outputValid = false;
  let parkTriangles: Awaited<ReturnType<typeof buildParkTriangleFrame>> | null = null;
  const sourceFrameBuffer = resources.createBuffer(
    'finite-source-frame',
    Float32Array.from(finitePositions)
  );
  const sourceCategoryBuffer = resources.createBuffer(
    'finite-source-categories',
    Uint32Array.from(finiteStrata)
  );

  const build = async (state: SamplingFramesOptions): Promise<SampleVariant> => {
    const generation = ++buildGeneration;
    const finiteSampleCount = Number(state.sampleCount);
    const iidSampleCount = Number(state.iidCount);
    const latticeSpacing = Number(state.spacing);
    const waveCount = Number(state.waves);
    const outputCapacity = Number(state.capacity);
    const candidateLimit = Number(state.candidateBudget);
    const variantResources = new SpatialAnalysisResources(
      ctx.device,
      `sampling-frames-${generation}`
    );
    buildingResources.set(generation, variantResources);
    const graph = new GPUCommandGraph<void>(ctx.device, {
      id: `sampling-frames-${state.design}-${generation}`
    });
    const importView = createGraphBufferImporter(graph);
    const upload = (name: string, values: Float32Array | Uint32Array) =>
      variantResources.createBuffer(name, values);
    const output = (name: string, byteLength: number) =>
      variantResources.createBuffer(name, byteLength);
    const summaryBuffers: {buffer: Buffer; size: number}[] = [];
    const addSummary = (buffer: Buffer) => summaryBuffers.push({buffer, size: buffer.byteLength});
    let samplePositions: Buffer | undefined;
    let renderedPositions: Buffer | undefined;
    let lineBuffer: Buffer | undefined;
    let renderedLineCount = 0;
    let sampleValues: Buffer | undefined;
    let residentParameters: {write: (values: Uint32Array) => void} | undefined;
    let sampleCount: number = finiteSampleCount;
    let activeOrigin: readonly [number, number] = FACILITY_ORIGIN;
    let sourceIds = finiteIds.map(String);

    if (state.design === 'finite') {
      activeOrigin = FACILITY_ORIGIN;
      const positions = upload('finite-positions', Float32Array.from(finitePositions));
      const ids = upload('frame-ids', Uint32Array.from(finiteIds));
      const strata = upload('strata', Uint32Array.from(finiteStrata));
      const tickets = new Uint32Array(sourceCount);
      for (let index = 0; index < sourceCount; index++) {
        // Libraries have 41 rows per parity, fire stations 46: their ticket totals
        // divide category allocations 5/10; pooled totals 1740/3480 divide 10/20.
        const sourceRow = finiteIds[index] - 1;
        const parityMass = state.unequal ? (sourceRow % 2 === 0 ? 0.5 : 1.5) : 1;
        tickets[index] = Math.round(finiteSampleCount * parityMass);
      }
      const inclusionTickets = upload('inclusion-tickets', tickets);
      const selectedCount = finiteSampleCount;
      const allocations =
        state.strata === 'category'
          ? [
              {stratumId: 0, sampleCount: selectedCount / 2},
              {stratumId: 1, sampleCount: selectedCount / 2}
            ]
          : [{stratumId: 0, sampleCount: selectedCount}];
      const selectedIds = output('selected-ids', selectedCount * 4);
      const probabilities = output('inclusion-probabilities', selectedCount * 4);
      const weights = output('design-weights', selectedCount * 4);
      const status = output('design-status', allocations.length * 8);
      const selectedPoints = output('selected-positions', selectedCount * 8);
      renderedPositions = selectedPoints;
      const selectedCategories = output('selected-categories', selectedCount * 4);
      sampleValues = selectedCategories;
      const parameters = variantResources.createParameterBuffer(
        'seed-replicate',
        'uint32',
        2,
        Uint32Array.of(state.seed, state.replicate)
      );
      residentParameters = parameters;
      graph.add(
        new GPUSpatialBalancedSample({
          id: 'balanced-finite-sample',
          positions: importView(positions, 'float32x2', sourceCount),
          frameIds: importView(ids, 'uint32', sourceCount),
          ...(state.strata === 'category'
            ? {strata: importView(strata, 'uint32', sourceCount)}
            : {}),
          inclusionTickets: importView(inclusionTickets, 'uint32', sourceCount),
          allocations,
          bounds: finiteBounds,
          seed: state.seed,
          parameters: parameters.importToGraph(graph),
          output: {
            frameIds: importView(selectedIds, 'uint32', selectedCount),
            inclusionProbabilities: importView(probabilities, 'float32', selectedCount),
            designWeights: importView(weights, 'float32', selectedCount),
            status: importView(status, 'uint32', allocations.length * 2)
          }
        })
      );
      addKernelPass(graph, {
        id: 'gather-selected-sites',
        invocationCount: selectedCount * 2,
        bindings: [
          {
            name: 'source',
            view: importView(positions, 'float32x2', sourceCount),
            type: 'f32',
            access: 'read'
          },
          {
            name: 'selected',
            view: importView(selectedIds, 'uint32', selectedCount),
            type: 'u32',
            access: 'read'
          },
          {
            name: 'frameIds',
            view: importView(ids, 'uint32', sourceCount),
            type: 'u32',
            access: 'read'
          },
          {
            name: 'sourceStrata',
            view: importView(strata, 'uint32', sourceCount),
            type: 'u32',
            access: 'read'
          },
          {
            name: 'points',
            view: importView(selectedPoints, 'float32x2', selectedCount),
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'categories',
            view: importView(selectedCategories, 'uint32', selectedCount),
            type: 'u32',
            access: 'read_write'
          }
        ],
        body: `let row = index / 2u; let axis = index % 2u; var sourceRow = 0xffffffffu; let frameId = selected[selectedOffset + row];
          for (var candidate = 0u; candidate < ${sourceCount}u; candidate++) { if (frameIds[frameIdsOffset + candidate] == frameId) { sourceRow = candidate; } }
          if (sourceRow < ${sourceCount}u) { points[pointsOffset + row * 2u + axis] = source[sourceOffset + sourceRow * 2u + axis]; if (axis == 0u) { categories[categoriesOffset + row] = sourceStrata[sourceStrataOffset + sourceRow]; } }
          else { points[pointsOffset + row * 2u + axis] = bitcast<f32>(0x7fc00000u | (index & 0u)); if (axis == 0u) { categories[categoriesOffset + row] = 0xffffffffu; } }`
      });
      samplePositions = selectedPoints;
      selectedRows = [];
      sourceIds = finiteIds.map(String);
      addSummary(selectedIds);
      addSummary(probabilities);
      addSummary(weights);
      addSummary(status);
    } else if (state.design === 'line' || state.design === 'iid-line') {
      activeOrigin = PARK_ORIGIN;
      const positions = upload('park-boundary-vertices', outer);
      const offsets = upload('park-line-offsets', lineOffsets);
      lineBuffer = upload('park-boundary-segments', lineSegments);
      renderedLineCount = lineSegments.length / 4;
      const count = state.design === 'iid-line' ? iidSampleCount : outputCapacity;
      sampleCount = state.design === 'line' ? count * waveCount : count;
      const positionsOutput = output(
        'line-sample-positions',
        count * (state.design === 'line' ? waveCount : 1) * 8
      );
      renderedPositions = positionsOutput;
      if (state.design === 'line') {
        const weights = output('line-sample-weights', count * waveCount * 4);
        sampleValues = weights;
        const provenance = output('line-sample-provenance', count * waveCount * 16);
        const roles = output('line-sample-roles', count * waveCount * 4);
        sampleValues = roles;
        const status = output('line-sample-status', waveCount * 16);
        const seedParameters = variantResources.createParameterBuffer(
          'seed-replicate',
          'uint32',
          2,
          Uint32Array.of(state.seed, state.replicate)
        );
        residentParameters = seedParameters;
        graph.add(
          new GPUContinuousSpatialSample({
            id: 'continuous-boundary-sample',
            frame: {
              type: 'line',
              positions: importView(positions, 'float32x2', outer.length / 2),
              ...(state.includeLegacy
                ? {
                    legacySites: importView(uploadLegacyLine(variantResources), 'float32', 1)
                  }
                : {})
            },
            spacing: latticeSpacing,
            seed: state.seed,
            replicate: state.replicate,
            waves: waveCount,
            capacity: count,
            maximumCandidates: candidateLimit,
            parameters: seedParameters.importToGraph(graph),
            output: {
              positions: importView(positionsOutput, 'float32x2', count * waveCount),
              designWeights: importView(weights, 'float32', count * waveCount),
              provenance: importView(provenance, 'uint32x4', count * waveCount),
              status: importView(status, 'uint32x4', waveCount)
            }
          })
        );
        addKernelPass(graph, {
          id: 'decode-line-sample-roles',
          invocationCount: count * waveCount,
          bindings: [
            {
              name: 'provenance',
              view: importView(provenance, 'uint32x4', count * waveCount),
              type: 'u32',
              access: 'read'
            },
            {
              name: 'roles',
              view: importView(roles, 'uint32', count * waveCount),
              type: 'u32',
              access: 'read_write'
            }
          ],
          body: 'roles[rolesOffset + index] = provenance[provenanceOffset + index * 4u + 3u];'
        });
        addSummary(weights);
        addSummary(provenance);
        addSummary(status);
      } else {
        const counts = upload('iid-counts', Uint32Array.of(iidSampleCount));
        const dotParameters = getGPUDotDensityParameterValues({
          seed: (state.seed + state.replicate) >>> 0
        });
        const parameters = variantResources.createParameterBuffer(
          'iid-parameters',
          'uint32',
          dotParameters.length,
          dotParameters
        );
        residentParameters = parameters;
        const ids = output('iid-line-ids', count * 4);
        const fractions = output('iid-fractions', count * 4);
        const statusCount = output('iid-line-count', 4);
        const statusOverflow = output('iid-line-overflow', 4);
        const statusRequired = output('iid-line-required', 4);
        graph.add(
          new GPURandomPointsOnLine({
            id: 'iid-arc-length-sample',
            positions: importView(positions, 'float32x2', outer.length / 2),
            lineOffsets: importView(offsets, 'uint32', 2),
            spatialContext: park.projection.spatialContext,
            counts: importView(counts, 'uint32', 1),
            parameters: parameters.importToGraph(graph),
            output: {
              positions: importView(positionsOutput, 'float32x2', count),
              points: {
                ids: importView(ids, 'uint32', count),
                status: {
                  count: importView(statusCount, 'uint32', 1),
                  overflow: importView(statusOverflow, 'uint32', 1),
                  requiredCount: importView(statusRequired, 'uint32', 1)
                }
              },
              fractions: importView(fractions, 'float32', count)
            }
          })
        );
        addSummary(ids);
        addSummary(fractions);
        addSummary(statusCount);
        addSummary(statusOverflow);
        addSummary(statusRequired);
      }
      sourceIds = [];
    } else {
      // The selected OSM polygon is sent through the public GPU earcut contributor first. The
      // triangle readback below is only unpacking GPU triangulation output for the sampler input.
      activeOrigin = PARK_ORIGIN;
      parkTriangles ??= await buildParkTriangleFrame(ctx, park, variantResources);
      const triangleData = parkTriangles;
      lineBuffer = upload('park-boundary-context', lineSegments);
      renderedLineCount = lineSegments.length / 4;
      const positions = upload('park-triangle-positions', triangleData.positions);
      const maxLineCount = outputCapacity;
      const samplePositionsOutput = output('area-sample-positions', maxLineCount * waveCount * 8);
      renderedPositions = samplePositionsOutput;
      const weights = output('area-sample-weights', maxLineCount * waveCount * 4);
      const provenance = output('area-sample-provenance', maxLineCount * waveCount * 16);
      const roles = output('area-sample-roles', maxLineCount * waveCount * 4);
      sampleValues = roles;
      const status = output('area-sample-status', waveCount * 16);
      const legacy = state.includeLegacy
        ? upload('legacy-area-site', Float32Array.from(triangleData.legacyCentroid))
        : undefined;
      const parameters = variantResources.createParameterBuffer(
        'seed-replicate',
        'uint32',
        2,
        Uint32Array.of(state.seed, state.replicate)
      );
      residentParameters = parameters;
      graph.add(
        new GPUContinuousSpatialSample({
          id: 'continuous-northerly-area-sample',
          frame: {
            type: 'area',
            positions: importView(positions, 'float32x2', triangleData.positions.length / 2),
            bounds: triangleData.bounds,
            ...(legacy ? {legacySites: importView(legacy, 'float32x2', 1)} : {})
          },
          spacing: latticeSpacing,
          seed: state.seed,
          replicate: state.replicate,
          waves: waveCount,
          capacity: maxLineCount,
          maximumCandidates: candidateLimit,
          parameters: parameters.importToGraph(graph),
          output: {
            positions: importView(samplePositionsOutput, 'float32x2', maxLineCount * waveCount),
            designWeights: importView(weights, 'float32', maxLineCount * waveCount),
            provenance: importView(provenance, 'uint32x4', maxLineCount * waveCount),
            status: importView(status, 'uint32x4', waveCount)
          }
        })
      );
      addKernelPass(graph, {
        id: 'decode-area-sample-roles',
        invocationCount: maxLineCount * waveCount,
        bindings: [
          {
            name: 'provenance',
            view: importView(provenance, 'uint32x4', maxLineCount * waveCount),
            type: 'u32',
            access: 'read'
          },
          {
            name: 'roles',
            view: importView(roles, 'uint32', maxLineCount * waveCount),
            type: 'u32',
            access: 'read_write'
          }
        ],
        body: 'roles[rolesOffset + index] = provenance[provenanceOffset + index * 4u + 3u];'
      });
      samplePositions = samplePositionsOutput;
      sampleCount = maxLineCount * waveCount;
      addSummary(weights);
      addSummary(provenance);
      addSummary(status);
      sourceIds = [];
    }

    const compiled = variantResources.track(await graph.compileAsync());
    let variant!: SampleVariant;
    let revision = 0;
    let reader!: SummaryReader;
    const createReader = () => {
      const readerRevision = revision;
      const currentReader = new SummaryReader(
        variantResources,
        `summary-${generation}-${readerRevision}`,
        summaryBuffers,
        bytes => {
          if (
            destroyed ||
            active !== variant ||
            variant.generation !== generation ||
            variant.reader !== currentReader ||
            revision !== readerRevision
          )
            return;
          outputReady = true;
          const view = new DataView(bytes);
          if (state.design === 'finite') {
            const ids = decodeUint32(bytes, 0, finiteSampleCount);
            const probabilities = decodeFloat32(bytes, finiteSampleCount * 4, finiteSampleCount);
            const weights = decodeFloat32(bytes, finiteSampleCount * 8, finiteSampleCount);
            const status = decodeUint32(
              bytes,
              finiteSampleCount * 12,
              state.strata === 'category' ? 4 : 2
            );
            const invalidStatus = Array.from(status).some(
              (value, index) => index % 2 === 1 && value !== 0
            );
            const pairs = Array.from(ids, (id, index) => ({
              id,
              probability: probabilities[index],
              weight: weights[index]
            }));
            const unique = new Set(ids).size === ids.length;
            const knownIds = Array.from(ids).every(id => finiteIds.includes(id));
            const validValues = pairs.every(
              row =>
                Number.isFinite(row.probability) &&
                row.probability > 0 &&
                row.probability <= 1 &&
                Number.isFinite(row.weight) &&
                row.weight > 0 &&
                Math.abs(row.weight * row.probability - 1) < 0.001
            );
            const invalid = invalidStatus || !unique || !knownIds || !validValues;
            outputValid = !invalid;
            selectedRows = invalid ? [] : pairs.map(row => row.id - 1);
            const total = weights.reduce((sum, weight) => sum + weight, 0);
            summaryText = invalid
              ? 'Invalid frame/design; output withheld'
              : `${ids.length} unique sites · Σ1/π = ${formatCount(total)} estimated frame total`;
            ctx.setReadout('selected', invalid ? null : ids.length);
            ctx.setReadout(
              'ids',
              invalid
                ? 'invalid design'
                : pairs
                    .map(row => {
                      const frameRow = finiteIds.indexOf(row.id);
                      return `${facilityNames[frameRow]} · source row ${row.id - 1} · ID ${row.id}`;
                    })
                    .join('\n')
            );
            ctx.setReadout(
              'weights',
              invalid
                ? null
                : pairs
                    .slice(0, 8)
                    .map(
                      row =>
                        `ID ${row.id}: π=${row.probability.toFixed(3)}, 1/π=${row.weight.toFixed(2)}`
                    )
                    .join('\n')
            );
            ctx.setReadout('estimate', invalid ? null : `Σ1/π ${formatCount(total)} frame sites`);
            ctx.setReadout(
              'status',
              invalid
                ? 'invalid whole design'
                : `valid · ${ids.length} / ${sourceCount} frame sites`
            );
          } else if (state.design === 'line' || state.design === 'area') {
            const statusOffset = bytes.byteLength - waveCount * 16;
            const statuses = decodeUint32(bytes, statusOffset, waveCount * 4);
            const flags = Array.from({length: waveCount}, (_, wave) => statuses[wave * 4 + 3]);
            const bad = flags.some(value => value !== 0);
            const writtenRows = statuses.reduce(
              (sum, value, index) => (index % 4 === 0 ? sum + value : sum),
              0
            );
            const acceptedRows = statuses.reduce(
              (sum, value, index) => (index % 4 === 2 ? sum + value : sum),
              0
            );
            const provenanceOffset = outputCapacity * waveCount * 4;
            const provenance = decodeUint32(
              bytes,
              provenanceOffset,
              outputCapacity * waveCount * 4
            );
            const weights = decodeFloat32(bytes, 0, outputCapacity * waveCount);
            const legacyRows = Array.from(
              {length: outputCapacity * waveCount},
              (_, row) => provenance[row * 4 + 3]
            ).filter(role => role === 2).length;
            const roleRows = Array.from(
              {length: outputCapacity * waveCount},
              (_, row) => provenance[row * 4 + 3]
            );
            const validRows = Array.from({length: waveCount}, (_, wave) =>
              Array.from({length: statuses[wave * 4]}, (_, slot) => {
                const row = wave * outputCapacity + slot;
                return {role: roleRows[row], weight: weights[row]};
              })
            ).flat();
            const validMetrics =
              validRows.length === writtenRows &&
              validRows.every(row => row.role <= 2) &&
              validRows.filter(row => row.role < 2).length === acceptedRows &&
              validRows.every(row =>
                row.role === 2 ? row.weight === 0 : Number.isFinite(row.weight) && row.weight > 0
              );
            const valid = !bad && validMetrics;

            summaryText = !valid
              ? bad
                ? `withheld · GPU flags ${flags.join(', ')}`
                : 'withheld · invalid GPU output values'
              : `${writtenRows} written rows (${acceptedRows} random accepted, ${legacyRows} legacy) · ${waveCount} complete wave${waveCount > 1 ? 's' : ''}`;
            outputValid = valid;
            ctx.setReadout('selected', valid ? acceptedRows : null);
            const examples: string[] = [];
            for (let row = 0; row < outputCapacity * waveCount && examples.length < 8; row++) {
              const source = provenance[row * 4];
              const stream = provenance[row * 4 + 1];
              const candidate = provenance[row * 4 + 2];
              const role = provenance[row * 4 + 3];
              if (role > 2) continue;
              examples.push(
                `source ${source} · stream ${stream} · candidate ${candidate} · ${role === 0 ? 'primary' : role === 1 ? 'replacement' : 'synthetic legacy'}`
              );
            }
            ctx.setReadout(
              'ids',
              !valid ? 'invalid / capacity-limited output withheld' : examples.join('\n')
            );
            ctx.setReadout(
              'weights',
              !valid
                ? null
                : `${state.design === 'line' ? `arc length ${latticeSpacing} m` : `area ${latticeSpacing * latticeSpacing} m²`}\n${Array.from(
                    weights
                  )
                    .filter(value => value > 0)
                    .slice(0, 8)
                    .map(value => value.toPrecision(5))
                    .join(', ')}`
            );
            ctx.setReadout(
              'estimate',
              !valid
                ? null
                : state.design === 'line'
                  ? `${lineLength.toFixed(0)} m mapped boundary`
                  : `${Number(park.feature.properties?.areaKm2).toFixed(3)} km² OSM area attribute`
            );
            ctx.setReadout(
              'status',
              Array.from({length: waveCount}, (_, wave) => {
                let waveLegacyRows = 0;
                for (let slot = 0; slot < statuses[wave * 4]; slot++) {
                  if (roleRows[wave * outputCapacity + slot] === 2) waveLegacyRows++;
                }
                return `wave ${wave}: ${statuses[wave * 4]} written (${statuses[wave * 4 + 2]} random accepted, ${waveLegacyRows} legacy); ${statuses[wave * 4 + 1]} required-so-far; flags ${flags[wave]}`;
              }).join('\n') +
                (state.design === 'area' &&
                parkTriangles &&
                new URLSearchParams(window.location.search).has('geometryDiagnostics')
                  ? `\nGPU triangle frame: ${parkTriangles.positions.length / 6} triangles; ${parkTriangles.invalidIndexCount} invalid indices; ${parkTriangles.nonFiniteCoordinateCount} non-finite coordinates; ${parkTriangles.outOfBoundsCoordinateCount} coordinates outside triangle bounds; ${parkTriangles.zeroDeterminantCount} zero-determinant triangles (f32 emulation); max |coordinate| ${parkTriangles.maximumCoordinateMagnitude.toPrecision(5)} m (spacing precision threshold ${(parkTriangles.maximumCoordinateMagnitude * 0.00000095367431640625).toPrecision(5)} m, selected spacing ${latticeSpacing} m).\n${parkTriangles.triangleDiagnostics}`
                  : '') +
                (!valid && !bad
                  ? '\nInvalid role, count, or weight values; output withheld.'
                  : '') +
                (flags.some(value => (value & 4) !== 0)
                  ? '\nFull count is unknown after a candidate-budget stop.'
                  : '')
            );
          } else {
            const ids = decodeUint32(bytes, 0, iidSampleCount);
            const fractions = decodeFloat32(bytes, iidSampleCount * 4, iidSampleCount);
            const countOffset = iidSampleCount * 8;
            const count = view.getUint32(countOffset, true);
            const overflow = view.getUint32(countOffset + 4, true);
            const required = view.getUint32(countOffset + 8, true);
            const valid =
              !overflow &&
              count === iidSampleCount &&
              Array.from(ids.slice(0, count)).every(id => id === 0) &&
              Array.from(fractions.slice(0, count)).every(
                value => Number.isFinite(value) && value >= 0 && value < 1
              );
            outputValid = valid;
            summaryText = !valid
              ? `${count}/${required} written · overflow or invalid output; display withheld`
              : `${count} IID draws · boundary ${lineLength.toFixed(0)} m`;
            ctx.setReadout('selected', valid ? count : null);
            ctx.setReadout(
              'ids',
              !valid
                ? 'overflow or invalid output: withheld'
                : `line ids ${Array.from(ids.slice(0, count)).join(', ')}`
            );
            ctx.setReadout(
              'weights',
              !valid
                ? null
                : `fractions ${Array.from(fractions.slice(0, Math.min(8, count)), value => value.toFixed(3)).join(', ')}`
            );
            ctx.setReadout(
              'estimate',
              !valid ? null : `${lineLength.toFixed(0)} m mapped boundary`
            );
            ctx.setReadout(
              'status',
              !valid
                ? `capacity ${count} < required ${required}`
                : 'valid · IID uniform in arc length'
            );
          }
          ctx.requestLayers();
          ctx.setFurniture({
            title: {
              title: 'What population can this sample represent?',
              subtitle: summaryText,
              sample:
                state.design === 'finite'
                  ? '174 Chicago facility sites · source row IDs'
                  : 'Northerly Island · OSM feature 2914325 · local metric frame',
              chips: ['Sampling design']
            }
          });
        }
      );
      return currentReader;
    };
    reader = createReader();
    samplePositions = renderedPositions ?? samplePositions;
    if (!samplePositions) throw new Error('Sampling design did not allocate a position output');
    variant = {
      graph: compiled,
      design: state.design,
      resources: variantResources,
      reader,
      output: samplePositions,
      count: sampleCount,
      origin: activeOrigin,
      line: lineBuffer,
      lineCount: renderedLineCount,
      values: sampleValues,
      parameters: residentParameters!,
      dirty: true,
      refreshReader: () => {
        revision++;
        reader.release();
        reader = createReader();
        variant.reader = reader;
      },
      generation,
      summary: summaryText,
      ids: sourceIds
    };
    return variant;
  };

  async function rebuild(state: SamplingFramesOptions) {
    const generation = buildGeneration + 1;
    buildPending = true;
    outputReady = false;
    outputValid = false;
    summaryText = 'Building GPU design…';
    selectedRows = [];
    active?.reader.release();
    ctx.setFurniture({
      title: {
        title: 'What population can this sample represent?',
        subtitle: summaryText,
        sample:
          state.design === 'finite'
            ? '174 Chicago facility sites · source row IDs'
            : 'Northerly Island · OSM feature 2914325 · local metric frame',
        chips: ['Sampling design']
      }
    });
    ctx.setReadout('frame', null);
    ctx.setReadout('selected', null);
    ctx.setReadout('ids', null);
    ctx.setReadout('weights', null);
    ctx.setReadout('estimate', null);
    ctx.setReadout('status', 'pending');
    ctx.requestLayers();
    let next: SampleVariant;
    try {
      next = await build(state);
    } catch (error) {
      buildingResources.get(generation)?.destroy();
      buildingResources.delete(generation);
      if (destroyed || generation !== buildGeneration) return;
      buildPending = false;
      if (active) {
        active.reader.release();
        active.resources.destroy();
        active = null;
      }
      summaryText = `GPU design error · ${error instanceof Error ? error.message : String(error)}`;
      ctx.setStatus(summaryText);
      ctx.setReadout(
        'frame',
        state.design === 'area' ? 'Northerly Island OSM Polygon · 2914325' : null
      );
      ctx.setReadout('selected', null);
      ctx.setReadout('ids', null);
      ctx.setReadout('weights', null);
      ctx.setReadout('estimate', null);
      ctx.setReadout('status', summaryText);
      ctx.setFurniture({
        title: {
          title: 'What population can this sample represent?',
          subtitle: summaryText,
          sample: 'Northerly Island · OSM feature 2914325 · local metric frame',
          chips: ['Sampling design']
        }
      });
      ctx.requestLayers();
      return;
    }
    if (destroyed || next.generation !== buildGeneration || generation !== next.generation) {
      next.reader.release();
      next.resources.destroy();
      buildingResources.delete(next.generation);
      return;
    }
    buildingResources.delete(next.generation);
    buildPending = false;
    const previous = active;
    active = next;
    ctx.setStatus('');
    ctx.setReadout(
      'frame',
      state.design === 'finite'
        ? 'Chicago libraries + fire stations'
        : 'Northerly Island OSM park boundary'
    );
    ctx.setFurniture({
      title: {
        title: 'What population can this sample represent?',
        subtitle: next.summary,
        sample:
          state.design === 'finite'
            ? '174 Chicago facility sites · source row IDs'
            : 'Northerly Island · OSM feature 2914325 · local metric frame',
        chips: ['Sampling design']
      }
    });
    ctx.requestLayers();
    if (previous) {
      previous.reader.release();
      previous.resources.destroy();
    }
  }

  await rebuild(ctx.options);
  return {
    getCompiledGraphs: () => (active ? [active.graph] : []),
    encode: (encoder, frame) => {
      if (!active) return;
      if (active.dirty) {
        active.graph.encode(encoder, {parameters: undefined});
        active.reader.request(encoder);
        active.dirty = false;
      } else {
        active.reader.flush(encoder);
      }
      void frame;
    },
    getLayers: () => {
      if (!active || buildPending) return [];
      const layers = [];
      if (ctx.options.design === 'finite') {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'sampling-finite-source-sites',
            data: {length: sourceCount},
            positions: sourceFrameBuffer,
            values: sourceCategoryBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: [LIBRARY_COLOR, FIRE_COLOR],
            instanceCount: sourceCount,
            coordinateOrigin: [FACILITY_ORIGIN[0], FACILITY_ORIGIN[1], 0],
            radiusPixels: 3.5,
            opacity: 0.35,
            pickable: false
          })
        );
      }
      if (
        ctx.options.design === 'line' ||
        ctx.options.design === 'iid-line' ||
        ctx.options.design === 'area'
      ) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'sampling-frame-boundary',
            segments: active.line!,
            instanceCount: active.lineCount,
            coordinateOrigin: [active.origin[0], active.origin[1], 0],
            color: ctx.ground() === 'dark' ? [132, 156, 125, 190] : [78, 104, 78, 205],
            widthPixels: 1.5,
            cap: 'round',
            pickable: false
          })
        );
      }
      if (!outputReady || !outputValid) return layers;
      layers.push(
        new SpatialAnalysisPointLayer({
          id: `sampling-output-${ctx.options.design}`,
          data: {length: active.count},
          positions: active.output,
          instanceCount: active.count,
          coordinateOrigin: [active.origin[0], active.origin[1], 0],
          color: ctx.options.design === 'iid-line' ? [73, 139, 120, 225] : [181, 110, 75, 240],
          ...(ctx.options.design === 'iid-line'
            ? {}
            : {
                values: active.values,
                valueFormat: 'uint32' as const,
                colormap: 'category' as const,
                palette:
                  ctx.options.design === 'finite'
                    ? [LIBRARY_COLOR, FIRE_COLOR]
                    : [PRIMARY_ROLE_COLOR, REPLACEMENT_ROLE_COLOR, LEGACY_ROLE_COLOR],
                noDataValue: 0xffffffff
              }),
          outlineColor: ctx.ground() === 'dark' ? [244, 231, 203, 235] : [250, 247, 232, 235],
          outlineWidthPixels: 1,
          radiusPixels: ctx.options.design === 'finite' ? 6 : 4.2,
          pickable: true
        })
      );
      return layers;
    },
    setOption: (id, _value, state) => {
      if ((id === 'seed' || id === 'replicate') && active && !buildPending) {
        active.refreshReader();
        active.dirty = true;
        outputReady = false;
        outputValid = false;
        selectedRows = [];
        active.parameters.write(
          active.design === 'iid-line'
            ? getGPUDotDensityParameterValues({seed: (state.seed + state.replicate) >>> 0})
            : Uint32Array.of(state.seed, state.replicate)
        );
        summaryText = 'Replay requested · waiting for GPU readback';
        ctx.setReadout('selected', null);
        ctx.setReadout('ids', null);
        ctx.setReadout('weights', null);
        ctx.setReadout('estimate', null);
        ctx.setReadout('status', 'pending replay');

        ctx.setFurniture({
          title: {
            title: 'What population can this sample represent?',
            subtitle: summaryText,
            sample:
              active.design === 'finite'
                ? '174 Chicago facility sites · source row IDs'
                : 'Northerly Island · OSM feature 2914325 · local metric frame',
            chips: ['Sampling design']
          }
        });
        ctx.requestLayers();
        return;
      }
      if (id === 'seed' || id === 'replicate') {
        void rebuild(state);
        return;
      }
      if (
        id === 'design' ||
        id === 'sampleCount' ||
        id === 'iidCount' ||
        id === 'strata' ||
        id === 'unequal' ||
        id === 'spacing' ||
        id === 'waves' ||
        id === 'capacity' ||
        id === 'candidateBudget' ||
        id === 'includeLegacy'
      )
        void rebuild(state);
    },
    getTooltip: (event: ScenePointerEvent): TooltipContent | null => {
      if (ctx.options.design !== 'finite' || !active) return null;
      const index = event.info.index;
      const sourceRow = selectedRows[index];
      if (sourceRow === undefined) return null;
      const frameRow = finiteIds.indexOf(sourceRow + 1);
      return {
        title: `${facilityNames[frameRow] ?? 'Facility'} · source row ${sourceRow}`,
        subtitle: 'Chicago public-facility inventory',
        rows: [
          {label: 'Frame ID', value: String(sourceRow + 1)},
          {
            label: 'Category stratum',
            value: finiteStrata[frameRow] === 0 ? 'library' : 'fire station'
          }
        ],
        note: 'Inventory membership only; not demand, quality or coverage.'
      };
    },
    destroy: () => {
      destroyed = true;
      for (const buildingResource of buildingResources.values()) buildingResource.destroy();
      buildingResources.clear();
      if (active) {
        active.reader.release();
        active.resources.destroy();
        active = null;
      }
      resources.destroy();
    }
  };
}

function uploadLegacyLine(resources: SpatialAnalysisResources): Buffer {
  return resources.createBuffer('legacy-line-measure', Float32Array.of(0));
}

async function buildParkTriangleFrame(
  ctx: SceneContext<SamplingFramesOptions>,
  park: ReturnType<typeof buildParkBoundary>,
  resources: SpatialAnalysisResources
): Promise<{
  positions: Float32Array;
  bounds: [number, number, number, number];
  legacyCentroid: [number, number];
  invalidIndexCount: number;
  nonFiniteCoordinateCount: number;
  outOfBoundsCoordinateCount: number;
  zeroDeterminantCount: number;
  maximumCoordinateMagnitude: number;
  triangleDiagnostics: string;
}> {
  const graph = new GPUCommandGraph<void>(ctx.device, {id: 'sampling-frames-park-triangulation'});
  const importView = createGraphBufferImporter(graph);
  const positions = resources.createBuffer('park-polygon-vertices', park.positions);
  const ringOffsets = resources.createBuffer('park-ring-offsets', park.ringOffsets);
  const polygonOffsets = resources.createBuffer(
    'park-polygon-offsets',
    Uint32Array.of(0, park.ringOffsets.length - 1)
  );
  const featureOffsets = resources.createBuffer('park-feature-offsets', Uint32Array.of(0, 1));
  const vertexCount = park.positions.length / 2;
  const ringCount = park.ringOffsets.length - 1;
  const indexCapacity = Math.max(3, 3 * (vertexCount + 2 * ringCount - 4));
  const indices = resources.createBuffer('park-triangle-indices', indexCapacity * 4);
  const valid = resources.createBuffer('park-triangle-valid', 4);
  const triangleCount = resources.createBuffer('park-triangle-count', 4);
  const triangleOffsets = resources.createBuffer('park-triangle-offsets', 8);
  graph.add(
    new GPUPolygonTriangulation({
      id: 'northerly-park-polygon-triangulation',
      positions: importView(positions, 'float32x2', park.positions.length / 2),
      featureOffsets: importView(featureOffsets, 'uint32', 2),
      polygonOffsets: importView(polygonOffsets, 'uint32', 2),
      ringOffsets: importView(ringOffsets, 'uint32', park.ringOffsets.length),
      maximumWork: 100_000,
      output: {
        indices: importView(indices, 'uint32', indexCapacity),
        valid: importView(valid, 'uint32', 1),
        triangleCount: importView(triangleCount, 'uint32', 1),
        triangleOffsets: importView(triangleOffsets, 'uint32', 2)
      }
    })
  );
  const compiled = resources.track(await graph.compileAsync());
  const readback = resources.track(
    new GPUReadbackRing(ctx.device, {
      id: 'sampling-frames-park-triangulation-readback',
      byteLength: indexCapacity * 4 + 8
    })
  );
  const ticket = readback.tryAcquire();
  if (!ticket) throw new Error('Unable to acquire bounded GPU triangulation readback');
  const encoder = ctx.device.createCommandEncoder({id: 'sampling-frames-triangulation-encoder'});
  compiled.encode(encoder, {parameters: undefined});
  encoder.copyBufferToBuffer({
    sourceBuffer: indices,
    destinationBuffer: ticket.buffer,
    size: indexCapacity * 4
  });
  encoder.copyBufferToBuffer({
    sourceBuffer: triangleCount,
    destinationBuffer: ticket.buffer,
    destinationOffset: indexCapacity * 4,
    size: 4
  });
  encoder.copyBufferToBuffer({
    sourceBuffer: valid,
    destinationBuffer: ticket.buffer,
    destinationOffset: indexCapacity * 4 + 4,
    size: 4
  });
  ticket.markEncoded({byteOffset: 0, byteLength: indexCapacity * 4 + 8});
  ctx.device.submit(encoder.finish());
  const bytes = await ticket.read();
  // read() returns Uint8Array: reinterpret packed words, never convert each byte to an index.
  const triangles = new Uint32Array(bytes.slice(0, indexCapacity * 4).buffer);
  const count = new Uint32Array(bytes.slice(indexCapacity * 4, indexCapacity * 4 + 4).buffer);
  const validity = new Uint32Array(
    bytes.slice(indexCapacity * 4 + 4, indexCapacity * 4 + 8).buffer
  );
  if (validity[0] !== 1 || count[0] === 0)
    throw new Error('Northerly Island polygon triangulation failed validity gate');
  const trianglePositions = new Float32Array(count[0] * 6);
  let invalidIndexCount = 0;
  for (let index = 0; index < count[0] * 3; index++) {
    const vertex = triangles[index];
    if (vertex >= park.positions.length / 2) invalidIndexCount++;
    trianglePositions[index * 2] = park.positions[vertex * 2];
    trianglePositions[index * 2 + 1] = park.positions[vertex * 2 + 1];
  }
  const bounds: [number, number, number, number] = [
    Math.min(...trianglePositions.filter((_, index) => index % 2 === 0)),
    Math.min(...trianglePositions.filter((_, index) => index % 2 === 1)),
    Math.max(...trianglePositions.filter((_, index) => index % 2 === 0)),
    Math.max(...trianglePositions.filter((_, index) => index % 2 === 1))
  ];
  const diagnosticsEnabled = new URLSearchParams(window.location.search).has('geometryDiagnostics');
  let nonFiniteCoordinateCount = 0;
  let outOfBoundsCoordinateCount = 0;
  let zeroDeterminantCount = 0;
  let maximumCoordinateMagnitude = 0;
  for (let index = 0; diagnosticsEnabled && index < trianglePositions.length; index += 2) {
    const coordinateX = trianglePositions[index];
    const coordinateY = trianglePositions[index + 1];
    if (!Number.isFinite(coordinateX) || !Number.isFinite(coordinateY)) {
      nonFiniteCoordinateCount++;
      continue;
    }
    maximumCoordinateMagnitude = Math.max(
      maximumCoordinateMagnitude,
      Math.abs(coordinateX),
      Math.abs(coordinateY)
    );
    if (
      coordinateX < bounds[0] ||
      coordinateY < bounds[1] ||
      coordinateX > bounds[2] ||
      coordinateY > bounds[3]
    )
      outOfBoundsCoordinateCount++;
  }
  for (let index = 0; diagnosticsEnabled && index < trianglePositions.length; index += 6) {
    const firstX = trianglePositions[index];
    const firstY = trianglePositions[index + 1];
    const secondX = trianglePositions[index + 2];
    const secondY = trianglePositions[index + 3];
    const thirdX = trianglePositions[index + 4];
    const thirdY = trianglePositions[index + 5];
    const determinant = Math.fround(
      Math.fround(Math.fround(secondX - firstX) * Math.fround(thirdY - firstY)) -
        Math.fround(Math.fround(secondY - firstY) * Math.fround(thirdX - firstX))
    );
    if (!Number.isFinite(determinant) || determinant === 0) zeroDeterminantCount++;
  }
  const formatCoordinates = (coordinates: ArrayLike<number>) =>
    Array.from(coordinates, coordinate => Number(coordinate).toPrecision(7)).join(', ');
  const firstRawIndices = diagnosticsEnabled ? triangles.slice(0, 18) : new Uint32Array(0);
  const firstReferencedSourceCoordinates = new Float32Array(firstRawIndices.length * 2);
  for (let index = 0; index < firstRawIndices.length; index++) {
    const sourceVertex = firstRawIndices[index];
    firstReferencedSourceCoordinates[index * 2] = park.positions[sourceVertex * 2];
    firstReferencedSourceCoordinates[index * 2 + 1] = park.positions[sourceVertex * 2 + 1];
  }
  const triangleDiagnostics = diagnosticsEnabled
    ? `First 18 raw indices: ${Array.from(firstRawIndices).join(', ')}. First 18 gathered xy scalars: ${formatCoordinates(trianglePositions.subarray(0, 18))}. Source xy for those indexed vertices: ${formatCoordinates(firstReferencedSourceCoordinates)}. First 18 source xy scalars: ${formatCoordinates(park.positions.subarray(0, 18))}.`
    : '';
  return {
    positions: trianglePositions,
    bounds,
    invalidIndexCount,
    nonFiniteCoordinateCount,
    outOfBoundsCoordinateCount,
    zeroDeterminantCount,
    maximumCoordinateMagnitude,
    triangleDiagnostics,
    legacyCentroid: [
      (trianglePositions[0] + trianglePositions[2] + trianglePositions[4]) / 3,
      (trianglePositions[1] + trianglePositions[3] + trianglePositions[5]) / 3
    ]
  };
}
