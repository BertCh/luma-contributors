// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ResidentSpatialJoinPointReprojector} from './resident-spatial-join-reproject';
import {Buffer, type CommandEncoder, type Fence} from '@luma.gl/core';
import type {PreparedKeplerPolygonTopologySource} from './geometry-preparation';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {
  GPUCommandGraph,
  GPUGather,
  type CompiledGPUCommandGraph,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {
  GPUSpatialJoinPrepared,
  GPUSpatialPredicateJoin
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  estimateKeplerResidentRowCacheByteLength,
  type KeplerArrowDataset,
  type KeplerComputeSession,
  type KeplerDatasetSnapshot,
  type KeplerMemoryReservation,
  type KeplerResidentColumnSchema
} from '@luma-contributors/kepler';
import type {ResidentJoinColumnFormat} from './resident-join-plan';
import {hasResidentNonFiniteValues} from './resident-nonfinite-check';
import {
  estimateResidentReadbackCpuByteLength,
  getResidentDomainBudgetBytes,
  makeResidentReadbackTable,
  readResidentResultChunks,
  shouldReadBackResidentResult
} from './resident-result-readback';
import {
  MAXIMUM_RESIDENT_SPATIAL_JOIN_MERGE_PAIR_COUNT,
  type ResidentSpatialJoinPlan,
  type ResidentSpatialJoinProvenance
} from './resident-spatial-join-plan';
import {
  MAXIMUM_RESIDENT_SPATIAL_JOIN_MERGE_LABEL_LENGTH,
  mergeSpatialJoinGeometries,
  readSpatialJoinPolygonGeometry,
  type SpatialJoinPolygonGeometry
} from './resident-spatial-join-merge';
import type {PackedSpatialJoinPolygons} from './resident-spatial-join-polygons';
import {
  fitResidentSpatialJoinCapacities,
  getInitialResidentSpatialJoinCapacities,
  growResidentSpatialJoinCapacities,
  RESIDENT_SPATIAL_JOIN_WORKING_BYTES,
  ResidentSpatialJoinAggregate,
  type ResidentSpatialJoinObservation
} from './resident-spatial-join-aggregate';

export type ResidentSpatialJoinResult = {
  readonly dataset: KeplerDatasetSnapshot<KeplerArrowDataset>;
  readonly provenance: ResidentSpatialJoinProvenance;
  readonly rowCount: number;
  /** Total matched (point, polygon) pairs, counted on the GPU across chunks and never stored. */
  readonly pairCount: number;
  /** Number of GPU chunks (encodings) used to join the points. */
  readonly chunkCount: number;
  /** Bytes read back: status words, plus result columns when at or below the readback cap. */
  readonly readbackBytes: number;
  isCurrent(): boolean;
  transfer(): void;
  destroy(): Promise<void>;
};

/** CPU bound for the accumulated merge pairs (row and polygon index as JavaScript numbers). */
const MERGE_PAIR_BYTES = MAXIMUM_RESIDENT_SPATIAL_JOIN_MERGE_PAIR_COUNT * 16;
const STORAGE_USAGE = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST | Buffer.VERTEX;
/** Attempts that grow or shrink capacities before the join gives up. */
const MAXIMUM_ATTEMPTS = 64;

type ChunkStatus = {
  pairCount: number;
  requiredCount: number;
  overflow: number;
  candidateOverflow: number;
  candidateCount: number;
  uncertainCount: number;
};

/**
 * Point-in-polygon enrichment with GPU-resident aggregation and no row, polygon or pair caps.
 *
 * Left points are joined in chunks. Each chunk runs GPUSpatialPredicateJoin against one prepared
 * polygon index (bounding-box BVH, robust Float32 classifier, closed intersects or open within),
 * whose capacity-bounded sorted pairs are consumed on the GPU by one fused kernel per output
 * column; count, sum, average, minimum and maximum never leave the GPU as pairs. Candidate and pair
 * capacities start from a small estimate and grow from observed counts (a few status words are the
 * only per-chunk readback). The chunk shrinks, never the input, when the working set would exceed
 * the documented reservation. Left columns are gathered on the GPU. Publication is size-gated:
 * results of at most `MAXIMUM_RESIDENT_READBACK_ROW_COUNT` rows are read back, validated (validity
 * 0 or 1, dictionary code range, non-finite aggregates) and adopted with `adoptArrowDataset`
 * (full cells, device recovery, exports). Larger results stay GPU-resident
 * (`adoptResidentDataset`): rows page in lazily under a bounded CPU row-cache reservation, numeric
 * domains are reduced on the GPU, a four byte GPU flag still refuses non-finite aggregates, and
 * they are not device-recoverable.
 */
export async function executeResidentSpatialJoin(props: {
  session: KeplerComputeSession;
  plan: ResidentSpatialJoinPlan;
  /** Resident left longitude/latitude positions, one chunk per source batch. */
  positions: GPUVector<'float32x2'>;
  polygons: PackedSpatialJoinPolygons;
  /** Retained polygon rows; required only when the plan merges the polygon geometry. */
  mergeSource?: PreparedKeplerPolygonTopologySource;
  signal?: AbortSignal;
  isCurrent(): boolean;
}): Promise<ResidentSpatialJoinResult> {
  const {session, positions, polygons} = props;
  const plan = structuredClone(props.plan);
  const left = session.registry.getDataset<KeplerArrowDataset>(plan.left.datasetId);
  const right = session.registry.getDataset<KeplerArrowDataset>(plan.right.datasetId);
  if (
    !left ||
    !right ||
    left.revision !== plan.left.revision ||
    right.revision !== plan.right.revision ||
    left.rowCount !== plan.left.rowCount ||
    right.rowCount !== plan.right.rowCount ||
    polygons.rowCount !== right.rowCount ||
    positions.length !== left.rowCount ||
    plan.resultId === left.id ||
    plan.resultId === right.id ||
    session.deviceLost ||
    props.signal?.aborted ||
    !props.isCurrent()
  )
    throw new Error('Spatial Join sources or request are stale');
  if (
    session.registry.getDataset(plan.resultId) ||
    session.registry.getRecoverySources().some(source => source.id === plan.resultId)
  )
    throw new Error('Spatial Join result ID is already occupied');
  const request = session.registry.beginRequest(`resident-spatial-join:${plan.resultId}`, [
    left.id,
    right.id
  ]);
  const cancel = () => request.cancel();
  props.signal?.addEventListener('abort', cancel, {once: true});
  const device = session.device;
  const current = () =>
    !request.signal.aborted &&
    !props.signal?.aborted &&
    !session.deviceLost &&
    session.device === device &&
    session.registry.getDataset(left.id)?.revision === left.revision &&
    session.registry.getDataset(right.id)?.revision === right.revision &&
    props.isCurrent();
  const assertCurrent = () => {
    if (!current()) throw new Error('Spatial Join was cancelled or its sources changed');
  };
  let gpuReservation: KeplerMemoryReservation | undefined;
  let cpuReservation: KeplerMemoryReservation | undefined;
  let readbackReservation: KeplerMemoryReservation | undefined;
  let compiled: CompiledGPUCommandGraph | undefined;
  let encoder: CommandEncoder | undefined;
  const fenceHolder: {current?: Fence} = {};
  let adopted: KeplerDatasetSnapshot<KeplerArrowDataset> | undefined;
  const scratch: Buffer[] = [];
  const retained: Buffer[] = [];
  let readbackBytes = 0;
  let pairCount = 0;
  let chunkCount = 0;
  const createBuffer = (
    name: string,
    byteLength: number,
    data?: ArrayBufferView,
    target: Buffer[] = scratch
  ) => {
    const buffer = device.createBuffer({
      id: `spatial-join-${name}`,
      byteLength: Math.max(4, byteLength),
      usage: STORAGE_USAGE,
      ...(data && data.byteLength ? {data} : {})
    });
    target.push(buffer);
    return buffer;
  };
  /** Submits and waits; the fence is kept so a failure path still drains outstanding work. */
  const submitAndWait = async (commands: CommandEncoder) => {
    fenceHolder.current?.destroy();
    device.submit(commands.finish());
    const nextFence = device.createFence();
    fenceHolder.current = nextFence;
    await Promise.race([nextFence.signaled, device.lost]);
    assertCurrent();
  };
  const resizeGpu = (byteLength: number) => {
    try {
      gpuReservation!.resize(byteLength);
    } catch (error) {
      throw new Error(
        `Spatial Join does not fit the GPU memory reservation (${error instanceof Error ? error.message : error}). Use SQL for larger joins.`
      );
    }
  };
  try {
    assertCurrent();
    const rowCount = left.rowCount;
    const columnCount = plan.columns.length;
    const outputBytes = rowCount * columnCount * 8;
    const geometryBytes =
      polygons.positions.byteLength +
      polygons.ringOffsets.byteLength +
      polygons.polygonOffsets.byteLength +
      polygons.featureOffsets.byteLength;
    const aggregateSources = [
      ...new Set(
        plan.columns.flatMap(column =>
          column.kind === 'aggregate' &&
          column.technique !== 'count' &&
          column.technique !== 'merge'
            ? [column.sourceColumn]
            : []
        )
      )
    ];
    // Documented reservation: retained result columns (values + validity), identity indices,
    // polygon geometry and packed right-hand values; join working memory is added per attempt.
    const fixedGpuBytes =
      outputBytes + rowCount * 4 + geometryBytes + aggregateSources.length * right.rowCount * 8;
    const mergeIndices = plan.columns.flatMap((column, index) =>
      column.kind === 'aggregate' && column.technique === 'merge' ? [index] : []
    );
    // Merge unions geometry on the CPU from accounted pair readback, so it needs the readback path.
    if (mergeIndices.length && (!props.mergeSource || !shouldReadBackResidentResult(rowCount)))
      throw new Error(
        'Spatial Join merge needs the retained polygon rows and a result within the readback cap. Uncheck merge or use SQL.'
      );
    const residentSchema: KeplerResidentColumnSchema[] = plan.columns.map((column, index) =>
      mergeIndices.includes(index)
        ? {name: column.outputColumn, format: 'uint32', dictionary: []}
        : column.kind === 'left' && column.dictionary
          ? {
              name: column.outputColumn,
              format: 'uint32',
              dictionary: left.value.prepared.dictionaries[column.sourceColumn].values
            }
          : {
              name: column.outputColumn,
              format:
                column.kind === 'left'
                  ? column.format
                  : column.kind === 'count' || column.technique === 'count'
                    ? 'uint32'
                    : 'float32'
            }
    );
    const readBack = shouldReadBackResidentResult(rowCount);
    try {
      gpuReservation = session.registry.reserveMemory(fixedGpuBytes + 65536, 'gpu', 'scratch');
      // At or below the cap: readback words and Arrow copies. Above: a bounded lazy row cache.
      cpuReservation = session.registry.reserveMemory(
        readBack
          ? estimateResidentReadbackCpuByteLength(residentSchema, rowCount) +
              (mergeIndices.length ? MERGE_PAIR_BYTES : 0)
          : estimateKeplerResidentRowCacheByteLength(residentSchema, [rowCount]),
        'cpu',
        'readback'
      );
    } catch (error) {
      throw new Error(
        `Spatial Join does not fit the memory reservation (${error instanceof Error ? error.message : error}). Use SQL for larger joins.`
      );
    }

    // Retained result columns (adopted by the dataset) and packed right-hand values.
    const ones = new Uint32Array(rowCount).fill(1);
    const outputs = plan.columns.map((column, index) => ({
      column,
      values: createBuffer(`values-${index}`, rowCount * 4, undefined, retained),
      validity: createBuffer(
        `validity-${index}`,
        rowCount * 4,
        column.kind === 'left' && left.value.analytics.validity[column.sourceColumn]
          ? undefined
          : ones,
        retained
      )
    }));
    const sources = new Map<string, {values: Buffer; validity?: Buffer}>();
    for (const name of aggregateSources)
      sources.set(name, {
        values: createBuffer(`source-${name}`, right.rowCount * 4),
        ...(right.value.analytics.validity[name]
          ? {validity: createBuffer(`source-validity-${name}`, right.rowCount * 4)}
          : {})
      });

    // Stage 1: left columns gathered with identity indices, and right values packed, on the GPU.
    {
      const graph = new GPUCommandGraph(device, {
        id: `resident-spatial-join-left-${plan.operationId}`
      });
      const identity = createBuffer(
        'identity',
        rowCount * 4,
        Uint32Array.from({length: rowCount}, (_, i) => i)
      );
      const identityView = makeView(graph, 'identity', identity, 'uint32', rowCount);
      const imports = new Map<string, GraphVectorView<ResidentJoinColumnFormat>>();
      const importVector = (
        key: string,
        vector: GPUVector | undefined,
        format: ResidentJoinColumnFormat
      ): GraphVectorView<ResidentJoinColumnFormat> => {
        if (!vector || vector.format !== format)
          throw new Error('Spatial Join requires prepared source storage matching its plan');
        let imported = imports.get(key);
        if (!imported) {
          imported = graph.importGPUVector(
            `spatial-source-${imports.size}`,
            vector as GPUVector<ResidentJoinColumnFormat>
          );
          imports.set(key, imported);
        }
        return imported;
      };
      outputs.forEach(({column, values, validity}, index) => {
        if (column.kind !== 'left') return;
        const sourceValidity = left.value.analytics.validity[column.sourceColumn];
        graph.add(
          new GPUGather({
            id: `left-values-${index}`,
            source: importVector(
              `values:${column.sourceColumn}`,
              left.value.columns[column.sourceColumn],
              column.format
            ),
            indices: identityView,
            output: makeView(graph, `out-values-${index}`, values, column.format, rowCount)
          })
        );
        if (sourceValidity)
          graph.add(
            new GPUGather({
              id: `left-validity-${index}`,
              source: importVector(`validity:${column.sourceColumn}`, sourceValidity, 'uint32'),
              indices: identityView,
              output: makeView(graph, `out-validity-${index}`, validity, 'uint32', rowCount)
            })
          );
      });
      const ownedBytes = [...scratch, ...retained].reduce(
        (total, buffer) => total + buffer.byteLength,
        0
      );
      // Published luma.gl has no pre-compile resource plan: compile, then reserve the measured
      // transient bytes. The cleanup destroys the compiled graph if the reservation fails.
      const leftCompiled = graph.compile();
      compiled = leftCompiled;
      resizeGpu(leftCompiled.stats.physicalTransientResourceBytes + ownedBytes + 65536);
      encoder = device.createCommandEncoder();
      // Packing right-hand values is a GPU buffer copy per source chunk.
      for (const [name, source] of sources) {
        copyVector(encoder, right.value.columns[name], source.values, right.rowCount);
        const validity = right.value.analytics.validity[name];
        if (validity && source.validity)
          copyVector(encoder, validity, source.validity, right.rowCount);
      }
      leftCompiled.encode(encoder, {parameters: undefined});
      assertCurrent();
      const commands = encoder;
      encoder = undefined;
      await submitAndWait(commands);
      leftCompiled.destroy();
      compiled = undefined;
    }

    // Stage 2: chunked prepared-index join with fused aggregation.
    const polygonBuffers = {
      positions: createBuffer(
        'polygon-positions',
        polygons.positions.byteLength,
        polygons.positions
      ),
      ringOffsets: createBuffer(
        'ring-offsets',
        polygons.ringOffsets.byteLength,
        polygons.ringOffsets
      ),
      polygonOffsets: createBuffer(
        'polygon-offsets',
        polygons.polygonOffsets.byteLength,
        polygons.polygonOffsets
      ),
      featureOffsets: createBuffer(
        'feature-offsets',
        polygons.featureOffsets.byteLength,
        polygons.featureOffsets
      )
    };
    const baseBytes = [...scratch, ...retained].reduce(
      (total, buffer) => total + buffer.byteLength,
      0
    );
    const aggregateOutputs = outputs.filter(
      ({column}) =>
        column.kind !== 'left' && !(column.kind === 'aggregate' && column.technique === 'merge')
    );
    // Matched right rows per left row, accumulated from the pairs of each successful chunk.
    const mergeMatches = new Map<number, number[]>();
    let mergePairCount = 0;
    let firstPoint: Float32Array | undefined;
    let capacities = fitResidentSpatialJoinCapacities(
      getInitialResidentSpatialJoinCapacities(rowCount, right.rowCount),
      right.rowCount
    );
    let doneRows = 0;
    for (let attempt = 0; doneRows < rowCount; attempt++) {
      // Each attempt runs ever larger capacities or smaller chunks, so it must terminate; the
      // bound only protects against a device that keeps reporting overflow.
      if (attempt >= MAXIMUM_ATTEMPTS)
        throw new Error('Spatial Join could not size its working set. Use SQL for this join.');
      assertCurrent();
      const attemptScratch: Buffer[] = [];
      let attemptCompiled: CompiledGPUCommandGraph | undefined;
      let prepared: GPUSpatialJoinPrepared | undefined;
      let pointReprojector: ResidentSpatialJoinPointReprojector | undefined;
      let overflowed: ResidentSpatialJoinObservation | undefined;
      try {
        const {chunkRowCount, candidateCapacity, pairCapacity} = capacities;
        const graph = new GPUCommandGraph(device, {
          id: `resident-spatial-join-${plan.operationId}-${attempt}`
        });
        const create = (name: string, byteLength: number, data?: ArrayBufferView) =>
          createBuffer(name, byteLength, data, attemptScratch);
        const scalar = (name: string) => create(name, 4, new Uint32Array(1));
        const view = <Format extends ResidentJoinColumnFormat | 'float32x2'>(
          name: string,
          buffer: Buffer,
          format: Format,
          length: number
        ) => makeView(graph, name, buffer, format, length);
        const chunkPoints = create('chunk-points', chunkRowCount * 8);
        const pointTransform = plan.provenance.crs?.pointTransform ?? 'none';
        if (pointTransform !== 'none')
          pointReprojector = new ResidentSpatialJoinPointReprojector(
            device,
            pointTransform,
            chunkRowCount,
            plan.provenance.crs
          );
        const leftRows = create('left-ids', pairCapacity * 4);
        const rightRows = create('right-ids', pairCapacity * 4);
        const statusBuffers = {
          pairCount: scalar('count'),
          requiredCount: scalar('required'),
          overflow: scalar('overflow'),
          candidateOverflow: scalar('candidate-overflow'),
          candidateCount: scalar('candidate-count'),
          uncertainCount: scalar('uncertain')
        };
        const rightGeometry = {
          kind: 'polygons' as const,
          positions: view(
            'polygon-positions',
            polygonBuffers.positions,
            'float32x2',
            polygons.vertexCount
          ),
          ringOffsets: view(
            'ring-offsets',
            polygonBuffers.ringOffsets,
            'uint32',
            polygons.ringOffsets.length
          ),
          polygonOffsets: view(
            'polygon-offsets',
            polygonBuffers.polygonOffsets,
            'uint32',
            polygons.polygonOffsets.length
          ),
          featureOffsets: view(
            'feature-offsets',
            polygonBuffers.featureOffsets,
            'uint32',
            polygons.featureOffsets.length
          )
        };
        // Polygon geometry stays resident; each chunk encoding replays its preparation
        // (including the BVH build) before the predicate join.
        prepared = new GPUSpatialJoinPrepared({
          id: 'spatial-join-prepared',
          geometry: rightGeometry
        });
        graph.add(prepared);
        const pairCountView = view('pair-count', statusBuffers.pairCount, 'uint32', 1);
        const leftRowsView = view('pair-left', leftRows, 'uint32', pairCapacity);
        const rightRowsView = view('pair-right', rightRows, 'uint32', pairCapacity);
        graph.add(
          new GPUSpatialPredicateJoin({
            id: 'spatial-join',
            left: {
              kind: 'points',
              positions: view('chunk-points', chunkPoints, 'float32x2', chunkRowCount)
            },
            right: rightGeometry,
            prepared,
            // Native turf intersects counts the boundary; native within excludes it.
            predicate: plan.predicate,
            // Planar point-in-polygon in either CRS; geographic bookkeeping only for EPSG:4326 polygons.
            spatialContext:
              plan.provenance.crs?.polygons !== 'EPSG:4326'
                ? {coordinateSpace: 'planar', metric: 'none'}
                : {coordinateSpace: 'longitude-latitude', metric: 'none', edgeModel: 'linear'},
            candidateCapacity,
            output: {
              pairs: {
                leftRows: leftRowsView,
                rightRows: rightRowsView,
                status: {
                  count: pairCountView,
                  requiredCount: view('pair-required', statusBuffers.requiredCount, 'uint32', 1),
                  overflow: view('pair-overflow', statusBuffers.overflow, 'uint32', 1),
                  candidateOverflow: view(
                    'pair-candidate-overflow',
                    statusBuffers.candidateOverflow,
                    'uint32',
                    1
                  )
                }
              },
              status: {
                overflow: view(
                  'overflow',
                  create('status-overflow', 4, new Uint32Array(1)),
                  'uint32',
                  1
                ),
                candidateCount: view('candidates', statusBuffers.candidateCount, 'uint32', 1),
                uncertainCount: view('uncertain', statusBuffers.uncertainCount, 'uint32', 1)
              }
            }
          })
        );
        const dummy = view('dummy', create('dummy', 4, new Uint32Array(1)), 'uint32', 1);
        const chunkOutputs = aggregateOutputs.map(({column}, index) => {
          const values = create(`chunk-values-${index}`, chunkRowCount * 4);
          const validity = create(`chunk-validity-${index}`, chunkRowCount * 4);
          const source = column.kind === 'aggregate' ? sources.get(column.sourceColumn) : undefined;
          graph.add(
            new ResidentSpatialJoinAggregate({
              id: `aggregate-${index}`,
              technique:
                column.kind === 'aggregate' && column.technique !== 'merge'
                  ? column.technique
                  : 'count',
              ...(column.kind === 'aggregate' && column.sourceFormat
                ? {sourceFormat: column.sourceFormat}
                : {}),
              leftRows: leftRowsView,
              rightRows: rightRowsView,
              pairCount: pairCountView,
              sourceValues: source
                ? view(`source-${index}`, source.values, 'uint32', right.rowCount)
                : dummy,
              ...(source?.validity
                ? {
                    sourceValidity: view(
                      `source-validity-${index}`,
                      source.validity,
                      'uint32',
                      right.rowCount
                    )
                  }
                : {}),
              dummy,
              outputValues: view(`chunk-out-values-${index}`, values, 'uint32', chunkRowCount),
              outputValidity: view(`chunk-out-validity-${index}`, validity, 'uint32', chunkRowCount)
            })
          );
          return {values, validity};
        });
        // Published luma.gl has no pre-compile resource plan, and compile allocates the transient
        // buffers. So the reservation is raised to the attempt's working-set cap BEFORE compiling
        // (the registry budget rejects an oversized attempt before any scratch exists), the compile
        // failing on device memory halves the chunk like an over-cap measurement, and the measured
        // bytes shrink the reservation after compile. A rejected attempt is destroyed by the
        // finally block before the retry.
        resizeGpu(baseBytes + RESIDENT_SPATIAL_JOIN_WORKING_BYTES + 65536);
        try {
          attemptCompiled = graph.compile();
        } catch (error) {
          if (
            !/out of memory|OutOfMemory/i.test(
              `${(error as Error)?.name} ${(error as Error)?.message}`
            )
          )
            throw error;
          attemptCompiled = undefined;
        }
        const attemptBytes = attemptCompiled
          ? attemptCompiled.stats.physicalTransientResourceBytes +
            attemptScratch.reduce((total, buffer) => total + buffer.byteLength, 0)
          : Infinity;
        if (attemptBytes > RESIDENT_SPATIAL_JOIN_WORKING_BYTES) {
          // The estimate under-reported the primitives' scratch: halve the chunk and retry.
          if (capacities.chunkRowCount === 1)
            throw new Error(
              'Spatial Join working set exceeds its memory reservation. Use SQL for this polygon set.'
            );
          capacities = fitResidentSpatialJoinCapacities(
            {
              chunkRowCount: Math.ceil(capacities.chunkRowCount / 2),
              candidateCapacity: Math.ceil(capacities.candidateCapacity / 2),
              pairCapacity: Math.ceil(capacities.pairCapacity / 2)
            },
            right.rowCount
          );
          continue;
        }
        resizeGpu(baseBytes + attemptBytes + 65536);
        if (!attemptCompiled) throw new Error('Spatial Join compile produced no graph.');
        for (let start = doneRows; start < rowCount; start += chunkRowCount) {
          assertCurrent();
          const validRows = Math.min(chunkRowCount, rowCount - start);
          const chunkEncoder = device.createCommandEncoder();
          encoder = chunkEncoder;
          copyPositionRows(chunkEncoder, positions, start, validRows, chunkPoints);
          if (validRows < chunkRowCount) {
            // Padding repeats a real point so every slot holds a finite, valid coordinate.
            if (!firstPoint) {
              const firstChunk = positions.data.find(chunk => chunk.length)!;
              const bytes = await getChunkBuffer(firstChunk).readAsync(firstChunk.byteOffset, 8);
              readbackBytes += 8;
              firstPoint = new Float32Array(bytes.slice().buffer);
            }
            chunkPoints.write(
              Float32Array.from(
                {length: (chunkRowCount - validRows) * 2},
                (_, i) => firstPoint![i % 2]
              ),
              validRows * 8
            );
          }
          if (pointReprojector) {
            // Points enter the polygon CRS on the GPU; padded rows are transformed too (finite).
            const reprojectPass = chunkEncoder.beginComputePass({});
            pointReprojector.encode(reprojectPass, chunkPoints);
            reprojectPass.end();
          }
          attemptCompiled.encode(chunkEncoder, {parameters: undefined});
          aggregateOutputs.forEach(({values, validity}, index) => {
            chunkEncoder.copyBufferToBuffer({
              sourceBuffer: chunkOutputs[index].values,
              sourceOffset: 0,
              destinationBuffer: values,
              destinationOffset: start * 4,
              size: validRows * 4
            });
            chunkEncoder.copyBufferToBuffer({
              sourceBuffer: chunkOutputs[index].validity,
              sourceOffset: 0,
              destinationBuffer: validity,
              destinationOffset: start * 4,
              size: validRows * 4
            });
          });
          encoder = undefined;
          await submitAndWait(chunkEncoder);
          const words = (
            await readBuffers(
              [
                statusBuffers.pairCount,
                statusBuffers.requiredCount,
                statusBuffers.overflow,
                statusBuffers.candidateOverflow,
                statusBuffers.candidateCount,
                statusBuffers.uncertainCount
              ],
              4
            )
          ).map(bytes => unsignedValues(bytes)[0]);
          readbackBytes += 24;
          assertCurrent();
          const status: ChunkStatus = {
            pairCount: words[0],
            requiredCount: words[1],
            overflow: words[2],
            candidateOverflow: words[3],
            candidateCount: words[4],
            uncertainCount: words[5]
          };
          if (status.overflow || status.candidateOverflow || status.pairCount > pairCapacity) {
            // Candidate/pair capacity was too small for this chunk; grow from observed counts.
            overflowed = {
              rowCount: validRows,
              candidateCount: status.candidateCount,
              requiredCount: status.requiredCount,
              candidateOverflow: Boolean(status.candidateOverflow)
            };
            break;
          }
          if (status.uncertainCount)
            throw new Error(
              `Spatial Join has ${status.uncertainCount} geometry classifications that Float32 cannot certify; use SQL`
            );
          if (mergeIndices.length && status.pairCount) {
            mergePairCount += status.pairCount;
            if (mergePairCount > MAXIMUM_RESIDENT_SPATIAL_JOIN_MERGE_PAIR_COUNT)
              throw new Error(
                `Spatial Join merge supports at most ${MAXIMUM_RESIDENT_SPATIAL_JOIN_MERGE_PAIR_COUNT} matched pairs. Uncheck merge or use SQL.`
              );
            const [leftBytes, rightBytes] = await readBuffers(
              [leftRows, rightRows],
              status.pairCount * 4
            );
            readbackBytes += leftBytes.byteLength + rightBytes.byteLength;
            assertCurrent();
            const leftWords = unsignedValues(leftBytes.slice());
            const rightWords = unsignedValues(rightBytes.slice());
            for (let pair = 0; pair < status.pairCount; pair++) {
              // Padding slots repeat a real point and are not rows.
              if (leftWords[pair] >= validRows) continue;
              const row = start + leftWords[pair];
              const matches = mergeMatches.get(row);
              if (matches) matches.push(rightWords[pair]);
              else mergeMatches.set(row, [rightWords[pair]]);
            }
          }
          pairCount += status.pairCount;
          chunkCount++;
          doneRows = start + validRows;
        }
      } finally {
        encoder?.destroy();
        encoder = undefined;
        attemptCompiled?.destroy();
        prepared?.destroy();
        pointReprojector?.destroy();
        const pending = fenceHolder.current
          ? Promise.race([fenceHolder.current.signaled, device.lost])
          : Promise.resolve();
        await Promise.allSettled([pending]);
        for (const buffer of attemptScratch) buffer.destroy();
      }
      if (overflowed) {
        capacities = fitResidentSpatialJoinCapacities(
          growResidentSpatialJoinCapacities(capacities, overflowed),
          right.rowCount
        );
      }
    }
    for (const buffer of scratch.splice(0)) buffer.destroy();
    resizeGpu(retained.reduce((bytes, buffer) => bytes + buffer.byteLength, 0));

    assertCurrent();

    for (const index of mergeIndices) {
      // Native mergeGeometries over the matched polygon rows, in right row order.
      const geometries = new Map<number, SpatialJoinPolygonGeometry>();
      const codes = new Uint32Array(rowCount);
      const validWords = new Uint32Array(rowCount);
      const labels: string[] = [];
      const labelCodes = new Map<string, number>();
      let labelLength = 0;
      for (const [row, matches] of mergeMatches) {
        matches.sort((a, b) => a - b);
        const key = matches.join(',');
        let code = labelCodes.get(key);
        if (code === undefined) {
          const label = mergeSpatialJoinGeometries(
            matches.map(match => {
              let geometry = geometries.get(match);
              if (!geometry) {
                geometry = readSpatialJoinPolygonGeometry(props.mergeSource!, match);
                geometries.set(match, geometry);
              }
              return geometry;
            })
          )!;
          labelLength += label.length;
          if (labelLength > MAXIMUM_RESIDENT_SPATIAL_JOIN_MERGE_LABEL_LENGTH)
            throw new Error(
              'Spatial Join merge geometry is too large to read back. Uncheck merge or use SQL.'
            );
          code = labels.length;
          labels.push(label);
          labelCodes.set(key, code);
        }
        codes[row] = code;
        validWords[row] = 1;
      }
      residentSchema[index] = {
        name: plan.columns[index].outputColumn,
        format: 'uint32',
        dictionary: labels
      };
      try {
        cpuReservation!.resize(
          estimateResidentReadbackCpuByteLength(residentSchema, rowCount) + MERGE_PAIR_BYTES
        );
      } catch (error) {
        throw new Error(
          `Spatial Join does not fit the memory reservation (${error instanceof Error ? error.message : error}). Use SQL for larger joins.`
        );
      }
      outputs[index].values.write(codes);
      outputs[index].validity.write(validWords);
    }

    const gpuColumns: Record<string, GPUVector<ResidentJoinColumnFormat>> = Object.create(null);
    const gpuValidity: Record<string, GPUVector<'uint32'>> = Object.create(null);
    for (const [index, {column, values, validity}] of outputs.entries()) {
      const name = column.outputColumn;
      gpuColumns[name] = outputVector(name, values, residentSchema[index].format, rowCount);
      gpuValidity[name] = outputVector(`${name}-validity`, validity, 'uint32', rowCount);
    }
    const isFloatAggregate = (column: ResidentSpatialJoinPlan['columns'][number]) =>
      column.kind === 'aggregate' && column.technique !== 'count';
    if (readBack) {
      // Staging buffers for one value and one validity column at a time.
      readbackReservation = session.registry.reserveMemory(rowCount * 8, 'gpu', 'readback');
      const readback = await readResidentResultChunks({
        session,
        columns: residentSchema.map((column, index) => ({
          ...column,
          refuseNonFinite: isFloatAggregate(plan.columns[index]),
          values: gpuColumns[column.name],
          validity: gpuValidity[column.name]
        })),
        assertCurrent
      });
      readbackReservation.release();
      readbackBytes += readback.readbackBytes;
      const result = makeResidentReadbackTable({
        columns: readback.columns,
        metadata: {'luma.gl.spatialJoin': JSON.stringify(plan.provenance)}
      });
      assertCurrent();
      adopted =
        session.adoptArrowDataset({
          id: plan.resultId,
          table: result.table,
          prepared: result.prepared,
          columns: gpuColumns,
          validity: gpuValidity,
          normalization: result.normalization,
          request,
          cpuReservation,
          gpuReservation,
          metadata: {spatialJoin: plan.provenance, derivedDataset: plan.derivedDataset},
          destroy: () => {
            for (const buffer of retained) buffer.destroy();
          }
        }) ?? undefined;
      if (!adopted) throw new Error('Spatial Join result was superseded before publication');
    } else {
      // The lazy view never scans cells, so refuse non-finite aggregates on the GPU first.
      readbackReservation = session.registry.reserveMemory(4096, 'gpu', 'readback');
      const hasNonFinite = await hasResidentNonFiniteValues({
        device,
        id: `resident-spatial-join-${plan.operationId}`,
        rowCount,
        columns: outputs
          .filter(({column}) => isFloatAggregate(column))
          .map(({values, validity}) => ({values, validity})),
        assertCurrent
      });
      readbackReservation.release();
      readbackBytes += 4;
      if (hasNonFinite)
        throw new Error('Result has a non-finite value; use SQL for unsupported values');
      adopted =
        session.adoptResidentDataset({
          id: plan.resultId,
          schema: residentSchema,
          columns: gpuColumns,
          validity: gpuValidity,
          schemaMetadata: {'luma.gl.spatialJoin': JSON.stringify(plan.provenance)},
          request,
          cpuReservation,
          gpuReservation,
          destroy: () => {
            for (const buffer of retained) buffer.destroy();
          }
        }) ?? undefined;
      if (!adopted) throw new Error('Spatial Join result was superseded before publication');
    }
    const snapshot = adopted;
    if (!readBack)
      try {
        // GPU min/max reduction; only a few bytes per numeric column return.
        const domains = await session.readResidentDomains(snapshot, {
          workingMemoryBudgetBytes: getResidentDomainBudgetBytes(
            rowCount,
            Object.keys(snapshot.value.columns).length
          )
        });
        if (!domains) throw new Error('Spatial Join result was superseded before publication');
        assertCurrent();
      } catch (error) {
        await removeGeneration(session, snapshot);
        throw error;
      }
    let transferred = false;
    let destruction: Promise<void> | undefined;
    const isResultCurrent = () =>
      !destruction &&
      !session.deviceLost &&
      session.device === device &&
      session.registry.getDataset(snapshot.id)?.revision === snapshot.revision &&
      (transferred ||
        (!props.signal?.aborted &&
          session.registry.getDataset(left.id)?.revision === left.revision &&
          session.registry.getDataset(right.id)?.revision === right.revision &&
          props.isCurrent()));
    return {
      dataset: snapshot,
      provenance: plan.provenance,
      rowCount,
      pairCount,
      chunkCount,
      readbackBytes,
      isCurrent: isResultCurrent,
      transfer() {
        if (transferred) return;
        if (!isResultCurrent())
          throw new Error('Spatial Join became stale before attachment committed');
        transferred = true;
      },
      destroy() {
        if (transferred) return Promise.resolve();
        destruction ??= removeGeneration(session, snapshot);
        return destruction;
      }
    };
  } finally {
    props.signal?.removeEventListener('abort', cancel);
    encoder?.destroy();
    try {
      const completion = fenceHolder.current
        ? Promise.race([fenceHolder.current.signaled, device.lost])
        : Promise.resolve();
      await Promise.allSettled([completion]);
    } finally {
      fenceHolder.current?.destroy();
      compiled?.destroy();
      for (const buffer of scratch) buffer.destroy();
      if (!adopted) for (const buffer of retained) buffer.destroy();
      gpuReservation?.release();
      readbackReservation?.release();
      cpuReservation?.release();
      await session.registry.finishRequest(request);
    }
  }
}

function outputVector<Format extends ResidentJoinColumnFormat>(
  name: string,
  buffer: Buffer,
  format: Format,
  length: number
): GPUVector<Format> {
  return new GPUVector({
    name,
    format,
    type: 'data',
    ownsData: true,
    data: [new GPUData({format, buffer, length, ownsBuffer: false})]
  });
}
function unsignedValues(bytes: Uint8Array): Uint32Array {
  return new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}
function makeView<Format extends ResidentJoinColumnFormat | 'float32x2'>(
  graph: GPUCommandGraph,
  name: string,
  buffer: Buffer,
  format: Format,
  length: number
): GraphDataView<Format> {
  return graph.createDataView(
    graph.importBuffer({id: name, usage: buffer.usage, byteLength: buffer.byteLength}, buffer),
    {format, length}
  );
}
/** Source chunks may borrow a DynamicBuffer; GPU copies need the underlying GPUBuffer wrapper. */
function getChunkBuffer(chunk: GPUData): Buffer {
  return chunk.buffer instanceof Buffer ? chunk.buffer : chunk.buffer.buffer;
}
/** Copies a chunked 4-byte-per-row vector into one contiguous buffer on the GPU. */
function copyVector(
  encoder: CommandEncoder,
  vector: GPUVector | undefined,
  destination: Buffer,
  rowCount: number
): void {
  if (!vector) throw new Error('Spatial Join column storage is missing');
  let rowOffset = 0;
  for (const chunk of vector.data) {
    if (!chunk.length) continue;
    encoder.copyBufferToBuffer({
      sourceBuffer: getChunkBuffer(chunk),
      sourceOffset: chunk.byteOffset,
      destinationBuffer: destination,
      destinationOffset: rowOffset * 4,
      size: chunk.length * 4
    });
    rowOffset += chunk.length;
  }
  if (rowOffset !== rowCount)
    throw new Error('Spatial Join column storage does not match its rows');
}
/** Copies `rowCount` points starting at source row `start`, across source chunk boundaries. */
function copyPositionRows(
  encoder: CommandEncoder,
  positions: GPUVector<'float32x2'>,
  start: number,
  rowCount: number,
  destination: Buffer
): void {
  let chunkStart = 0;
  let copied = 0;
  for (const chunk of positions.data) {
    const chunkEnd = chunkStart + chunk.length;
    const first = Math.max(start, chunkStart);
    const last = Math.min(start + rowCount, chunkEnd);
    if (last > first) {
      encoder.copyBufferToBuffer({
        sourceBuffer: getChunkBuffer(chunk),
        sourceOffset: chunk.byteOffset + (first - chunkStart) * 8,
        destinationBuffer: destination,
        destinationOffset: (first - start) * 8,
        size: (last - first) * 8
      });
      copied += last - first;
    }
    chunkStart = chunkEnd;
  }
  // Positions are validated against the dataset row count before execution.
  if (copied !== rowCount) throw new Error('Spatial Join point positions changed');
}
async function readBuffers(buffers: Buffer[], byteLength: number): Promise<Uint8Array[]> {
  const results = await Promise.allSettled(
    buffers.map(buffer => Promise.resolve().then(() => buffer.readAsync(0, byteLength)))
  );
  const failure = results.find(result => result.status === 'rejected');
  if (failure?.status === 'rejected') throw failure.reason;
  return results.map(result => {
    if (result.status === 'rejected') throw result.reason;
    return result.value;
  });
}
async function removeGeneration(
  session: KeplerComputeSession,
  snapshot: KeplerDatasetSnapshot<KeplerArrowDataset>
): Promise<void> {
  if (session.registry.getDataset(snapshot.id)?.revision !== snapshot.revision) {
    if (
      session.deviceLost &&
      session.registry
        .getRecoverySources()
        .some(source => source.id === snapshot.id && source.revision === snapshot.revision)
    )
      session.registry.removeDataset(snapshot.id);
    return;
  }
  if (session.deviceLost) {
    session.registry.removeDataset(snapshot.id);
    return;
  }
  const fence = session.device.createFence();
  try {
    await session.removeDataset(snapshot.id, Promise.race([fence.signaled, session.device.lost]));
  } finally {
    fence.destroy();
  }
}
