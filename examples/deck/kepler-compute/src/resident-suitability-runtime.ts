// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '@luma-contributors/kepler';
import {Buffer, type CommandEncoder} from '@luma.gl/core';
import type {DynamicBuffer} from '@luma.gl/engine';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {
  addGPUCommandNodes,
  createTransientView,
  GPUCommandGraph,
  GPUReduction,
  type GraphBufferHandle,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {GPUVariableStandardization} from '@luma-contributors/analysis/dataframe-analysis';
import {
  estimateKeplerResidentRowCacheByteLength,
  type KeplerArrowDataset,
  type KeplerComputeSession,
  type KeplerDatasetSnapshot,
  type KeplerMemoryReservation,
  type KeplerResidentColumnSchema
} from '@luma-contributors/kepler';
import {
  createSuitabilityAccumulateNode,
  createSuitabilityFinishNodes,
  createSuitabilityPrepareNodes,
  createSuitabilityReclassifyNodes
} from './resident-suitability-kernels';
import {
  estimateResidentReadbackCpuByteLength,
  getResidentDomainBudgetBytes,
  makeResidentReadbackTable,
  readResidentResultChunks,
  shouldReadBackResidentResult
} from './resident-result-readback';
import {assertResidentSuitabilityStatistics} from './resident-suitability-pack';
import {
  getResidentSuitabilityEffectiveWeights,
  type ResidentSuitabilityColumnFormat,
  type ResidentSuitabilityProvenance,
  type ResidentSuitabilityResultPlan
} from './resident-suitability-plan';

export type ResidentSuitabilityResult = {
  readonly dataset: KeplerDatasetSnapshot<KeplerArrowDataset>;
  readonly provenance: ResidentSuitabilityProvenance;
  readonly rowCount: number;
  /** Rows missing in some but not all weighted fields, scored per the missing-value policy. */
  readonly partialMissingRowCount: number;
  readonly readbackBytes: number;
  /** Includes source/request callback checks until the host commits attachment. */
  isCurrent(): boolean;
  /** Transfer result generation ownership only after successful native attachment. */
  transfer(): void;
  /** Roll back only this generation; never remove a newer result using the same ID. */
  destroy(): Promise<void>;
};

const STORAGE_USAGE = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST | Buffer.VERTEX;
/** Slack for kernel scratch (reduction partials, small statistics) beyond the row-sized buffers. */
const SCRATCH_MARGIN_BYTES = 4 * 1024 * 1024;
/** Row-sized Float32/Uint32 working buffers alive at once while one column is standardized. */
const TRANSIENT_ROW_BUFFERS = 8;
const WORD_BYTES = 4;

/**
 * Single-source weighted-criteria score with no row cap. Each weighted column is standardized on
 * the GPU straight from its resident chunks (`GPUVariableStandardization`, population statistics
 * like native) and accumulated into a weighted sum, one column at a time so working memory stays
 * at a handful of row-sized buffers. Only small statistics (extent and defined count per column,
 * one mixed-missing flag) are read back, to refuse constant columns, inexact integers and
 * differently-missing fields before anything is published. The score is written directly into
 * chunk-preserving GPU columns; carried columns are copied chunk by chunk on the GPU. Nothing is
 * packed or repartitioned.
 *
 * Publication is size-gated: results of at most `MAXIMUM_RESIDENT_READBACK_ROW_COUNT` rows are
 * read back, validated (validity 0 or 1, dictionary code range) and adopted with
 * `adoptArrowDataset` (full cells, device recovery, exports). Larger results are adopted
 * GPU-resident (`adoptResidentDataset`): rows page in lazily under a bounded CPU row-cache
 * reservation, numeric domains are reduced on the GPU, and they are not device-recoverable. No
 * source column is ever read back. The source stays borrowed.
 */
export async function executeResidentSuitability(props: {
  session: KeplerComputeSession;
  plan: ResidentSuitabilityResultPlan;
  signal?: AbortSignal;
  isCurrent(): boolean;
}): Promise<ResidentSuitabilityResult> {
  const {session} = props;
  const plan = structuredClone(props.plan);
  const source = session.registry.getDataset<KeplerArrowDataset>(plan.source.datasetId);
  if (
    !source ||
    source.revision !== plan.source.revision ||
    source.rowCount !== plan.source.rowCount ||
    plan.resultId === source.id ||
    session.deviceLost ||
    props.signal?.aborted ||
    !props.isCurrent()
  )
    throw new Error('Suitability source or request is stale');
  if (
    session.registry.getDataset(plan.resultId) ||
    session.registry.getRecoverySources().some(recovery => recovery.id === plan.resultId)
  )
    throw new Error('Suitability result ID is already occupied');
  const request = session.registry.beginRequest(`resident-suitability:${plan.resultId}`, [
    source.id
  ]);
  const cancel = () => request.cancel();
  props.signal?.addEventListener('abort', cancel, {once: true});
  const device = session.device;
  const current = () =>
    !request.signal.aborted &&
    !props.signal?.aborted &&
    !session.deviceLost &&
    session.device === device &&
    session.registry.getDataset(source.id)?.revision === source.revision &&
    props.isCurrent();
  const assertCurrent = () => {
    if (!current()) throw new Error('Suitability was cancelled or its source changed');
  };
  let gpuReservation: KeplerMemoryReservation | undefined;
  let cpuReservation: KeplerMemoryReservation | undefined;
  let readbackReservation: KeplerMemoryReservation | undefined;
  let adopted: KeplerDatasetSnapshot<KeplerArrowDataset> | undefined;
  const outputs: Buffer[] = [];
  const retainedOutputs = new Set<Buffer>();
  let resultReadbackBytes = 0;
  try {
    assertCurrent();
    const sourceValue = source.value;
    const rowCount = source.rowCount;
    const indicatorCount = plan.scoring.indicators.length;
    const names = plan.scoring.indicators.map(indicator => indicator.column);
    const carriedCount = plan.columns.length;

    // The source's chunk layout is the result's layout; empty chunks carry no rows.
    const layoutVector =
      sourceValue.columns[plan.columns[0]?.sourceColumn ?? names[0]] ?? undefined;
    const chunkLengths = (layoutVector?.data ?? [])
      .map(chunk => chunk.length)
      .filter(length => length > 0);
    // Matching chunk layouts are an invariant of prepared sources; the message names the cause.
    if (chunkLengths.reduce((total, length) => total + length, 0) !== rowCount)
      throw new Error('Suitability source storage changed');

    const outputBytes = rowCount * (carriedCount + 1) * 8;
    const persistentBytes = rowCount * WORD_BYTES * 2 + outputBytes + 4096;
    const transientBytes = rowCount * WORD_BYTES * TRANSIENT_ROW_BUFFERS;
    gpuReservation = session.registry.reserveMemory(
      persistentBytes + transientBytes + SCRATCH_MARGIN_BYTES,
      'gpu',
      'scratch'
    );
    const residentSchema: KeplerResidentColumnSchema[] = [
      ...plan.columns.map(column => ({
        name: column.outputColumn,
        format: column.format,
        ...(column.dictionary
          ? {dictionary: sourceValue.prepared.dictionaries[column.sourceColumn].values}
          : {})
      })),
      {name: plan.scoring.scoreColumn, format: 'float32' as const}
    ];
    const readBack = shouldReadBackResidentResult(rowCount);
    // At or below the cap: readback words and Arrow copies. Above: a bounded lazy row cache.
    cpuReservation = session.registry.reserveMemory(
      readBack
        ? estimateResidentReadbackCpuByteLength(residentSchema, rowCount)
        : estimateKeplerResidentRowCacheByteLength(residentSchema, chunkLengths),
      'cpu',
      'readback'
    );
    const allocate = (
      name: string,
      byteLength: number,
      data?: ArrayBufferView,
      retained = false
    ): Buffer => {
      const buffer = device.createBuffer({
        id: `suitability-${name}`,
        byteLength: Math.max(4, byteLength),
        usage: STORAGE_USAGE,
        ...(data ? {data} : {})
      });
      outputs.push(buffer);
      if (retained) retainedOutputs.add(buffer);
      return buffer;
    };
    const importView = <Format extends 'uint32' | 'float32'>(
      graph: GPUCommandGraph,
      name: string,
      buffer: Buffer,
      format: Format,
      length: number,
      byteOffset = 0
    ): GraphDataView<Format> => {
      const handle: GraphBufferHandle = graph.importBuffer(
        {id: name, usage: buffer.usage, byteLength: buffer.byteLength},
        buffer
      );
      return graph.createDataView(handle, {format, length, byteOffset});
    };
    const runGraph = async (graph: GPUCommandGraph): Promise<void> => {
      // Published luma.gl has no pre-compile resource plan: compile, then admit the measured bytes.
      const compiled = graph.compile();
      let encoder: CommandEncoder | undefined;
      try {
        const plannedBytes = compiled.stats.physicalTransientResourceBytes;
        if (persistentBytes + plannedBytes > gpuReservation!.byteLength)
          throw new KeplerAllocationError('memory-budget', {
            message: 'Suitability exceeds admitted GPU working memory'
          });
        assertCurrent();
        encoder = device.createCommandEncoder();
        compiled.encode(encoder, {parameters: undefined});
        assertCurrent();
        device.submit(encoder.finish());
        encoder = undefined;
        await waitForSubmittedWork();
        assertCurrent();
      } finally {
        encoder?.destroy();
        compiled.destroy();
      }
    };
    const waitForSubmittedWork = async (): Promise<void> => {
      const fence = device.createFence();
      try {
        await Promise.race([fence.signaled, device.lost]);
      } finally {
        fence.destroy();
      }
    };

    // Row-sized accumulators and the small statistics read back before publication.
    const scoreAccumulator = allocate('score-accumulator', rowCount * WORD_BYTES);
    const definedCounts = allocate('defined-counts', rowCount * WORD_BYTES);
    const missingValues = plan.scoring.missingValuePolicy;
    const weightSums =
      missingValues === 'renormalize' ? allocate('weight-sums', rowCount * WORD_BYTES) : undefined;
    const extentBuffer = allocate('extents', indicatorCount * 2 * WORD_BYTES);
    const countBuffer = allocate(
      'counts',
      (indicatorCount + 1) * WORD_BYTES,
      new Uint32Array(indicatorCount + 1)
    );
    const effectiveWeights = getResidentSuitabilityEffectiveWeights(
      plan.scoring,
      plan.provenance.weightStandardization
    );

    // Phase 1: standardize and accumulate one weighted column at a time, straight from chunks.
    for (const [index, name] of names.entries()) {
      const vector = sourceValue.columns[name];
      if (!vector || vector.format !== plan.indicatorFormats[index])
        throw new Error('Suitability source storage changed');
      const graph = new GPUCommandGraph(device, {
        id: `resident-suitability-${plan.operationId}-${index}`
      });
      const sourceView = graph.importGPUVector(
        `suitability-source-${index}`,
        vector as GPUVector<ResidentSuitabilityColumnFormat>
      );
      const sourceValidity = sourceValue.analytics.validity[name];
      const validityView = sourceValidity
        ? graph.importGPUVector(`suitability-validity-${index}`, sourceValidity)
        : undefined;
      const clean = createTransientView(graph, 'clean', 'float32', rowCount);
      const defined = createTransientView(graph, 'defined', 'uint32', rowCount);
      // Raw data standardization weights the cleaned values unchanged (no standardization pass).
      const standardized =
        plan.scoring.scaler === 'none'
          ? clean
          : createTransientView(graph, 'standardized', 'float32', rowCount);
      const reclassification = plan.indicatorReclassifications[index];
      if (reclassification) {
        // Categorical field: dictionary codes become scores through a per-code lookup table.
        const lookup = allocate(
          `reclassify-${index}`,
          reclassification.length * WORD_BYTES,
          Float32Array.from(reclassification, score => score ?? Number.NaN)
        );
        addGPUCommandNodes(
          graph,
          createSuitabilityReclassifyNodes(graph, {
            id: `suitability-${index}`,
            source: sourceView as GraphVectorView<'uint32'>,
            validity: validityView,
            lookup: importView(graph, 'reclassify', lookup, 'float32', reclassification.length),
            clean,
            defined
          })
        );
      } else
        addGPUCommandNodes(
          graph,
          createSuitabilityPrepareNodes(graph, {
            id: `suitability-${index}`,
            format: plan.indicatorFormats[index],
            source: sourceView,
            validity: validityView,
            clean,
            defined
          })
        );
      graph.add(
        new GPUReduction({
          id: `suitability-extent-${index}`,
          input: clean,
          mask: defined,
          operation: 'extent',
          output: importView(graph, 'extents', extentBuffer, 'float32', 2, index * 2 * WORD_BYTES)
        })
      );
      graph.add(
        new GPUReduction({
          id: `suitability-count-${index}`,
          input: defined,
          operation: 'sum',
          output: importView(graph, 'counts', countBuffer, 'uint32', 1, index * WORD_BYTES)
        })
      );
      if (plan.scoring.scaler !== 'none')
        graph.add(
          new GPUVariableStandardization({
            id: `suitability-standardize-${index}`,
            values: clean,
            mask: defined,
            method: plan.scoring.scaler === 'min-max' ? 'range-standardize' : 'z-score',
            variance: 'population',
            output: standardized
          })
        );
      addGPUCommandNodes(graph, [
        createSuitabilityAccumulateNode(graph, {
          id: `suitability-accumulate-${index}`,
          first: index === 0,
          weight: effectiveWeights[index],
          standardized,
          defined,
          score: importView(graph, 'score-accumulator', scoreAccumulator, 'float32', rowCount),
          definedCount: importView(graph, 'defined-counts', definedCounts, 'uint32', rowCount),
          weightSum: weightSums
            ? importView(graph, 'weight-sums', weightSums, 'float32', rowCount)
            : undefined
        })
      ]);
      await runGraph(graph);
    }

    // Phase 2: write the chunk-preserving score columns and copy the carried columns on the GPU.
    const makeChunkedVector = <Format extends ResidentSuitabilityColumnFormat>(
      name: string,
      format: Format
    ): GPUVector<Format> =>
      new GPUVector({
        name,
        format,
        type: 'data',
        ownsData: true,
        data: chunkLengths.map(
          (length, chunkIndex) =>
            new GPUData({
              format,
              buffer: allocate(`${name}-${chunkIndex}`, length * WORD_BYTES, undefined, true),
              length,
              ownsBuffer: false
            })
        )
      });
    const scoreName = plan.scoring.scoreColumn;
    const scoreValueVector = makeChunkedVector(scoreName, 'float32');
    const scoreValidityVector = makeChunkedVector(`${scoreName}-validity`, 'uint32');
    const mixedMissingFlag = allocate('mixed-missing-flag', WORD_BYTES, new Uint32Array(1));
    const finishGraph = new GPUCommandGraph(device, {
      id: `resident-suitability-${plan.operationId}-finish`
    });
    addGPUCommandNodes(
      finishGraph,
      createSuitabilityFinishNodes(finishGraph, {
        id: 'suitability',
        indicatorCount,
        score: importView(finishGraph, 'score-accumulator', scoreAccumulator, 'float32', rowCount),
        definedCount: importView(finishGraph, 'defined-counts', definedCounts, 'uint32', rowCount),
        scoreValues: finishGraph.importGPUVector('suitability-score-values', scoreValueVector),
        scoreValidity: finishGraph.importGPUVector(
          'suitability-score-validity',
          scoreValidityVector
        ),
        mixedMissingFlag: importView(
          finishGraph,
          'mixed-missing-flag',
          mixedMissingFlag,
          'uint32',
          1
        ),
        missingValues,
        totalWeight: effectiveWeights.reduce((sum, weight) => sum + weight, 0),
        weightSum: weightSums
          ? importView(finishGraph, 'weight-sums', weightSums, 'float32', rowCount)
          : undefined
      })
    );
    await runGraph(finishGraph);

    // Small statistics refuse unreproducible inputs before any result is read or published.
    const [extentBytes, countBytes, flagBytes] = await Promise.all([
      extentBuffer.readAsync(0, indicatorCount * 2 * WORD_BYTES),
      countBuffer.readAsync(0, indicatorCount * WORD_BYTES),
      mixedMissingFlag.readAsync(0, WORD_BYTES)
    ]);
    resultReadbackBytes += (indicatorCount * 3 + 1) * WORD_BYTES;
    assertCurrent();
    const partialMissingRowCount = new Uint32Array(flagBytes.slice().buffer)[0];
    assertResidentSuitabilityStatistics({
      names,
      formats: plan.indicatorFormats.map((format, index) =>
        plan.indicatorReclassifications[index] ? 'float32' : format
      ),
      extents: new Float32Array(extentBytes.slice().buffer),
      counts: new Uint32Array(countBytes.slice().buffer),
      mixedMissingRows: partialMissingRowCount,
      missingValues,
      isConstantAllowed: plan.scoring.scaler === 'none'
    });

    const gpuColumns: Record<string, GPUVector<ResidentSuitabilityColumnFormat>> = Object.create(
      null
    );
    const gpuValidity: Record<string, GPUVector<'uint32'>> = Object.create(null);
    const copyEncoder = device.createCommandEncoder();
    try {
      for (const column of plan.columns) {
        const sourceVector = sourceValue.columns[column.sourceColumn];
        if (!sourceVector || sourceVector.format !== column.format)
          throw new Error('Suitability requires prepared source storage matching its plan');
        const sourceValidity = sourceValue.analytics.validity[column.sourceColumn];
        const values = makeChunkedVector(column.outputColumn, column.format);
        const validity = makeChunkedVector(`${column.outputColumn}-validity`, 'uint32');
        copyChunks(copyEncoder, sourceVector, values, chunkLengths);
        if (sourceValidity) copyChunks(copyEncoder, sourceValidity, validity, chunkLengths);
        else
          for (const [chunkIndex, chunk] of validity.data.entries())
            getCoreBuffer(chunk.buffer).write(new Uint32Array(chunkLengths[chunkIndex]).fill(1), 0);
        gpuColumns[column.outputColumn] = values;
        gpuValidity[column.outputColumn] = validity;
      }
      device.submit(copyEncoder.finish());
    } catch (error) {
      copyEncoder.destroy();
      throw error;
    }
    await waitForSubmittedWork();
    assertCurrent();
    gpuColumns[scoreName] = scoreValueVector;
    gpuValidity[scoreName] = scoreValidityVector;

    assertCurrent();
    // Retain only the immutable dataset allocation; accumulators and statistics are finished.
    for (let index = outputs.length - 1; index >= 0; index--) {
      if (retainedOutputs.has(outputs[index])) continue;
      outputs[index].destroy();
      outputs.splice(index, 1);
    }
    gpuReservation.resize(
      [...retainedOutputs].reduce((bytes, buffer) => bytes + buffer.byteLength, 0)
    );
    if (readBack) {
      // Staging buffers for one value and one validity column at a time.
      readbackReservation = session.registry.reserveMemory(rowCount * 8, 'gpu', 'readback');
      const readback = await readResidentResultChunks({
        session,
        columns: residentSchema.map(column => ({
          ...column,
          values: gpuColumns[column.name],
          validity: gpuValidity[column.name]
        })),
        assertCurrent
      });
      readbackReservation.release();
      resultReadbackBytes += readback.readbackBytes;
      const result = makeResidentReadbackTable({
        columns: readback.columns,
        metadata: {'luma.gl.suitability': JSON.stringify(plan.provenance)}
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
          metadata: {suitability: plan.provenance, derivedDataset: plan.derivedDataset},
          destroy: () => {
            for (const buffer of retainedOutputs) buffer.destroy();
          }
        }) ?? undefined;
      if (!adopted) throw new Error('Suitability result was superseded before publication');
    } else {
      adopted =
        session.adoptResidentDataset({
          id: plan.resultId,
          schema: residentSchema,
          columns: gpuColumns,
          validity: gpuValidity,
          schemaMetadata: {'luma.gl.suitability': JSON.stringify(plan.provenance)},
          request,
          cpuReservation,
          gpuReservation,
          destroy: () => {
            for (const buffer of retainedOutputs) buffer.destroy();
          }
        }) ?? undefined;
      if (!adopted) throw new Error('Suitability result was superseded before publication');
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
        if (!domains) throw new Error('Suitability result was superseded before publication');
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
          session.registry.getDataset(source.id)?.revision === source.revision &&
          props.isCurrent()));
    return {
      dataset: snapshot,
      provenance: plan.provenance,
      rowCount,
      partialMissingRowCount,
      readbackBytes: resultReadbackBytes,
      isCurrent: isResultCurrent,
      transfer() {
        if (transferred) return;
        if (!isResultCurrent())
          throw new Error('Suitability became stale before attachment committed');
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
    // Every graph and copy awaited its own fence, so no submitted work still touches these.
    try {
      for (const buffer of outputs) if (!adopted || !retainedOutputs.has(buffer)) buffer.destroy();
      gpuReservation?.release();
      readbackReservation?.release();
      cpuReservation?.release();
    } finally {
      await session.registry.finishRequest(request);
    }
  }
}

function getCoreBuffer(buffer: Buffer | DynamicBuffer): Buffer {
  return buffer instanceof Buffer ? buffer : buffer.buffer;
}
/** Copies each source chunk into the matching dedicated output chunk, on the GPU. */
function copyChunks(
  encoder: CommandEncoder,
  sourceVector: GPUVector,
  target: GPUVector,
  chunkLengths: readonly number[]
): void {
  const sourceChunks = sourceVector.data.filter(chunk => chunk.length > 0);
  // Prepared sources share one chunk layout across columns; a mismatch means storage changed.
  if (
    sourceChunks.length !== chunkLengths.length ||
    sourceChunks.some(
      (chunk, index) => chunk.length !== chunkLengths[index] || chunk.byteStride !== WORD_BYTES
    )
  )
    throw new Error('Suitability source storage changed');
  for (const [index, chunk] of sourceChunks.entries())
    encoder.copyBufferToBuffer({
      sourceBuffer: getCoreBuffer(chunk.buffer),
      sourceOffset: chunk.byteOffset,
      destinationBuffer: getCoreBuffer(target.data[index].buffer),
      size: chunk.length * WORD_BYTES
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
