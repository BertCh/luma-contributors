// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '@luma-contributors/kepler';
import {Buffer, type CommandEncoder, type Device, type Fence} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {
  GPUCommandGraph,
  GPUGather,
  GPUScan,
  GPUScanUint64,
  GPUSort,
  type CompiledGPUCommandGraph,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {
  estimateKeplerResidentRowCacheByteLength,
  type KeplerArrowDataset,
  type KeplerComputeSession,
  type KeplerDatasetSnapshot,
  type KeplerMemoryReservation,
  type KeplerResidentColumnSchema
} from '@luma-contributors/kepler';
import {
  estimateResidentReadbackCpuByteLength,
  getResidentDomainBudgetBytes,
  makeResidentReadbackTable,
  readResidentResultChunks,
  shouldReadBackResidentResult
} from './resident-result-readback';
import {
  addResidentJoinFillPass,
  addResidentJoinPass,
  addResidentJoinKeyPass,
  addResidentJoinProbePass,
  addResidentJoinRunPass,
  addResidentJoinSummaryPass,
  addResidentJoinUnmatchedPass,
  RESIDENT_JOIN_SUMMARY_WORDS
} from './resident-join-gpu-kernels';
import {
  RESIDENT_JOIN_STATUS_DICTIONARY_CODE,
  RESIDENT_JOIN_STATUS_FLOAT_KEY,
  RESIDENT_JOIN_STATUS_OUTPUT_OVERFLOW,
  type ResidentJoinKeyMode
} from './resident-join-gpu-reference';
import {
  addResidentJoinCompositeFinishPass,
  RESIDENT_JOIN_COMPOSITE_MATCH_COLUMN,
  addResidentJoinCompositeKeyPass,
  type ResidentJoinCompositeKeyPlan
} from './resident-join-composite-key';
import {
  makeResidentJoinDictionaryKeyTables,
  MAXIMUM_RESIDENT_JOIN_ROW_INDEX,
  RESIDENT_JOIN_SQL_GUIDANCE,
  shouldDropResidentJoinHighKeys,
  type ResidentJoinColumnFormat,
  type ResidentJoinPlan,
  type ResidentJoinProvenance
} from './resident-join-plan';

export type ResidentJoinResult = {
  readonly dataset: KeplerDatasetSnapshot<KeplerArrowDataset>;
  readonly provenance: ResidentJoinProvenance;
  readonly rowCount: number;
  readonly readbackBytes: number;
  /** Includes source/request callback checks until the host commits attachment. */
  isCurrent(): boolean;
  /** Transfer result generation ownership only after successful native attachment. */
  transfer(): void;
  /** Roll back only this generation; never remove a newer result using the same ID. */
  destroy(): Promise<void>;
};

const STORAGE_USAGE = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST | Buffer.VERTEX;

/**
 * Two-input attribute join with no row cap. Keys are copied GPU-to-GPU, canonicalized, and joined
 * by two stable GPUSort passes (right rows by key, valid rows first), a duplicate-aware range
 * probe, GPUScan/GPUScanUint64 offsets and an output-parallel fill, so only a 16 byte summary
 * (output total, unmatched total, status, valid right rows) is read back before allocation. Row
 * mappings and gathered payload columns are produced by GPUGather into dedicated resident buffers;
 * sources stay borrowed. Size-gated publication: results of at most
 * `MAXIMUM_RESIDENT_READBACK_ROW_COUNT` rows are read back, validated (validity 0 or 1, dictionary
 * code range) and adopted with `adoptArrowDataset` (full cells, device recovery, exports). Larger
 * results stay GPU-resident (`adoptResidentDataset`): no output-sized readback, rows page in
 * lazily on demand, numeric domains are reduced on the GPU, and they are not device-recoverable.
 * Working memory is admitted by `reserveMemory`, not by a row count.
 */
export async function executeResidentJoin(props: {
  session: KeplerComputeSession;
  plan: ResidentJoinPlan;
  signal?: AbortSignal;
  /** Folds several key columns into the single key word (`plan.leftKey` is its first column). */
  compositeKey?: ResidentJoinCompositeKeyPlan;
  isCurrent(): boolean;
}): Promise<ResidentJoinResult> {
  const {session} = props;
  const composite = props.compositeKey;
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
    plan.resultId === left.id ||
    plan.resultId === right.id ||
    session.deviceLost ||
    props.signal?.aborted ||
    !props.isCurrent()
  )
    throw new Error('Join sources or request are stale');
  if (
    session.registry.getDataset(plan.resultId) ||
    session.registry.getRecoverySources().some(source => source.id === plan.resultId)
  )
    throw new Error('Join result ID is already occupied');
  const request = session.registry.beginRequest(`resident-join:${plan.resultId}`, [
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
    if (!current()) throw new Error('Join was cancelled or its sources changed');
  };
  let keyCpuReservation: KeplerMemoryReservation | undefined;
  let gpuReservation: KeplerMemoryReservation | undefined;
  let cpuReservation: KeplerMemoryReservation | undefined;
  let readbackReservation: KeplerMemoryReservation | undefined;
  let compiled: CompiledGPUCommandGraph | undefined;
  let compiledKeys: CompiledGPUCommandGraph | undefined;
  let encoder: CommandEncoder | undefined;
  let fence: Fence | undefined;
  let submitted = false;
  let adopted: KeplerDatasetSnapshot<KeplerArrowDataset> | undefined;
  const outputs: Buffer[] = [];
  const retainedOutputs = new Set<Buffer>();
  let resultReadbackBytes = 0; // Summary words, plus result columns when read back (<= cap).
  try {
    assertCurrent();
    const leftSource = left.value;
    const rightSource = right.value;
    const leftKeyVector = leftSource.columns[plan.leftKey];
    const rightKeyVector = rightSource.columns[plan.rightKey];
    const leftDictionary = leftSource.prepared.dictionaries[plan.leftKey];
    const rightDictionary = rightSource.prepared.dictionaries[plan.rightKey];
    if (composite) {
      for (const component of composite.components) {
        const leftVector = leftSource.columns[component.leftColumn];
        const rightVector = rightSource.columns[component.rightColumn];
        const leftLabels = leftSource.prepared.dictionaries[component.leftColumn];
        const rightLabels = rightSource.prepared.dictionaries[component.rightColumn];
        if (
          leftVector?.format !== component.leftFormat ||
          rightVector?.format !== component.rightFormat ||
          (component.kind === 'string'
            ? leftLabels?.values.length !== component.leftTable?.length ||
              rightLabels?.values.length !== component.rightTable?.length
            : Boolean(leftLabels || rightLabels))
        )
          throw new Error('Join key storage changed');
      }
    } else if (
      plan.keyKind === 'string'
        ? !leftDictionary ||
          !rightDictionary ||
          leftDictionary.ordered ||
          rightDictionary.ordered ||
          leftKeyVector?.format !== 'uint32' ||
          rightKeyVector?.format !== 'uint32'
        : !leftKeyVector || !rightKeyVector || leftDictionary || rightDictionary
    )
      throw new Error('Join key storage changed');
    const leftCount = left.rowCount;
    const rightCount = right.rowCount;
    const leftValidity = leftSource.analytics.validity[plan.leftKey];
    const rightValidity = rightSource.analytics.validity[plan.rightKey];
    // Row IDs are uint32 with one sentinel value.
    if (leftCount > MAXIMUM_RESIDENT_JOIN_ROW_INDEX || rightCount > MAXIMUM_RESIDENT_JOIN_ROW_INDEX)
      throw new Error(`Join input exceeds uint32 row indices.${RESIDENT_JOIN_SQL_GUIDANCE}`);
    assertBufferFits(device, Math.max(leftCount, rightCount, 4) * 4);

    // Phase A: key canonicalization, sort, probe, offsets; GPU-resident, one 16 byte readback.
    // A composite key is one numeric uint32 word per row by the time the key pass reads it.
    const keyKind = composite ? 'numeric' : plan.keyKind;
    const leftMode: ResidentJoinKeyMode = composite
      ? 'uint32'
      : keyKind === 'string'
        ? 'string'
        : (leftKeyVector.format as ResidentJoinColumnFormat);
    const rightMode: ResidentJoinKeyMode = composite
      ? 'uint32'
      : keyKind === 'string'
        ? 'string'
        : (rightKeyVector.format as ResidentJoinColumnFormat);
    const tables =
      keyKind === 'string'
        ? makeResidentJoinDictionaryKeyTables(leftDictionary.values, rightDictionary.values)
        : undefined;
    const tableBytes = ((tables?.left.length ?? 0) + (tables?.right.length ?? 0)) * 4;
    keyCpuReservation = session.registry.reserveMemory(
      tableBytes * 16 +
        65536 +
        (tables ? (leftDictionary.values.length + rightDictionary.values.length) * 64 : 0),
      'cpu',
      'scratch'
    );
    const compositeBytes = composite
      ? 4 * (leftCount + rightCount) * (2 * composite.components.length + 2) +
        composite.components.reduce(
          (bytes, component) =>
            bytes + 4 * ((component.leftTable?.length ?? 0) + (component.rightTable?.length ?? 0)),
          0
        )
      : 0;
    const ownedBytesA =
      compositeBytes +
      4 * (10 * leftCount + 19 * rightCount) +
      tableBytes +
      RESIDENT_JOIN_SUMMARY_WORDS * 4 +
      64;
    // Sort scratch is planned by the graph; admit a conservative estimate first and resize below.
    gpuReservation = session.registry.reserveMemory(
      ownedBytesA + 16 * (leftCount + rightCount) + 1048576,
      'gpu',
      'scratch'
    );
    const graphA = new GPUCommandGraph(device, {id: `resident-join-keys-${plan.operationId}`});
    const allocate = (name: string, wordCount: number, data?: Uint32Array, retained = false) => {
      const buffer = device.createBuffer({
        id: `join-${name}`,
        byteLength: Math.max(4, wordCount * 4),
        usage: STORAGE_USAGE,
        ...(data ? {data} : {})
      });
      outputs.push(buffer);
      if (retained) retainedOutputs.add(buffer);
      return buffer;
    };
    const makeView = (
      graph: GPUCommandGraph<undefined>,
      name: string,
      buffer: Buffer,
      length: number
    ): GraphDataView<'uint32'> =>
      graph.createDataView(
        graph.importBuffer({id: name, usage: buffer.usage, byteLength: buffer.byteLength}, buffer),
        {format: 'uint32', length}
      );
    const aBuffers = (name: string, length: number) => {
      const buffer = allocate(name, length);
      return {buffer, view: makeView(graphA, name, buffer, length)};
    };
    const leftWords = aBuffers('left-words', leftCount);
    const rightWords = aBuffers('right-words', rightCount);
    const leftValid = leftValidity || composite ? aBuffers('left-valid', leftCount) : undefined;
    const rightValid = rightValidity || composite ? aBuffers('right-valid', rightCount) : undefined;
    const makeTable = (name: string, table: Uint32Array) => {
      const buffer = allocate(name, table.length, table.length ? table : new Uint32Array([0]));
      return {buffer, view: makeView(graphA, name, buffer, Math.max(1, table.length))};
    };
    const leftTable = tables ? makeTable('left-table', tables.left) : undefined;
    const rightTable = tables ? makeTable('right-table', tables.right) : undefined;
    keyCpuReservation.release();
    keyCpuReservation = undefined;
    const leftKey = aBuffers('left-key', leftCount);
    const leftOk = aBuffers('left-ok', leftCount);
    const rightKey = aBuffers('right-key', rightCount);
    const rightOk = aBuffers('right-ok', rightCount);
    const rightNotOk = aBuffers('right-not-ok', rightCount);
    const rowIds = aBuffers('right-row-ids', rightCount);
    const keysByKey = aBuffers('right-keys-by-key', rightCount);
    const rowsByKey = aBuffers('right-rows-by-key', rightCount);
    const notOkByKey = aBuffers('right-not-ok-by-key', rightCount);
    const notOkSorted = aBuffers('right-not-ok-sorted', rightCount);
    const sortedRows = aBuffers('right-sorted-rows', rightCount);
    const sortedKeys = aBuffers('right-sorted-keys', rightCount);
    const validAt = aBuffers('right-valid-at', rightCount);
    const runStart = aBuffers('right-run-start', rightCount);
    const validIncluded = aBuffers('right-valid-included', rightCount);
    const runIncluded = aBuffers('right-run-included', rightCount);
    const matchedRun = aBuffers('right-matched-run', rightCount);
    const unmatched = aBuffers('right-unmatched', rightCount);
    const unmatchedIncluded = aBuffers('right-unmatched-included', rightCount);
    const lowerBound = aBuffers('left-lower-bound', leftCount);
    const matches = aBuffers('left-matches', leftCount);
    const emit = aBuffers('left-emit', leftCount);
    const leftIncludedLow = aBuffers('left-included-low', leftCount);
    const leftIncludedHigh = aBuffers('left-included-high', leftCount);
    // Never written: zero high words for the 64-bit emit scan (new buffers are zero-filled).
    const emitHigh = aBuffers('left-emit-high', leftCount);
    const summary = aBuffers('summary', RESIDENT_JOIN_SUMMARY_WORDS);

    const compositeCopies: {vector: GPUVector; buffer: Buffer; count: number}[] = [];
    if (composite) {
      for (const side of ['left', 'right'] as const) {
        const sideSource = side === 'left' ? leftSource : rightSource;
        const count = side === 'left' ? leftCount : rightCount;
        const keyWords = side === 'left' ? leftWords : rightWords;
        const valid = (side === 'left' ? leftValid : rightValid)!;
        const hashes = composite.exact ? undefined : aBuffers(`${side}-composite-hash`, count);
        composite.components.forEach((component, index) => {
          const column = side === 'left' ? component.leftColumn : component.rightColumn;
          const words = aBuffers(`${side}-c${index}-words`, count);
          compositeCopies.push({vector: sideSource.columns[column], buffer: words.buffer, count});
          const validityVector = sideSource.analytics.validity[column];
          const validity = validityVector ? aBuffers(`${side}-c${index}-valid`, count) : undefined;
          if (validityVector && validity)
            compositeCopies.push({vector: validityVector, buffer: validity.buffer, count});
          const sideTable = side === 'left' ? component.leftTable : component.rightTable;
          const table = sideTable ? makeTable(`${side}-c${index}-table`, sideTable) : undefined;
          addResidentJoinCompositeKeyPass(graphA, {
            id: `join-${side}-composite-${index}`,
            count,
            first: index === 0,
            component,
            side,
            words: words.view,
            validity: validity?.view,
            table: table?.view,
            keyWords: keyWords.view,
            hashes: hashes?.view,
            valid: valid.view,
            summary: summary.view
          });
        });
        if (hashes)
          addResidentJoinCompositeFinishPass(graphA, {
            id: `join-${side}-composite-finish`,
            count,
            plan: composite,
            keyWords: keyWords.view,
            hashes: hashes.view
          });
      }
    }
    addResidentJoinKeyPass(graphA, {
      id: 'join-left-keys',
      count: leftCount,
      mode: leftMode,
      dropHighKeys:
        keyKind === 'numeric' &&
        shouldDropResidentJoinHighKeys(
          leftMode as ResidentJoinColumnFormat,
          rightMode as ResidentJoinColumnFormat
        ),
      dictionarySize: leftDictionary?.values.length ?? 0,
      words: leftWords.view,
      validity: leftValid?.view,
      table: leftTable?.view,
      key: leftKey.view,
      ok: leftOk.view,
      summary: summary.view
    });
    addResidentJoinKeyPass(graphA, {
      id: 'join-right-keys',
      count: rightCount,
      mode: rightMode,
      dropHighKeys:
        keyKind === 'numeric' &&
        shouldDropResidentJoinHighKeys(
          rightMode as ResidentJoinColumnFormat,
          leftMode as ResidentJoinColumnFormat
        ),
      dictionarySize: rightDictionary?.values.length ?? 0,
      words: rightWords.view,
      validity: rightValid?.view,
      table: rightTable?.view,
      key: rightKey.view,
      ok: rightOk.view,
      notOk: rightNotOk.view,
      rowIds: rowIds.view,
      summary: summary.view
    });
    // Stable sort by key, then stable partition valid rows first (not-ok flag as a 1 bit key).
    graphA.add(
      new GPUSort({
        id: 'join-sort-right-keys',
        keys: rightKey.view,
        values: rowIds.view,
        outputKeys: keysByKey.view,
        outputValues: rowsByKey.view,
        keyBits:
          keyKind === 'string'
            ? Math.max(1, 32 - Math.clz32(Math.max(1, rightDictionary.values.length - 1)))
            : 32
      })
    );
    graphA.add(
      new GPUGather({
        id: 'join-gather-not-ok',
        source: rightNotOk.view,
        indices: rowsByKey.view,
        output: notOkByKey.view
      })
    );
    graphA.add(
      new GPUSort({
        id: 'join-partition-right-valid',
        keys: notOkByKey.view,
        values: rowsByKey.view,
        outputKeys: notOkSorted.view,
        outputValues: sortedRows.view,
        keyBits: 1
      })
    );
    addResidentJoinRunPass(graphA, {
      count: rightCount,
      sortedRows: sortedRows.view,
      rightKey: rightKey.view,
      rightOk: rightOk.view,
      sortedKeys: sortedKeys.view,
      validAt: validAt.view,
      runStart: runStart.view,
      matchedRun: matchedRun.view
    });
    graphA.add(
      new GPUScan({
        id: 'join-scan-valid',
        input: validAt.view,
        output: validIncluded.view,
        mode: 'inclusive'
      })
    );
    graphA.add(
      new GPUScan({
        id: 'join-scan-runs',
        input: runStart.view,
        output: runIncluded.view,
        mode: 'inclusive'
      })
    );
    addResidentJoinProbePass(graphA, {
      count: leftCount,
      rightCount,
      keepUnmatchedLeft: plan.joinType !== 'INNER',
      duplicateRightMatches: plan.duplicateRightMatches,
      leftKey: leftKey.view,
      leftOk: leftOk.view,
      sortedKeys: sortedKeys.view,
      validIncluded: validIncluded.view,
      runIncluded: runIncluded.view,
      lowerBound: lowerBound.view,
      matches: matches.view,
      emit: emit.view,
      matchedRun: matchedRun.view
    });
    addResidentJoinUnmatchedPass(graphA, {
      count: rightCount,
      emitUnmatchedRight: plan.joinType === 'FULL',
      sortedRows: sortedRows.view,
      validIncluded: validIncluded.view,
      runIncluded: runIncluded.view,
      matchedRun: matchedRun.view,
      unmatched: unmatched.view
    });
    // 64-bit emit scan: a many-to-many total can exceed uint32, which the summary reports.
    graphA.add(
      new GPUScanUint64({
        id: 'join-scan-emit',
        inputLow: emit.view,
        inputHigh: emitHigh.view,
        outputLow: leftIncludedLow.view,
        outputHigh: leftIncludedHigh.view
      })
    );
    graphA.add(
      new GPUScan({
        id: 'join-scan-unmatched',
        input: unmatched.view,
        output: unmatchedIncluded.view,
        mode: 'inclusive'
      })
    );
    addResidentJoinSummaryPass(graphA, {
      leftCount,
      rightCount,
      leftIncludedLow: leftIncludedLow.view,
      leftIncludedHigh: leftIncludedHigh.view,
      unmatchedIncluded: unmatchedIncluded.view,
      validIncluded: validIncluded.view,
      summary: summary.view
    });
    // Published luma.gl has no pre-compile resource plan: compile, then reserve the measured
    // transient bytes. The cleanup destroys the compiled graph if the reservation fails.
    compiledKeys = graphA.compile();
    const plannedBytesA =
      compiledKeys.stats.physicalTransientResourceBytes +
      outputs.reduce((bytes, buffer) => bytes + buffer.byteLength, 0);
    // resize enforces the registry budget; this is the documented memory bound, not a row cap.
    gpuReservation.resize(Math.max(gpuReservation.byteLength, plannedBytesA));
    assertCurrent();
    encoder = device.createCommandEncoder();
    if (composite) {
      for (const copy of compositeCopies)
        copyVectorToBuffer(encoder, copy.vector, copy.buffer, copy.count);
    } else {
      copyVectorToBuffer(encoder, leftKeyVector, leftWords.buffer, leftCount);
      copyVectorToBuffer(encoder, rightKeyVector, rightWords.buffer, rightCount);
      if (leftValidity && leftValid)
        copyVectorToBuffer(encoder, leftValidity, leftValid.buffer, leftCount);
      if (rightValidity && rightValid)
        copyVectorToBuffer(encoder, rightValidity, rightValid.buffer, rightCount);
    }
    compiledKeys.encode(encoder, {parameters: undefined});
    assertCurrent();
    submitted = true;
    device.submit(encoder.finish());
    encoder = undefined;
    let summaryWords: Uint32Array;
    try {
      summaryWords = unsignedValues((await readBuffers([summary.buffer], 16))[0]);
    } finally {
      compiledKeys.destroy();
      compiledKeys = undefined;
    }
    resultReadbackBytes += 16;
    assertCurrent();
    const [leftTotal, unmatchedTotal, status] = summaryWords;
    if (status & RESIDENT_JOIN_STATUS_FLOAT_KEY)
      throw new Error(
        `Join Float32 keys must be integers within +/-16777216 to match native string forms.${RESIDENT_JOIN_SQL_GUIDANCE}`
      );
    if (status & RESIDENT_JOIN_STATUS_DICTIONARY_CODE)
      throw new Error('Join key has an invalid dictionary code');
    const rowCount = leftTotal + unmatchedTotal;
    if (status & RESIDENT_JOIN_STATUS_OUTPUT_OVERFLOW || rowCount > MAXIMUM_RESIDENT_JOIN_ROW_INDEX)
      throw new Error(`Join output exceeds uint32 row indices.${RESIDENT_JOIN_SQL_GUIDANCE}`);
    // An empty result cannot be represented by one-batch adoption; refuse rather than approximate.
    if (!rowCount)
      throw new Error('Join produced no rows; resident results require at least one row');
    assertBufferFits(device, rowCount * 4);

    // Release key/sort scratch; the fill and gathers need only these Phase A results.
    const keptFromPhaseA = new Set<Buffer>([
      leftIncludedLow.buffer,
      emit.buffer,
      matches.buffer,
      lowerBound.buffer,
      sortedRows.buffer,
      unmatchedIncluded.buffer
    ]);
    for (let index = outputs.length - 1; index >= 0; index--) {
      if (keptFromPhaseA.has(outputs[index])) continue;
      outputs[index].destroy();
      outputs.splice(index, 1);
    }

    // Phase B: dedicated row mappings and outputs, GPU fill and gather.
    // A hashed composite key adds one 0/1 column: all numeric key parts equal in the joined row.
    const verifiedComponents =
      composite && !composite.exact
        ? composite.components.filter(component => component.kind === 'numeric')
        : [];
    const columnCount =
      plan.columns.length + (verifiedComponents.length ? 1 + 2 * verifiedComponents.length : 0);
    const indexBytes = rowCount * 16;
    const keptBytes = outputs.reduce((bytes, buffer) => bytes + buffer.byteLength, 0);
    const admittedBytesB = keptBytes + rowCount * columnCount * 16 + indexBytes + 65536;
    const residentSchema: KeplerResidentColumnSchema[] = plan.columns.map(column => ({
      name: column.outputColumn,
      format: column.format,
      ...(column.dictionary
        ? {
            dictionary: (column.side === 'left' ? left : right).value.prepared.dictionaries[
              column.sourceColumn
            ].values
          }
        : {})
    }));
    if (verifiedComponents.length)
      residentSchema.push({name: RESIDENT_JOIN_COMPOSITE_MATCH_COLUMN, format: 'uint32'});
    const readBack = shouldReadBackResidentResult(rowCount);
    // At or below the cap: readback words and Arrow copies. Above: a bounded lazy row cache.
    cpuReservation = session.registry.reserveMemory(
      readBack
        ? estimateResidentReadbackCpuByteLength(residentSchema, rowCount)
        : estimateKeplerResidentRowCacheByteLength(residentSchema, [rowCount]),
      'cpu',
      'readback'
    );
    // Outputs, indices and gather scratch.
    gpuReservation.resize(admittedBytesB);
    const graph = new GPUCommandGraph(device, {id: `resident-join-${plan.operationId}`});
    const view = <Format extends ResidentJoinColumnFormat>(
      name: string,
      buffer: Buffer,
      format: Format,
      length = rowCount
    ): GraphDataView<Format> =>
      graph.createDataView(
        graph.importBuffer({id: name, usage: buffer.usage, byteLength: buffer.byteLength}, buffer),
        {format, length}
      );
    const rowMappings = {
      left: allocate('indices-left', rowCount),
      right: allocate('indices-right', rowCount)
    };
    const sideValid = {
      left: allocate('valid-left', rowCount),
      right: allocate('valid-right', rowCount)
    };
    // One view per resource id: the graph rejects importing the same id twice.
    const indices = {
      left: view('indices-left', rowMappings.left, 'uint32'),
      right: view('indices-right', rowMappings.right, 'uint32')
    };
    addResidentJoinFillPass(graph, {
      count: rowCount,
      leftCount,
      rightCount,
      leftTotal,
      leftIncludedLow: view('left-included-low', leftIncludedLow.buffer, 'uint32', leftCount),
      emit: view('left-emit', emit.buffer, 'uint32', leftCount),
      matches: view('left-matches', matches.buffer, 'uint32', leftCount),
      lowerBound: view('left-lower-bound', lowerBound.buffer, 'uint32', leftCount),
      sortedRows: view('right-sorted-rows', sortedRows.buffer, 'uint32', rightCount),
      unmatchedIncluded: view(
        'right-unmatched-included',
        unmatchedIncluded.buffer,
        'uint32',
        rightCount
      ),
      leftRows: indices.left,
      rightRows: indices.right,
      leftValid: view('valid-left', sideValid.left, 'uint32'),
      rightValid: view('valid-right', sideValid.right, 'uint32')
    });
    const imports = new Map<string, GraphVectorView<ResidentJoinColumnFormat>>();
    const importVector = (
      key: string,
      vector: GPUVector | undefined,
      format: ResidentJoinColumnFormat
    ): GraphVectorView<ResidentJoinColumnFormat> => {
      if (!vector || vector.format !== format)
        throw new Error('Join requires prepared source storage matching its plan');
      let imported = imports.get(key);
      if (!imported) {
        imported = graph.importGPUVector(
          `join-source-${imports.size}`,
          vector as GPUVector<ResidentJoinColumnFormat>
        );
        imports.set(key, imported);
      }
      return imported;
    };
    const columns = plan.columns.map((column, index) => {
      const source = column.side === 'left' ? left : right;
      const values = allocate(`values-${index}`, rowCount, undefined, true);
      const hasValidity = Boolean(source.value.analytics.validity[column.sourceColumn]);
      const validity = allocate(`validity-${index}`, rowCount, undefined, true);
      graph.add(
        new GPUGather({
          id: `join-values-${index}`,
          source: importVector(
            `${column.side}:${column.sourceColumn}`,
            source.value.columns[column.sourceColumn],
            column.format
          ),
          indices: indices[column.side],
          output: view(`join-output-values-${index}`, values, column.format)
        })
      );
      if (hasValidity)
        // Out-of-range rows gather zero, which is exactly invalid.
        graph.add(
          new GPUGather({
            id: `join-validity-${index}`,
            source: importVector(
              `${column.side}:${column.sourceColumn}:validity`,
              source.value.analytics.validity[column.sourceColumn],
              'uint32'
            ),
            indices: indices[column.side],
            output: view(`join-output-validity-${index}`, validity, 'uint32')
          })
        );
      return {column, values, validity, hasValidity};
    });
    let matchColumn: {values: Buffer; validity: Buffer} | undefined;
    if (verifiedComponents.length) {
      const values = allocate('composite-match', rowCount, undefined, true);
      const validity = allocate('composite-match-validity', rowCount, undefined, true);
      matchColumn = {values, validity};
      const importWhole = (name: string, buffer: Buffer) =>
        graph.importBuffer({id: name, usage: buffer.usage, byteLength: buffer.byteLength}, buffer);
      const matchHandle = importWhole('composite-match', values);
      verifiedComponents.forEach((component, index) => {
        const handles = {
          left: importWhole(`verify-left-${index}`, allocate(`verify-left-${index}`, rowCount)),
          right: importWhole(`verify-right-${index}`, allocate(`verify-right-${index}`, rowCount))
        };
        for (const side of ['left', 'right'] as const) {
          const format = side === 'left' ? component.leftFormat : component.rightFormat;
          const column = side === 'left' ? component.leftColumn : component.rightColumn;
          graph.add(
            new GPUGather({
              id: `join-verify-${side}-${index}`,
              source: importVector(
                `${side}:${column}`,
                (side === 'left' ? left : right).value.columns[column],
                format
              ),
              indices: indices[side],
              output: graph.createDataView(handles[side], {format, length: rowCount})
            })
          );
        }
        addResidentJoinPass(graph, {
          id: `join-verify-${index}`,
          count: rowCount,
          bindings: {
            leftValues: {
              view: graph.createDataView(handles.left, {format: 'uint32', length: rowCount}),
              access: 'read'
            },
            rightValues: {
              view: graph.createDataView(handles.right, {format: 'uint32', length: rowCount}),
              access: 'read'
            },
            matches: {
              view: graph.createDataView(matchHandle, {format: 'uint32', length: rowCount}),
              access: 'read-write'
            }
          },
          body: `  matches[index] = ${index === 0 ? '' : 'matches[index] * '}select(0u, 1u, leftValues[index] == rightValues[index]);`
        });
      });
    }
    const ownedBytes = outputs.reduce((bytes, buffer) => bytes + buffer.byteLength, 0);
    // Published luma.gl has no pre-compile resource plan: compile, then admit the measured bytes.
    compiled = graph.compile();
    const plannedBytes = compiled.stats.physicalTransientResourceBytes + ownedBytes;
    if (plannedBytes > admittedBytesB)
      throw new KeplerAllocationError('memory-budget', {
        message: 'Join exceeds admitted GPU working memory'
      });
    gpuReservation.resize(plannedBytes);
    assertCurrent();
    encoder = device.createCommandEncoder();
    compiled.encode(encoder, {parameters: undefined});
    // Columns whose source has no validity are valid exactly where their side has a row.
    for (const {column, validity, hasValidity} of columns)
      if (!hasValidity)
        encoder.copyBufferToBuffer({
          sourceBuffer: sideValid[column.side],
          destinationBuffer: validity,
          size: rowCount * 4
        });
    if (matchColumn)
      encoder.copyBufferToBuffer({
        sourceBuffer: sideValid.left,
        destinationBuffer: matchColumn.validity,
        size: rowCount * 4
      });
    assertCurrent();
    submitted = true;
    device.submit(encoder.finish());
    encoder = undefined;
    fence = device.createFence();
    const gpuColumns: Record<string, GPUVector<ResidentJoinColumnFormat>> = Object.create(null);
    const gpuValidity: Record<string, GPUVector<'uint32'>> = Object.create(null);
    for (const {column, values, validity} of columns) {
      const name = column.outputColumn;
      gpuColumns[name] = outputVector(name, values, column.format, rowCount);
      gpuValidity[name] = outputVector(`${name}-validity`, validity, 'uint32', rowCount);
    }
    if (matchColumn) {
      gpuColumns[RESIDENT_JOIN_COMPOSITE_MATCH_COLUMN] = outputVector(
        RESIDENT_JOIN_COMPOSITE_MATCH_COLUMN,
        matchColumn.values,
        'uint32',
        rowCount
      );
      gpuValidity[RESIDENT_JOIN_COMPOSITE_MATCH_COLUMN] = outputVector(
        `${RESIDENT_JOIN_COMPOSITE_MATCH_COLUMN}-validity`,
        matchColumn.validity,
        'uint32',
        rowCount
      );
    }
    await Promise.race([fence.signaled, device.lost]);
    assertCurrent();
    // Retain only the immutable dataset allocation; compiled scratch and indices are finished.
    compiled.destroy();
    compiled = undefined;
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
        metadata: {'luma.gl.join': JSON.stringify(plan.provenance)}
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
          metadata: {join: plan.provenance, derivedDataset: plan.derivedDataset},
          destroy: () => {
            for (const buffer of retainedOutputs) buffer.destroy();
          }
        }) ?? undefined;
      if (!adopted) throw new Error('Join result was superseded before publication');
    } else {
      adopted =
        session.adoptResidentDataset({
          id: plan.resultId,
          schema: residentSchema,
          columns: gpuColumns,
          validity: gpuValidity,
          schemaMetadata: {'luma.gl.join': JSON.stringify(plan.provenance)},
          request,
          cpuReservation,
          gpuReservation,
          destroy: () => {
            for (const buffer of retainedOutputs) buffer.destroy();
          }
        }) ?? undefined;
      if (!adopted) throw new Error('Join result was superseded before publication');
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
        if (!domains) throw new Error('Join result was superseded before publication');
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
      readbackBytes: resultReadbackBytes,
      isCurrent: isResultCurrent,
      transfer() {
        if (transferred) return;
        if (!isResultCurrent()) throw new Error('Join became stale before attachment committed');
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
      if (submitted && !fence && !device.isLost) fence = device.createFence();
      const completion = fence ? Promise.race([fence.signaled, device.lost]) : Promise.resolve();
      await Promise.allSettled([completion]);
    } finally {
      fence?.destroy();
      compiledKeys?.destroy();
      compiled?.destroy();
      for (const buffer of outputs) if (!adopted || !retainedOutputs.has(buffer)) buffer.destroy();
      keyCpuReservation?.release();
      readbackReservation?.release();
      gpuReservation?.release();
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
/** GPU-to-GPU copy of every chunk of a 4 byte scalar vector into one packed buffer, in row order. */
function copyVectorToBuffer(
  encoder: CommandEncoder,
  vector: GPUVector,
  destination: Buffer,
  rowCount: number
): void {
  let rowOffset = 0;
  for (const chunk of vector.data) {
    if (!chunk.length) continue;
    encoder.copyBufferToBuffer({
      sourceBuffer: chunk.buffer instanceof Buffer ? chunk.buffer : chunk.buffer.buffer,
      sourceOffset: chunk.byteOffset,
      destinationBuffer: destination,
      destinationOffset: rowOffset * 4,
      size: chunk.length * 4
    });
    rowOffset += chunk.length;
  }
  if (rowOffset !== rowCount) throw new Error('Join key storage does not match its row count');
}
/** Per-buffer device limits are the only size bound besides the memory reservation. */
function assertBufferFits(device: Device, byteLength: number): void {
  const {maxBufferSize, maxStorageBufferBindingSize} = device.limits;
  if (byteLength > Math.min(maxBufferSize, maxStorageBufferBindingSize))
    throw new KeplerAllocationError('capacity', {
      message: `Join buffers exceed the device storage limit.${RESIDENT_JOIN_SQL_GUIDANCE}`
    });
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
