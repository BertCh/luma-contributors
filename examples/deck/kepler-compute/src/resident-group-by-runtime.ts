// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type CommandEncoder, type Fence} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {GPUCommandGraph, type CompiledGPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {prepareArrowTableForGPUAnalytics} from '@luma.gl/graph/arrow';
import {
  addKeplerGroupBy,
  getKeplerGroupByMemoryEstimate,
  mergeKeplerGroupByPasses,
  parseKeplerGroupBySettings,
  planKeplerGroupByPasses,
  type KeplerGroupByPass,
  type KeplerGroupByPassResult,
  type KeplerArrowDataset,
  type KeplerComputeSession,
  type KeplerDatasetReference,
  type KeplerDatasetSnapshot,
  type KeplerMemoryReservation
} from '@luma-contributors/kepler';
import * as arrow from 'apache-arrow';
import {
  MAXIMUM_RESIDENT_GROUP_BY_AGGREGATIONS,
  MAXIMUM_RESIDENT_GROUP_BY_KEY_TEXT_BYTES,
  type ResidentGroupByPlan,
  type ResidentGroupByProvenance
} from './resident-group-by-plan';

export type ResidentGroupByResult = {
  readonly dataset: KeplerDatasetSnapshot<KeplerArrowDataset>;
  readonly provenance: ResidentGroupByProvenance;
  readonly readbackBytes: number;
  /** Includes source/request callback checks until the host commits attachment. */
  isCurrent(): boolean;
  /** Transfer result generation ownership only after successful native attachment. */
  transfer(): void;
  /** Roll back only this generation; never remove a newer result using the same ID. */
  destroy(): Promise<void>;
};

/** All-source aggregation. Only bounded result rows return to CPU; source buffers stay borrowed. */
export async function executeResidentGroupBy(props: {
  session: KeplerComputeSession;
  source: KeplerDatasetReference;
  plan: ResidentGroupByPlan;
  signal?: AbortSignal;
  isCurrent(): boolean;
}): Promise<ResidentGroupByResult> {
  const {session} = props;
  const sourceReference = {...props.source};
  const plan = structuredClone(props.plan);
  const settings = parseKeplerGroupBySettings(
    plan.settings,
    MAXIMUM_RESIDENT_GROUP_BY_AGGREGATIONS
  );
  if (
    plan.source.datasetId !== sourceReference.datasetId ||
    plan.source.revision !== sourceReference.revision ||
    plan.resultId === sourceReference.datasetId ||
    settings.keyColumn !== plan.key.sourceColumn ||
    plan.key.outputColumn !== settings.keyColumn ||
    plan.measures.length !== settings.aggregations.length ||
    plan.measures.some(
      (measure, index) =>
        measure.outputColumn !== settings.aggregations[index].name ||
        measure.sourceColumn !== settings.aggregations[index].column
    )
  )
    throw new Error('Group By plan does not match its source and output schema');
  const dataset = session.registry.getDataset<KeplerArrowDataset>(sourceReference.datasetId);
  if (
    !dataset ||
    dataset.revision !== sourceReference.revision ||
    session.deviceLost ||
    props.signal?.aborted ||
    !props.isCurrent()
  )
    throw new Error('Group By source or request is stale');
  if (
    session.registry.getDataset(plan.resultId) ||
    session.registry.getRecoverySources().some(source => source.id === plan.resultId)
  )
    throw new Error('Group By result ID is already occupied');
  const request = session.registry.beginRequest(`resident-group-by:${plan.resultId}`, [dataset.id]);
  const cancel = () => request.cancel();
  props.signal?.addEventListener('abort', cancel, {once: true});
  const device = session.device;
  const current = () =>
    !request.signal.aborted &&
    !props.signal?.aborted &&
    !session.deviceLost &&
    session.device === device &&
    session.registry.getDataset(dataset.id)?.revision === dataset.revision &&
    props.isCurrent();
  const assertCurrent = () => {
    if (!current()) throw new Error('Group By was cancelled or its source changed');
  };
  let gpuReservation: KeplerMemoryReservation | undefined;
  let readbackReservation: KeplerMemoryReservation | undefined;
  let cpuReservation: KeplerMemoryReservation | undefined;
  let compiled: CompiledGPUCommandGraph | undefined;
  let encoder: CommandEncoder | undefined;
  let fence: Fence | undefined;
  let submitted = false;
  let adopted: KeplerDatasetSnapshot<KeplerArrowDataset> | undefined;
  const outputs: Buffer[] = [];
  const retainedOutputs = new Set<Buffer>();
  let resultReadbackBytes = 0;
  try {
    assertCurrent();
    const source = dataset.value;
    const dictionary = source.prepared.dictionaries[settings.keyColumn];
    const key = columnWithFormat(source.columns[settings.keyColumn], 'uint32');
    if (!dictionary || dictionary.ordered || plan.keyCapacity !== dictionary.values.length + 1)
      throw new Error('Group By dictionary metadata changed');
    let keyTextBytes = 0;
    for (const label of dictionary.values) {
      keyTextBytes += label.length * 2;
      if (keyTextBytes > MAXIMUM_RESIDENT_GROUP_BY_KEY_TEXT_BYTES * 2)
        throw new Error('Group By dictionary text exceeds its bounded result budget');
    }
    const passPlan = planKeplerGroupByPasses({
      keySlotCount: plan.keyCapacity,
      measureCount: settings.aggregations.length,
      rowCount: dataset.rowCount
    });
    const multiPass = passPlan.passes.length > 1;
    const passSettings = (pass: KeplerGroupByPass) => ({
      ...settings,
      aggregations: settings.aggregations.slice(pass.measureStart, pass.measureEnd)
    });
    const getPassInputs = (pass: KeplerGroupByPass) => {
      const names = [
        ...new Set(
          passSettings(pass)
            .aggregations.filter(measure => measure.operation !== 'count')
            .map(measure => measure.column)
        )
      ];
      const columns = names.map(name => columnWithFormat(source.columns[name], 'float32'));
      const vectors = [
        key,
        source.analytics.validity[settings.keyColumn],
        ...names.flatMap((name, index) => [columns[index], source.analytics.validity[name]])
      ].filter(vector => vector !== undefined);
      return {names, columns, vectors};
    };
    const estimates = passPlan.passes.map(pass =>
      getKeplerGroupByMemoryEstimate({
        rowCount: dataset.rowCount,
        chunkCount: getPassInputs(pass).vectors.reduce(
          (total, vector) => total + vector.data.length,
          0
        ),
        dictionaryCount: pass.keyCapacity - 1,
        measureCount: pass.measureEnd - pass.measureStart
      })
    );
    const mergedBytes = multiPass
      ? plan.keyCapacity * (12 + settings.aggregations.length * 8) + 16
      : 0;
    // Passes run one after another, so scratch is reserved for the largest pass only.
    const passBytes = Math.max(...estimates.map(item => item.totalByteLength));
    const widestPassCapacity = Math.max(...passPlan.passes.map(pass => pass.keyCapacity));
    const widestBatch = Math.max(
      ...passPlan.passes.map(pass => pass.measureEnd - pass.measureStart)
    );
    gpuReservation = session.registry.reserveMemory(passBytes + mergedBytes, 'gpu', 'scratch');
    // readAsync creates one MAP_READ GPU buffer per simultaneous read; CPU copies are separate.
    // Status completes before the result batch starts, so reserve the larger bounded phase.
    readbackReservation = session.registry.reserveMemory(
      Math.max(16, widestPassCapacity * 4 * (3 + widestBatch * 2)),
      'gpu',
      'readback'
    );
    // Cover readbacks, Arrow numeric/mask copies, explicit dictionary preparation and bounded strings.
    const cpuBytes =
      plan.keyCapacity * (12 + settings.aggregations.length * 8) * 6 +
      keyTextBytes * 4 +
      plan.keyCapacity * 128 +
      65536;
    cpuReservation = session.registry.reserveMemory(cpuBytes, 'cpu', 'readback');
    const passResults: KeplerGroupByPassResult[] = [];
    let singlePassBuffers:
      | {
          keys: Buffer;
          keyValidity: Buffer;
          measures: {values: Buffer; validity: Buffer}[];
        }
      | undefined;
    for (const [passIndex, pass] of passPlan.passes.entries()) {
      assertCurrent();
      const batchSettings = passSettings(pass);
      const {names: numericNames, columns: numericColumns} = getPassInputs(pass);
      const estimate = estimates[passIndex];
      const passBuffers: Buffer[] = [];
      const allocate = (name: string, length: number, retained = false): Buffer => {
        const buffer = device.createBuffer({
          id: `group-by-${passIndex}-${name}`,
          byteLength: Math.max(4, length * 4),
          usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST | Buffer.VERTEX
        });
        outputs.push(buffer);
        passBuffers.push(buffer);
        if (retained && !multiPass) retainedOutputs.add(buffer);
        return buffer;
      };
      const graph = new GPUCommandGraph(device, {
        id: `resident-group-by-${plan.operationId}-${passIndex}`
      });
      const view = <Format extends 'uint32' | 'float32'>(
        name: string,
        buffer: Buffer,
        format: Format,
        length: number
      ): GraphDataView<Format> =>
        graph.createDataView(
          graph.importBuffer(
            {id: name, usage: buffer.usage, byteLength: buffer.byteLength},
            buffer
          ),
          {format, length}
        );
      const capacity = estimate.capacity;
      const keys = allocate('keys', capacity, true);
      const keyValidity = allocate('key-validity', capacity, true);
      const firstRows = allocate('first-source-rows', capacity);
      const status = allocate('status', 4);
      const measures = batchSettings.aggregations.map(measure => ({
        ...measure,
        values: allocate(measure.name, capacity, true),
        validity: allocate(`${measure.name}-validity`, capacity, true)
      }));
      const keyInput = graph.importGPUVector('group-key', key);
      const validityInput = source.analytics.validity[settings.keyColumn];
      addKeplerGroupBy(graph, {
        id: `kepler-group-by-${passIndex}`,
        source: sourceReference,
        settings: batchSettings,
        sourceFields: source.source.schema.fields.map(field => field.name),
        key: keyInput,
        keyValidity: validityInput
          ? graph.importGPUVector('group-key-validity', validityInput)
          : undefined,
        dictionaryCount: dictionary.values.length,
        keyWindow: {start: pass.keyStart, capacity: pass.keyCapacity},
        columns: Object.fromEntries(
          numericNames.map((name, index) => [
            name,
            {
              values: graph.importGPUVector(`group-input-${index}`, numericColumns[index]),
              ...(source.analytics.validity[name]
                ? {
                    validity: graph.importGPUVector(
                      `group-input-validity-${index}`,
                      source.analytics.validity[name]!
                    )
                  }
                : {})
            }
          ])
        ),
        outputs: {
          keys: view('group-output-keys', keys, 'uint32', capacity),
          keyValidity: view('group-output-key-validity', keyValidity, 'uint32', capacity),
          firstSourceRows: view('group-output-first-rows', firstRows, 'uint32', capacity),
          status: view('group-output-status', status, 'uint32', 4),
          measures: Object.fromEntries(
            measures.map((measure, index) => [
              measure.name,
              {
                values:
                  measure.operation === 'count'
                    ? view(`group-output-measure-${index}`, measure.values, 'uint32', capacity)
                    : view(`group-output-measure-${index}`, measure.values, 'float32', capacity),
                validity: view(
                  `group-output-measure-validity-${index}`,
                  measure.validity,
                  'uint32',
                  capacity
                )
              }
            ])
          )
        }
      });
      const ownedBytes = passBuffers.reduce((bytes, buffer) => bytes + buffer.byteLength, 0);
      const plannedBytes = graph.planResources().stats.physicalTransientResourceBytes + ownedBytes;
      if (plannedBytes > estimate.totalByteLength)
        throw new Error('Group By exceeds admitted GPU working memory');
      if (!multiPass) gpuReservation.resize(plannedBytes);
      compiled = graph.compile();
      assertCurrent();
      encoder = device.createCommandEncoder();
      compiled.encode(encoder, {parameters: undefined});
      assertCurrent();
      submitted = true;
      device.submit(encoder.finish());
      encoder = undefined;
      fence = device.createFence();
      const statusBytes = await status.readAsync(0, 16);
      resultReadbackBytes += 16;
      assertCurrent();
      const diagnostics = unsignedValues(statusBytes);
      const passGroupCount = diagnostics[0];
      if (passGroupCount > capacity || diagnostics[1] || diagnostics[2] || diagnostics[3])
        throw new Error(
          'Group By has invalid dictionary codes or nonfinite numerical inputs/results; use SQL for unsupported values'
        );
      const passReadbacks = await readBuffers(
        [keys, firstRows, ...measures.flatMap(measure => [measure.values, measure.validity])],
        passGroupCount * 4
      );
      resultReadbackBytes += passGroupCount * 4 * passReadbacks.length;
      assertCurrent();
      passResults.push({
        pass,
        keys: unsignedValues(passReadbacks[0]).slice(),
        firstRows: unsignedValues(passReadbacks[1]).slice(),
        values: measures.map((measure, index) =>
          measure.operation === 'count'
            ? unsignedValues(passReadbacks[2 + index * 2]).slice()
            : floatValues(passReadbacks[2 + index * 2]).slice()
        ),
        validity: measures.map((_, index) => unsignedValues(passReadbacks[3 + index * 2]).slice())
      });
      if (!multiPass) {
        singlePassBuffers = {
          keys,
          keyValidity,
          measures: measures.map(measure => ({values: measure.values, validity: measure.validity}))
        };
      } else {
        await Promise.race([fence.signaled, device.lost]);
        assertCurrent();
        compiled.destroy();
        compiled = undefined;
        fence.destroy();
        fence = undefined;
        for (const buffer of passBuffers) {
          buffer.destroy();
          outputs.splice(outputs.indexOf(buffer), 1);
        }
      }
    }
    // readBuffers settles every sibling read, including staging cleanup, before returning.
    readbackReservation.release();
    const merged = mergeKeplerGroupByPasses(
      passResults,
      settings.aggregations.length,
      settings.aggregations.map(measure => measure.operation === 'count')
    );
    const groupCount = merged.groupCount;
    const keyCodes = merged.slots;
    const firstSourceRows = merged.firstRows;
    const keyValid = new Uint32Array(groupCount);
    for (let row = 0; row < groupCount; row++)
      keyValid[row] = keyCodes[row] < dictionary.values.length ? 1 : 0;
    const keyRows: (string | null)[] = [];
    const seenCodes = new Set<number>();
    for (let row = 0; row < groupCount; row++) {
      if (
        firstSourceRows[row] >= dataset.rowCount ||
        (row && firstSourceRows[row] <= firstSourceRows[row - 1]) ||
        seenCodes.has(keyCodes[row]) ||
        keyCodes[row] > dictionary.values.length
      )
        throw new Error('Group By key identity or first-occurrence order is invalid');
      seenCodes.add(keyCodes[row]);
      keyRows.push(keyValid[row] ? dictionary.values[keyCodes[row]] : null);
    }
    const arrowColumns: Record<string, arrow.Vector> = Object.create(null);
    arrowColumns[plan.key.outputColumn] = arrow.vectorFromArray(keyRows, new arrow.Utf8());
    const columns: Record<string, GPUVector<'float32' | 'uint32'>> = Object.create(null);
    const validity: Record<string, GPUVector<'uint32'>> = Object.create(null);
    // One pass adopts its GPU outputs directly; several passes upload the merged, ordered result.
    const upload = (name: string, data: Uint32Array | Float32Array): Buffer => {
      const buffer = device.createBuffer({
        id: `group-by-merged-${name}`,
        data: data.length ? data : new Uint32Array(1),
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST | Buffer.VERTEX
      });
      outputs.push(buffer);
      retainedOutputs.add(buffer);
      return buffer;
    };
    const keyBuffer = singlePassBuffers?.keys ?? upload('keys', keyCodes);
    const keyValidityBuffer = singlePassBuffers?.keyValidity ?? upload('key-validity', keyValid);
    columns[plan.key.outputColumn] = outputVector(
      plan.key.outputColumn,
      keyBuffer,
      'uint32',
      groupCount
    );
    validity[plan.key.outputColumn] = outputVector(
      `${plan.key.outputColumn}-validity`,
      keyValidityBuffer,
      'uint32',
      groupCount
    );
    for (const [index, aggregation] of settings.aggregations.entries()) {
      const isCount = aggregation.operation === 'count';
      const values = merged.values[index];
      const valid = merged.validity[index];
      const rows: (number | null)[] = [];
      for (let row = 0; row < groupCount; row++) {
        if (
          valid[row] > 1 ||
          (isCount && valid[row] !== 1) ||
          (valid[row] && !Number.isFinite(values[row]))
        )
          throw new Error('Group By output validity is inconsistent');
        rows.push(valid[row] ? values[row] : null);
      }
      const valueBuffer =
        singlePassBuffers?.measures[index].values ?? upload(aggregation.name, values);
      const validityBuffer =
        singlePassBuffers?.measures[index].validity ??
        upload(`${aggregation.name}-validity`, valid);
      if (isCount) {
        arrowColumns[aggregation.name] = arrow.vectorFromArray(rows, new arrow.Uint32());
        columns[aggregation.name] = outputVector(
          aggregation.name,
          valueBuffer,
          'uint32',
          groupCount
        );
      } else {
        arrowColumns[aggregation.name] = arrow.vectorFromArray(rows, new arrow.Float32());
        columns[aggregation.name] = outputVector(
          aggregation.name,
          valueBuffer,
          'float32',
          groupCount
        );
      }
      validity[aggregation.name] = outputVector(
        `${aggregation.name}-validity`,
        validityBuffer,
        'uint32',
        groupCount
      );
    }
    const table = new arrow.Table(arrowColumns);
    table.schema.metadata.set('luma.gl.groupBy', JSON.stringify(plan.provenance));
    const normalization = {dictionaries: {[plan.key.outputColumn]: dictionary}};
    const prepared = prepareArrowTableForGPUAnalytics(table, normalization);
    // Seeded preparation must preserve exact GPU codes, including unobserved dictionary labels.
    const preparedKeys = prepared.table.getChild(plan.key.outputColumn)!;
    const preparedData = preparedKeys.data[0];
    for (let row = 0; row < groupCount; row++) {
      if (
        keyValid[row]
          ? preparedData.values[preparedData.offset + row] !== keyCodes[row]
          : preparedKeys.get(row) !== null
      )
        throw new Error('Prepared result dictionary changed adopted GPU codes');
    }
    if (fence) await Promise.race([fence.signaled, device.lost]);
    assertCurrent();
    // Retain only the immutable dataset allocation; compiled scratch and diagnostics are finished.
    compiled?.destroy();
    compiled = undefined;
    for (let index = outputs.length - 1; index >= 0; index--) {
      if (retainedOutputs.has(outputs[index])) continue;
      outputs[index].destroy();
      outputs.splice(index, 1);
    }
    gpuReservation.resize(
      [...retainedOutputs].reduce((bytes, buffer) => bytes + buffer.byteLength, 0)
    );
    adopted =
      session.adoptArrowDataset({
        id: plan.resultId,
        table,
        prepared,
        columns,
        validity,
        normalization,
        request,
        cpuReservation,
        gpuReservation,
        metadata: {groupBy: plan.provenance, derivedDataset: plan.derivedDataset},
        destroy: () => {
          for (const buffer of retainedOutputs) buffer.destroy();
        }
      }) ?? undefined;
    if (!adopted) throw new Error('Group By result was superseded before publication');
    const snapshot = adopted;
    let transferred = false;
    let destruction: Promise<void> | undefined;
    const isResultCurrent = () =>
      !destruction &&
      !session.deviceLost &&
      session.device === device &&
      session.registry.getDataset(snapshot.id)?.revision === snapshot.revision &&
      (transferred ||
        (!props.signal?.aborted &&
          session.registry.getDataset(dataset.id)?.revision === dataset.revision &&
          props.isCurrent()));
    return {
      dataset: snapshot,
      provenance: plan.provenance,
      readbackBytes: resultReadbackBytes,
      isCurrent: isResultCurrent,
      transfer() {
        if (transferred) return;
        if (!isResultCurrent())
          throw new Error('Group By became stale before attachment committed');
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
      compiled?.destroy();
      for (const buffer of outputs) if (!adopted || !retainedOutputs.has(buffer)) buffer.destroy();
      gpuReservation?.release();
      readbackReservation?.release();
      cpuReservation?.release();
      await session.registry.finishRequest(request);
    }
  }
}

function outputVector<Format extends 'float32' | 'uint32'>(
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
function columnWithFormat<Format extends 'float32' | 'uint32'>(
  column: GPUVector | undefined,
  format: Format
): GPUVector<Format> {
  if (!column || column.format !== format)
    throw new Error(`Group By requires prepared ${format} source storage`);
  // Canonical format checked at the borrowed-source boundary; no conversion or upload.
  return column as GPUVector<Format>;
}
function unsignedValues(bytes: Uint8Array): Uint32Array {
  return new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}
function floatValues(bytes: Uint8Array): Float32Array {
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}
async function readBuffers(buffers: Buffer[], byteLength: number): Promise<Uint8Array[]> {
  if (!byteLength) return buffers.map(() => new Uint8Array());
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
