// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import type {
  GPUComputationRecorder,
  GPUComputationGraph
} from '@luma-contributors/analysis/computation-record';

/** Version the capture configuration when its inputs, parameters or rendering change. */
export const WORKFLOW_CAPTURE_ID = 'drainage-grand-canyon-v2';
export const WORKFLOW_CAPTURE_COLUMNS = 96;
export const WORKFLOW_CAPTURE_ROWS = 60;
export const WORKFLOW_CAPTURE_PARAMETERS = {
  streamThreshold: 32,
  fillEpsilon: 0.02,
  maximumSteps: 512,
  fillDepressions: true
} as const;

export type WorkflowViewIdentity = {
  resourceId: string;
  format: string;
  length: number;
  byteOffset: number;
  byteStride: number;
  rowByteLength: number;
};
export type WorkflowSnapshot = {
  source: WorkflowViewIdentity;
  snapshot: WorkflowViewIdentity;
  copyNodeId: string;
  /** Exact bytes copied from the chain output, retained for independent verification. */
  bytes: number[];
  sha256: string;
};
export type WorkflowPicture = {
  runId: string;
  graphId: string;
  stageId: string;
  stepIndex: number;
  entryId: string;
  theme: 'dark' | 'light';
  path: string;
  sha256: string;
  /** Ordered snapshot resource identities actually passed to the renderer. */
  sources: WorkflowViewIdentity[];
  sourceHashes: string[];
};
export type WorkflowIntermediateRun = {
  runId: string;
  captureId: string;
  templateId: string;
  submissionCount: number;
  input: {source: string; columns: number; rows: number; cellSize: number; sha256: string};
  parameters: typeof WORKFLOW_CAPTURE_PARAMETERS;
  graph: GPUComputationGraph;
  stages: {id: string; stepIndex: number; entryId: string; snapshots: WorkflowSnapshot[]}[];
  pictures: WorkflowPicture[];
  status: {stageId: string; name: string; value: number; source: WorkflowViewIdentity}[];
};
export type WorkflowPictureManifest = {
  schemaVersion: number;
  runs: (Pick<WorkflowIntermediateRun, 'runId' | 'templateId' | 'pictures'> & {
    recordPath: string;
    recordSha256: string;
  })[];
};

export function identifyWorkflowView(view: GraphDataView): WorkflowViewIdentity {
  return {
    resourceId: view.buffer.id,
    format: view.format,
    length: view.length,
    byteOffset: view.byteOffset,
    byteStride: view.byteStride,
    rowByteLength: view.rowByteLength
  };
}

/** Build the real recipe once; every stage refers to the recipe's returned views by identity. */
export async function buildWorkflowIntermediateChain(
  graph: GPUCommandGraph<void>,
  elevation: GraphDataView<'float32'>,
  parameters: GraphDataView<'float32'>,
  cellSize: number
) {
  const [{addDrainageNetworkRecipe}, {makeGPURasterMetadata}, {GPUComputationRecorder}] =
    await Promise.all([
      import('@luma-contributors/analysis/gpu-spatial-analysis'),
      import('@luma-contributors/analysis/raster-analysis'),
      import('@luma-contributors/analysis/computation-record')
    ]);
  const recorder = new GPUComputationRecorder();
  recorder.recordKernels(graph);
  const grid = makeGPURasterMetadata({
    width: WORKFLOW_CAPTURE_COLUMNS,
    height: WORKFLOW_CAPTURE_ROWS,
    cellSize,
    origin: [0, WORKFLOW_CAPTURE_ROWS * cellSize]
  });
  const recipe = addDrainageNetworkRecipe(graph, {
    id: WORKFLOW_CAPTURE_ID,
    flow: {
      grid,
      elevation: {id: 'elevation', format: 'float32', storage: {kind: 'buffer', values: elevation}},
      parameters,
      fillDepressions: WORKFLOW_CAPTURE_PARAMETERS.fillDepressions
    },
    maximumSteps: WORKFLOW_CAPTURE_PARAMETERS.maximumSteps
  });
  const outputs = recipe.outputs;
  const stages = [
    {id: 'accumulation', entryId: 'terrain-flow', views: [outputs.accumulation]},
    {id: 'stream-order', entryId: 'terrain-stream-order', views: [outputs.streamOrder]},
    {id: 'catchments', entryId: 'terrain-watersheds', views: [outputs.catchments]},
    {
      id: 'downstream-distance',
      entryId: 'terrain-flow-distance',
      views: [outputs.distance.distance]
    },
    {
      id: 'stream-network',
      entryId: 'terrain-stream-network',
      views: [outputs.network.positions, outputs.network.vertexOffsets, outputs.network.reachCount]
    },
    {
      id: 'catchment-summary',
      entryId: 'group-statistics',
      views: [outputs.summary.meanDownstreamDistance, outputs.summary.count]
    }
  ];
  return {recorder, grid, recipe, stages};
}

/** Selection is scoped to a workflow and its step, never to an operation poster. */
export function findWorkflowStepPictures(
  manifest: WorkflowPictureManifest,
  templateId: string,
  stepIndex: number,
  entryId?: string
): WorkflowPicture[] {
  const run = manifest.runs.find(candidate => candidate.templateId === templateId);
  return (
    run?.pictures.filter(
      picture => picture.stepIndex === stepIndex && picture.entryId === entryId
    ) ?? []
  );
}

/** Storage-only recipe transients are copied as uint32 words, preserving float/vector bits. */
export async function addWorkflowSnapshotCopy(
  graph: GPUCommandGraph<void>,
  source: GraphDataView,
  snapshot: GraphDataView,
  recorder: GPUComputationRecorder
): Promise<string> {
  const {GPUElementwise} = await import('@luma.gl/graph/gpu-core');
  const copyNodeId = `${snapshot.buffer.id}-copy`;
  const createWordView = (view: GraphDataView) =>
    graph.createDataView(view.buffer, {
      format: 'uint32',
      length: (view.length * view.rowByteLength) / 4,
      byteOffset: view.byteOffset
    });
  const copy = new GPUElementwise({
    id: copyNodeId,
    operation: 'copy',
    input: createWordView(source),
    output: createWordView(snapshot)
  });
  for (const node of copy.getCommandNodes(graph)) recorder.addNode(graph, node);
  return copyNodeId;
}

async function hashWorkflowBytes(bytes: Uint8Array): Promise<string> {
  const hash = await crypto.subtle.digest('SHA-256', Uint8Array.from(bytes).buffer);
  return Array.from(new Uint8Array(hash), value => value.toString(16).padStart(2, '0')).join('');
}

/**
 * Browser capture entry point. One submission produces every snapshot; changing the theme never
 * reruns the recipe. Rendering consumes only these exact readbacks, including compact counts.
 */
export async function captureWorkflowIntermediates(canvas: HTMLCanvasElement) {
  const [{createGpuDevice}, kit, {GPUCommandGraph}, {sampleElevation}, terrain, recording] =
    await Promise.all([
      import('../device'),
      import('../demos/kit'),
      import('@luma.gl/graph/gpu-core'),
      import('../data/elevation'),
      import('@luma-contributors/analysis/gpu-terrain'),
      import('@luma-contributors/analysis/computation-record')
    ]);
  const device = await createGpuDevice(canvas);
  if (!device || device.type !== 'webgpu') {
    device?.destroy();
    throw new Error('workflow-intermediates requires a WebGPU device');
  }
  const errors: string[] = [];
  const nativeDevice = (device as unknown as {handle: GPUDevice}).handle;
  nativeDevice.addEventListener('uncapturederror', event => {
    errors.push((event as GPUUncapturedErrorEvent).error.message);
  });
  const buffers: Buffer[] = [];
  let compiled: Awaited<ReturnType<GPUCommandGraph['compileAsync']>> | undefined;
  try {
    const graph = new GPUCommandGraph<void>(device, {id: WORKFLOW_CAPTURE_ID});
    const sampled = sampleElevation(WORKFLOW_CAPTURE_COLUMNS, WORKFLOW_CAPTURE_ROWS);
    const elevationBuffer = kit.createStorageBuffer(device, sampled.elevation);
    const parameterBuffer = kit.createEmptyStorageBuffer(device, 8);
    buffers.push(elevationBuffer, parameterBuffer);
    const elevation = kit.importBufferView(
      graph,
      'elevation',
      elevationBuffer,
      'float32',
      sampled.elevation.length
    );
    const parameters = kit.importBufferView(graph, 'parameters', parameterBuffer, 'float32', 8);
    const chain = await buildWorkflowIntermediateChain(
      graph,
      elevation,
      parameters,
      sampled.cellSize
    );
    parameterBuffer.write(
      terrain.getGPUTerrainFlowParameterValues(WORKFLOW_CAPTURE_PARAMETERS, chain.grid)
    );
    const stages = [];
    for (const [stepIndex, stage] of chain.stages.entries()) {
      const copies = [];
      for (const [sourceIndex, source] of stage.views.entries()) {
        const snapshotId = `picture-${stage.id}-${sourceIndex}`;
        const byteLength = source.length * source.rowByteLength;
        const buffer = kit.createEmptyStorageBuffer(device, byteLength / 4);
        buffers.push(buffer);
        const snapshot = kit.importBufferView(
          graph,
          snapshotId,
          buffer,
          source.format,
          source.length
        );
        const copyNodeId = await addWorkflowSnapshotCopy(graph, source, snapshot, chain.recorder);
        copies.push({source, snapshot, copyNodeId, buffer});
      }
      stages.push({...stage, stepIndex, copies});
    }
    const statusCopies = [];
    for (const stage of chain.recipe.status.stages) {
      for (const [name, source] of Object.entries(stage.status)) {
        if (!source) continue;
        const buffer = kit.createEmptyStorageBuffer(device, 1);
        buffers.push(buffer);
        const snapshot = kit.importBufferView(
          graph,
          `status-${stage.stage}-${name}`,
          buffer,
          'uint32',
          1
        );
        await addWorkflowSnapshotCopy(graph, source, snapshot, chain.recorder);
        statusCopies.push({stageId: stage.stage, name, source, buffer});
      }
    }
    compiled = await graph.compileAsync();
    const encoder = device.createCommandEncoder({id: 'workflow-intermediate-capture'});
    compiled.encode(encoder, {parameters: undefined});
    device.submit(encoder.finish());
    const runId = `${WORKFLOW_CAPTURE_ID}-${crypto.randomUUID()}`;
    const recordedStages = [];
    for (const stage of stages) {
      const snapshots: WorkflowSnapshot[] = [];
      for (const copy of stage.copies) {
        const bytes = new Uint8Array(await copy.buffer.readAsync());
        snapshots.push({
          source: identifyWorkflowView(copy.source),
          snapshot: identifyWorkflowView(copy.snapshot),
          copyNodeId: copy.copyNodeId,
          bytes: Array.from(bytes),
          sha256: await hashWorkflowBytes(bytes)
        });
      }
      recordedStages.push({
        id: stage.id,
        entryId: stage.entryId,
        stepIndex: stage.stepIndex,
        snapshots
      });
    }
    const status = [];
    for (const copy of statusCopies) {
      const value = new Uint32Array(await copy.buffer.readAsync())[0];
      status.push({
        stageId: copy.stageId,
        name: copy.name,
        value,
        source: identifyWorkflowView(copy.source)
      });
      if (
        (copy.name === 'converged' && value !== 1) ||
        ((copy.name === 'overflow' || copy.name === 'uncertainCount') && value !== 0)
      ) {
        throw new Error(
          `workflow-intermediates ${copy.stageId}.${copy.name} = ${value}; capture requires converged, complete outputs`
        );
      }
    }
    const run: WorkflowIntermediateRun = {
      runId,
      captureId: WORKFLOW_CAPTURE_ID,
      templateId: 'drainage-network',
      submissionCount: 1,
      input: {
        source:
          'USGS 3DEP Grand Canyon crop; website/src/gpu/data/elevation.ts (THIRD-PARTY-DATA.md)',
        columns: WORKFLOW_CAPTURE_COLUMNS,
        rows: WORKFLOW_CAPTURE_ROWS,
        cellSize: sampled.cellSize,
        sha256: await hashWorkflowBytes(
          new Uint8Array(
            sampled.elevation.buffer,
            sampled.elevation.byteOffset,
            sampled.elevation.byteLength
          )
        )
      },
      parameters: WORKFLOW_CAPTURE_PARAMETERS,
      graph: recording.describeGPUCommandGraph(compiled, {recorder: chain.recorder}).graph,
      stages: recordedStages,
      status,
      pictures: []
    };
    const images: {path: string; dataUrl: string}[] = [];
    for (const stage of run.stages) {
      for (const theme of ['dark', 'light'] as const) {
        const pictureCanvas = document.createElement('canvas');
        pictureCanvas.width = 640;
        pictureCanvas.height = 360;
        const pixels = renderWorkflowStagePixels(stage, theme, sampled.cellSize);
        pictureCanvas
          .getContext('2d')!
          .putImageData(new ImageData(pixels.data, pixels.width, pixels.height), 0, 0);
        const dataUrl = pictureCanvas.toDataURL('image/png');
        const bytes = new Uint8Array(await (await fetch(dataUrl)).arrayBuffer());
        const path = `workflow-intermediates/${runId}/${stage.id}-${theme}.png`;
        run.pictures.push({
          runId,
          graphId: run.graph.id,
          stageId: stage.id,
          stepIndex: stage.stepIndex,
          entryId: stage.entryId,
          theme,
          path,
          sha256: await hashWorkflowBytes(bytes),
          sources: stage.snapshots.map(snapshot => snapshot.snapshot),
          sourceHashes: stage.snapshots.map(snapshot => snapshot.sha256)
        });
        images.push({path, dataUrl});
      }
    }
    await nativeDevice.queue.onSubmittedWorkDone();
    if (errors.length) throw new Error(`workflow-intermediates GPU errors: ${errors.join('\n')}`);
    return {run, images};
  } finally {
    compiled?.destroy();
    for (const buffer of buffers) buffer.destroy();
    device.destroy();
  }
}

export type WorkflowStagePixels = {
  width: number;
  height: number;
  data: Uint8ClampedArray<ArrayBuffer>;
};

/**
 * Deterministic, lossless stage rendering. Both capture and Node replay consume these recorded
 * bytes; no browser fonts, external posters or platform-dependent canvas strokes participate.
 */
export function renderWorkflowStagePixels(
  stage: WorkflowIntermediateRun['stages'][number],
  theme: 'dark' | 'light',
  cellSize: number
): WorkflowStagePixels {
  const width = 640;
  const height = 360;
  const data = new Uint8ClampedArray(width * height * 4);
  const dark = theme === 'dark';
  const background = dark ? [12, 19, 35] : [246, 244, 239];
  const ink = dark ? [113, 216, 255] : [18, 103, 143];
  const low = dark ? [24, 45, 80] : [220, 232, 247];
  const high = dark ? [255, 211, 99] : [20, 81, 123];
  const categories = dark
    ? [
        [113, 216, 255],
        [255, 211, 99],
        [186, 137, 255],
        [105, 219, 151]
      ]
    : [
        [18, 103, 143],
        [163, 99, 0],
        [117, 60, 172],
        [16, 122, 68]
      ];
  const writePixel = (column: number, row: number, color: number[]) => {
    const offset = 4 * (row * width + column);
    data.set([...color, 255], offset);
  };
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) writePixel(column, row, background);
  }
  const arrays = stage.snapshots.map(({bytes, snapshot}) => {
    if (bytes.length !== snapshot.length * snapshot.rowByteLength) {
      throw new Error(`${stage.id} snapshot ${snapshot.resourceId} byte length mismatch`);
    }
    const buffer = Uint8Array.from(bytes).buffer;
    return snapshot.format.startsWith('float32')
      ? new Float32Array(buffer)
      : new Uint32Array(buffer);
  });
  const values = arrays[0];
  const left = 20;
  const top = 20;
  const plotWidth = 600;
  const plotHeight = 300;
  if (stage.id === 'stream-network') {
    const offsets = arrays[1];
    const reachCount = arrays[2][0];
    if (reachCount < 1 || reachCount >= offsets.length) {
      throw new Error(`stream-network invalid reachCount ${reachCount}`);
    }
    const getVertex = (vertex: number) => {
      const coordinateX =
        left + (plotWidth * values[2 * vertex]) / (WORKFLOW_CAPTURE_COLUMNS * cellSize);
      const coordinateY =
        top + plotHeight * (1 - values[2 * vertex + 1] / (WORKFLOW_CAPTURE_ROWS * cellSize));
      if (
        !Number.isFinite(coordinateX) ||
        !Number.isFinite(coordinateY) ||
        coordinateX < left ||
        coordinateX >= left + plotWidth ||
        coordinateY < top ||
        coordinateY >= top + plotHeight
      ) {
        throw new Error(`stream-network invalid vertex ${vertex}`);
      }
      return [Math.round(coordinateX), Math.round(coordinateY)];
    };
    for (let reach = 0; reach < reachCount; reach++) {
      if (offsets[reach] > offsets[reach + 1] || offsets[reach + 1] * 2 > values.length) {
        throw new Error(`stream-network invalid vertex range for reach ${reach}`);
      }
      for (let vertex = offsets[reach] + 1; vertex < offsets[reach + 1]; vertex++) {
        const start = getVertex(vertex - 1);
        const end = getVertex(vertex);
        const steps = Math.max(Math.abs(end[0] - start[0]), Math.abs(end[1] - start[1]), 1);
        for (let step = 0; step <= steps; step++) {
          const column = Math.round(start[0] + ((end[0] - start[0]) * step) / steps);
          const row = Math.round(start[1] + ((end[1] - start[1]) * step) / steps);
          writePixel(column, row, ink);
          writePixel(column, row + 1, ink);
          writePixel(column + 1, row, ink);
        }
      }
    }
  } else if (stage.id === 'catchment-summary') {
    const count = arrays[1][0];
    if (count < 1 || count > values.length)
      throw new Error(`catchment-summary invalid count ${count}`);
    let maximum = 1;
    for (let index = 0; index < count; index++) {
      if (!Number.isFinite(values[index]) || values[index] < 0)
        throw new Error(`catchment-summary invalid mean ${index}`);
      maximum = Math.max(maximum, values[index]);
    }
    for (let row = 0; row < plotHeight; row++) {
      for (let column = 0; column < plotWidth; column++) {
        const index = Math.floor((column * count) / plotWidth);
        if (row >= plotHeight * (1 - values[index] / maximum))
          writePixel(left + column, top + row, ink);
      }
    }
  } else {
    if (!['accumulation', 'stream-order', 'catchments', 'downstream-distance'].includes(stage.id)) {
      throw new Error(`Unknown workflow stage ${stage.id}`);
    }
    if (values.length !== WORKFLOW_CAPTURE_ROWS * WORKFLOW_CAPTURE_COLUMNS)
      throw new Error(`${stage.id} invalid cell count`);
    let maximum = 1;
    for (const value of values) {
      if (!Number.isFinite(value) || value < 0)
        throw new Error(`${stage.id} invalid cell value ${value}`);
      maximum = Math.max(maximum, value);
    }
    for (let row = 0; row < plotHeight; row++) {
      for (let column = 0; column < plotWidth; column++) {
        const index =
          Math.floor((row * WORKFLOW_CAPTURE_ROWS) / plotHeight) * WORKFLOW_CAPTURE_COLUMNS +
          Math.floor((column * WORKFLOW_CAPTURE_COLUMNS) / plotWidth);
        const value = values[index];
        const fraction =
          stage.id === 'accumulation' ? Math.log1p(value) / Math.log1p(maximum) : value / maximum;
        const color =
          stage.id === 'catchments'
            ? categories[value % categories.length]
            : low.map((channel, colorIndex) =>
                Math.round(channel + (high[colorIndex] - channel) * fraction)
              );
        writePixel(left + column, top + row, color);
      }
    }
  }
  let visible = 0;
  for (let offset = 0; offset < data.length; offset += 4) {
    if (
      Math.abs(data[offset] - background[0]) +
        Math.abs(data[offset + 1] - background[1]) +
        Math.abs(data[offset + 2] - background[2]) >
      24
    )
      visible++;
  }
  if (visible < plotWidth * plotHeight * 0.002)
    throw new Error(`${stage.id} blank ${theme} picture`);
  return {width, height, data};
}

/** Decode the published image separately before calling: declarations or image hashes cannot pass this check. */
export function verifyWorkflowStagePixels(
  stage: WorkflowIntermediateRun['stages'][number],
  theme: 'dark' | 'light',
  cellSize: number,
  decoded: {width: number; height: number; data: ArrayLike<number>}
): void {
  const expected = renderWorkflowStagePixels(stage, theme, cellSize);
  if (
    decoded.width !== expected.width ||
    decoded.height !== expected.height ||
    decoded.data.length !== expected.data.length
  ) {
    throw new Error(`${stage.id} ${theme} picture dimensions differ from snapshot replay`);
  }
  for (let offset = 0; offset < expected.data.length; offset++) {
    if (decoded.data[offset] !== expected.data[offset]) {
      const pixel = Math.floor(offset / 4);
      throw new Error(
        `${stage.id} ${theme} picture differs from snapshot replay at ${pixel % expected.width},${Math.floor(pixel / expected.width)} channel ${offset % 4}: expected ${expected.data[offset]}, received ${decoded.data[offset]}`
      );
    }
  }
}
