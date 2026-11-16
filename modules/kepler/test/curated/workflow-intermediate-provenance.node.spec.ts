// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {existsSync, readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import type {
  GPUComputationGraph,
  GPUComputationNode
} from '@luma-contributors/analysis/computation-record';
import {afterAll, beforeAll, describe, expect, it, vi} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../../analysis/src/testing/gpu-contributor-test-utils';
import {
  addWorkflowSnapshotCopy,
  buildWorkflowIntermediateChain,
  findWorkflowStepPictures,
  identifyWorkflowView,
  renderWorkflowStagePixels,
  verifyWorkflowStagePixels,
  WORKFLOW_CAPTURE_COLUMNS,
  WORKFLOW_CAPTURE_ID,
  WORKFLOW_CAPTURE_PARAMETERS,
  WORKFLOW_CAPTURE_ROWS,
  type WorkflowIntermediateRun,
  type WorkflowPictureManifest,
  type WorkflowStagePixels
} from '../../../../website/src/gpu/runs/workflow-intermediates';
import {WORKFLOWS} from '../../../../website/src/components/meta/pages/workflows/workflows';
import {createElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {ChainFigure} from '../../../../website/src/components/meta/pages/workflows/chain-figure';
import pictureManifest from '../../../../website/src/gpu/runs/workflow-intermediates.manifest.json';

// The website's catalog summaries are supplied by a build plugin, absent in the node project.
vi.mock('../../../../website/src/catalog/index', () => ({
  getEntry: (id: string) => ({id, title: id, symbol: id})
}));
vi.mock('../../../../website/src/components/catalog/entry-preview', () => ({
  EntryPreview: ({entry}: {entry: {id: string; title: string}}) =>
    createElement('img', {'data-operation-poster': entry.id, alt: entry.title})
}));
vi.mock('@tanstack/react-router', () => ({
  Link: ({children}: {children: import('react').ReactNode}) => createElement('a', {}, children)
}));

// pngjs is the existing Node PNG codec supplied with the browser-test tooling.
const {PNG} = createRequire(import.meta.url)('pngjs') as {
  PNG: {
    sync: {
      read(bytes: Uint8Array): {width: number; height: number; data: Uint8Array};
      write(image: {width: number; height: number; data: Uint8Array}): Uint8Array;
    };
  };
};
type ProducerIdentity = Pick<GPUComputationNode, 'id' | 'kind' | 'reads' | 'writes'>;
let recipeNodes: ProducerIdentity[] = [];

const publicDirectory = new URL('../../../../website/public/', import.meta.url);
const manifest = pictureManifest as WorkflowPictureManifest;
const hashBytes = (bytes: Uint8Array | string) => createHash('sha256').update(bytes).digest('hex');
const device = createNullWebGPUDevice();
let chain: Awaited<ReturnType<typeof buildWorkflowIntermediateChain>>;

beforeAll(async () => {
  const graph = new GPUCommandGraph(device, {id: WORKFLOW_CAPTURE_ID});
  const computePasses = vi.spyOn(graph, 'addComputePass');
  const copyPasses = vi.spyOn(graph, 'addCopyPass');
  const renderPasses = vi.spyOn(graph, 'addRenderPass');
  chain = await buildWorkflowIntermediateChain(
    graph,
    createTransientView(
      graph,
      'elevation',
      'float32',
      WORKFLOW_CAPTURE_COLUMNS * WORKFLOW_CAPTURE_ROWS
    ),
    createTransientView(graph, 'parameters', 'float32', 8),
    70
  );
  // Observe the public pass declarations emitted by the real recipe, without compiling a GPU.
  const declarations = [
    ...computePasses.mock.calls.map(([node]) => ({...node, kind: 'compute' as const})),
    ...copyPasses.mock.calls.map(([node]) => ({...node, kind: 'copy' as const})),
    ...renderPasses.mock.calls.map(([node]) => ({...node, kind: 'render' as const}))
  ];
  recipeNodes = declarations.map(node => {
    const reads: string[] = [];
    const writes: string[] = [];
    for (const use of node.resources ?? []) {
      if (!('buffer' in use)) continue;
      const resourceId = ('buffer' in use.buffer ? use.buffer.buffer : use.buffer).id;
      if (
        [
          'storage-read',
          'storage-read-write',
          'uniform',
          'copy-source',
          'indirect',
          'vertex',
          'index'
        ].includes(use.usage)
      )
        reads.push(resourceId);
      if (['storage-write', 'storage-read-write', 'copy-destination'].includes(use.usage))
        writes.push(resourceId);
    }
    return {id: node.id, kind: node.kind, reads: [...new Set(reads)], writes: [...new Set(writes)]};
  });
  computePasses.mockRestore();
  copyPasses.mockRestore();
  renderPasses.mockRestore();
}, 60000);
afterAll(() => device.destroy());

function verifyRecipeProducers(
  graph: GPUComputationGraph,
  resourceId: string,
  copyNodeId: string
): void {
  const identifyProducer = ({id, kind, reads, writes}: ProducerIdentity) => ({
    id,
    kind,
    reads,
    writes
  });
  const expected = recipeNodes.filter(node => node.writes.includes(resourceId));
  expect(expected.length, `${resourceId} real recipe producers`).toBeGreaterThan(0);
  const actual = graph.nodes.filter(node => node.writes.includes(resourceId));
  const compareProducerIds = (first: ProducerIdentity, second: ProducerIdentity) =>
    first.id.localeCompare(second.id);
  expect(
    actual.map(identifyProducer).sort(compareProducerIds),
    `${resourceId} real recipe producer identities`
  ).toEqual([...expected].sort(compareProducerIds));
  const copyIndex = graph.nodes.findIndex(node => node.id === copyNodeId);
  for (const producer of actual) expect(graph.nodes.indexOf(producer)).toBeLessThan(copyIndex);
  const lastProducer = actual[actual.length - 1];
  expect(
    graph.edges.filter(
      edge => edge.to === copyNodeId && edge.kind === 'raw' && edge.resource === resourceId
    )
  ).toEqual([{from: lastProducer.id, to: copyNodeId, kind: 'raw', resource: resourceId}]);
}

function createStageFixture(
  stepIndex: number,
  changed = false
): WorkflowIntermediateRun['stages'][number] {
  const stage = chain.stages[stepIndex];
  const snapshots = stage.views.map((view, sourceIndex) => {
    const values = view.format.startsWith('float32')
      ? new Float32Array((view.length * view.rowByteLength) / 4)
      : new Uint32Array((view.length * view.rowByteLength) / 4);
    if (stage.id === 'stream-network') {
      if (sourceIndex === 0) {
        // Horizontal and vertical reaches have distinguishable interior pixels, away from borders.
        values.set(
          changed
            ? [
                WORKFLOW_CAPTURE_COLUMNS / 2,
                WORKFLOW_CAPTURE_ROWS - 1,
                WORKFLOW_CAPTURE_COLUMNS / 2,
                1
              ]
            : [
                1,
                WORKFLOW_CAPTURE_ROWS / 2,
                WORKFLOW_CAPTURE_COLUMNS - 1,
                WORKFLOW_CAPTURE_ROWS / 2
              ]
        );
      } else if (sourceIndex === 1) values.set([0, 2]);
      else values[0] = 1;
    } else if (stage.id === 'catchment-summary') {
      if (sourceIndex === 0) values.set(changed ? [3, 1] : [1, 3]);
      else values[0] = 2;
    } else {
      values[0] = changed ? 4 : 0;
      values[1] = changed ? 0 : 4;
      if (stage.id === 'catchments') {
        values[0] = changed ? 1 : 0;
        values[1] = changed ? 0 : 1;
      }
    }
    const identity = identifyWorkflowView(view);
    const bytes = Array.from(new Uint8Array(values.buffer));
    return {
      source: identity,
      snapshot: {...identity, resourceId: `fixture-${sourceIndex}`},
      copyNodeId: 'fixture',
      bytes,
      sha256: hashBytes(Uint8Array.from(bytes))
    };
  });
  return {id: stage.id, entryId: stage.entryId, stepIndex, snapshots};
}

function readPixel(image: WorkflowStagePixels, column: number, row: number): number[] {
  return Array.from(
    image.data.slice(4 * (row * image.width + column), 4 * (row * image.width + column) + 4)
  );
}

describe('workflow intermediate picture provenance', () => {
  it('prefers a recorded stage over its operation poster while other stages retain their posters', () => {
    const workflow = WORKFLOWS.find(candidate => candidate.templateId === 'hot-spot-map')!;
    const stepIndex = workflow.steps.findIndex(step => step.entry);
    const entryId = workflow.steps[stepIndex].entry!;
    const pictures = (['dark', 'light'] as const).map(theme => ({
      runId: 'render-fixture',
      graphId: 'fixture-graph',
      stageId: 'fixture-stage',
      stepIndex,
      entryId,
      theme,
      path: `workflow-intermediates/render-fixture/stage-${theme}.png`,
      sha256: 'fixture',
      sources: [],
      sourceHashes: []
    }));
    const published = {
      runId: 'render-fixture',
      templateId: workflow.templateId,
      recordPath: 'fixture',
      recordSha256: 'fixture',
      pictures
    };
    manifest.runs.push(published);
    try {
      const markup = renderToStaticMarkup(
        createElement(ChainFigure, {
          steps: workflow.steps,
          label: 'Hot spot workflow'
        })
      );
      expect(markup.match(/data-run-id="render-fixture"/g)).toHaveLength(2);
      expect(markup.match(/data-operation-poster=/g)).toHaveLength(
        workflow.steps.filter(step => step.entry).length - 1
      );
    } finally {
      manifest.runs.splice(manifest.runs.indexOf(published), 1);
    }
  });

  it('renders operation posters for uncaptured contributor stages and identifies them as examples', () => {
    const workflow = WORKFLOWS.find(candidate => candidate.templateId === 'hot-spot-map')!;
    const markup = renderToStaticMarkup(
      createElement(ChainFigure, {
        steps: workflow.steps,
        label: 'Hot spot workflow'
      })
    );
    const contributorSteps = workflow.steps.filter(step => step.entry);
    expect(markup.match(/data-operation-poster=/g)).toHaveLength(contributorSteps.length);
    for (const step of contributorSteps)
      expect(markup).toContain(`data-operation-poster="${step.entry}"`);
    expect(markup).toContain('individual operation examples');
    expect(markup).toContain('WEB-HIW-03');
  });

  it('selects the returned views of a single real six-stage drainage recipe by object identity', () => {
    const outputs = chain.recipe.outputs;
    const expected = [
      [outputs.accumulation],
      [outputs.streamOrder],
      [outputs.catchments],
      [outputs.distance.distance],
      [outputs.network.positions, outputs.network.vertexOffsets, outputs.network.reachCount],
      [outputs.summary.meanDownstreamDistance, outputs.summary.count]
    ];
    expect(chain.stages).toHaveLength(expected.length);
    const workflow = WORKFLOWS.find(candidate => candidate.templateId === 'drainage-network')!;
    for (const [stepIndex, stage] of chain.stages.entries()) {
      expect(stage.entryId).toBe(workflow.steps[stepIndex].entry);
      expect(chain.recipe.status.stages.some(candidate => candidate.stage === stage.id)).toBe(true);
      expect(stage.views).toHaveLength(expected[stepIndex].length);
      stage.views.forEach((view, index) => {
        expect(view).toBe(expected[stepIndex][index]);
        expect(view.buffer.id).toMatch(new RegExp(`^${WORKFLOW_CAPTURE_ID}-`));
      });
    }
  });

  it('copies the exact packed words of vector outputs using the public primitive and storage usages', async () => {
    const graph = new GPUCommandGraph(device);
    const {GPUComputationRecorder} = await import('@luma-contributors/analysis/computation-record');
    const recorder = new GPUComputationRecorder();
    const recordedNode = vi.spyOn(recorder, 'recordNode');
    const source = createTransientView(graph, 'chain-positions', 'float32x2', 20);
    const snapshot = createTransientView(graph, 'picture-positions', 'float32x2', 20);
    const copyNodeId = await addWorkflowSnapshotCopy(graph, source, snapshot, recorder);
    expect(copyNodeId).toBe('picture-positions-copy');
    const node = recordedNode.mock.calls[0][0];
    expect(node.workload?.readByteLength).toBe(160);
    expect(node.workload?.writeByteLength).toBe(160);
    expect(node.resources?.map(use => ('buffer' in use ? use.buffer : undefined))).toMatchObject([
      {buffer: source.buffer, format: 'uint32', length: 40},
      {buffer: snapshot.buffer, format: 'uint32', length: 40}
    ]);
    expect(recorder.getResource(source.buffer.id)).toBe(source.buffer);
    expect(recorder.getResource(snapshot.buffer.id)).toBe(snapshot.buffer);
    expect(recorder.getNodeRecord(copyNodeId)).toBeDefined();
    // The node can be added with STORAGE-only source and destination; raw copy usages would throw.
    expect(source.buffer.byteLength).toBe(160);
    expect(snapshot.buffer.byteLength).toBe(160);
  });

  it('never substitutes an entry poster, a different workflow, or a different step', () => {
    expect(findWorkflowStepPictures(manifest, 'hot-spot-map', 0, 'terrain-flow')).toEqual([]);
    expect(findWorkflowStepPictures(manifest, 'drainage-network', 0, 'cell-aggregation')).toEqual(
      []
    );
    expect(findWorkflowStepPictures(manifest, 'drainage-network', 20, 'terrain-flow')).toEqual([]);
    for (const run of manifest.runs) {
      for (const picture of run.pictures) {
        expect(picture.path).toBe(
          `workflow-intermediates/${run.runId}/${picture.stageId}-${picture.theme}.png`
        );
        expect(
          findWorkflowStepPictures(manifest, run.templateId, picture.stepIndex, picture.entryId)
        ).toContain(picture);
      }
    }
  });

  it('scopes repeated operation entries to their run and step, while keeping both themes together', () => {
    const makePicture = (runId: string, stepIndex: number, theme: 'dark' | 'light') => ({
      runId,
      graphId: `${runId}-graph`,
      stageId: `stage-${stepIndex}`,
      stepIndex,
      entryId: 'cell-aggregation',
      theme,
      path: `workflow-intermediates/${runId}/stage-${stepIndex}-${theme}.png`,
      sha256: 'fixture',
      sources: [],
      sourceHashes: []
    });
    const firstStep = [makePicture('first-run', 0, 'dark'), makePicture('first-run', 0, 'light')];
    const secondStep = [makePicture('first-run', 1, 'dark'), makePicture('first-run', 1, 'light')];
    const otherWorkflow = [
      makePicture('second-run', 0, 'dark'),
      makePicture('second-run', 0, 'light')
    ];
    const fixture: WorkflowPictureManifest = {
      schemaVersion: 1,
      runs: [
        {
          runId: 'first-run',
          templateId: 'first-template',
          recordPath: 'fixture',
          recordSha256: 'fixture',
          pictures: [...firstStep, ...secondStep]
        },
        {
          runId: 'second-run',
          templateId: 'second-template',
          recordPath: 'fixture',
          recordSha256: 'fixture',
          pictures: otherWorkflow
        }
      ]
    };
    expect(findWorkflowStepPictures(fixture, 'first-template', 0, 'cell-aggregation')).toEqual(
      firstStep
    );
    expect(findWorkflowStepPictures(fixture, 'first-template', 1, 'cell-aggregation')).toEqual(
      secondStep
    );
    expect(findWorkflowStepPictures(fixture, 'second-template', 0, 'cell-aggregation')).toEqual(
      otherWorkflow
    );
  });

  it('rejects a synthetic writer even when it has the correct resource and a preceding RAW edge', () => {
    const resourceId = chain.stages[0].views[0].buffer.id;
    const writers = recipeNodes.filter(node => node.writes.includes(resourceId));
    const makeNode = (node: ProducerIdentity): GPUComputationNode => ({
      ...node,
      label: node.id,
      namespace: '',
      attrs: {}
    });
    const copyNodeId = 'picture-copy';
    const valid: GPUComputationGraph = {
      id: 'fixture',
      form: 'compiled',
      domain: 'gpu-core',
      nodes: [
        ...writers.map(makeNode),
        makeNode({id: copyNodeId, kind: 'compute', reads: [resourceId], writes: ['snapshot']})
      ],
      edges: [
        {from: writers[writers.length - 1].id, to: copyNodeId, kind: 'raw', resource: resourceId}
      ]
    };
    expect(() => verifyRecipeProducers(valid, resourceId, copyNodeId)).not.toThrow();
    const forged = structuredClone(valid);
    forged.nodes[writers.length - 1].id = 'synthetic-producer';
    forged.edges[0].from = 'synthetic-producer';
    expect(() => verifyRecipeProducers(forged, resourceId, copyNodeId)).toThrow(
      'real recipe producer identities'
    );
    const rewired = structuredClone(valid);
    rewired.nodes[writers.length - 1].reads = ['independent-operation-buffer'];
    expect(() => verifyRecipeProducers(rewired, resourceId, copyNodeId)).toThrow(
      'real recipe producer identities'
    );
  });

  for (const theme of ['dark', 'light'] as const) {
    it(`renders distinguishable snapshot bytes for every stage in ${theme}, with hand-checked interior pixels`, () => {
      const background = theme === 'dark' ? [12, 19, 35, 255] : [246, 244, 239, 255];
      const low = theme === 'dark' ? [24, 45, 80, 255] : [220, 232, 247, 255];
      const high = theme === 'dark' ? [255, 211, 99, 255] : [20, 81, 123, 255];
      const ink = theme === 'dark' ? [113, 216, 255, 255] : [18, 103, 143, 255];
      for (let stepIndex = 0; stepIndex < chain.stages.length; stepIndex++) {
        const first = renderWorkflowStagePixels(createStageFixture(stepIndex), theme, 1);
        const second = renderWorkflowStagePixels(createStageFixture(stepIndex, true), theme, 1);
        expect(readPixel(first, 0, 0)).toEqual(background);
        expect(hashBytes(Uint8Array.from(first.data)), chain.stages[stepIndex].id).not.toBe(
          hashBytes(Uint8Array.from(second.data))
        );
        if (stepIndex === 4) {
          expect(readPixel(first, 100, 170)).toEqual(ink);
          expect(readPixel(second, 100, 170)).toEqual(background);
          expect(readPixel(first, 320, 100)).toEqual(background);
          expect(readPixel(second, 320, 100)).toEqual(ink);
        } else if (stepIndex === 5) {
          expect(readPixel(first, 100, 100)).toEqual(background);
          expect(readPixel(second, 100, 100)).toEqual(ink);
          expect(readPixel(first, 500, 100)).toEqual(ink);
          expect(readPixel(second, 500, 100)).toEqual(background);
        } else if (stepIndex === 2) {
          const secondClass = theme === 'dark' ? [255, 211, 99, 255] : [163, 99, 0, 255];
          expect(readPixel(first, 22, 22)).toEqual(ink);
          expect(readPixel(second, 22, 22)).toEqual(secondClass);
        } else {
          expect(readPixel(first, 22, 22)).toEqual(low);
          expect(readPixel(second, 22, 22)).toEqual(high);
          expect(readPixel(first, 29, 22)).toEqual(high);
          expect(readPixel(second, 29, 22)).toEqual(low);
        }
      }
    });

    it(`decodes and replays all six ${theme} pictures, rejecting a repeated poster with matching file hashes`, () => {
      const posterPixels = renderWorkflowStagePixels(createStageFixture(0), theme, 1);
      const poster = PNG.sync.write({...posterPixels, data: Buffer.from(posterPixels.data)});
      const decodedPoster = PNG.sync.read(poster);
      for (let stepIndex = 0; stepIndex < chain.stages.length; stepIndex++) {
        const stage = createStageFixture(stepIndex, true);
        const expected = renderWorkflowStagePixels(stage, theme, 1);
        const image = PNG.sync.write({...expected, data: Buffer.from(expected.data)});
        expect(() =>
          verifyWorkflowStagePixels(stage, theme, 1, PNG.sync.read(image))
        ).not.toThrow();
        // Rehashing a substituted poster satisfies the old file-hash check but fails pixel replay.
        const substituted = {image: poster, sha256: hashBytes(poster)};
        expect(hashBytes(substituted.image)).toBe(substituted.sha256);
        expect(() => verifyWorkflowStagePixels(stage, theme, 1, decodedPoster)).toThrow(
          'picture differs from snapshot replay'
        );
      }
    });
  }

  it.skip('CAPTURE GATE (WEB-HIW-03): every workflow needs chain captures; the capture script currently supports drainage only', () => {
    expect(manifest.schemaVersion).toBe(1);
    const missingPictures = WORKFLOWS.flatMap(workflow =>
      workflow.steps.flatMap((step, stepIndex) => {
        // Adapters have textual cards; every contributor card needs both themes from its chain.
        if (!step.entry) return [];
        const pictures = findWorkflowStepPictures(
          manifest,
          workflow.templateId,
          stepIndex,
          step.entry
        );
        const published = manifest.runs.find(run => run.templateId === workflow.templateId);
        const hasRecord = published && existsSync(new URL(published.recordPath, publicDirectory));
        return hasRecord &&
          pictures.length === 2 &&
          pictures.every(picture => existsSync(new URL(picture.path, publicDirectory))) &&
          new Set(pictures.map(picture => picture.theme)).size === 2
          ? []
          : [`${workflow.templateId} step ${stepIndex + 1} (${step.entry})`];
      })
    );
    expect(
      missingPictures,
      `Required workflow captures are missing: ${missingPictures.join('; ')}. Capture and commit real chain pictures and records; website/scripts/snapshot-workflow-intermediates.mjs supplies drainage only.`
    ).toEqual([]);
  });

  it('verifies recorded chain stage/resource identities and exact snapshot bytes for published captures', () => {
    expect(manifest.schemaVersion).toBe(1);
    // New capture producers need their own real-chain oracle before joining this manifest.
    expect(manifest.runs.every(run => run.templateId === 'drainage-network')).toBe(true);
    const drainageRuns = manifest.runs.filter(run => run.templateId === 'drainage-network');
    // No captures have landed yet (WEB-HIW-03); every published capture must still pass below.
    for (const published of drainageRuns) {
      expect(published.recordPath).toBe(`workflow-intermediates/${published.runId}/record.json`);
      const recordBytes = readFileSync(new URL(published.recordPath, publicDirectory));
      expect(hashBytes(recordBytes)).toBe(published.recordSha256);
      const run: WorkflowIntermediateRun = JSON.parse(recordBytes.toString('utf8'));
      expect(run.runId).toBe(published.runId);
      expect(run.templateId).toBe(published.templateId);
      expect(run.captureId).toBe(WORKFLOW_CAPTURE_ID);
      expect(run.graph.id).toBe(`${WORKFLOW_CAPTURE_ID}-compiled`);
      expect(run.submissionCount).toBe(1);
      expect(run.parameters).toEqual(WORKFLOW_CAPTURE_PARAMETERS);
      expect(run.input.columns).toBe(WORKFLOW_CAPTURE_COLUMNS);
      expect(run.input.rows).toBe(WORKFLOW_CAPTURE_ROWS);
      expect(run.input.sha256).toMatch(/^[a-f0-9]{64}$/);
      const expectedStatus = chain.recipe.status.stages.flatMap(stage =>
        Object.entries(stage.status)
          .filter(([, source]) => source)
          .map(([name, source]) => ({
            stageId: stage.stage,
            name,
            source: identifyWorkflowView(source!)
          }))
      );
      expect(run.status.map(({value, ...identity}) => identity)).toEqual(expectedStatus);
      for (const status of run.status) {
        if (status.name === 'converged') expect(status.value, status.stageId).toBe(1);
        if (status.name === 'overflow' || status.name === 'uncertainCount')
          expect(status.value, status.stageId).toBe(0);
      }
      expect(run.pictures).toEqual(published.pictures);
      expect(run.pictures).toHaveLength(12);
      expect(run.stages).toHaveLength(chain.stages.length);
      expect(new Set(run.pictures.map(picture => picture.path)).size).toBe(12);
      for (const [stepIndex, stage] of run.stages.entries()) {
        const expectedStage = chain.stages[stepIndex];
        expect(stage.stepIndex).toBe(stepIndex);
        expect(stage.id).toBe(expectedStage.id);
        expect(stage.entryId).toBe(expectedStage.entryId);
        expect(stage.snapshots).toHaveLength(expectedStage.views.length);
        for (const [sourceIndex, snapshot] of stage.snapshots.entries()) {
          expect(snapshot.source).toEqual(identifyWorkflowView(expectedStage.views[sourceIndex]));
          expect(snapshot.snapshot).toEqual({
            ...snapshot.source,
            resourceId: `picture-${stage.id}-${sourceIndex}`,
            byteOffset: 0
          });
          expect(snapshot.bytes).toHaveLength(
            snapshot.source.length * snapshot.source.rowByteLength
          );
          expect(hashBytes(Uint8Array.from(snapshot.bytes))).toBe(snapshot.sha256);
          expect(snapshot.copyNodeId).toBe(`${snapshot.snapshot.resourceId}-copy`);
          const copy = run.graph.nodes.find(node => node.id === snapshot.copyNodeId)!;
          expect(copy.kind).toBe('compute');
          expect(copy.reads).toEqual([snapshot.source.resourceId]);
          expect(copy.writes).toEqual([snapshot.snapshot.resourceId]);
          verifyRecipeProducers(run.graph, snapshot.source.resourceId, snapshot.copyNodeId);
          for (const identity of [snapshot.source, snapshot.snapshot]) {
            const resource = run.graph.resources!.find(
              candidate => candidate.id === identity.resourceId
            )!;
            expect(resource, identity.resourceId).toBeDefined();
            expect(resource.bytes).toBeGreaterThanOrEqual(identity.length * identity.rowByteLength);
          }
        }
        const pictures = findWorkflowStepPictures(
          manifest,
          run.templateId,
          stepIndex,
          stage.entryId
        );
        expect(pictures.map(picture => picture.theme).sort()).toEqual(['dark', 'light']);
        for (const picture of pictures) {
          expect(picture.runId).toBe(run.runId);
          expect(picture.graphId).toBe(run.graph.id);
          expect(picture.stageId).toBe(stage.id);
          expect(picture.sources).toEqual(stage.snapshots.map(snapshot => snapshot.snapshot));
          expect(picture.sourceHashes).toEqual(stage.snapshots.map(snapshot => snapshot.sha256));
          const imageBytes = readFileSync(new URL(picture.path, publicDirectory));
          expect(Array.from(imageBytes.subarray(0, 8))).toEqual([137, 80, 78, 71, 13, 10, 26, 10]);
          verifyWorkflowStagePixels(
            stage,
            picture.theme,
            run.input.cellSize,
            PNG.sync.read(imageBytes)
          );
          expect(hashBytes(imageBytes)).toBe(picture.sha256);
        }
      }
    }
  });
});
