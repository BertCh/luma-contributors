// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, test, vi} from 'vitest';
import {createNullWebGPUDevice} from '@luma-contributors/analysis/testing';
import {computeNetworkLouvainOnCPU} from '@luma-contributors/analysis/gpu-network';
import '../../../src/computation-lenses/scenarios/gpu-network/index';
import {fromRecordedLensFile, toRecordedLensFile} from '../../../src/computation-lenses/registry';
import {getLensScenario} from '../../../src/computation-lenses/registry-implementation';
import {buildDecodedLensTwin} from '../../../src/computation-views/twin/static-twin';
import {createNetworkExpansionScenario} from '../../../src/computation-lenses/scenarios/gpu-network/expansion.lens';
import {describeNetworkOptimizationCoverage} from '../../../src/computation-lenses/scenarios/gpu-network/optimization-coverage.lens';
import * as louvainExample from '../../../src/computation-lenses/scenarios/gpu-network/network-louvain.example';

function readRecord(scenarioId: string) {
  return fromRecordedLensFile(
    JSON.parse(
      readFileSync(
        new URL(`../../../../../docs/lens-records/${scenarioId}.json`, import.meta.url),
        'utf8'
      )
    )
  );
}

test('sparse Louvain docs budget includes coarsening and converges on the two-pair input', async () => {
  const device = createNullWebGPUDevice();
  const exampleSpy = vi.spyOn(louvainExample, 'runExample');
  try {
    const structure = await describeNetworkOptimizationCoverage(device, 'louvain-sparse');
    const properties = exampleSpy.mock.calls[0][1];
    expect(properties.maximumLevels).toBe(2);
    expect(properties.maximumIterations).toBe(4);
    expect(properties.output.objectiveTrace.length).toBe(9);
    const recorded = readRecord('gpu-network.network-louvain.sparse');
    const values = (name: string) =>
      recorded.record.captures!.snapshots.find(capture => capture.name === name)!.values;
    const result = computeNetworkLouvainOnCPU(
      {
        topology: {
          offsets: Uint32Array.from(values('offsets')),
          neighbors: Uint32Array.from(values('neighbors')),
          weights: Float32Array.from(values('weights'))
        },
        parameters: Float32Array.from(values('parameters'))
      },
      properties
    );
    expect(Array.from(result.communities)).toEqual([1, 1, 3, 3]);
    expect(Array.from(result.summary)).toEqual([1, 1, 2, 3]);
    expect(Array.from(result.objectiveTrace).filter(Number.isFinite)).toEqual([
      -0.25, 0.5, 0.5, 0.5
    ]);
    // Fresh graph and archived readbacks/timings: a size estimate, never a replacement capture.
    const nodeIds = new Set(structure.graphs.flatMap(graph => graph.nodes.map(node => node.id)));
    const metrics = recorded.record.metrics?.map(metric => ({
      ...metric,
      nodes: Object.fromEntries(
        Object.entries(metric.nodes).filter(([nodeId]) => nodeIds.has(nodeId))
      )
    }));
    const estimate = toRecordedLensFile(
      {...recorded.record, graphs: structure.graphs, refs: structure.refs, metrics},
      {
        scenarioId: recorded.scenarioId,
        recordedAt: recorded.recordedAt,
        adapterLabel: recorded.adapterLabel
      }
    );
    const bytes = Buffer.byteLength(JSON.stringify(estimate));
    console.info(`Sparse Louvain estimated encoded record: ${bytes} bytes`);
    expect(bytes).toBeLessThan(450 * 1024);
  } finally {
    exampleSpy.mockRestore();
    device.destroy();
  }
});

test('SCC docs capture labels as primary output and retain status within both twin budgets', () => {
  const recorded = readRecord('gpu-network.network-strongly-connected-components');
  const scenario = createNetworkExpansionScenario('network-strongly-connected-components');
  const captures = recorded.record.captures!.snapshots;
  const snapshots = scenario.capture(
    name => captures.find(capture => capture.name === name)!,
    {} as Parameters<typeof scenario.capture>[1],
    captures
  );
  const labels = snapshots.find(capture => capture.name === 'strongComponents')!;
  expect(labels.role).toBe('output');
  expect(Array.from(labels.values)).toEqual([0, 0, 2, 3]);
  expect(snapshots).toHaveLength(6);
  for (const name of ['converged', 'iterationCount', 'iterationLimitReached']) {
    expect(snapshots.find(capture => capture.name === name)).toMatchObject({
      role: 'output',
      keySpace: 'query'
    });
  }
  const projected = {
    ...recorded,
    record: {...recorded.record, captures: {...recorded.record.captures!, snapshots}}
  };
  const recordBytes = Buffer.byteLength(
    JSON.stringify(
      toRecordedLensFile(projected.record, {
        scenarioId: recorded.scenarioId,
        recordedAt: recorded.recordedAt,
        adapterLabel: recorded.adapterLabel
      })
    )
  );
  console.info(`SCC estimated encoded record: ${recordBytes} bytes`);
  expect(recordBytes).toBeLessThan(512 * 1024);
  expect(getLensScenario(recorded.scenarioId)!.presentation?.primaryOutputCaptureId).toBe(
    labels.id
  );
  for (const size of ['full', 'thumbnail'] as const) {
    const example =
      size === 'full'
        ? readFileSync(
            new URL(
              '../../../src/computation-lenses/scenarios/gpu-network/network-strongly-connected-components.example.ts',
              import.meta.url
            ),
            'utf8'
          )
        : undefined;
    const twin = buildDecodedLensTwin(
      projected,
      {scenario: recorded.scenarioId, rung: 'result'},
      {size, example}
    );
    expect(twin.disabled).toBeUndefined();
    expect(twin.llms).toContain('strongComponents');
    expect(twin.keyNumber.value).toBe('4');
    expect(twin.figure.html).not.toContain('data-section="graph:');
    if (size === 'full') {
      for (const name of ['converged', 'iterationCount', 'iterationLimitReached'])
        expect(twin.figure.html).toContain(name);
    }
    const bytes = Buffer.byteLength(
      `${JSON.stringify({schema: 'luma.lens-twin/1', scenario: recorded.scenarioId, ...twin}, null, 2)}\n`
    );
    console.info(`SCC estimated ${size} twin: ${bytes} bytes`);
    expect(bytes).toBeLessThan(size === 'full' ? 256 * 1024 : 48 * 1024);
  }
});
