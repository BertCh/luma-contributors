// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {Buffer} from '@luma.gl/core';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {createFillNode} from '@luma-contributors/analysis/analysis-core';
import {requireTestDevice} from '@luma-contributors/analysis/testing';
import {
  capturePairStatisticsViews,
  createDataFrameLensGraph,
  preparePairStatisticsViews,
  runDataFrameContributorLens,
  runDataFrameLensRecord
} from '../../src/computation-lenses/scenarios/gpu-dataframe/dataframe-lens-support';

test.each([
  'contributor',
  'pair-record',
  'capture-wrapper'
] as const)('%s retains real GPU readback after declaration is separated from execution', async mode => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scenarioId = 'gpu-dataframe.declaration-fixture';
  const graph = createDataFrameLensGraph<undefined>(device, scenarioId);
  const values = createTransientView(
    graph,
    'values',
    'float32',
    3,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  graph.add(
    createFillNode(graph, {id: 'fill', operation: 'Fill', view: values, type: 'f32', value: '7.0'})
  );
  const specs = {values: {view: values, role: 'output' as const, keySpace: 'row'}};
  const keySpaces = [{id: 'row', label: 'row', size: 3, indexSpace: '1d' as const}];
  const captures =
    mode === 'capture-wrapper'
      ? (await capturePairStatisticsViews(device, graph, `${scenarioId}-compiled`, specs)).captures
      : (
          await (mode === 'contributor'
            ? runDataFrameContributorLens(device, {
                scenarioId,
                graph,
                element: 'GPUClassBreaks',
                specs,
                keySpaces,
                datumKeySpace: 'row',
                choices: [],
                datasetVersion: 'fixture'
              })
            : runDataFrameLensRecord({
                scenarioId,
                device,
                keySpaces,
                choices: [],
                prepared: preparePairStatisticsViews(device, graph, `${scenarioId}-compiled`, specs)
              }))
        ).captures!.snapshots;
  expect(captures.map(capture => capture.name)).toEqual(['values']);
  expect(Array.from(captures[0].values)).toEqual([7, 7, 7]);
  expect(captures[0].provenance).toBe('gpu-readback');
});

test('derived capture names are checked against actual GPU-derived captures', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scenarioId = 'gpu-dataframe.derived-fixture';
  const graph = createDataFrameLensGraph<undefined>(device, scenarioId);
  const values = createTransientView(
    graph,
    'values',
    'float32',
    1,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  graph.add(
    createFillNode(graph, {id: 'fill', operation: 'Fill', view: values, type: 'f32', value: '7.0'})
  );
  await expect(
    runDataFrameContributorLens(device, {
      scenarioId,
      graph,
      element: 'GPUClassBreaks',
      specs: {values: {view: values, role: 'output', keySpace: 'row'}},
      keySpaces: [{id: 'row', label: 'row', size: 1, indexSpace: '1d'}],
      datumKeySpace: 'row',
      choices: [],
      datasetVersion: 'fixture',
      deriveCaptures: captures => captures.map(capture => ({...capture, name: 'renamed'})),
      derivedCaptureNames: ['wrong-name']
    })
  ).rejects.toThrow(/derivedCaptureNames must equal/);
});
