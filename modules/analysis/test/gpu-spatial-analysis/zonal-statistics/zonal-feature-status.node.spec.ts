// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPUZonalStatistics} from '../../../src/gpu-spatial-analysis/zonal-statistics/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('feature-row membership status requires matching output diagnostics', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const rows = createTransientView(graph, 'rows', 'uint32', 2);
  const counts = createTransientView(graph, 'counts', 'uint32', 1);
  for (const name of [
    'overflow',
    'candidateOverflow',
    'uncertainCount',
    'antimeridianUncertainCount',
    'poleUncertainCount'
  ] as const) {
    const status = {[name]: createTransientView(graph, `input-${name}`, 'uint32', 1)};
    expect(
      () =>
        new GPUZonalStatistics({
          features: {kind: 'feature-rows', pointFeatureRows: rows, featureCount: 1, status},
          output: {counts}
        })
    ).toThrow(new RegExp(`output.status.${name}`));
  }
});

it('feature-row membership status snapshots and validates separate scalar views', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const scalar = (name: string, length = 1) => createTransientView(graph, name, 'uint32', length);
  const input = scalar('input-candidate');
  const status = {candidateOverflow: input};
  const outputStatus = {candidateOverflow: scalar('output-candidate')};
  const originalOutput = outputStatus.candidateOverflow;
  const props = {
    features: {
      kind: 'feature-rows' as const,
      pointFeatureRows: scalar('rows', 2),
      featureCount: 1,
      status
    },
    output: {counts: scalar('counts'), status: outputStatus}
  };
  const contributor = new GPUZonalStatistics(props);
  status.candidateOverflow = scalar('replacement');
  outputStatus.candidateOverflow = scalar('replacement-output');
  expect(contributor.props.features).toHaveProperty('status.candidateOverflow', input);
  expect(contributor.props.output.status?.candidateOverflow).toBe(originalOutput);
  const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
  const statusSource = sources.find(source => source.id.endsWith('-candidate-overflow'))!.source;
  expect(statusSource).toContain('inputStatus[inputStatusOffset]');
  expect(new WgslReflect(statusSource).storage).toHaveLength(2);
  expect(
    () =>
      new GPUZonalStatistics({
        ...props,
        features: {...props.features, status: {candidateOverflow: scalar('short', 0)}}
      })
  ).toThrow(/features.status.*candidateOverflow/);
  expect(
    () =>
      new GPUZonalStatistics({
        ...props,
        output: {...props.output, status}
      })
  ).toThrow(/overlap|alias|disjoint/i);
  const otherGraph = new GPUCommandGraph(device);
  const foreign = createTransientView(otherGraph, 'foreign', 'uint32', 1);
  expect(() =>
    new GPUZonalStatistics({
      ...props,
      features: {...props.features, status: {candidateOverflow: foreign}}
    }).getCommandNodes(graph)
  ).toThrow(/graph/i);
  device.destroy();
});

it('feature-row snapshots retain and copy geographic membership diagnostics (B1)', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
  const sourceStatus = {
    overflow: scalar('overflow'),
    antimeridianUncertainCount: scalar('antimeridian'),
    poleUncertainCount: scalar('pole')
  };
  const targetStatus = {
    overflow: scalar('out-overflow'),
    antimeridianUncertainCount: scalar('out-antimeridian'),
    poleUncertainCount: scalar('out-pole')
  };
  const contributor = new GPUZonalStatistics({
    features: {
      kind: 'feature-rows',
      featureCount: 1,
      pointFeatureRows: createTransientView(graph, 'rows', 'uint32', 2),
      status: sourceStatus
    },
    output: {status: targetStatus}
  });
  expect(contributor.props.features).toHaveProperty(
    'status.antimeridianUncertainCount',
    sourceStatus.antimeridianUncertainCount
  );
  expect(contributor.props.features).toHaveProperty(
    'status.poleUncertainCount',
    sourceStatus.poleUncertainCount
  );
  const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
  for (const name of ['antimeridianUncertainCount', 'poleUncertainCount']) {
    expect(sources.find(source => source.id.endsWith(`-${name}`))?.source).toContain(
      'inputStatus[inputStatusOffset]'
    );
  }
});

it('feature-row status copying rejects disjoint ranges of the same physical buffer', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const backing = createTransientView(graph, 'status-backing', 'uint32', 65);
  const source = graph.createDataView(backing.buffer, {format: 'uint32', byteOffset: 0, length: 1});
  const target = graph.createDataView(backing.buffer, {
    format: 'uint32',
    byteOffset: 256,
    length: 1
  });
  expect(
    () =>
      new GPUZonalStatistics({
        features: {
          kind: 'feature-rows',
          featureCount: 1,
          pointFeatureRows: createTransientView(graph, 'rows', 'uint32', 2),
          status: {candidateOverflow: source}
        },
        output: {status: {candidateOverflow: target}}
      })
  ).toThrow(/physical buffer/);
});

it('feature-row snapshots retain inherited output views', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const counts = createTransientView(graph, 'inherited-counts', 'uint32', 1);
  const contributor = new GPUZonalStatistics({
    features: {
      kind: 'feature-rows',
      featureCount: 1,
      pointFeatureRows: createTransientView(graph, 'rows', 'uint32', 2)
    },
    output: Object.create({counts})
  });
  expect(contributor.props.output.counts).toBe(counts);
  expect(
    contributor.getCommandNodes(graph).some(node => node.id.endsWith('-count-aggregation-clear'))
  ).toBe(true);
});
