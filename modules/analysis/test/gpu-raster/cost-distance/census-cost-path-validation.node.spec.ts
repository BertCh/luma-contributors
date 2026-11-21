// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {Buffer} from '@luma.gl/core';
import {expect, it} from 'vitest';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  GPUCostDistancePath,
  type GPUCostDistancePathProps
} from '../../../src/raster-analysis/cost-distance/gpu-cost-distance-path';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  GPUCostDistancePathOperation,
  type CostDistancePathOperationProps
} from '../../../src/raster-analysis/cost-distance/gpu-cost-distance-path-operation';
import {CPUCostDistancePathOperation} from '../../../src/raster-analysis/cost-distance/cpu-cost-distance-path-operation';

function createPathFixture() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props: GPUCostDistancePathProps = {
    id: 'census-path',
    grid: makeGPURasterMetadata({width: 3, height: 1}),
    backLinks: createTransientView(graph, 'links', 'uint32', 3),
    target: createTransientView(graph, 'target', 'uint32', 1),
    output: {
      path: {
        ids: createTransientView(graph, 'ids', 'uint32', 3),
        status: {
          count: createTransientView(graph, 'count', 'uint32', 1),
          overflow: createTransientView(graph, 'overflow', 'uint32', 1)
        }
      }
    }
  };
  return {graph, props};
}

for (const invalid of ['float', 'two rows', 'overlapping IDs', 'overlapping input'] as const) {
  it(`census path rejects candidateOverflow with ${invalid}`, () => {
    const {graph, props} = createPathFixture();
    const candidateOverflow =
      invalid === 'float'
        ? (createTransientView(
            graph,
            'candidate',
            'float32',
            1
          ) as unknown as GraphDataView<'uint32'>)
        : invalid === 'two rows'
          ? createTransientView(graph, 'candidate', 'uint32', 2)
          : graph.createDataView(
              invalid === 'overlapping IDs' ? props.output.path.ids.buffer : props.backLinks.buffer,
              {format: 'uint32', length: 1}
            );
    expect(
      () =>
        new GPUCostDistancePath({
          ...props,
          output: {
            path: {...props.output.path, status: {...props.output.path.status, candidateOverflow}}
          }
        })
    ).toThrow(invalid.startsWith('overlapping') ? /overlap/ : /candidateOverflow/);
  });
}

it('census path refuses candidateOverflow from another graph', () => {
  const {graph, props} = createPathFixture();
  const otherGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const candidateOverflow = createTransientView(otherGraph, 'foreign-candidate', 'uint32', 1);
  expect(() =>
    new GPUCostDistancePath({
      ...props,
      output: {
        path: {...props.output.path, status: {...props.output.path.status, candidateOverflow}}
      }
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
});

it('census path accepts a separate uint32 candidateOverflow without requiredCount', () => {
  const {graph, props} = createPathFixture();
  const candidateOverflow = createTransientView(graph, 'candidate', 'uint32', 1);
  expect(() =>
    new GPUCostDistancePath({
      ...props,
      output: {
        path: {...props.output.path, status: {...props.output.path.status, candidateOverflow}}
      }
    }).getCommandNodes(graph)
  ).not.toThrow();
});

it('census path snapshots candidate binding before caller replacement', () => {
  const {graph, props} = createPathFixture();
  recordWGSLKernelNodes(graph);
  const original = createTransientView(graph, 'original-candidate', 'uint32', 1);
  props.output.path.status.candidateOverflow = original;
  const contributor = new GPUCostDistancePath(props);
  props.output.path.status.candidateOverflow = graph.createDataView(props.output.path.ids.buffer, {
    format: 'uint32',
    length: 1
  });
  contributor.getCommandNodes(graph);
  const publisher = getWGSLKernelNodeRecord(graph, 'census-path-publish');
  const binding = publisher?.bindings.find(binding => binding.name === 'candidateOverflowOut');
  // The contract binds caller storage at construction, independently of later props mutation.
  expect(binding?.view === original).toBe(true);
  expect(binding?.view.byteOffset).toBe(0);
});

it('census path accepts disjoint back-links and candidate word used by separate nodes', () => {
  const {graph, props} = createPathFixture();
  recordWGSLKernelNodes(graph);
  const storage = graph.createTransientBuffer({
    id: 'shared-links-candidate',
    byteLength: 512,
    usage: Buffer.STORAGE
  });
  props.backLinks = graph.createDataView(storage, {format: 'uint32', length: 3});
  const candidateOverflow = graph.createDataView(storage, {
    format: 'uint32',
    byteOffset: 256,
    length: 1
  });
  props.output.path.status.candidateOverflow = candidateOverflow;
  new GPUCostDistancePath(props).getCommandNodes(graph);
  const walk = getWGSLKernelNodeRecord(graph, 'census-path-walk');
  const publish = getWGSLKernelNodeRecord(graph, 'census-path-publish');
  const links = walk?.bindings.find(binding => binding.name === 'backLinks');
  const candidate = publish?.bindings.find(binding => binding.name === 'candidateOverflowOut');
  expect(links?.view.buffer === candidate?.view.buffer).toBe(true);
  expect([
    links?.view.byteOffset,
    links?.view.length,
    candidate?.view.byteOffset,
    candidate?.view.length
  ]).toEqual([0, 3, 256, 1]);
  expect(walk?.bindings.some(binding => binding.view === candidateOverflow)).toBe(false);
  expect(publish?.bindings.some(binding => binding.view === props.backLinks)).toBe(false);
});

for (const Operation of [GPUCostDistancePathOperation, CPUCostDistancePathOperation]) {
  for (const required of [false, true])
    for (const candidate of [false, true]) {
      it(`census ${Operation.name} declares independent status names (required=${required}, candidate=${candidate})`, async () => {
        const props: CostDistancePathOperationProps = {
          grid: makeGPURasterMetadata({width: 3, height: 1}),
          backLinks: 'links',
          target: 'target',
          capacity: 2,
          output: {
            path: {
              ids: 'ids',
              status: {
                count: 'count',
                overflow: 'overflow',
                ...(required ? {requiredCount: 'required'} : {}),
                ...(candidate ? {candidateOverflow: 'candidate'} : {})
              }
            }
          }
        };
        const operation = new Operation(props);
        const expectedNames = [
          'ids',
          'count',
          'overflow',
          ...(required ? ['required'] : []),
          ...(candidate ? ['candidate'] : [])
        ].sort();
        const declaredOutputs = operation.contract.outputs;
        if (!declaredOutputs || typeof declaredOutputs === 'function')
          throw new Error('path operation must declare static output descriptors');
        expect(Object.keys(declaredOutputs).sort()).toEqual(expectedNames);
        props.output.path.ids = 'replaced-ids';
        props.output.path.status.requiredCount = 'replaced-required';
        props.output.path.status.candidateOverflow = 'replaced-candidate';
        if (operation instanceof CPUCostDistancePathOperation) {
          const rows = (values: Uint32Array) => ({
            format: 'uint32' as const,
            length: values.length,
            chunks: [{values, length: values.length}],
            pack: () => values
          });
          const result = await operation.execute({
            inputs: {backLinks: rows(Uint32Array.of(0, 16, 16)), target: rows(Uint32Array.of(2))}
          });
          // Independent westward 2 -> 1 -> 0: two rows fit, full length three, no candidates.
          expect(Array.from(result['ids'] as Uint32Array)).toEqual([2, 1]);
          expect(Array.from(result['count'] as Uint32Array)).toEqual([2]);
          expect(Array.from(result['overflow'] as Uint32Array)).toEqual([1]);
          if (required) expect(Array.from(result['required'] as Uint32Array)).toEqual([3]);
          if (candidate) expect(Array.from(result['candidate'] as Uint32Array)).toEqual([0]);
          expect(Object.keys(result).sort()).toEqual(expectedNames);
        } else {
          const graph = new GPUCommandGraph(createNullWebGPUDevice());
          recordWGSLKernelNodes(graph);
          const views = Object.fromEntries(
            Object.entries(operation.outputs).map(([port, output]) => [
              port,
              createTransientView(graph, port, output.format, output.length)
            ])
          );
          views['backLinks'] = createTransientView(graph, 'input-links', 'uint32', 3);
          views['target'] = createTransientView(graph, 'input-target', 'uint32', 1);
          const contributor = operation.createContributor(
            port => views[port] as never
          ) as GPUCostDistancePath;
          contributor.getCommandNodes(graph);
          const bindings = getWGSLKernelNodeRecord(graph, 'cost-distance-path-publish')?.bindings;
          expect(bindings?.some(binding => binding.name === 'candidateOverflowOut')).toBe(
            candidate
          );
          expect(bindings?.some(binding => binding.name === 'totalOut')).toBe(required);
        }
      });
    }
}
