// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUSpatialWeightsAlgebra,
  type GPUSpatialWeightsAlgebraProps
} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra';
import {GPUChunkedSpatialWeightsAlgebra} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-algebra';
import type {CPUSpatialWeightsAlgebraRows} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-algebra-operation';
import type {GPUSpatialWeightsOutput} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from './spatial-weights-harness';
import {AUDIT_ALGEBRA_CASES} from './audit-weights-a-algebra-fixture';
import {
  GPUSpatialWeightsAlgebraOperation,
  type SpatialWeightsAlgebraOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra-operation';
import {runOperationGraph, type OperationGraphInput} from '../../utils/mixed-execution-test-utils';

type Fixture = (typeof AUDIT_ALGEBRA_CASES)[number];

function uploadRows(
  rig: WeightsRig,
  source: CPUSpatialWeightsAlgebraRows,
  rowStart = 0,
  rowCount = 8
) {
  const begin = source.offsets[rowStart];
  const end = source.offsets[rowStart + rowCount];
  return rig.uploadWeights(
    {
      offsets: Array.from(
        source.offsets.slice(rowStart, rowStart + rowCount + 1),
        value => value - begin
      ),
      neighbors: Array.from(source.neighbors.slice(begin, end)),
      weights: Array.from(source.weights.slice(begin, end)),
      distances: Array.from(source.distances?.slice(begin, end) ?? [])
    },
    1
  );
}

function createProps(
  rig: WeightsRig,
  fixture: Fixture,
  output: GPUSpatialWeightsOutput
): GPUSpatialWeightsAlgebraProps {
  const {options, input} = fixture;
  switch (options.operation) {
    case 'union':
    case 'intersection':
    case 'difference':
    case 'symmetric-difference':
      return {
        ...options,
        left: uploadRows(rig, input.left!),
        right: uploadRows(rig, input.right!),
        output
      };
    case 'higherOrder':
    case 'spaceTime':
      return {...options, weights: uploadRows(rig, input.weights!), output};
    case 'selfWeight':
      return {
        ...options,
        weights: uploadRows(rig, input.weights!),
        selfWeight: rig.input(input.selfWeights!, 'float32', 8),
        output
      };
    case 'subgraph':
      return {
        ...options,
        weights: uploadRows(rig, input.weights!),
        mask: rig.input(input.mask!, 'uint32', 8),
        output
      };
    case 'block':
      return {...options, groupIds: rig.input(input.groupIds!, 'uint32', 8), output};
  }
}

function createOutput(rig: WeightsRig, rows: number, capacity: number, distances: boolean) {
  const output = rig.weightsOutput(rows, capacity, distances);
  const overflow = rig.output('uint32', 1);
  const required = rig.output('uint32', 1);
  const candidate = rig.output('uint32', 1);
  return {
    output: {
      ...output.spatialWeights,
      status: {
        overflow: overflow.view,
        requiredCount: required.view,
        candidateOverflow: candidate.view
      }
    },
    async check(fixture: Fixture, rowStart = 0) {
      const begin = fixture.offsets[rowStart];
      const end = fixture.offsets[rowStart + rows];
      const usedEnd = Math.min(end, begin + capacity);
      const actual = await output.read();
      expect(actual.offsets, fixture.name).toEqual(
        fixture.offsets
          .slice(rowStart, rowStart + rows + 1)
          .map(value => Math.min(capacity, value - begin))
      );
      expect(actual.neighbors, fixture.name).toEqual(fixture.neighbors.slice(begin, usedEnd));
      expect(actual.weights, fixture.name).toEqual(fixture.weights.slice(begin, usedEnd));
      if (fixture.distances)
        expect(actual.distances, fixture.name).toEqual(fixture.distances.slice(begin, usedEnd));
      expect(await readUint32(required.buffer, 1), fixture.name).toEqual([end - begin]);
      expect(await readUint32(overflow.buffer, 1), fixture.name).toEqual([
        Number(end - begin > capacity)
      ]);
      expect(await readUint32(candidate.buffer, 1), fixture.name).toEqual([0]);
    }
  };
}

it('audits every whole algebra operation and output against literal dense-set/shortest-path/Kronecker values', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Small dyadic weights, sums and products are exactly representable; compare without tolerance.
  for (const fixture of AUDIT_ALGEBRA_CASES) {
    for (const capacity of [2, 64]) {
      const rig = new WeightsRig(device);
      try {
        const output = createOutput(
          rig,
          fixture.offsets.length - 1,
          capacity,
          Boolean(fixture.distances)
        );
        rig.run(new GPUSpatialWeightsAlgebra(createProps(rig, fixture, output.output)));
        await output.check(fixture);
      } finally {
        rig.destroy();
      }
    }
  }
});

it('audits named algebra operation wiring against literal results on every output', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const fixture of AUDIT_ALGEBRA_CASES) {
    const inputs: Record<string, OperationGraphInput> = {};
    const addWeights = (prefix: string, source: CPUSpatialWeightsAlgebraRows) => {
      inputs[`${prefix}Offsets`] = {format: 'uint32', values: source.offsets};
      inputs[`${prefix}Neighbors`] = {format: 'uint32', values: source.neighbors};
      inputs[`${prefix}Weights`] = {format: 'float32', values: source.weights};
      if (source.distances)
        inputs[`${prefix}Distances`] = {format: 'float32', values: source.distances};
      return {
        offsets: `${prefix}Offsets`,
        neighbors: `${prefix}Neighbors`,
        weights: `${prefix}Weights`,
        distances: source.distances ? `${prefix}Distances` : undefined
      };
    };
    const {input, options} = fixture;
    const operands = input.left
      ? {left: addWeights('left', input.left), right: addWeights('right', input.right!)}
      : input.weights
        ? {weights: addWeights('source', input.weights)}
        : {};
    if (input.mask) inputs['mask'] = {format: 'uint32', values: input.mask};
    if (input.groupIds) inputs['groups'] = {format: 'uint32', values: input.groupIds};
    if (input.selfWeights) inputs['self'] = {format: 'float32', values: input.selfWeights};
    const props = {
      ...options,
      ...operands,
      rowCount: fixture.offsets.length - 1,
      ...(options.operation === 'subgraph' ? {mask: 'mask'} : {}),
      ...(options.operation === 'block' ? {groupIds: 'groups'} : {}),
      ...(options.operation === 'selfWeight' ? {selfWeight: 'self'} : {}),
      output: {
        offsets: 'offsets',
        neighbors: 'neighbors',
        weights: 'weights',
        distances: fixture.distances ? 'distances' : undefined,
        status: {overflow: 'overflow', requiredCount: 'required', candidateOverflow: 'candidate'}
      }
    } as SpatialWeightsAlgebraOperationProps;
    const result = await runOperationGraph(device, {
      inputs,
      operations: [new GPUSpatialWeightsAlgebraOperation(props)],
      outputs: [
        'offsets',
        'neighbors',
        'weights',
        'overflow',
        'required',
        'candidate',
        ...(fixture.distances ? ['distances'] : [])
      ]
    });
    const used = fixture.neighbors.length;
    expect(Array.from(result['offsets']), fixture.name).toEqual(fixture.offsets);
    expect(Array.from(result['neighbors'].slice(0, used)), fixture.name).toEqual(fixture.neighbors);
    expect(Array.from(result['weights'].slice(0, used)), fixture.name).toEqual(fixture.weights);
    if (fixture.distances)
      expect(Array.from(result['distances'].slice(0, used)), fixture.name).toEqual(
        fixture.distances
      );
    expect(Array.from(result['required'])).toEqual([used]);
    expect(Array.from(result['overflow'])).toEqual([0]);
    expect(Array.from(result['candidate'])).toEqual([0]);
  }
});

it('audits chunk-local CSR/status outputs and aggregate loss against the same literal algebra values', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const fixture of AUDIT_ALGEBRA_CASES) {
    const {options, input} = fixture;
    if (
      options.operation === 'higherOrder' ||
      options.operation === 'spaceTime' ||
      options.operation === 'block'
    )
      continue;
    for (const capacity of [2, 64]) {
      const rig = new WeightsRig(device);
      const ranges = [
        {rowStart: 0, rowCount: 2},
        {rowStart: 2, rowCount: 3},
        {rowStart: 5, rowCount: 3}
      ];
      const upload = (source: CPUSpatialWeightsAlgebraRows) => ({
        rowCount: 8,
        information: {
          rowCount: 8,
          transform: 'raw' as const,
          symmetric: 'unknown' as const,
          hasSelfWeights: true
        },
        chunks: ranges.map(range => ({
          ...range,
          weights: uploadRows(rig, source, range.rowStart, range.rowCount)
        }))
      });
      const outputs = ranges.map(range =>
        createOutput(rig, range.rowCount, capacity, Boolean(fixture.distances))
      );
      const output = {
        chunks: ranges.map((range, index) => ({...range, weights: outputs[index].output}))
      };
      const operation =
        options.operation === 'subgraph'
          ? new GPUChunkedSpatialWeightsAlgebra({
              ...options,
              weights: upload(input.weights!),
              mask: rig.input(input.mask!, 'uint32', 8),
              output
            })
          : options.operation === 'selfWeight'
            ? new GPUChunkedSpatialWeightsAlgebra({
                ...options,
                weights: upload(input.weights!),
                selfWeight: rig.input(input.selfWeights!, 'float32', 8),
                output
              })
            : new GPUChunkedSpatialWeightsAlgebra({
                ...options,
                left: upload(input.left!),
                right: upload(input.right!),
                output
              });
      const aggregate = rig.output('uint32', 2);
      try {
        rig.run(operation, {
          getCommandNodes: graph => [
            createWGSLKernelNode(graph, {
              id: 'audit-aggregate',
              operation: 'AuditAlgebra',
              invocationCount: 1,
              bindings: [
                {
                  name: 'overflow',
                  view: operation.outputWeights.status!.overflow!,
                  type: 'u32',
                  access: 'read'
                },
                {
                  name: 'candidate',
                  view: operation.outputWeights.status!.candidateOverflow!,
                  type: 'u32',
                  access: 'read'
                },
                {name: 'result', view: aggregate.view, type: 'u32', access: 'read_write'}
              ],
              body: 'result[resultOffset] = overflow[overflowOffset]; result[resultOffset + 1u] = candidate[candidateOffset];'
            })
          ]
        });
        for (const [index, range] of ranges.entries())
          await outputs[index].check(fixture, range.rowStart);
        expect(await readUint32(aggregate.buffer, 2)).toEqual([
          Number(
            ranges.some(
              range =>
                fixture.offsets[range.rowStart + range.rowCount] - fixture.offsets[range.rowStart] >
                capacity
            )
          ),
          0
        ]);
      } finally {
        rig.destroy();
      }
    }
  }
});
