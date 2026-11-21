// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {ROW_LOCAL_RULE_AUDIT_CAPACITY} from './weights-helpers-audit-fixtures';
import {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {expect, it} from 'vitest';
import {
  createIslandAttachmentFallbackNodes,
  ISLAND_ATTACHMENT_NEAREST_DECLARATIONS
} from '../../../src/gpu-spatial-analysis/spatial-weights/island-attachment-fallback';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from './spatial-weights-harness';
import {
  FALLBACK_MASK,
  FALLBACK_POSITIONS,
  FALLBACK_SELECTED_NEAREST
} from './weights-helpers-audit-fixtures';
import {
  getKeyPairSortNodes,
  getKeyGroupNodes
} from '../../../src/gpu-spatial-analysis/spatial-weights/key-pair-grouping';
import {
  PAIR_HIGH,
  PAIR_LOW,
  PAIR_ORDER,
  PAIR_GROUP_INDEX,
  PAIR_BOUNDARIES
} from './weights-helpers-audit-fixtures';
import {addWeightsAuditReadback} from './weights-helpers-audit-fixtures';
import {getSlotGroupingNodes} from '../../../src/gpu-spatial-analysis/spatial-weights/slot-grouping';
import {
  SLOT_KEYS,
  SLOT_ORDER,
  SORTED_SLOT_KEYS,
  COLUMN_OFFSETS
} from './weights-helpers-audit-fixtures';
import {createCollectiveAuditNode} from './weights-helpers-audit-fixtures';
import {
  createCountArithmeticAuditNode,
  EXACT_COUNTS,
  COUNT_VALUES,
  COUNT_RESULT_BITS
} from './weights-helpers-audit-fixtures';
import {createRowAuditNode, ROW_NEIGHBORS, ROW_RESULT} from './weights-helpers-audit-fixtures';
import {
  createCsrStageAuditNodes,
  createContainsAuditNode,
  STAGE_LEFT,
  STAGE_RIGHT,
  SET_EXPECTED
} from './weights-helpers-audit-fixtures';
import {createSortRowsAuditNodes} from './weights-helpers-audit-fixtures';
import {
  createWeightsLossNodes,
  createWeightsIncompleteNodes
} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights-loss';
import {
  getRowLocalAuditExpected,
  SELF_INPUT,
  SELF_WEIGHTS,
  SUBGRAPH_MASK
} from './weights-helpers-audit-fixtures';
import {GPUSpatialWeightsAlgebra} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra';
import {GPUChunkedSpatialWeightsAlgebra} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-algebra';
import {
  GPUSpatialWeightsAlgebraOperation,
  type SpatialWeightsAlgebraOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra-operation';
import {CPUSpatialWeightsAlgebraOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-algebra-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {
  SPACE_TIME_INPUT,
  SPACE_TIME_EXPECTED,
  SPACE_TIME_LAGS
} from './weights-helpers-audit-fixtures';
import {createSpatialWeightsSuppressionNodes} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights-suppression';
import {createStatisticsGuardAuditNode} from './weights-helpers-audit-fixtures';
import {GPUSpatialWeightsSummary} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-summary';
import {GPUSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transform';
import {GPUSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transpose';
import {GPUSpatialWeightsValidation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-validation';

it('construction snapshots retain original ports while their buffer contents remain live', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const weights = rig.uploadWeights(STAGE_LEFT);
    const originalValues = weights.weights;
    const statistics = rig.output('float32', 3),
      counts = rig.output('uint32', 6),
      cardinality = rig.output('uint32', 8);
    const summaryOutput = {
      statistics: statistics.view,
      counts: counts.view,
      cardinality: cardinality.view
    };
    const binary = rig.output('float32', 10);
    const transformProps = {operation: 'binary' as const, weights, output: binary.view};
    const transposed = rig.weightsOutput(8, 10, true);
    const asymmetric = rig.output('uint32', 1);
    const transposeOutput = {...transposed.spatialWeights, asymmetricSlots: asymmetric.view};
    const flags = rig.output('uint32', 8),
      invalidCount = rig.output('uint32', 1);
    const validationOutput = {flags: flags.view, status: {invalidCount: invalidCount.view}};
    const contributors = [
      new GPUSpatialWeightsSummary({weights, output: summaryOutput}),
      new GPUSpatialWeightsTransform(transformProps),
      new GPUSpatialWeightsTranspose({weights, output: transposeOutput}),
      new GPUSpatialWeightsValidation({weights, output: validationOutput})
    ];
    const replacement = rig.uploadWeights(STAGE_RIGHT);
    Object.assign(weights, replacement);
    summaryOutput.statistics = rig.output('float32', 3).view;
    summaryOutput.counts = rig.output('uint32', 6).view;
    summaryOutput.cardinality = rig.output('uint32', 8).view;
    transformProps.output = rig.output('float32', 10).view;
    Object.assign(transposeOutput, rig.weightsOutput(8, 10, true).spatialWeights);
    transposeOutput.asymmetricSlots = rig.output('uint32', 1).view;
    validationOutput.flags = rig.output('uint32', 8).view;
    validationOutput.status.invalidCount = rig.output('uint32', 1).view;
    for (const contributor of contributors) rig.graph.add(contributor);
    const compiled = rig.graph.compile();
    try {
      for (const scale of [1, 2]) {
        rig
          .bufferOf(originalValues)
          .write(Float32Array.from(STAGE_LEFT.weights, value => value * scale));
        submitGraph(device, compiled, undefined);
        // Independent dense NumPy moments in weights-helpers-audit.oracle.py; integers
        // below 2^24 make these sums and both positive rescalings exactly representable.
        expect(await readFloat32(statistics.buffer, 3)).toEqual([
          407 * scale,
          38253 * scale * scale,
          138782 * scale * scale
        ]);
        expect(await readUint32(counts.buffer, 6)).toEqual([10, 10, 3, 0, 3, 2]);
        expect(await readUint32(cardinality.buffer, 8)).toEqual([2, 1, 0, 3, 0, 0, 2, 1]);
        expect(await readFloat32(binary.buffer, 10)).toEqual([1, 1, 1, 1, 1, 1, 0, 1, 1, 1]);
        expect(await transposed.read()).toEqual({
          offsets: [0, 2, 4, 5, 6, 6, 7, 8, 10],
          neighbors: [1, 6, 0, 3, 3, 0, 4, 7, 3, 6],
          weights: [20, 70, 11, 41, 42, 13, 0, 86, 47, 77].map(value => value * scale),
          distances: [0, 0, 1, 1, 2, 3, 5, 6, 7, 7]
        });
        expect(await readUint32(asymmetric.buffer, 1)).toEqual([16]);
        expect(await readUint32(flags.buffer, 8)).toEqual(Array(8).fill(0));
        expect(await readUint32(invalidCount.buffer, 1)).toEqual([0]);
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});

it('fallback writes exact selected nearest rows and uncertainty counts across batches and repeated encodings', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [shift, scale] of [
    [0, 1],
    [100, 2]
  ]) {
    for (const batched of [false, true]) {
      const rig = new WeightsRig(device);
      try {
        const positions = rig.input(
          Float32Array.from(FALLBACK_POSITIONS, value => shift + scale * value),
          'float32x2',
          8
        );
        const unresolved = rig.input(Uint32Array.from(FALLBACK_MASK), 'uint32', 8);
        const attachment = rig.output('uint32', 8);
        const uncertainCount = rig.output('uint32', 1);
        rig.graph.add(
          createIslandAttachmentFallbackNodes(rig.graph, {
            id: 'fallback-oracle',
            positions,
            unresolved,
            attachment: attachment.view,
            uncertainCount: uncertainCount.view,
            nearestDeclarations: ISLAND_ATTACHMENT_NEAREST_DECLARATIONS,
            queryBatchSize: batched ? 2 : undefined,
            representativeBatchSize: batched ? 3 : undefined
          })
        );
        const compiled = rig.graph.compile();
        try {
          for (const selected of [true, false]) {
            attachment.buffer.write(new Uint32Array(8).fill(0xffffffff));
            rig
              .bufferOf(unresolved)
              .write(Uint32Array.from(selected ? FALLBACK_MASK : Array(8).fill(0)));
            submitGraph(device, compiled, undefined);
            expect(await readUint32(attachment.buffer, 8)).toEqual(
              selected ? FALLBACK_SELECTED_NEAREST : Array(8).fill(0xffffffff)
            );
            expect(await readUint32(uncertainCount.buffer, 1)).toEqual([selected ? 5 : 0]);
          }
        } finally {
          compiled.destroy();
        }
      } finally {
        rig.destroy();
      }
    }
  }
});

it('GPU cooperative guards preserve barriers, group outputs and zero-dispatch guards', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const invocationCount of [0, 32]) {
    const rig = new WeightsRig(device);
    try {
      const output = rig.output('uint32', 9);
      output.buffer.write(new Uint32Array(9).fill(99));
      rig.graph.add(createStatisticsGuardAuditNode(rig.graph, output.view, invocationCount));
      const compiled = rig.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        expect(await readUint32(output.buffer, 9)).toEqual(
          invocationCount === 0 ? Array(9).fill(99) : [1, 2, 3, 4, 5, 6, 7, 8, 99]
        );
      } finally {
        compiled.destroy();
      }
    } finally {
      rig.destroy();
    }
  }
});

it('GPU suppression writes all six public fields from either matrix or chunk loss and preserves every tail', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const overflow = rig.input(Uint32Array.of(0), 'uint32', 1);
    const candidate = rig.input(Uint32Array.of(0), 'uint32', 1);
    const source = rig.uploadWeights(STAGE_LEFT);
    source.status = {candidateOverflow: candidate};
    const weights = {
      rowCount: 8,
      status: {overflow},
      chunks: [{rowStart: 0, rowCount: 8, weights: source}]
    };
    const output = {
      output: rig.output('float32', 10),
      statistics: rig.output('float32', 5),
      summary: rig.output('float32', 6),
      counts: rig.output('uint32', 8),
      cardinality: rig.output('uint32', 10),
      permutationCount: rig.output('uint32', 3)
    };
    const lengths = {
      output: 8,
      statistics: 3,
      summary: 4,
      counts: 6,
      cardinality: 8,
      permutationCount: 1
    };
    rig.graph.add(
      createSpatialWeightsSuppressionNodes(
        rig.graph,
        'suppress',
        'audit',
        weights,
        Object.fromEntries(Object.entries(output).map(([name, port]) => [name, port.view])),
        lengths
      )
    );
    const compiled = rig.graph.compile();
    try {
      for (const flags of [
        [1, 0],
        [0, 7],
        [0, 0]
      ]) {
        rig.bufferOf(overflow).write(Uint32Array.of(flags[0]));
        rig.bufferOf(candidate).write(Uint32Array.of(flags[1]));
        for (const port of Object.values(output))
          port.buffer.write(
            port.view.format === 'float32'
              ? Float32Array.from({length: port.view.length}, (_, index) => index + 1)
              : Uint32Array.from({length: port.view.length}, (_, index) => index + 1)
          );
        submitGraph(device, compiled, undefined);
        for (const [name, port] of Object.entries(output)) {
          const values =
            port.view.format === 'float32'
              ? await readFloat32(port.buffer, port.view.length)
              : await readUint32(port.buffer, port.view.length);
          const logical = lengths[name as keyof typeof lengths];
          if (flags[0] || flags[1]) {
            if (port.view.format === 'float32')
              expect(values.slice(0, logical).every(Number.isNaN)).toBe(true);
            else expect(values.slice(0, logical)).toEqual(Array(logical).fill(0));
          } else
            expect(values.slice(0, logical)).toEqual(
              Array.from({length: logical}, (_, index) => index + 1)
            );
          expect(values.slice(logical)).toEqual(
            Array.from({length: port.view.length - logical}, (_, index) => logical + index + 1)
          );
        }
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});

it('space-time contributor and both operation forms match every independent Kronecker-product output', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const layout of ['cell-major', 'period-major'] as const)
    for (const scale of [1, 2])
      for (const capacity of [10, 48]) {
        const expected = SPACE_TIME_EXPECTED[layout];
        const used = Math.min(capacity, 43);
        const lags = SPACE_TIME_LAGS.map(lag => ({...lag, weight: lag.weight * scale}));
        const rig = new WeightsRig(device);
        try {
          const output = rig.weightsOutput(12, capacity);
          const overflow = rig.output('uint32', 1);
          const required = rig.output('uint32', 1);
          const candidate = rig.output('uint32', 1);
          rig.run(
            new GPUSpatialWeightsAlgebra({
              operation: 'spaceTime',
              weights: rig.uploadWeights({...SPACE_TIME_INPUT, distances: []}),
              periodCount: 3,
              layout,
              lags,
              output: {
                ...output.spatialWeights,
                status: {
                  overflow: overflow.view,
                  requiredCount: required.view,
                  candidateOverflow: candidate.view
                }
              }
            })
          );
          const result = await output.read();
          expect(result.offsets).toEqual(expected.offsets.map(value => Math.min(capacity, value)));
          expect(result.neighbors).toEqual(expected.neighbors.slice(0, used));
          expect(result.weights).toEqual(
            expected.weights.slice(0, used).map(value => value * scale)
          );
          expect(await readUint32(overflow.buffer, 1)).toEqual([Number(capacity < 43)]);
          expect(await readUint32(required.buffer, 1)).toEqual([43]);
          expect(await readUint32(candidate.buffer, 1)).toEqual([0]);
        } finally {
          rig.destroy();
        }
        for (const Operation of [
          GPUSpatialWeightsAlgebraOperation,
          CPUSpatialWeightsAlgebraOperation
        ]) {
          const instance = new Operation({
            operation: 'spaceTime',
            rowCount: 12,
            capacity,
            weights: {
              offsets: 'sourceOffsets',
              neighbors: 'sourceNeighbors',
              weights: 'sourceWeights'
            },
            periodCount: 3,
            layout,
            lags,
            output: {
              offsets: 'offsets',
              neighbors: 'neighbors',
              weights: 'weights',
              status: {
                overflow: 'overflow',
                requiredCount: 'required',
                candidateOverflow: 'candidate'
              }
            }
          });
          try {
            const result = await runOperationGraph(device, {
              inputs: {
                sourceOffsets: {
                  format: 'uint32',
                  values: Uint32Array.from(SPACE_TIME_INPUT.offsets)
                },
                sourceNeighbors: {
                  format: 'uint32',
                  values: Uint32Array.from(SPACE_TIME_INPUT.neighbors)
                },
                sourceWeights: {
                  format: 'float32',
                  values: Float32Array.from(SPACE_TIME_INPUT.weights)
                }
              },
              operations: [instance],
              outputs: ['offsets', 'neighbors', 'weights', 'overflow', 'required', 'candidate']
            });
            expect(Array.from(result['offsets'])).toEqual(
              expected.offsets.map(value => Math.min(capacity, value))
            );
            expect(Array.from(result['neighbors']).slice(0, used)).toEqual(
              expected.neighbors.slice(0, used)
            );
            expect(Array.from(result['weights']).slice(0, used)).toEqual(
              expected.weights.slice(0, used).map(value => value * scale)
            );
            expect(Array.from(result['overflow'])).toEqual([Number(capacity < 43)]);
            expect(Array.from(result['required'])).toEqual([43]);
            expect(Array.from(result['candidate'])).toEqual([0]);
          } finally {
            if (instance instanceof GPUSpatialWeightsAlgebraOperation) instance.destroy();
          }
        }
      }
}, 120000);

it('packed and chunked row-local contributors match independent weights, distances, offsets and all loss/count ports', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const operation of [
    'union',
    'intersection',
    'difference',
    'symmetric-difference',
    'selfWeight',
    'subgraph'
  ] as const)
    for (const weightRule of operation === 'union'
      ? (['left', 'right', 'sum', 'min', 'max', 'product', 'binary'] as const)
      : (['sum'] as const))
      for (const perRow of operation === 'selfWeight' ? [false, true] : [true])
        for (const scale of [1, 2])
          for (const chunked of [false, true]) {
            const rig = new WeightsRig(device);
            try {
              const capacity = operation === 'union' ? ROW_LOCAL_RULE_AUDIT_CAPACITY : 7;
              const original = operation === 'selfWeight' ? SELF_INPUT : STAGE_LEFT;
              const left = {...original, weights: original.weights.map(value => value * scale)};
              const right = {
                ...STAGE_RIGHT,
                weights: STAGE_RIGHT.weights.map(value => value * scale)
              };
              const expected = getRowLocalAuditExpected(operation, weightRule, scale, perRow);
              const mask = rig.input(Uint32Array.from(SUBGRAPH_MASK), 'uint32', 8);
              const self = perRow
                ? rig.input(
                    Float32Array.from(SELF_WEIGHTS, value => value * scale),
                    'float32',
                    8
                  )
                : 0.5 * scale;
              const ranges = chunked
                ? [
                    [0, 4],
                    [4, 4]
                  ]
                : [[0, 8]];
              const outputs = ranges.map(([rowStart, rowCount]) => ({
                rowStart,
                rowCount,
                csr: rig.weightsOutput(rowCount, capacity, true),
                overflow: rig.output('uint32', 1),
                required: rig.output('uint32', 1),
                candidate: rig.output('uint32', 1)
              }));
              const status = (entry: (typeof outputs)[number]) => ({
                overflow: entry.overflow.view,
                requiredCount: entry.required.view,
                candidateOverflow: entry.candidate.view
              });
              const outputWeights = (entry: (typeof outputs)[number]) => ({
                ...entry.csr.spatialWeights,
                status: status(entry)
              });
              const aggregateOverflow = rig.output('uint32', 1);
              const aggregateCandidate = rig.output('uint32', 1);
              if (chunked) {
                const upload = (source: typeof left) => ({
                  rowCount: 8,
                  chunks: ranges.map(([rowStart, rowCount]) => {
                    const begin = source.offsets[rowStart];
                    const end = source.offsets[rowStart + rowCount];
                    return {
                      rowStart,
                      rowCount,
                      weights: rig.uploadWeights(
                        {
                          offsets: source.offsets
                            .slice(rowStart, rowStart + rowCount + 1)
                            .map(value => value - begin),
                          neighbors: source.neighbors.slice(begin, end),
                          weights: source.weights.slice(begin, end),
                          distances: source.distances.slice(begin, end)
                        },
                        1
                      )
                    };
                  })
                });
                const output = {
                  chunks: outputs.map(entry => ({
                    rowStart: entry.rowStart,
                    rowCount: entry.rowCount,
                    weights: outputWeights(entry)
                  }))
                };
                const contributor = new GPUChunkedSpatialWeightsAlgebra(
                  operation === 'selfWeight'
                    ? {operation, weights: upload(left), selfWeight: self, output}
                    : operation === 'subgraph'
                      ? {operation, weights: upload(left), mask, output}
                      : {
                          operation,
                          left: upload(left),
                          right: upload(right),
                          weightRule,
                          output
                        }
                );
                rig.graph.add(contributor);
                addWeightsAuditReadback(
                  rig.graph,
                  'aggregate-overflow',
                  contributor.outputWeights.status!.overflow!,
                  aggregateOverflow.view
                );
                addWeightsAuditReadback(
                  rig.graph,
                  'aggregate-candidate',
                  contributor.outputWeights.status!.candidateOverflow!,
                  aggregateCandidate.view
                );
              } else {
                const output = outputWeights(outputs[0]);
                rig.graph.add(
                  new GPUSpatialWeightsAlgebra(
                    operation === 'selfWeight'
                      ? {operation, weights: rig.uploadWeights(left), selfWeight: self, output}
                      : operation === 'subgraph'
                        ? {operation, weights: rig.uploadWeights(left), mask, output}
                        : {
                            operation,
                            left: rig.uploadWeights(left),
                            right: rig.uploadWeights(right),
                            weightRule,
                            output
                          }
                  )
                );
              }
              const compiled = rig.graph.compile();
              try {
                submitGraph(device, compiled, undefined);
                let anyOverflow = 0;
                for (const entry of outputs) {
                  const begin = expected.offsets[entry.rowStart];
                  const end = expected.offsets[entry.rowStart + entry.rowCount];
                  const used = Math.min(capacity, end - begin);
                  const actual = await entry.csr.read();
                  expect(actual.offsets).toEqual(
                    expected.offsets
                      .slice(entry.rowStart, entry.rowStart + entry.rowCount + 1)
                      .map(value => Math.min(value - begin, capacity))
                  );
                  expect(actual.neighbors).toEqual(expected.neighbors.slice(begin, begin + used));
                  expect(actual.weights).toEqual(expected.weights.slice(begin, begin + used));
                  expect(actual.distances).toEqual(expected.distances.slice(begin, begin + used));
                  const overflow = Number(end - begin > capacity);
                  anyOverflow |= overflow;
                  expect(await readUint32(entry.overflow.buffer, 1)).toEqual([overflow]);
                  expect(await readUint32(entry.required.buffer, 1)).toEqual([end - begin]);
                  expect(await readUint32(entry.candidate.buffer, 1)).toEqual([0]);
                }
                if (chunked) {
                  expect(await readUint32(aggregateOverflow.buffer, 1)).toEqual([anyOverflow]);
                  expect(await readUint32(aggregateCandidate.buffer, 1)).toEqual([0]);
                }
              } finally {
                compiled.destroy();
              }
            } finally {
              rig.destroy();
            }
          }
}, 120000);

it('GPU and CPU row-local operation forms match the same independent oracle on every port', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const operation of [
    'union',
    'intersection',
    'difference',
    'symmetric-difference',
    'selfWeight',
    'subgraph'
  ] as const)
    for (const weightRule of operation === 'union'
      ? (['sum', 'min', 'max'] as const)
      : (['sum'] as const))
      for (const perRow of operation === 'selfWeight' ? [false, true] : [true]) {
        const original = operation === 'selfWeight' ? SELF_INPUT : STAGE_LEFT;
        const expected = getRowLocalAuditExpected(operation, weightRule, 1, perRow);
        const capacity = operation === 'union' ? ROW_LOCAL_RULE_AUDIT_CAPACITY : 7;
        const used = Math.min(capacity, expected.neighbors.length);
        const sourceNames = {
          offsets: 'leftOffsets',
          neighbors: 'leftNeighbors',
          weights: 'leftWeights',
          distances: 'leftDistances'
        };
        const common = {
          rowCount: 8,
          capacity,
          output: {
            offsets: 'offsets',
            neighbors: 'neighbors',
            weights: 'weights',
            distances: 'distances',
            status: {
              overflow: 'overflow',
              requiredCount: 'required',
              candidateOverflow: 'candidate'
            }
          }
        };
        const props: SpatialWeightsAlgebraOperationProps =
          operation === 'selfWeight'
            ? {...common, operation, weights: sourceNames, selfWeight: perRow ? 'self' : 0.5}
            : operation === 'subgraph'
              ? {...common, operation, weights: sourceNames, mask: 'mask'}
              : {
                  ...common,
                  operation,
                  left: sourceNames,
                  right: {
                    offsets: 'rightOffsets',
                    neighbors: 'rightNeighbors',
                    weights: 'rightWeights',
                    distances: 'rightDistances'
                  },
                  weightRule
                };
        for (const Operation of [
          GPUSpatialWeightsAlgebraOperation,
          CPUSpatialWeightsAlgebraOperation
        ]) {
          const instance = new Operation(props);
          try {
            const result = await runOperationGraph(device, {
              inputs: {
                leftOffsets: {format: 'uint32', values: Uint32Array.from(original.offsets)},
                leftNeighbors: {format: 'uint32', values: Uint32Array.from(original.neighbors)},
                leftWeights: {format: 'float32', values: Float32Array.from(original.weights)},
                leftDistances: {format: 'float32', values: Float32Array.from(original.distances)},
                rightOffsets: {format: 'uint32', values: Uint32Array.from(STAGE_RIGHT.offsets)},
                rightNeighbors: {format: 'uint32', values: Uint32Array.from(STAGE_RIGHT.neighbors)},
                rightWeights: {format: 'float32', values: Float32Array.from(STAGE_RIGHT.weights)},
                rightDistances: {
                  format: 'float32',
                  values: Float32Array.from(STAGE_RIGHT.distances)
                },
                mask: {format: 'uint32', values: Uint32Array.from(SUBGRAPH_MASK)},
                self: {format: 'float32', values: Float32Array.from(SELF_WEIGHTS)}
              },
              operations: [instance],
              outputs: [
                'offsets',
                'neighbors',
                'weights',
                'distances',
                'overflow',
                'required',
                'candidate'
              ]
            });
            expect(Array.from(result['offsets'])).toEqual(
              expected.offsets.map(value => Math.min(value, capacity))
            );
            for (const name of ['neighbors', 'weights', 'distances'] as const)
              expect(Array.from(result[name]).slice(0, used)).toEqual(
                expected[name].slice(0, used)
              );
            expect(Array.from(result['overflow'])).toEqual([
              Number(expected.neighbors.length > capacity)
            ]);
            expect(Array.from(result['required'])).toEqual([expected.neighbors.length]);
            expect(Array.from(result['candidate'])).toEqual([0]);
          } finally {
            if (instance instanceof GPUSpatialWeightsAlgebraOperation) instance.destroy();
          }
        }
      }
}, 120000);

it('GPU loss OR handles eight sources, matrix/chunk metadata, zero reruns and inherited output state', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const merge of [false, true]) {
    const rig = new WeightsRig(device);
    try {
      const inputs = Array.from({length: 8}, () => rig.input(Uint32Array.of(0), 'uint32', 1));
      const output = rig.output('uint32', 1);
      const aggregate = createWeightsIncompleteNodes(rig.graph, 'incomplete-audit', [
        {
          status: {overflow: inputs[0]},
          chunks: inputs.slice(1).map(view => ({weights: {status: {candidateOverflow: view}}}))
        }
      ]);
      const aggregateOutput = rig.output('uint32', 1);
      rig.graph.add([
        ...createWeightsLossNodes(rig.graph, 'loss-audit', inputs, output.view, merge),
        ...aggregate.nodes
      ]);
      addWeightsAuditReadback(rig.graph, 'read-loss', aggregate.view!, aggregateOutput.view);
      const compiled = rig.graph.compile();
      try {
        for (const active of [true, false]) {
          output.buffer.write(Uint32Array.of(1));
          rig.bufferOf(inputs[7]).write(Uint32Array.of(active ? 7 : 0));
          submitGraph(device, compiled, undefined);
          expect(await readUint32(output.buffer, 1)).toEqual([merge || active ? 1 : 0]);
          expect(await readUint32(aggregateOutput.buffer, 1)).toEqual([active ? 1 : 0]);
        }
      } finally {
        compiled.destroy();
      }
    } finally {
      rig.destroy();
    }
  }
});

it('GPU binary frontier sorting writes sorted IDs, aligned constant weights and exact count/status', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const output = rig.weightsOutput(8, 24);
    const overflow = rig.output('uint32', 1);
    const count = rig.output('uint32', 1);
    rig.graph.add(
      createSortRowsAuditNodes(rig.graph, {
        ...output.spatialWeights,
        status: {overflow: overflow.view, requiredCount: count.view}
      })
    );
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const result = await output.read();
      expect(result.offsets).toEqual([0, 3, 6, 9, 12, 15, 18, 21, 24]);
      expect(result.neighbors).toEqual(Array.from({length: 8}, () => [0, 1, 2]).flat());
      expect(result.weights).toEqual(Array(24).fill(1));
      expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
      expect(await readUint32(count.buffer, 1)).toEqual([24]);
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});

it('GPU CSR stages and merge visitors write every port against independent set and count literals', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const operation of [
    undefined,
    'union',
    'intersection',
    'difference',
    'symmetric-difference'
  ] as const) {
    for (const capacity of [7, 20]) {
      const rig = new WeightsRig(device);
      try {
        const output = rig.weightsOutput(8, capacity);
        const overflow = rig.output('uint32', 1);
        const count = rig.output('uint32', 1);
        const nodes = createCsrStageAuditNodes(
          rig.graph,
          rig.uploadWeights(STAGE_LEFT),
          rig.uploadWeights(STAGE_RIGHT),
          {...output.spatialWeights, status: {overflow: overflow.view, requiredCount: count.view}},
          operation
        );
        rig.graph.add(nodes);
        const compiled = rig.graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          const actual = await output.read();
          const expected = operation ? SET_EXPECTED[operation] : STAGE_LEFT;
          const used = Math.min(capacity, expected.neighbors.length);
          expect(actual.offsets).toEqual(expected.offsets.map(value => Math.min(capacity, value)));
          expect(actual.neighbors).toEqual(expected.neighbors.slice(0, used));
          const values = expected.offsets
            .slice(0, -1)
            .flatMap((begin, row) =>
              expected.neighbors
                .slice(begin, expected.offsets[row + 1])
                .map(id => (row + 1) * 10 + id)
            );
          expect(actual.weights).toEqual(values.slice(0, used));
          expect(await readUint32(overflow.buffer, 1)).toEqual([
            expected.neighbors.length > capacity ? 1 : 0
          ]);
          expect(await readUint32(count.buffer, 1)).toEqual([expected.neighbors.length]);
        } finally {
          compiled.destroy();
        }
      } finally {
        rig.destroy();
      }
    }
  }
  const rig = new WeightsRig(device);
  try {
    const answers = rig.output('uint32', 8);
    rig.graph.add(
      createContainsAuditNode(
        rig.graph,
        rig.input(Uint32Array.of(0, 2, 2, 4, 8, 10, 12, 14), 'uint32', 8),
        rig.input(Uint32Array.of(0, 1, 2, 7, 8, 9, 14, 15), 'uint32', 8),
        answers.view
      )
    );
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(answers.buffer, 8)).toEqual([1, 0, 1, 0, 1, 0, 1, 0]);
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});

it('row dispatch uses logical sliced bindings and global output rows on GPU', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const shift of [0, 2]) {
    const rig = new WeightsRig(device);
    try {
      const packedNeighbors = rig.input(
        Uint32Array.from([88, 88, ...ROW_NEIGHBORS.map(value => value + shift), 77, 77]),
        'uint32',
        12
      );
      const neighbors = rig.graph.createDataView(packedNeighbors.buffer, {
        format: 'uint32',
        byteOffset: 8,
        length: 8
      });
      const packedResult = rig.output('uint32', 15);
      packedResult.buffer.write(new Uint32Array(15).fill(99));
      const result = rig.graph.createDataView(packedResult.view.buffer, {
        format: 'uint32',
        byteOffset: 4,
        length: 14
      });
      rig.graph.add(createRowAuditNode(rig.graph, neighbors, result));
      const compiled = rig.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        expect(await readUint32(packedResult.buffer, 15)).toEqual(
          ROW_RESULT.map((value, index) => (index >= 4 && index < 12 ? value + shift : value))
        );
      } finally {
        compiled.destroy();
      }
    } finally {
      rig.destroy();
    }
  }
});

it('GPU count arithmetic matches exact-rational binary32 literals on all four outputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const result = rig.output('uint32', 32);
    rig.graph.add(
      createCountArithmeticAuditNode(
        rig.graph,
        rig.input(Uint32Array.from(EXACT_COUNTS), 'uint32', 8),
        rig.input(Float32Array.from(COUNT_VALUES), 'float32', 8),
        result.view
      )
    );
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(result.buffer, 32)).toEqual(COUNT_RESULT_BITS);
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});

it('collective reductions match closed-form sums with optional aligned control and untouched tails', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scale of [1, 2]) {
    for (const withControl of [false, true]) {
      const rig = new WeightsRig(device);
      let controlBuffer: Buffer | undefined;
      try {
        const values = rig.input(
          Float32Array.from({length: 32}, (_, index) => (index + 1) * scale),
          'float32',
          32
        );
        const output = rig.output('float32', 10);
        output.buffer.write(new Float32Array(10).fill(99));
        // A 256-byte prefix is distinct from the control's logical first word.
        const words = new Uint32Array(128).fill(99);
        words[64] = 7;
        controlBuffer = device.createBuffer({data: words, usage: Buffer.UNIFORM | Buffer.COPY_DST});
        const packed = importGraphBuffer(rig.graph, 'audit-control', controlBuffer, 'uint32', 128);
        const control = withControl
          ? rig.graph.createDataView(packed.buffer, {format: 'uint32', byteOffset: 256, length: 64})
          : undefined;
        rig.graph.add(createCollectiveAuditNode(rig.graph, values, output.view, control));
        const compiled = rig.graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          // Sum of consecutive blocks [1..4], [5..8], ..., [29..32].
          expect(await readFloat32(output.buffer, 10)).toEqual([
            ...[10, 26, 42, 58, 74, 90, 106, 122].map(
              value => value * scale + (withControl ? 7 : 0)
            ),
            99,
            99
          ]);
        } finally {
          compiled.destroy();
        }
      } finally {
        rig.destroy();
        controlBuffer?.destroy();
      }
    }
  }
});

it('stable slot grouping matches independent keys, slot IDs and empty-column offsets', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const offsets = rig.output('uint32', 6);
    const grouping = getSlotGroupingNodes(rig.graph, {
      id: 'slots',
      operation: 'audit',
      columns: 5,
      slotKeys: rig.input(Uint32Array.from(SLOT_KEYS), 'uint32', 8),
      slotIndices: rig.input(Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7), 'uint32', 8),
      columnOffsets: offsets.view
    });
    const keys = rig.output('uint32', 8);
    const slots = rig.output('uint32', 8);
    rig.graph.add(grouping.nodes);
    addWeightsAuditReadback(rig.graph, 'slot-keys', grouping.sortedKeys, keys.view);
    addWeightsAuditReadback(rig.graph, 'slot-ids', grouping.sortedSlots, slots.view);
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(keys.buffer, 8)).toEqual(SORTED_SLOT_KEYS);
      expect(await readUint32(slots.buffer, 8)).toEqual(SLOT_ORDER);
      expect(await readUint32(offsets.buffer, 6)).toEqual(COLUMN_OFFSETS);
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});

it('pair sort and groups match NumPy lexicographic literals through native GPU primitives', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const shift of [0, 8]) {
    for (const narrowed of [false, true]) {
      const rig = new WeightsRig(device);
      try {
        const high = rig.input(
          Uint32Array.from(PAIR_HIGH, value => (value === 0xffffffff ? value : value + shift)),
          'uint32',
          8
        );
        const low = rig.input(
          Uint32Array.from(PAIR_LOW, value => (value === 0xffffffff ? value : value + shift)),
          'uint32',
          8
        );
        const sorted = getKeyPairSortNodes(
          rig.graph,
          'pair',
          'audit',
          8,
          high,
          low,
          narrowed ? {high: 8, low: 8} : undefined
        );
        const groups = getKeyGroupNodes(
          rig.graph,
          'groups',
          'audit',
          8,
          high,
          low,
          sorted.sortedItems
        );
        // Read storage-only transient results into caller-owned readback views.
        const sortedOutput = rig.output('uint32', 8);
        const indexOutput = rig.output('uint32', 8);
        const startsOutput = rig.output('uint32', 9);
        rig.graph.add([...sorted.nodes, ...groups.nodes]);
        addWeightsAuditReadback(rig.graph, 'read-order', sorted.sortedItems, sortedOutput.view);
        addWeightsAuditReadback(rig.graph, 'read-index', groups.groupIndex, indexOutput.view);
        addWeightsAuditReadback(rig.graph, 'read-starts', groups.groupStarts, startsOutput.view);
        const compiled = rig.graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          expect(await readUint32(sortedOutput.buffer, 8)).toEqual(PAIR_ORDER);
          expect(await readUint32(indexOutput.buffer, 8)).toEqual(PAIR_GROUP_INDEX);
          expect((await readUint32(startsOutput.buffer, 9)).slice(0, 7)).toEqual(PAIR_BOUNDARIES);
        } finally {
          compiled.destroy();
        }
      } finally {
        rig.destroy();
      }
    }
  }
});
