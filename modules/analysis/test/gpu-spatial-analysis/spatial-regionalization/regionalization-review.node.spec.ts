// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeMaxPRegionsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-max-p-regions-operation';
import {createReviewR1} from './regionalization-review-fixtures';
import {createReviewR2} from './regionalization-review-fixtures';
import {createReviewR3} from './regionalization-review-fixtures';
import {createReviewR4} from './regionalization-review-fixtures';
import {computeAZPRegionsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-azp-regions-operation';
import {
  GPUCommandGraph,
  GPUGraph,
  gpuGraph,
  createTransientView,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPURegionPartitionEvaluation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-region-partition-evaluation';
import {GPUMaxPRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions-operation';
import {CPUMaxPRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-max-p-regions-operation';
import {GPUAZPRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-azp-regions-operation';
import {CPUAZPRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-azp-regions-operation';
import {GPUMaxPRegions} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions';
import {GPUAZPRegions} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-azp-regions';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createEnclaveRemainderComponentNodes,
  getEnclaveRemainderComponentNodeCount
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/enclave-remainder-components';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {getRegionalizationOperationInputs} from './regionalization-operation-test-utils';
import type {GPUContributorOperation} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import type {CPUMaxPRegionsInput} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-max-p-regions-operation';
import type {CPUAZPRegionsInput} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-azp-regions-operation';
import type {CPUVector} from '@luma.gl/graph/gpu-data';
import {
  makeCPUVector,
  getCPUVectorRows,
  runOperationGraph
} from '../../utils/mixed-execution-test-utils';
import {createAuditAZPInput, createAuditAZPProps} from './regionalization-audit-azp-fixtures';
import {createAuditMaxPInput, createAuditMaxPProps} from './regionalization-audit-max-p-fixtures';

function lowerReviewOperation(
  graph: GPUCommandGraph,
  operation: GPUContributorOperation,
  input: CPUMaxPRegionsInput | CPUAZPRegionsInput
) {
  const arrays = getRegionalizationOperationInputs(input);
  return operation.createContributor(port => {
    const output = operation.outputs[port];
    const descriptor = output ?? {format: arrays[port].format, length: arrays[port].values.length};
    return createTransientView(graph, port, descriptor.format, descriptor.length) as never;
  });
}

it('R1 publishes feasibility from final refined memberships', () => {
  const fixture = createReviewR1();
  const result = computeMaxPRegionsOnCPU(fixture.input, fixture.props);
  expect(Array.from(result.labels)).toEqual(fixture.labels);
  expect(Array.from(result.objective)).toEqual([fixture.objective]);
  expect(Array.from(result.status)).toEqual(fixture.status);
});

it('R2 ranks only AZP destinations whose own relative cutoff admits the move', () => {
  const fixture = createReviewR2();
  const result = computeAZPRegionsOnCPU(fixture.input, fixture.props);
  expect(Array.from(result.labels)).toEqual(fixture.labels);
  expect(Array.from(result.objective)).toEqual([fixture.objective]);
  expect(Array.from(result.status)).toEqual(fixture.status);
});

it('R3 retains connected enclave remainders as one region', () => {
  const fixture = createReviewR3();
  const result = computeMaxPRegionsOnCPU(fixture.input, fixture.props);
  expect(Array.from(result.labels)).toEqual(fixture.labels);
  expect(Array.from(result.objective)).toEqual([fixture.objective]);
  expect(Array.from(result.status)).toEqual(fixture.status);
});

it('R4 retains the growth SSD of a region that enclaves fully absorb', () => {
  const fixture = createReviewR4();
  const result = computeMaxPRegionsOnCPU(fixture.input, fixture.props);
  expect(Array.from(result.status)).toEqual(fixture.status);
  expect(Array.from(result.labels)).toEqual(fixture.labels);
  expect(Array.from(result.objective)).toEqual([fixture.objective]);
});

for (const name of ['overflow', 'candidateOverflow'] as const) {
  for (const placement of ['outer chunk', 'whole CSR', 'chunk local'] as const) {
    it(`R14 rejects misaligned ${placement} ${name} at construction with the public field name`, () => {
      const device = createNullWebGPUDevice();
      try {
        const graph = new GPUCommandGraph(device);
        const buffer = createTransientView(graph, 'flag-storage', 'uint32', 2).buffer;
        const flag = graph.createDataView(buffer, {format: 'uint32', length: 1, byteOffset: 2});
        const matrix = {
          offsets: createTransientView(graph, 'offsets', 'uint32', 3),
          neighbors: createTransientView(graph, 'neighbors', 'uint32', 2),
          weights: createTransientView(graph, 'weights', 'float32', 2),
          ...(placement === 'chunk local' || placement === 'whole CSR'
            ? {status: {[name]: flag}}
            : {})
        };
        const weights =
          placement === 'whole CSR'
            ? matrix
            : {
                rowCount: 2,
                chunks: [{rowStart: 0, rowCount: 2, weights: matrix}],
                ...(placement === 'outer chunk' ? {status: {[name]: flag}} : {})
              };
        expect(
          () =>
            new GPURegionPartitionEvaluation({
              id: 'alignment',
              values: createTransientView(graph, 'values', 'float32', 2),
              labels: createTransientView(graph, 'labels', 'uint32', 2),
              weights,
              output: {summary: createTransientView(graph, 'summary', 'float32', 9)}
            })
        ).toThrow(
          `alignment ${placement === 'chunk local' ? 'weights.chunks[0].weights.status' : 'weights.status'}.${name}`
        );
      } finally {
        device.destroy();
      }
    });
  }
  for (const format of ['uint32', 'float32'] as const) {
    it(`R6 rejects outer chunk ${name} with ${format === 'uint32' ? 'zero length' : 'float32 format'}`, () => {
      const device = createNullWebGPUDevice();
      try {
        const graph = new GPUCommandGraph(device);
        const flag = createTransientView(
          graph,
          'bad-loss',
          format,
          format === 'uint32' ? 0 : 1
        ) as unknown as GraphDataView<'uint32'>;
        const weights = {
          offsets: createTransientView(graph, 'offsets', 'uint32', 3),
          neighbors: createTransientView(graph, 'neighbors', 'uint32', 2),
          weights: createTransientView(graph, 'weights', 'float32', 2)
        };
        expect(
          () =>
            new GPURegionPartitionEvaluation({
              values: createTransientView(graph, 'values', 'float32', 2),
              labels: createTransientView(graph, 'labels', 'uint32', 2),
              weights: {
                rowCount: 2,
                chunks: [{rowStart: 0, rowCount: 2, weights}],
                status: {[name]: flag}
              },
              output: {summary: createTransientView(graph, 'summary', 'float32', 9)}
            })
        ).toThrow(new RegExp(`weights.status.${name}`));
      } finally {
        device.destroy();
      }
    });
  }
}

it('R7 accepts inherited output names on all AZP and Max-P operation forms', () => {
  const output = Object.create({labels: 'L', objective: 'O', status: 'S'});
  for (const Operation of [GPUMaxPRegionsOperation, CPUMaxPRegionsOperation])
    expect(() => new Operation({...createReviewR1().props, output})).not.toThrow();
  for (const Operation of [GPUAZPRegionsOperation, CPUAZPRegionsOperation])
    expect(() => new Operation({...createReviewR2().props, output})).not.toThrow();
});

it('R7 captures inherited options/output getters once through GPU lowering and direct constructors', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const maxP of [false, true]) {
      const graph = new GPUCommandGraph(device);
      const reads = {option: 0, labels: 0, objective: 0, status: 0};
      const optionPrototype = {
        get standardize() {
          reads.option++;
          return false;
        }
      };
      const outputPrototype = {
        get labels() {
          reads.labels++;
          return 'L';
        },
        get objective() {
          reads.objective++;
          return 'O';
        },
        get status() {
          reads.status++;
          return 'S';
        }
      };
      const fixture = maxP ? createReviewR1() : createReviewR2();
      const {standardize: ignored, ...own} = fixture.props;
      void ignored;
      const inherited = Object.assign(Object.create(optionPrototype), own, {
        output: Object.create(outputPrototype)
      });
      const operation = maxP
        ? new GPUMaxPRegionsOperation(inherited)
        : new GPUAZPRegionsOperation(inherited);
      expect(reads).toEqual({option: 1, labels: 1, objective: 1, status: 1});
      Object.defineProperty(optionPrototype, 'standardize', {value: true});
      const contributor = lowerReviewOperation(graph, operation, fixture.input) as
        | GPUMaxPRegions
        | GPUAZPRegions;
      expect(contributor.props.standardize).toBe(false);
      expect(operation.outputs['labels'].value).toBe('L');
      expect(reads).toEqual({option: 1, labels: 1, objective: 1, status: 1});
      const {standardize: ignoredDirect, output, ...directOwn} = contributor.props;
      void ignoredDirect;
      const direct = Object.assign(Object.create({standardize: false}), directOwn, {
        output: Object.create(output)
      });
      const rebuilt = maxP ? new GPUMaxPRegions(direct) : new GPUAZPRegions(direct);
      expect(rebuilt.props.standardize).toBe(false);
      expect(rebuilt.props.output.labels).toBe(output.labels);
      expect(rebuilt.getCommandNodes(graph).some(node => node.id.includes('standardize'))).toBe(
        false
      );
      operation.destroy();
    }
    const graph = new GPUCommandGraph(device);
    const views = {
      summary: createTransientView(graph, 'summary', 'float32', 9),
      regionSizes: createTransientView(graph, 'sizes', 'uint32', 3),
      regionMeans: createTransientView(graph, 'means', 'float32', 6),
      regionWithinSumOfSquares: createTransientView(graph, 'within', 'float32', 3),
      invalidRowCount: createTransientView(graph, 'invalid', 'uint32', 1)
    };
    const evaluation = new GPURegionPartitionEvaluation(
      Object.assign(Object.create({columnCount: 2, labelCapacity: 3}), {
        values: createTransientView(graph, 'values', 'float32', 4),
        labels: createTransientView(graph, 'labels', 'uint32', 2),
        output: Object.create(views)
      })
    );
    expect(evaluation.props.columnCount).toBe(2);
    expect(evaluation.props.output.regionMeans).toBe(views.regionMeans);
    expect(evaluation.getCommandNodes(graph).length).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});

it('R7 executes CPU operations with the inherited raw-unit option and getter output names', async () => {
  const device = createNullWebGPUDevice();
  try {
    for (const maxP of [false, true]) {
      const fixture = maxP ? createReviewR1() : createReviewR2();
      let optionReads = 0;
      const prototype = {
        get standardize() {
          optionReads++;
          return false;
        }
      };
      const {standardize: ignored, ...own} = fixture.props;
      void ignored;
      const inherited = Object.assign(Object.create(prototype), own, {
        output: Object.create({
          get labels() {
            return 'L';
          },
          get objective() {
            return 'O';
          },
          get status() {
            return 'S';
          }
        })
      });
      const operation = maxP
        ? new CPUMaxPRegionsOperation(inherited)
        : new CPUAZPRegionsOperation(inherited);
      Object.defineProperty(prototype, 'standardize', {value: true});
      const result = await runOperationGraph(device, {
        inputs: getRegionalizationOperationInputs(fixture.input),
        operations: [operation],
        outputs: ['L', 'O', 'S']
      });
      expect(optionReads).toBe(1);
      expect(Array.from(result['L'])).toEqual(fixture.labels);
      expect(Array.from(result['S'])).toEqual(fixture.status);
      expect(Array.from(result['O'])).toEqual([0.5]);
    }
  } finally {
    device.destroy();
  }
});

it('R3 executes the component kernels on connected and disconnected remainders against hand components', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const connected of [false, true]) {
      const graph = new GPUCommandGraph(device);
      const memory = createCPUGraphMemory();
      const adjacency = [
        [1, 2, 3, 4],
        connected ? [0, 3] : [0],
        [0],
        connected ? [0, 1] : [0],
        [0, 5],
        [4]
      ];
      const input = <Format extends 'uint32' | 'float32'>(
        name: string,
        format: Format,
        data: Uint32Array | Float32Array
      ) => {
        const view = createTransientView(graph, name, format, data.length);
        memory.write(view, data);
        return view;
      };
      const rowState = input('rows', 'uint32', Uint32Array.of(0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0));
      const nodes = createEnclaveRemainderComponentNodes(graph, {
        id: 'components',
        rowCount: 6,
        startCount: 1,
        rowState,
        regionPhase: input('phase', 'uint32', Uint32Array.of(2, 0, 0, 0, 3, 0)),
        claims: input('claims', 'uint32', new Uint32Array(24)),
        weights: {
          offsets: input(
            'offsets',
            'uint32',
            Uint32Array.from([
              0,
              ...adjacency.map((_, row) => adjacency.slice(0, row + 1).flat().length)
            ])
          ),
          neighbors: input('neighbors', 'uint32', Uint32Array.from(adjacency.flat())),
          weights: input('weights', 'float32', new Float32Array(adjacency.flat().length).fill(1))
        }
      });
      expect(nodes.length).toBe(getEnclaveRemainderComponentNodeCount(6));
      executeCommandNodesOnCPU(nodes, memory);
      expect(Array.from(memory.readUint32(rowState)).filter((_, index) => index % 2 === 0)).toEqual(
        connected ? [0, 4, 0, 4, 0, 0] : [0, 4, 0, 3, 0, 0]
      );
    }
  } finally {
    device.destroy();
  }
});

it('R8 declares atomic shared region-presence writes in finalization', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const fixture = createReviewR3();
    const operation = new GPUMaxPRegionsOperation({...fixture.props, id: 'presence'});
    lowerReviewOperation(graph, operation, fixture.input).getCommandNodes(graph);
    const record = getWGSLKernelNodeRecord(graph, 'presence-finalize-starts')!;
    expect(record.bindings.find(binding => binding.name === 'claims')?.type).toBe('atomic<u32>');
    operation.destroy();
  } finally {
    device.destroy();
  }
});

it('R11 includes six AZP and two evaluator suppression nodes with queen-style loss ports', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const fixture = createReviewR2();
    const operation = new GPUAZPRegionsOperation(fixture.props);
    const contributor = lowerReviewOperation(graph, operation, fixture.input) as GPUAZPRegions;
    const weights = {
      ...contributor.props.weights,
      status: {
        overflow: createTransientView(graph, 'overflow', 'uint32', 1),
        candidateOverflow: createTransientView(graph, 'candidate-overflow', 'uint32', 1)
      }
    };
    const azpNodes = new GPUAZPRegions({...contributor.props, weights}).getCommandNodes(graph);
    expect(azpNodes.filter(node => node.id.includes('-incomplete-'))).toHaveLength(6);
    const evaluationNodes = new GPURegionPartitionEvaluation({
      id: 'evaluation',
      weights,
      values: contributor.props.values,
      labels: contributor.props.initialLabels,
      output: {summary: createTransientView(graph, 'summary', 'float32', 9)}
    }).getCommandNodes(graph);
    expect(evaluationNodes.filter(node => node.id.includes('-incomplete-'))).toHaveLength(2);
    operation.destroy();
  } finally {
    device.destroy();
  }
});

it('v2-1 publishes the evaluator summary without serial tile scans', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    new GPURegionPartitionEvaluation({
      id: 'parallel-summary',
      values: createTransientView(graph, 'values', 'float32', 260),
      labels: createTransientView(graph, 'labels', 'uint32', 260),
      labelCapacity: 260,
      output: {summary: createTransientView(graph, 'summary', 'float32', 9)}
    }).getCommandNodes(graph);
    const record = getWGSLKernelNodeRecord(graph, 'parallel-summary-summary')!;
    expect(record.body).not.toMatch(/\b(?:for|while)\s*\(|\bloop\s*\{/);
  } finally {
    device.destroy();
  }
});

it('R15 evaluates the actual summary shader at normal and subnormal SSD scales', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const exponent of [-74, -63]) {
      const graph = new GPUCommandGraph(device);
      recordWGSLKernelNodes(graph);
      const summary = createTransientView(graph, 'summary', 'float32', 9);
      const nodes = new GPURegionPartitionEvaluation({
        id: 'exponents',
        values: createTransientView(graph, 'values', 'float32', 8),
        labels: createTransientView(graph, 'labels', 'uint32', 8),
        labelCapacity: 2,
        output: {summary}
      }).getCommandNodes(graph);
      const record = getWGSLKernelNodeRecord(graph, 'exponents-summary')!;
      const countView = record.bindings.find(binding => binding.name === 'countTotals')!.view;
      const squareView = record.bindings.find(binding => binding.name === 'squareTotals')!.view;
      const stride = countView.length / 7;
      const counts = new Uint32Array(countView.length);
      [2, 4, 4, 0, 0, 0, 0].forEach((value, index) => {
        counts[index * stride] = value;
      });
      const squares = new Float32Array(squareView.length);
      const scale = 2 ** exponent;
      [2, 32, 34].forEach((value, index) => {
        squares[index * stride] = value * scale ** 2;
      });
      const memory = createCPUGraphMemory();
      memory.write(countView, counts);
      memory.write(squareView, squares);
      // Runtime-zero label-offset binding is left zero-initialized.
      executeCommandNodesOnCPU(
        nodes.filter(node => node.id === 'exponents-summary'),
        memory
      );
      expect(memory.readFloat32(summary)[8]).toBe(96);
    }
  } finally {
    device.destroy();
  }
});

it('R12/R13 executes origin selection, centroids and between/total kernels against exact moments', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const exponent of [0, -24, null]) {
      const constant = exponent === null;
      const scale = constant ? 1 : 2 ** exponent;
      const values = Float32Array.from(
        constant
          ? [1000000.125, 1000000.125, 1000000.125]
          : [16777216, -16777215, -0.5, -0.5].map(value => value * scale)
      );
      const labels = Uint32Array.from(constant ? [0, 0, 0] : [0, 0, 1, 1]);
      const graph = new GPUCommandGraph(device);
      recordWGSLKernelNodes(graph);
      const means = createTransientView(graph, 'means-output', 'float32', 2);
      const nodes = new GPURegionPartitionEvaluation({
        id: 'moments',
        values: createTransientView(graph, 'values', 'float32', values.length),
        labels: createTransientView(graph, 'labels', 'uint32', values.length),
        labelCapacity: 2,
        output: {summary: createTransientView(graph, 'summary', 'float32', 9), regionMeans: means}
      }).getCommandNodes(graph);
      const binding = (suffix: string, name: string) =>
        getWGSLKernelNodeRecord(graph, `moments-${suffix}`)!.bindings.find(
          entry => entry.name === name
        )!.view;
      const memory = createCPUGraphMemory();
      memory.write(binding('region-origins', 'values'), values);
      memory.write(
        binding('region-origins', 'sortedRows'),
        Uint32Array.from(values, (_, row) => row)
      );
      memory.write(binding('region-origins', 'sortedKeys'), labels);
      memory.write(
        binding('region-origins', 'labelOffsets'),
        Uint32Array.from(constant ? [0, 3, 3] : [0, 2, 4])
      );
      memory.write(binding('origin-signs', 'labels'), labels);
      memory.write(binding('row-tiles', 'sourceLabels'), labels);
      memory.write(binding('between-tiles', 'sizes'), Uint32Array.from(constant ? [3, 0] : [2, 2]));
      // Execute the actual parallel split reduction, including the input-subtraction residuals.
      executeCommandNodesOnCPU(
        nodes.filter(
          node =>
            ['region-origins', 'origin-signs', 'select-origins'].some(
              suffix => node.id === `moments-${suffix}`
            ) || node.id.startsWith('moments-region-sums-')
        ),
        memory
      );
      const suffixes = [
        'region-origins',
        'origin-signs',
        'select-origins',
        'region-centroids',
        'row-tiles',
        'means',
        'label-tiles',
        'between-tiles',
        'deviation-tiles'
      ];
      executeCommandNodesOnCPU(
        suffixes.map(suffix => nodes.find(node => node.id === `moments-${suffix}`)!),
        memory
      );
      const actualMeans = Array.from(memory.readFloat32(means));
      if (constant) {
        expect(actualMeans[0]).toBe(1000000.125);
        expect(Number.isNaN(actualMeans[1])).toBe(true);
        expect(memory.readFloat32(binding('between-tiles', 'labelBetween'))[0]).toBe(0);
        expect(memory.readFloat32(binding('deviation-tiles', 'deviations'))[0]).toBe(0);
      } else {
        expect(actualMeans).toEqual([0.5 * scale, -0.5 * scale]);
        expect(memory.readFloat32(binding('between-tiles', 'labelBetween'))[0]).toBe(scale ** 2);
      }
    }
  } finally {
    device.destroy();
  }
});

for (const lossName of ['overflow', 'candidateOverflow'] as const) {
  it(`R10 recovers both CPU graph operations after clearing the same ${lossName} array`, async () => {
    const device = createNullWebGPUDevice();
    try {
      for (const maxP of [false, true]) {
        const flag = Uint32Array.of(1);
        const input = maxP ? createAuditMaxPInput() : createAuditAZPInput();
        const options = maxP ? createAuditMaxPProps() : createAuditAZPProps();
        const arrays = getRegionalizationOperationInputs({...input, status: {[lossName]: flag}});
        const graph = new GPUGraph({id: `cpu-recovery-${maxP}`});
        const vectors = Object.fromEntries(
          Object.entries(arrays).map(([name, array]) => {
            graph.addInput(name, gpuGraph.vector(array.format, {length: array.values.length}));
            return [name, makeCPUVector(array.format, array.values)];
          })
        );
        const weights = {
          ...options.weights,
          status: {
            [lossName]: lossName === 'overflow' ? 'weightsOverflow' : 'weightsCandidateOverflow'
          }
        };
        graph.add(
          maxP
            ? new CPUMaxPRegionsOperation({...createAuditMaxPProps(), weights})
            : new CPUAZPRegionsOperation({...createAuditAZPProps(), weights})
        );
        const compiled = graph.compile(device, {
          outputs: ['labels', 'objective', 'status'],
          outputLocations: {labels: 'cpu', objective: 'cpu', status: 'cpu'}
        });
        try {
          for (const flagValue of [1, 0, 1]) {
            flag[0] = flagValue;
            const result = (await compiled.run({inputs: vectors})) as unknown as Record<
              string,
              CPUVector
            >;
            const labels = Array.from(getCPUVectorRows(result['labels']));
            const status = Array.from(getCPUVectorRows(result['status']));
            const objective = getCPUVectorRows(result['objective'])[0];
            expect(labels).toEqual(
              flagValue
                ? new Array(options.rowCount).fill(4294967295)
                : maxP
                  ? [0, 0, 1, 1, 2, 2, 3, 3]
                  : [0, 0, 1, 1, 1, 1, 2, 2, 3]
            );
            expect(status).toEqual(
              flagValue ? new Array(maxP ? 6 : 4).fill(0) : maxP ? [4, 1, 0, 0, 0, 0] : [1, 1, 0, 0]
            );
            if (flagValue) expect(Number.isNaN(objective)).toBe(true);
            else expect(objective).toBe(maxP ? 28 : 24);
          }
        } finally {
          compiled.destroy();
        }
      }
    } finally {
      device.destroy();
    }
  });
}
