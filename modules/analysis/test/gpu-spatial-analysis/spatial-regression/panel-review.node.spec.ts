// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUSpatialPanelLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood';
import {
  computeSpatialPanelLagMaximumLikelihoodOnCPU,
  CPUSpatialPanelLagMaximumLikelihoodOperation
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-lag-maximum-likelihood-operation';
import {
  createOffsetPanelLagInput,
  expectOffsetPanelLag,
  createConstantPanelLagInput,
  createPanelReviewLagProps,
  createBoundaryPanelLagInput,
  expectBoundaryPanelLag,
  createExtremeOffsetPanelLagInput,
  EXTREME_OFFSET_PANEL_LAG_EXPECTED,
  createExtremeFirstPanelLagInput
} from './panel-review.test-utils';

it('review 2 R4 CPU unit operation ignores extreme pooled rank and matches within SVD', async () => {
  const input = createExtremeOffsetPanelLagInput();
  const operation = new CPUSpatialPanelLagMaximumLikelihoodOperation({
    entityCount: 4,
    periodCount: 4,
    predictorCount: 2,
    effects: 'unit',
    predictors: 'predictors',
    response: 'response',
    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    output: {
      coefficients: 'coefficients',
      covariance: 'covariance',
      summary: 'summary',
      fitted: 'fitted',
      residuals: 'residuals',
      status: 'status'
    }
  });
  const rows = {
    predictors: input.predictors,
    response: input.response,
    offsets: input.weights.offsets,
    neighbors: input.weights.neighbors,
    weights: input.weights.weights
  };
  const result = await operation.execute({
    inputs: Object.fromEntries(
      Object.entries(rows).map(([name, values]) => [
        name,
        {
          format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
          length: values.length,
          chunks: [{values, length: values.length}],
          pack: () => values
        }
      ])
    )
  });
  const actual = {
    coefficients: result['coefficients'],
    covariance: result['covariance'],
    summary: result['summary'],
    fitted: result['fitted'],
    residuals: result['residuals'],
    status: result['status']
  } as Record<
    'coefficients' | 'covariance' | 'summary' | 'fitted' | 'residuals' | 'status',
    ArrayLike<number>
  >;
  expect(Array.from(actual.status), JSON.stringify(Array.from(actual.summary))).toEqual([0]);
  expectOffsetPanelLag(actual, EXTREME_OFFSET_PANEL_LAG_EXPECTED);
  expectOffsetPanelLag(
    computeSpatialPanelLagMaximumLikelihoodOnCPU(input, {
      periodCount: 4,
      predictorCount: 2,
      effects: 'unit'
    }),
    EXTREME_OFFSET_PANEL_LAG_EXPECTED
  );
});

for (const fitMode of ['reference', 'tiled'] as const)
  it(`review kernels parse with all completeness scalars within eight storage bindings: ${fitMode}`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      recordWGSLKernelNodes(graph);
      const props = createPanelReviewLagProps(graph, createConstantPanelLagInput(), 3, 1);
      const scalar = (name: string) => createTransientView(graph, `review-${name}`, 'uint32', 1);
      const nodes = new GPUSpatialPanelLagMaximumLikelihood({
        ...props,
        fitMode,
        eigenvalues:
          fitMode === 'tiled'
            ? createTransientView(graph, 'review-spectrum', 'float32', 4)
            : undefined,
        weights: {
          ...props.weights,
          status: {
            count: scalar('count'),
            requiredCount: scalar('required'),
            overflow: scalar('overflow'),
            candidateOverflow: scalar('candidate')
          }
        }
      }).getCommandNodes(graph);
      for (const node of nodes) {
        const record = getWGSLKernelNodeRecord(graph, node.id);
        if (!record) continue;
        expect(record.bindings.length, node.id).toBeLessThanOrEqual(8);
        expect(() => new WgslReflect(record.source), node.id).not.toThrow();
      }
    } finally {
      device.destroy();
    }
  });

for (const fitMode of ['reference', 'tiled'] as const)
  for (const field of ['count', 'requiredCount', 'overflow'] as const)
    for (const invalid of ['foreign', 'alias'] as const)
      it(`review 2 R7 rejects ${invalid} ${field} on ${fitMode}`, () => {
        const device = createNullWebGPUDevice();
        try {
          const graph = new GPUCommandGraph(device),
            otherGraph = new GPUCommandGraph(device);
          const props = createPanelReviewLagProps(graph, createConstantPanelLagInput(), 3, 1);
          const eigenvalues =
            fitMode === 'tiled'
              ? createTransientView(graph, 'ownership-spectrum', 'float32', 4)
              : undefined;
          const own = createTransientView(graph, 'ownership-count', 'uint32', 1);
          const make = (view: typeof own) =>
            new GPUSpatialPanelLagMaximumLikelihood({
              ...props,
              fitMode,
              eigenvalues,
              weights: {...props.weights, status: {[field]: view}}
            });
          expect(() => make(own).getCommandNodes(graph)).not.toThrow();
          if (invalid === 'alias') expect(() => make(props.output.status)).toThrow();
          else {
            const foreign = createTransientView(otherGraph, 'foreign-count', 'uint32', 1);
            expect(() => make(foreign).getCommandNodes(graph)).toThrow();
          }
        } finally {
          device.destroy();
        }
      });

for (const budget of [1, 32, 64])
  it(`R6 CPU retained endpoint observes explicit search budget ${budget}`, () => {
    const result = computeSpatialPanelLagMaximumLikelihoodOnCPU(createBoundaryPanelLagInput(), {
      periodCount: 4,
      predictorCount: 1,
      effects: 'unit',
      maximumSearchIterations: budget
    });
    expect(
      result.status[0],
      JSON.stringify({summary: Array.from(result.summary), rho: result.coefficients[2]})
    ).toBe(budget === 1 ? 3 : 0);
    expectBoundaryPanelLag(result, budget);
  });

it('R5 CPU refuses an exactly time-invariant predictor', () => {
  const result = computeSpatialPanelLagMaximumLikelihoodOnCPU(createConstantPanelLagInput(), {
    periodCount: 3,
    predictorCount: 1,
    effects: 'unit'
  });
  expect(result.status[0]).toBe(2);
  for (const port of ['covariance', 'fitted', 'residuals'] as const)
    expect(Array.from(result[port]).every(Number.isNaN), port).toBe(true);
});

it('R5 emitted pack preserves twelve exactly zero within predictor values', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device),
      memory = createCPUGraphMemory();
    recordWGSLKernelNodes(graph);
    const input = createConstantPanelLagInput(),
      props = createPanelReviewLagProps(graph, input, 3, 1);
    memory.write(props.predictors, input.predictors);
    memory.write(props.response, input.response);
    memory.write(props.weights.offsets, input.weights.offsets);
    memory.write(props.weights.neighbors, input.weights.neighbors);
    memory.write(props.weights.weights, input.weights.weights);
    const allNodes = new GPUSpatialPanelLagMaximumLikelihood(props).getCommandNodes(graph);
    const nodes = allNodes.filter(
      node =>
        node.id.includes('-within-pack-') ||
        ['-pooled-means', '-dense-weights', '-pack'].some(suffix => node.id.endsWith(suffix))
    );
    executeCommandNodesOnCPU(
      nodes.map(node => {
        const record = getWGSLKernelNodeRecord(graph, node.id)!;
        return createWGSLKernelNode(graph, {
          ...record,
          id: `${node.id}-interpreter`,
          workgroupSize: record.workgroupSize[0],
          declarations: `fn reviewLdexp(value:f32,power:i32)->f32{return value*exp2(f32(power));}\n${(record.declarations ?? '').replace(/\bldexp\(/g, 'reviewLdexp(')}`
        });
      }),
      memory
    );
    const pack = getWGSLKernelNodeRecord(graph, nodes.find(node => node.id.endsWith('-pack'))!.id)!;
    const values = memory.readFloat32(pack.bindings.find(binding => binding.name === 'data')!.view);
    expect(Array.from({length: 12}, (_, row) => values[row * 8 + 1])).toEqual(
      new Array(12).fill(0)
    );
    const fit = getWGSLKernelNodeRecord(
      graph,
      allNodes.find(node => node.id.endsWith('-unit-fit'))!.id
    )!;
    // The real device already confirmed status 2. A zero scalar Gram cannot be positive
    // definite. Substitute only that shared factorization (the interpreter cannot run its
    // nested structs), retaining the actual panel control flow and every publication writer.
    const declarations = fit.declarations!.replace(
      /fn factorEquilibrated_1\([\s\S]*?\n}\n/,
      'fn factorEquilibrated_1(a:ptr<function,array<f32,1>>,scale:ptr<function,array<f32,1>>)->bool{return false;}\n'
    );
    expect(declarations).not.toBe(fit.declarations);
    executeCommandNodesOnCPU(
      [
        createWGSLKernelNode(graph, {
          ...fit,
          id: 'r5-singular-control-flow',
          workgroupSize: fit.workgroupSize[0],
          declarations: `fn reviewLdexp(value:f32,power:i32)->f32{return value*exp2(f32(power));}\n${declarations.replace(/\bldexp\(/g, 'reviewLdexp(')}`
        }),
        ...allNodes.filter(node => node.id.endsWith('-publish') || node.id.endsWith('-residuals'))
      ],
      memory
    );
    expect(Array.from(memory.readUint32(props.output.status))).toEqual([2]);
    // Singular conditional fits retain their initial diagnostic iterate. The old R5 test
    // incorrectly equated that status-2 contract with the all-NaN exact-pole contract.
    expect(Array.from(memory.readFloat32(props.output.coefficients))).toEqual([
      NaN,
      0,
      Math.fround(0.1),
      0
    ]);
    expect(Array.from(memory.readFloat32(props.output.summary))).toEqual([
      0,
      Math.fround(-3.402823e38),
      0,
      0,
      2,
      0
    ]);
    for (const name of ['covariance', 'fitted', 'residuals'] as const) {
      const actual = Array.from(memory.readFloat32(props.output[name]));
      expect(actual.every(Number.isNaN), `${name}: ${JSON.stringify(actual)}`).toBe(true);
    }
  } finally {
    device.destroy();
  }
});

it('review 2 R5 an extreme first predictor does not erase third-period within values', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device),
      memory = createCPUGraphMemory();
    recordWGSLKernelNodes(graph);
    const input = createExtremeFirstPanelLagInput();
    const props = createPanelReviewLagProps(graph, input, 3, 1);
    memory.write(props.predictors, input.predictors);
    memory.write(props.response, input.response);
    memory.write(props.weights.offsets, input.weights.offsets);
    memory.write(props.weights.neighbors, input.weights.neighbors);
    memory.write(props.weights.weights, input.weights.weights);
    const nodes = new GPUSpatialPanelLagMaximumLikelihood(props)
      .getCommandNodes(graph)
      .filter(
        node =>
          node.id.includes('-within-pack-') ||
          ['-pooled-means', '-dense-weights', '-pack'].some(suffix => node.id.endsWith(suffix))
      );
    executeCommandNodesOnCPU(
      nodes.map(node => {
        const record = getWGSLKernelNodeRecord(graph, node.id)!;
        return createWGSLKernelNode(graph, {
          ...record,
          id: `${node.id}-interpreter`,
          workgroupSize: record.workgroupSize[0],
          declarations: `fn reviewLdexp(value:f32,power:i32)->f32{return value*exp2(f32(power));}\n${(record.declarations ?? '').replace(/\bldexp\(/g, 'reviewLdexp(')}`
        });
      }),
      memory
    );
    const record = getWGSLKernelNodeRecord(
      graph,
      nodes.find(node => node.id.endsWith('-pack'))!.id
    )!;
    const values = memory.readFloat32(
      record.bindings.find(binding => binding.name === 'data')!.view
    );
    // Exact mean is +/-1/3; the final within value is +/-2/3. The first observation
    // must not become an arithmetic reference that rounds away the later unit value.
    for (let entity = 0; entity < 4; entity++) {
      const value = values[(8 + entity) * 8 + 1];
      expect(Number.isFinite(value)).toBe(true);
      expect(Math.abs(value - (entity % 2 ? -2 / 3 : 2 / 3))).toBeLessThan(2 ** -24);
    }
    const means = memory.readFloat32(
      record.bindings.find(binding => binding.name === 'entityCenters')!.view
    );
    for (let entity = 0; entity < 4; entity++)
      for (let component = 0; component < 2; component++) {
        const offset = (entity * 2 + component) * 2;
        const expected =
          component === 0 ? [8 / 3, 8 / 3, 2, 11 / 3][entity] : entity % 2 ? -1 / 3 : 1 / 3;
        expect(Number.isFinite(means[offset] + means[offset + 1])).toBe(true);
        expect(Math.abs(means[offset] + means[offset + 1] - expected)).toBeLessThan(1e-13);
      }
    expect(means[16]).toBe(0);
  } finally {
    device.destroy();
  }
});

it('R4 CPU unit fit matches the identifiable within SVD oracle despite collinear entity offsets', () => {
  expectOffsetPanelLag(
    computeSpatialPanelLagMaximumLikelihoodOnCPU(createOffsetPanelLagInput(), {
      periodCount: 4,
      predictorCount: 2,
      effects: 'unit'
    })
  );
});

it('R4 reference unit initialization never depends on the unused pooled factorization', () => {
  // Exact pooled Gram entries from sixteen common +/-2048 means plus the orthogonal
  // +/-1 within columns. Its second normalized pivot is below the 1e-6 squared floor.
  const pooledDiagonal = 16 * 2048 ** 2 + 16,
    pooledCross = 16 * 2048 ** 2;
  expect(1 - (pooledCross / pooledDiagonal) ** 2).toBeLessThan(1e-6);
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device),
      memory = createCPUGraphMemory();
    recordWGSLKernelNodes(graph);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `r4-${serial++}`, format, length);
    const props = {
      effects: 'unit' as const,
      periodCount: 4,
      predictorCount: 2,
      predictors: view('float32', 32),
      response: view('float32', 16),
      weights: {
        offsets: view('uint32', 5),
        neighbors: view('uint32', 8),
        weights: view('float32', 8)
      },
      output: {
        coefficients: view('float32', 5),
        covariance: view('float32', 16),
        summary: view('float32', 6),
        fitted: view('float32', 16),
        residuals: view('float32', 16),
        status: view('uint32', 1)
      }
    };
    const nodes = new GPUSpatialPanelLagMaximumLikelihood(props).getCommandNodes(graph);
    const record = getWGSLKernelNodeRecord(
      graph,
      nodes.find(node => node.id.endsWith('-unit-fit'))!.id
    )!;
    const data = record.bindings.find(binding => binding.name === 'data')!.view;
    const packed = new Float32Array(data.length);
    // Exact packed temporal contrasts: both within columns are orthogonal +/-1 vectors,
    // while both entity means carry the identical +/-2048 offset. Within Gram=diag(16,16).
    for (let row = 0; row < 16; row++)
      packed.set(
        [
          0,
          [1, -1, 1, -1][Math.floor(row / 4)],
          [1, 1, -1, -1][Math.floor(row / 4)],
          0,
          0,
          1,
          row % 2 ? 2048 : -2048,
          row % 2 ? 2048 : -2048,
          0,
          0
        ],
        row * 10
      );
    memory.write(data, packed);
    const output = view('float32', 3);
    const start = record.body.indexOf('var beta:');
    const end = record.body.indexOf('var rho=');
    expect(start).toBeGreaterThan(0);
    expect(end).toBeGreaterThan(start);
    const quasiMoment = record
      .declarations!.split('\n')
      .find(line => line.startsWith('fn getQuasiMoment('))!;
    // Deliberately fail the delegated pooled factorization. No solver emulation is evidence
    // for numerical results: this probe checks only that a unit fit never calls that gate.
    const probe = createWGSLKernelNode(graph, {
      id: 'r4-unused-pooled-gate',
      operation: 'PanelReview',
      invocationCount: 1,
      bindings: [
        ...record.bindings.filter(binding => ['data', 'results'].includes(binding.name)),
        {name: 'output', view: output, type: 'f32', access: 'read_write'}
      ],
      declarations: `${quasiMoment}\nfn factorEquilibrated_2(a:ptr<function,array<f32,4>>,scale:ptr<function,array<f32,2>>)->bool{return false;}\nfn solveEquilibrated_2(a:ptr<function,array<f32,4>>,scale:ptr<function,array<f32,2>>,value:ptr<function,array<f32,2>>) {}`,
      body: `let base=0u;${record.body.slice(start, end)}output[outputOffset]=1.0;output[outputOffset+1u]=deviationMoments[0];output[outputOffset+2u]=deviationMoments[5];`
    });
    executeCommandNodesOnCPU([probe], memory);
    expect(Array.from(memory.readFloat32(output))).toEqual([1, 16, 16]);
  } finally {
    device.destroy();
  }
});

it('W5-36 counts the accessible CSR span after a leading unused slot', () => {
  // Use the ordinary four-entity fixture: eight reachable edges, one unused leading slot.
  const base = createOffsetPanelLagInput();
  const weights = {
    ...base.weights,
    offsets: Uint32Array.of(1, 3, 5, 7, 9),
    neighbors: Uint32Array.of(0, ...base.weights.neighbors),
    weights: Float32Array.of(0, ...base.weights.weights)
  };
  for (const count of [8, 9]) {
    const actual = computeSpatialPanelLagMaximumLikelihoodOnCPU(
      {
        ...base,
        weights: {
          ...weights,
          status: {count: Uint32Array.of(count), requiredCount: Uint32Array.of(count)}
        }
      },
      {periodCount: 4, predictorCount: 2, effects: 'random', hausman: true}
    );
    expect(actual.status[0]).toBe(count === 8 ? 0 : 1);
    if (count === 9) {
      expect(actual.hausman!.status[0]).toBe(1);
      expect(Array.from(actual.coefficients).every(Number.isNaN)).toBe(true);
    }
  }
});
