// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {getRegionalizationWeightsReadout} from '../../../../../examples/deck/spatial-analysis-showcase/src/scenes/regression/regionalization-readout';
import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  GPULocalSpatialHeteroscedasticity,
  GPUMultivariateLocalGeary
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-local-variability';
import {GPUNeighborSearch} from '../../../src/gpu-spatial-analysis/neighbor-search/gpu-neighbor-search';
import {GPUSpatialWeightsAlgebra} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra';
import {GPUChunkedSpatialWeightsAlgebra} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-algebra';
import {GPUSpatialWeightsAlgebraOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra-operation';
import {
  CPUSpatialWeightsAlgebraOperation,
  computeSpatialWeightsAlgebraOnCPU
} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-algebra-operation';
import {GPUContiguityWeightsOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-contiguity-weights-operation';
import {GPUContiguityWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-contiguity-weights';
import {GPUSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transform';
import {GPUSpatialLag} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-lag';
import {GPUSpatialWeightsMinimumSpanningTree} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-spatial-weights-minimum-spanning-tree';
import {
  CPUContiguityWeightsOperation,
  computeContiguityWeightsOnCPU
} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-contiguity-weights-operation';
import {createSubdividedContiguityFixture} from './contiguity-loss.test-utils';
import {buildMoranScatter} from '../../../../../examples/deck/spatial-analysis-showcase/src/scenes/weights/global-autocorrelation.stats';
import {runExample as runForest} from '../../../../../modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-weights-minimum-spanning-tree.example';
import {runExample as runSkater} from '../../../../../modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/skater-regions.example';

function scene() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <F extends 'uint32' | 'float32' | 'float32x2'>(format: F, length: number) =>
    createTransientView(graph, `final2-${serial++}`, format, length);
  const weights = (rows = 4, capacity = 6) => ({
    offsets: view('uint32', rows + 1),
    neighbors: view('uint32', capacity),
    weights: view('float32', capacity)
  });
  const output = (rows = 4, capacity = 6) => ({
    ...weights(rows, capacity),
    status: {overflow: view('uint32', 1), candidateOverflow: view('uint32', 1)}
  });
  return {device, graph, view, weights, output};
}

for (const kind of ['losh', 'geary'] as const)
  it(`catalog-docs:G1 ${kind} captures inherited CSR and mask getters once`, () => {
    const {device, graph, view, weights} = scene();
    try {
      const initial = weights();
      const mask = view('uint32', 4);
      let current = initial;
      let currentMask = mask;
      let reads = 0;
      const port = Object.create({
        get offsets() {
          reads++;
          return current.offsets;
        },
        get neighbors() {
          reads++;
          return current.neighbors;
        },
        get weights() {
          reads++;
          return current.weights;
        }
      });
      const props = Object.assign(
        Object.create({
          get mask() {
            reads++;
            return currentMask;
          }
        }),
        {
          weights: port,
          values: view('float32', 4),
          variableCount: 1,
          output: {statistic: view('float32', 4), status: view('uint32', 4)}
        }
      );
      const contributor =
        kind === 'losh'
          ? new GPULocalSpatialHeteroscedasticity(props)
          : new GPUMultivariateLocalGeary(props);
      current = weights();
      currentMask = view('uint32', 4);
      expect((contributor.props.weights as typeof initial).offsets).toBe(initial.offsets);
      expect(contributor.props.mask).toBe(mask);
      const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
      expect(sources.some(record => record.source.includes('mask[maskOffset'))).toBe(true);
      expect(reads).toBe(4);
    } finally {
      device.destroy();
    }
  });

for (const [name, run] of [
  ['forest', runForest],
  ['skater', runSkater]
] as const)
  it(`core:G2 real ${name} lens builds with producer loss ports`, async () => {
    const {device, graph, view} = scene();
    try {
      await expect(
        run(device, {
          graph,
          tracts: {
            kind: 'polygons',
            positions: view('float32x2', 64),
            ringOffsets: view('uint32', 17),
            polygonOffsets: view('uint32', 17),
            featureOffsets: view('uint32', 17)
          },
          values: view('float32', 16),
          parameters: view('uint32', 2)
        })
      ).resolves.toBeDefined();
    } finally {
      device.destroy();
    }
  });

for (const inherited of ['invalidCount', 'candidateOverflow'] as const)
  it(`spatial-2:G5 rejects inherited ${inherited} writer aliases`, () => {
    const {device, view, output} = scene();
    try {
      const shared = view('uint32', 1);
      const status = Object.assign(
        Object.create({
          get [inherited]() {
            return shared;
          }
        }),
        {
          overflow: view('uint32', 1),
          [inherited === 'invalidCount' ? 'candidateOverflow' : 'invalidCount']: shared
        }
      );
      expect(
        () =>
          new GPUNeighborSearch({
            mode: 'radius',
            gridSize: [2, 2],
            positions: view('float32x2', 4),
            parameters: view('float32', 12),
            output: {...output(), status}
          })
      ).toThrow(/overlap/);
    } finally {
      device.destroy();
    }
  });

it('core:G2 forest candidate destinations are independent for packed tree outputs', () => {
  const {device, graph, view, weights} = scene();
  try {
    const flags = view('uint32', 12);
    const source = {...weights(), status: {candidateOverflow: view('uint32', 1)}};
    const create = (byteOffset: number) =>
      new GPUSpatialWeightsMinimumSpanningTree({
        weights: source,
        values: view('float32', 4),
        output: {
          treeEdgeFlags: graph.createDataView(flags.buffer, {
            format: 'uint32',
            length: 6,
            byteOffset
          }),
          componentLabels: view('uint32', 4),
          edges: {
            ids: view('uint32', 3),
            status: {count: view('uint32', 1), overflow: view('uint32', 1)}
          }
        }
      });
    const first = create(0);
    const second = create(24);
    expect(second.output.edges!.status.candidateOverflow).toBeDefined();
    expect(second.output.edges!.status.candidateOverflow!.buffer).not.toBe(
      first.output.edges!.status.candidateOverflow!.buffer
    );
  } finally {
    device.destroy();
  }
});

for (const Backend of [GPUContiguityWeightsOperation, CPUContiguityWeightsOperation])
  it(`spatial-4:G2 ${Backend.name} refuses omitted candidate destination`, () => {
    expect(
      () =>
        new Backend({
          criterion: 'queen',
          featureCount: 2,
          capacity: 8,
          pairCapacity: 1,
          polygons: {positions: 'p', ringOffsets: 'r', polygonOffsets: 'o', featureOffsets: 'f'},
          output: {
            offsets: 'offsets',
            neighbors: 'neighbors',
            weights: 'weights',
            status: {overflow: 'overflow'}
          }
        } as never)
    ).toThrow(/candidateOverflow/);
  });
for (const Backend of [GPUSpatialWeightsAlgebraOperation, CPUSpatialWeightsAlgebraOperation])
  it(`spatial-4:G2 ${Backend.name} refuses hidden inherited loss`, () => {
    expect(
      () =>
        new Backend({
          operation: 'subgraph',
          rowCount: 4,
          capacity: 8,
          mask: 'mask',
          weights: {
            offsets: 'o',
            neighbors: 'n',
            weights: 'w',
            status: {candidateOverflow: 'loss'}
          },
          output: {
            offsets: 'out-o',
            neighbors: 'out-n',
            weights: 'out-w',
            status: {overflow: 'overflow'}
          }
        })
    ).toThrow(/candidateOverflow/);
  });

it('spatial-4:G6 rejects shared loss words across chunks before allocation and permits corrected retry', () => {
  const {device, graph, view, weights} = scene();
  try {
    const input = {
      rowCount: 4,
      chunks: [0, 2].map(rowStart => ({rowStart, rowCount: 2, weights: weights(2, 4)}))
    };
    const shared = view('uint32', 1);
    const chunks = [1, 4].map((capacity, index) => ({
      rowStart: index * 2,
      rowCount: 2,
      weights: {...weights(2, capacity), status: {overflow: shared}}
    }));
    expect(
      () =>
        new GPUChunkedSpatialWeightsAlgebra({
          operation: 'selfWeight',
          selfWeight: 1,
          weights: input,
          output: {chunks}
        })
    ).toThrow(/overlap|share/);
    chunks[1].weights.status.overflow = view('uint32', 1);
    expect(() =>
      new GPUChunkedSpatialWeightsAlgebra({
        operation: 'selfWeight',
        selfWeight: 1,
        weights: input,
        output: {chunks}
      }).getCommandNodes(graph)
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

for (const getterBacked of [false, true])
  it(`spatial-4:G7 captures ${getterBacked ? 'getter' : 'own'} discriminator and nested output maps`, () => {
    const {device, graph, view, weights, output} = scene();
    try {
      let operation = 'selfWeight';
      let reads = 0;
      const props = Object.assign(
        Object.create(
          getterBacked
            ? {
                get operation() {
                  reads++;
                  return operation;
                }
              }
            : {}
        ),
        {selfWeight: 1, weights: weights(), output: output()},
        getterBacked ? {} : {operation}
      );
      const producer = new GPUSpatialWeightsAlgebra(props);
      operation = 'union';
      if (!getterBacked) props.operation = 'union';
      props.output.status.overflow = view('uint32', 1);
      expect(producer.props.operation).toBe('selfWeight');
      expect(reads).toBe(getterBacked ? 1 : 0);
      expect(
        getGPUCommandNodeKernelSources(producer.getCommandNodes(graph)).some(record =>
          record.source.includes('var inserted = false')
        )
      ).toBe(true);
    } finally {
      device.destroy();
    }
  });

for (const Backend of [GPUSpatialWeightsAlgebraOperation, CPUSpatialWeightsAlgebraOperation])
  it(`final2 ${Backend.name} snapshots inherited input and output value names`, () => {
    let operation = 'selfWeight';
    let candidateName = 'loss';
    const weights = {
      offsets: 'o',
      neighbors: 'n',
      weights: 'w',
      status: Object.create({
        get candidateOverflow() {
          return candidateName;
        }
      })
    };
    const output = {
      offsets: 'out-o',
      neighbors: 'out-n',
      weights: 'out-w',
      status: {overflow: 'overflow', candidateOverflow: 'candidate'}
    };
    const props = Object.assign(
      Object.create({
        get operation() {
          return operation;
        }
      }),
      {weights, selfWeight: 1, rowCount: 4, capacity: 12, output}
    );
    const contributor = new Backend(props);
    operation = 'union';
    candidateName = 'mutated-input';
    output.status.candidateOverflow = 'mutated-output';
    const names = contributor.contract.values?.map(value => value.value);
    expect(names).toContain('loss');
    expect(names).toContain('candidate');
    expect(names).not.toContain('mutated-input');
    expect(names).not.toContain('mutated-output');
  });

it('final2 search snapshots the validated status maps before later mutations', () => {
  const {device, graph, view, output} = scene();
  try {
    const destination = output();
    const original = destination.status.candidateOverflow;
    const props = {
      mode: 'radius' as const,
      gridSize: [2, 2] as const,
      positions: view('float32x2', 4),
      parameters: view('float32', 12),
      output: {...destination, status: {...destination.status, invalidCount: view('uint32', 1)}}
    };
    const contributor = new GPUNeighborSearch(props);
    props.output.status.candidateOverflow = props.output.status.invalidCount;
    expect(contributor.props.output.status.candidateOverflow).toBe(original);
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('final2 the contiguity contributor snapshots its validated loss maps', () => {
  const {device, graph, view, output} = scene();
  try {
    const destination = output();
    const original = destination.status.overflow;
    const producer = new GPUContiguityWeights({
      criterion: 'queen',
      polygons: {
        kind: 'polygons',
        positions: view('float32x2', 16),
        ringOffsets: view('uint32', 5),
        polygonOffsets: view('uint32', 5),
        featureOffsets: view('uint32', 5)
      },
      output: destination
    });
    destination.status.overflow = destination.status.candidateOverflow;
    expect(producer.props.output.status.overflow).toBe(original);
    expect(() => producer.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('spatial-4:G9 reserves final overflow for final capacity and reports inherited loss as candidate loss', () => {
  const weights = {
    offsets: Uint32Array.of(0, 1, 2),
    neighbors: Uint32Array.of(1, 0),
    weights: Float32Array.of(1, 1),
    status: {overflow: Uint32Array.of(1)}
  };
  for (const capacity of [4, 8]) {
    const result = computeSpatialWeightsAlgebraOnCPU(
      {weights, mask: Uint32Array.of(0, 0)},
      {operation: 'subgraph', capacity}
    );
    expect(Array.from(result.requiredCount)).toEqual([0]);
    expect(Array.from(result.overflow)).toEqual([0]);
    expect(Array.from(result.candidateOverflow)).toEqual([1]);
  }
});

it('core:G3 regionalization does not call candidate-only topology complete', () => {
  const source = readFileSync(
    'examples/deck/spatial-analysis-showcase/src/scenes/regression/regionalization.compute.ts',
    'utf8'
  );
  expect([...source.matchAll(/getRegionalizationWeightsReadout\(/g)]).toHaveLength(2);
  for (const [overflow, candidate] of [
    [0, 1],
    [1, 0],
    [0, 0]
  ]) {
    const readout = getRegionalizationWeightsReadout(overflow, candidate);
    expect(readout.complete).toBe(overflow === 0 && candidate === 0);
    expect(readout.capacity).toBe(
      overflow || candidate ? 'Incomplete weights: raise neighbor or pair capacity' : 'ok'
    );
    expect(readout.unavailable).toBe('Unavailable (incomplete weights)');
  }
});

it('spatial-1:G1 display classification refuses NaN means and lags', () => {
  const source = readFileSync(
    'examples/deck/spatial-analysis-showcase/src/scenes/weights/global-autocorrelation-display.ts',
    'utf8'
  );
  const classify = /const CLASSIFY_BODY =[^`]*`([\s\S]*?)`/.exec(source)![1];
  expect(/isFinite|isNan|isInf|0x7f800000/.test(classify)).toBe(true);
});

it('spatial-1:G1 the actual scatter consumer omits unavailable quadrants and recovers LH/HL', () => {
  const incomplete = buildMoranScatter(
    [1, 4],
    [Number.NaN, Number.NaN],
    [0, 1, 2],
    Number.NaN,
    Number.NaN
  );
  expect(incomplete.rows.length).toBe(0);
  expect(incomplete.counts).toEqual([0, 0, 0, 0]);
  const complete = buildMoranScatter([1, 4], [4, 1], [0, 1, 2], 2.5, 2.25);
  expect(Array.from(complete.quadrant)).toEqual([1, 3]);
  expect(complete.counts).toEqual([0, 1, 0, 1]);
});

it('spatial-4:G3 explorer and guide retain candidate diagnostics', () => {
  const regions = readFileSync(
    'examples/deck/spatial-analysis-explorer/modes/regions-mode.ts',
    'utf8'
  );
  const coverage = readFileSync(
    'examples/deck/spatial-analysis-explorer/modes/coverage-mode.ts',
    'utf8'
  );
  const guide = readFileSync('docs/api/gpu-spatial-analysis/operators-weights.md', 'utf8');
  expect(regions.includes('candidateOverflow')).toBe(true);
  expect(coverage.includes('candidateOverflow')).toBe(true);
  expect(guide.includes('weights: contiguity.outputWeights')).toBe(true);
});

it('spatial-4:G3 the executable lag guide binds both producer loss words', () => {
  const {device, graph, view, weights} = scene();
  try {
    const guide = readFileSync('docs/api/gpu-spatial-analysis/operators-weights.md', 'utf8');
    const snippet = /```ts\n(const contiguity = new GPUContiguityWeights[^]*?)```/.exec(guide)![1];
    const contributors: {getCommandNodes: (graph: GPUCommandGraph) => readonly unknown[]}[] = [];
    Function(
      'GPUContiguityWeights',
      'GPUSpatialWeightsTransform',
      'GPUSpatialLag',
      'graph',
      'polygons',
      'weights',
      'overflow',
      'values',
      'lag',
      snippet
    )(
      GPUContiguityWeights,
      GPUSpatialWeightsTransform,
      GPUSpatialLag,
      {add: (contributor: (typeof contributors)[number]) => contributors.push(contributor)},
      {
        kind: 'polygons',
        positions: view('float32x2', 16),
        ringOffsets: view('uint32', 5),
        polygonOffsets: view('uint32', 5),
        featureOffsets: view('uint32', 5)
      },
      weights(),
      view('uint32', 1),
      view('float32', 4),
      view('float32', 4)
    );
    const sources = getGPUCommandNodeKernelSources(contributors[2].getCommandNodes(graph));
    expect(sources.filter(record => record.id.includes('-incomplete-')).length).toBe(2);
  } finally {
    device.destroy();
  }
});

it('spatial-4:G3 both explorer readback guards reject candidate-only loss and recover', () => {
  const regions = readFileSync(
    'examples/deck/spatial-analysis-explorer/modes/regions-mode.ts',
    'utf8'
  );
  const coverage = readFileSync(
    'examples/deck/spatial-analysis-explorer/modes/coverage-mode.ts',
    'utf8'
  );
  const regionsGuard = /weightsComplete =\s*(new Uint32Array\(bytes,[^;]+);/.exec(regions)![1];
  const coverageGuard = /weightsComplete =\s*(at\(layoutWords.weightOverflow\)[^;]+);/.exec(
    coverage
  )![1];
  for (const candidate of [0, 1, 0]) {
    const words = Uint32Array.of(0, 0, 0, candidate);
    expect(Function('bytes', 'offset', `return ${regionsGuard}`)(words.buffer, 0)).toBe(
      candidate === 0
    );
    expect(
      Function(
        'at',
        'layoutWords',
        `return ${coverageGuard}`
      )((index: number) => words[index], {weightOverflow: 2, weightCandidateOverflow: 3})
    ).toBe(candidate === 0);
  }
});

for (const criterion of ['queen', 'rook'] as const)
  it(`spatial-4:G3 ${criterion} reproduces candidate-only loss at the explorer budgets`, () => {
    const polygons = createSubdividedContiguityFixture();
    const partial = computeContiguityWeightsOnCPU(polygons, {
      criterion,
      capacity: 72,
      pairCapacity: 288
    });
    expect(Array.from(partial.overflow)).toEqual([0]);
    expect(Array.from(partial.candidateOverflow)).toEqual([1]);
    expect(Array.from(partial.offsets)).toEqual([0, 1, 2, 2]);
    const complete = computeContiguityWeightsOnCPU(polygons, {
      criterion,
      capacity: 72,
      pairCapacity: 1024
    });
    expect(Array.from(complete.overflow)).toEqual([0]);
    expect(Array.from(complete.candidateOverflow)).toEqual([0]);
    expect(Array.from(complete.offsets)).toEqual([0, 1, 3, 4]);
  });
