// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPU_RASTER_FRACTIONAL_COVERAGE_PARAMETER_LENGTH,
  getGPURasterFractionalCoverageParameterValues
} from '../../../src/raster-analysis/gpu-raster-fractional-coverage';
import {addDasymetricRedistributionRecipe} from '../../../src/gpu-spatial-analysis/recipes/dasymetric-redistribution-recipe';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {RecipeTestFixture} from './recipe-harness';
import reference from '../../oracles/fixtures/dasymetric-redistribution.json';

const GRID: GPURasterMetadata = {
  width: 6,
  height: 1,
  affine: [1, 0, 0, 0, 1, 0],
  pixelInterpretation: 'area'
};

async function createFixture(name: string) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return new RecipeTestFixture(device, name);
}

it('F1/G5 prepared one-cell dasymetric input reads padded statuses and restores total 7', async () => {
  const fixture = await createFixture('final1-dasymetric-completeness');
  const count = fixture.output('count', 'uint32', 2);
  count.buffer.write(Uint32Array.of(1, 0xffffffff));
  const required = fixture.output('required', 'uint32', 2);
  const candidate = fixture.output('candidate', 'uint32', 4);
  const candidateView = fixture.graph.createDataView(candidate.view.buffer, {
    format: 'uint32',
    length: 2,
    byteOffset: 4
  });
  const result = addDasymetricRedistributionRecipe(fixture.graph, {
    grid: {...GRID, width: 1},
    sourceTotals: fixture.input('totals', Float32Array.of(7), 'float32', 1),
    targetZones: fixture.input('target-zones', Uint32Array.of(0), 'uint32', 1),
    targetZoneCount: 1,
    pairCapacity: 1,
    ancillary: {
      id: 'ancillary',
      format: 'float32',
      storage: {
        kind: 'buffer',
        values: fixture.input('ancillary', Float32Array.of(1), 'float32', 1)
      }
    },
    source: {
      kind: 'coverage',
      coverage: {
        zoneIds: fixture.input('zones', Uint32Array.of(0), 'uint32', 1),
        cellIds: fixture.input('cells', Uint32Array.of(0), 'uint32', 1),
        fractions: fixture.input('fractions', Float32Array.of(1), 'float32', 1),
        zoneValidity: fixture.input('zones-valid', Uint32Array.of(1), 'uint32', 1),
        status: {
          count: count.view,
          requiredCount: required.view,
          overflow: fixture.input('overflow', Uint32Array.of(0, 1), 'uint32', 2),
          candidateOverflow: candidateView
        }
      }
    }
  });
  const target = publish(fixture, 'published-target', result.outputs.targetTotals);
  const complete = publish(fixture, 'published-complete', result.status.complete);
  const compiled = fixture.graph.compile();
  try {
    for (const phase of ['clean', 'mismatch', 'clean', 'candidate', 'clean']) {
      required.buffer.write(Uint32Array.of(phase === 'mismatch' ? 2 : 1, 0xffffffff));
      const loss = Number(phase === 'candidate');
      candidate.buffer.write(Uint32Array.of(99, loss, 1 - loss, 99));
      submitGraph(fixture.device, compiled, undefined);
      expect(await fixture.readUint32(complete, 1)).toEqual([Number(phase === 'clean')]);
      await expectClose(fixture, target, [phase === 'clean' ? 7 : NaN]);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

function publish<Format extends 'float32' | 'uint32'>(
  fixture: RecipeTestFixture,
  name: string,
  source: GraphDataView<Format>
) {
  const output = fixture.output(name, source.format, source.length);
  const type = source.format === 'float32' ? 'f32' : 'u32';
  fixture.graph.add(
    createWGSLKernelNode(fixture.graph, {
      id: name,
      operation: 'dasymetric-test-readback',
      invocationCount: source.length,
      bindings: [
        {name: 'source', view: source, type, access: 'read'},
        {name: 'output', view: output.view, type, access: 'read_write'}
      ],
      body: 'output[outputOffset+index]=source[sourceOffset+index];'
    })
  );
  return output;
}

function createInputs(
  fixture: RecipeTestFixture,
  fixtureCase = reference.cases[0],
  order?: number[]
) {
  const indices = order ?? fixtureCase.coverage.zoneIds.map((_, index) => index);
  const createMutable = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    values: number[]
  ) => {
    const output = fixture.output(name, format, values.length);
    output.buffer.write(
      format === 'float32' ? Float32Array.from(values) : Uint32Array.from(values)
    );
    return output;
  };
  const count = createMutable('input-count', 'uint32', [indices.length]);
  const requiredCount = createMutable('input-required', 'uint32', [indices.length]);
  const overflow = createMutable('input-overflow', 'uint32', [0]);
  const candidateOverflow = createMutable('input-candidate-overflow', 'uint32', [0]);
  const zoneValidity = createMutable('input-source-validity', 'uint32', [1, 1, 1]);
  const fractions = createMutable(
    'input-fractions',
    'float32',
    indices.map(index => fixtureCase.coverage.fractions[index])
  );
  const coverage = {
    zoneIds: fixture.input(
      'input-sources',
      Uint32Array.from(indices.map(index => fixtureCase.coverage.zoneIds[index])),
      'uint32',
      indices.length
    ),
    cellIds: fixture.input(
      'input-cells',
      Uint32Array.from(indices.map(index => fixtureCase.coverage.cellIds[index])),
      'uint32',
      indices.length
    ),
    fractions: fractions.view,
    zoneValidity: zoneValidity.view,
    status: {
      count: count.view,
      requiredCount: requiredCount.view,
      overflow: overflow.view,
      candidateOverflow: candidateOverflow.view
    }
  };
  const ancillary = createMutable('input-ancillary', 'float32', fixtureCase.ancillary);
  const sourceTotals = createMutable('input-totals', 'float32', reference.sourceTotals);
  const targetZones = createMutable('input-targets', 'uint32', reference.targetZones);
  const mask = createMutable('input-mask', 'uint32', [1, 1, 1, 1, 1, 1]);
  return {
    coverage,
    count,
    requiredCount,
    overflow,
    candidateOverflow,
    zoneValidity,
    fractions,
    ancillary,
    sourceTotals,
    targetZones,
    mask
  };
}

function createRecipe(
  fixture: RecipeTestFixture,
  inputs: ReturnType<typeof createInputs>,
  denominator: 'zone' | 'overlap' = 'overlap',
  pairCapacity = 6
) {
  return addDasymetricRedistributionRecipe(fixture.graph, {
    grid: GRID,
    source: {kind: 'coverage', coverage: inputs.coverage},
    sourceTotals: inputs.sourceTotals.view,
    targetZones: inputs.targetZones.view,
    targetZoneCount: 4,
    ancillary: {
      id: 'ancillary',
      format: 'float32',
      storage: {kind: 'buffer', values: inputs.ancillary.view}
    },
    mask: inputs.mask.view,
    denominator,
    pairCapacity
  });
}

async function expectClose(
  fixture: RecipeTestFixture,
  output: {buffer: import('@luma.gl/core').Buffer},
  expected: (number | null)[]
) {
  const actual = await fixture.readFloat32(output, expected.length);
  for (const [index, value] of expected.entries()) {
    if (value === null || Number.isNaN(value)) expect(actual[index]).toBeNaN();
    else
      expect(
        Math.abs(actual[index] - value),
        `row ${index}: ${actual[index]} versus ${value}`
      ).toBeLessThanOrEqual(1e-4 + 1e-5 * Math.abs(value));
  }
}

it.each(
  reference.cases
)('dasymetric $name matches its independent geometric measure oracle and conserves mass', async fixtureCase => {
  const fixture = await createFixture(`dasymetric-${fixtureCase.name}`);
  try {
    const inputs = createInputs(fixture, fixtureCase);
    const result = createRecipe(
      fixture,
      inputs,
      fixtureCase.denominator === 'zone' ? 'zone' : 'overlap'
    );
    const target = publish(fixture, 'target', result.outputs.targetTotals);
    const support = publish(fixture, 'support', result.outputs.targetSupportWeights);
    const eligible = publish(fixture, 'eligible', result.outputs.sourceEligibleWeights);
    const fractions = publish(fixture, 'fractions', result.outputs.sourceTransferredFractions);
    const allocated = publish(fixture, 'allocated', result.outputs.sourceAllocatedTotals);
    const unallocated = publish(fixture, 'unallocated', result.outputs.sourceUnallocatedTotals);
    const valid = publish(fixture, 'valid', result.outputs.sourceValidity);
    const complete = publish(fixture, 'complete', result.status.complete);
    const empty = publish(fixture, 'empty', result.status.zeroEligibleCount);
    fixture.run();
    await expectClose(fixture, target, fixtureCase.targetTotals);
    await expectClose(fixture, support, fixtureCase.targetSupportWeights);
    await expectClose(fixture, eligible, fixtureCase.sourceEligibleWeights);
    await expectClose(fixture, fractions, fixtureCase.sourceTransferredFractions);
    await expectClose(fixture, allocated, fixtureCase.sourceAllocatedTotals);
    await expectClose(fixture, unallocated, fixtureCase.sourceUnallocatedTotals);
    expect(await fixture.readUint32(valid, 3)).toEqual([1, 1, 0]);
    expect(await fixture.readUint32(complete, 1)).toEqual([1]);
    expect(await fixture.readUint32(empty, 1)).toEqual([1]);
    const transferred = (await fixture.readFloat32(target, 4)).reduce(
      (sum, value) => sum + value,
      0
    );
    const residual = (await fixture.readFloat32(unallocated, 3)).reduce(
      (sum, value) => sum + value,
      0
    );
    expect(transferred + residual).toBeCloseTo(250, 3);
  } finally {
    fixture.destroy();
  }
});

it('dasymetric polygon-source composition preserves fractional boundaries before weighting', async () => {
  const fixture = await createFixture('dasymetric-polygons');
  try {
    const inputs = createInputs(fixture);
    const result = addDasymetricRedistributionRecipe(fixture.graph, {
      grid: GRID,
      source: {
        kind: 'polygons',
        tupleCapacity: 8,
        parameters: fixture.input(
          'coverage-parameters',
          getGPURasterFractionalCoverageParameterValues(GRID),
          'float32',
          GPU_RASTER_FRACTIONAL_COVERAGE_PARAMETER_LENGTH
        ),
        zones: {
          positions: fixture.input(
            'positions',
            Float32Array.from(reference.sourcePositions),
            'float32x2',
            12
          ),
          featureOffsets: fixture.input('features', new Uint32Array([0, 1, 2, 3]), 'uint32', 4),
          polygonOffsets: fixture.input('polygons', new Uint32Array([0, 1, 2, 3]), 'uint32', 4),
          ringOffsets: fixture.input('rings', new Uint32Array([0, 4, 8, 12]), 'uint32', 4)
        }
      },
      sourceTotals: inputs.sourceTotals.view,
      targetZones: inputs.targetZones.view,
      targetZoneCount: 4,
      ancillary: {
        id: 'density',
        format: 'float32',
        storage: {kind: 'buffer', values: inputs.ancillary.view}
      },
      pairCapacity: 6,
      denominator: 'zone'
    });
    const target = publish(fixture, 'target', result.outputs.targetTotals);
    const eligible = publish(fixture, 'eligible', result.outputs.sourceEligibleWeights);
    fixture.run();
    await expectClose(fixture, target, reference.cases[0].targetTotals);
    await expectClose(fixture, eligible, reference.cases[0].sourceEligibleWeights);
  } finally {
    fixture.destroy();
  }
});

it('dasymetric disjoint core partitions normalize together and equal complete-domain support', async () => {
  // Concatenate right core then left core; a source spans both. This also rejects averaging
  // independently normalized tile allocations, which would transfer source 0 twice.
  const fixture = await createFixture('dasymetric-partitions');
  try {
    const cells = reference.cases[0].coverage.cellIds;
    const order = [
      ...cells.map((_, index) => index).filter(index => cells[index] >= 2),
      ...cells.map((_, index) => index).filter(index => cells[index] < 2)
    ];
    const result = createRecipe(fixture, createInputs(fixture, reference.cases[0], order), 'zone');
    const target = publish(fixture, 'target', result.outputs.targetTotals);
    const allocated = publish(fixture, 'allocated', result.outputs.sourceAllocatedTotals);
    const residual = publish(fixture, 'residual', result.outputs.sourceUnallocatedTotals);
    fixture.run();
    await expectClose(fixture, target, reference.cases[0].targetTotals);
    await expectClose(fixture, allocated, reference.cases[0].sourceAllocatedTotals);
    await expectClose(fixture, residual, reference.cases[0].sourceUnallocatedTotals);
  } finally {
    fixture.destroy();
  }
});

it('dasymetric live masks, totals and target ownership update one compiled graph without losing source mass', async () => {
  const fixture = await createFixture('dasymetric-live-inputs');
  let compiled: ReturnType<typeof fixture.graph.compile> | undefined;
  try {
    const inputs = createInputs(fixture);
    const result = createRecipe(fixture, inputs);
    const target = publish(fixture, 'target', result.outputs.targetTotals);
    const residual = publish(fixture, 'residual', result.outputs.sourceUnallocatedTotals);
    const complete = publish(fixture, 'complete', result.status.complete);
    compiled = fixture.graph.compile();
    submitGraph(fixture.device, compiled, undefined);
    await expectClose(fixture, target, [80, 120, 0, 0]);
    inputs.mask.buffer.write(new Uint32Array([0, 0, 0, 0, 0, 0]));
    submitGraph(fixture.device, compiled, undefined);
    await expectClose(fixture, target, [0, 0, 0, 0]);
    await expectClose(fixture, residual, [120, 80, 50]);
    expect(await fixture.readUint32(complete, 1)).toEqual([1]);
    inputs.mask.buffer.write(new Uint32Array([1, 1, 1, 1, 1, 1]));
    inputs.ancillary.buffer.write(new Float32Array([2, NaN, 1, 3, 1, 0]));
    inputs.sourceTotals.buffer.write(new Float32Array([60, 40, 25]));
    inputs.targetZones.buffer.write(new Uint32Array([1, 0, 0, 0, 0xffffffff, 2]));
    submitGraph(fixture.device, compiled, undefined);
    await expectClose(fixture, target, [60, 40, 0, 0]);
    await expectClose(fixture, residual, [0, 0, 25]);
    expect(await fixture.readUint32(complete, 1)).toEqual([1]);
    // Shrink the active tuple count without clearing the old source-1/source-2 tail.
    inputs.count.buffer.write(new Uint32Array([3]));
    inputs.requiredCount.buffer.write(new Uint32Array([3]));
    submitGraph(fixture.device, compiled, undefined);
    await expectClose(fixture, target, [20, 40, 0, 0]);
    await expectClose(fixture, residual, [0, 40, 25]);
    expect(await fixture.readUint32(complete, 1)).toEqual([1]);
    // Positive eligible support with no target owner retains the entire source total.
    inputs.targetZones.buffer.write(new Uint32Array(6).fill(0xffffffff));
    submitGraph(fixture.device, compiled, undefined);
    await expectClose(fixture, target, [0, 0, 0, 0]);
    await expectClose(fixture, residual, [60, 40, 25]);
    expect(await fixture.readUint32(complete, 1)).toEqual([1]);
    inputs.count.buffer.write(new Uint32Array([0]));
    inputs.requiredCount.buffer.write(new Uint32Array([0]));
    submitGraph(fixture.device, compiled, undefined);
    await expectClose(fixture, target, [0, 0, 0, 0]);
    await expectClose(fixture, residual, [60, 40, 25]);
    expect(await fixture.readUint32(complete, 1)).toEqual([1]);
  } finally {
    compiled?.destroy();
    fixture.destroy();
  }
});

it('dasymetric incomplete coverage and malformed rows suppress totals and recover on the same graph', async () => {
  const fixture = await createFixture('dasymetric-failure-propagation');
  let compiled: ReturnType<typeof fixture.graph.compile> | undefined;
  try {
    const inputs = createInputs(fixture);
    const result = createRecipe(fixture, inputs);
    const target = publish(fixture, 'target', result.outputs.targetTotals);
    const complete = publish(fixture, 'complete', result.status.complete);
    const invalid = publish(fixture, 'invalid', result.status.invalidTupleCount);
    compiled = fixture.graph.compile();
    for (const failure of [
      'overflow',
      'required',
      'candidate-only',
      'invalid geometry',
      'invalid fraction',
      'invalid source'
    ]) {
      inputs.overflow.buffer.write(new Uint32Array([failure === 'overflow' ? 1 : 0]));
      inputs.candidateOverflow.buffer.write(Uint32Array.of(Number(failure === 'candidate-only')));
      inputs.requiredCount.buffer.write(new Uint32Array([failure === 'required' ? 7 : 6]));
      inputs.zoneValidity.buffer.write(
        new Uint32Array([1, failure === 'invalid geometry' ? 0 : 1, 1])
      );
      inputs.fractions.buffer.write(
        Float32Array.from(
          reference.cases[0].coverage.fractions.map((value, index) =>
            failure === 'invalid fraction' && index === 0 ? 1.5 : value
          )
        )
      );
      inputs.sourceTotals.buffer.write(
        new Float32Array([failure === 'invalid source' ? -1 : 120, 80, 50])
      );
      submitGraph(fixture.device, compiled, undefined);
      expect(await fixture.readUint32(complete, 1), failure).toEqual([0]);
      await expectClose(fixture, target, [NaN, NaN, NaN, NaN]);
      expect(await fixture.readUint32(invalid, 1)).toEqual([
        failure === 'invalid fraction' ? 1 : 0
      ]);
    }
    inputs.sourceTotals.buffer.write(new Float32Array([120, 80, 50]));
    submitGraph(fixture.device, compiled, undefined);
    expect(await fixture.readUint32(complete, 1)).toEqual([1]);
    await expectClose(fixture, target, [80, 120, 0, 0]);
  } finally {
    compiled?.destroy();
    fixture.destroy();
  }
});

it('dasymetric pair overflow does not publish a partial allocation as conserved mass', async () => {
  const fixture = await createFixture('dasymetric-pair-overflow');
  try {
    const result = createRecipe(fixture, createInputs(fixture), 'overlap', 1);
    const target = publish(fixture, 'target', result.outputs.targetTotals);
    const residual = publish(fixture, 'residual', result.outputs.sourceUnallocatedTotals);
    const complete = publish(fixture, 'complete', result.status.complete);
    fixture.run();
    expect(await fixture.readUint32(complete, 1)).toEqual([0]);
    await expectClose(fixture, target, [NaN, NaN, NaN, NaN]);
    await expectClose(fixture, residual, [NaN, NaN, NaN]);
  } finally {
    fixture.destroy();
  }
});

it.each([
  'cell product',
  'source support sum',
  'target support sum',
  'target mass sum'
])('dasymetric detects float32 overflow in %s', async failure => {
  const fixture = await createFixture(`dasymetric-overflow-${failure}`);
  try {
    const sourceCount = failure === 'target support sum' || failure === 'target mass sum' ? 2 : 1;
    const result = addDasymetricRedistributionRecipe(fixture.graph, {
      grid: {...GRID, width: 2, affine: [failure === 'cell product' ? 2 : 1, 0, 0, 0, 1, 0]},
      source: {
        kind: 'coverage',
        coverage: {
          zoneIds: fixture.input('sources', new Uint32Array([0, sourceCount - 1]), 'uint32', 2),
          cellIds: fixture.input('cells', new Uint32Array([0, 1]), 'uint32', 2),
          fractions: fixture.input('fractions', new Float32Array([1, 1]), 'float32', 2),
          zoneValidity: fixture.input(
            'validity',
            new Uint32Array(sourceCount).fill(1),
            'uint32',
            sourceCount
          ),
          status: {
            count: fixture.input('count', new Uint32Array([2]), 'uint32', 1),
            requiredCount: fixture.input('required', new Uint32Array([2]), 'uint32', 1),
            overflow: fixture.input('overflow', new Uint32Array([0]), 'uint32', 1)
          }
        }
      },
      sourceTotals: fixture.input(
        'totals',
        new Float32Array(sourceCount).fill(failure === 'target mass sum' ? 3e38 : 100),
        'float32',
        sourceCount
      ),
      targetZones: fixture.input('targets', new Uint32Array([0, 0]), 'uint32', 2),
      targetZoneCount: 1,
      ancillary: {
        id: 'density',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: fixture.input(
            'density',
            new Float32Array(2).fill(failure === 'target mass sum' ? 1 : 3e38),
            'float32',
            2
          )
        }
      },
      pairCapacity: 2
    });
    const target = publish(fixture, 'target', result.outputs.targetTotals);
    const complete = publish(fixture, 'complete', result.status.complete);
    const invalid = publish(fixture, 'invalid', result.status.arithmeticInvalidCount);
    fixture.run();
    expect(await fixture.readUint32(complete, 1)).toEqual([0]);
    expect((await fixture.readUint32(invalid, 1))[0]).toBeGreaterThan(0);
    await expectClose(fixture, target, [NaN]);
  } finally {
    fixture.destroy();
  }
});

it('dasymetric skewed support retains explicit float32 mass-accounting tolerance', async () => {
  const fixture = await createFixture('dasymetric-skewed-support');
  try {
    const cellCount = 257;
    const density = Float32Array.from({length: cellCount}, (_, index) =>
      index === 0 ? 1e8 : ((index % 7) + 1) / 32
    );
    const targets = Uint32Array.from({length: cellCount}, (_, index) =>
      index % 3 === 2 ? 0xffffffff : index % 3
    );
    const sourceTotal = 1234567;
    const eligibleWeight = Array.from(density).reduce((sum, value) => sum + value, 0);
    const targetWeights = [0, 0];
    for (const [index, weight] of density.entries())
      if (targets[index] < 2) targetWeights[targets[index]] += weight;
    const expectedTargets = targetWeights.map(weight => (sourceTotal * weight) / eligibleWeight);
    const result = addDasymetricRedistributionRecipe(fixture.graph, {
      grid: {...GRID, width: cellCount},
      source: {
        kind: 'coverage',
        coverage: {
          zoneIds: fixture.input('sources', new Uint32Array(cellCount), 'uint32', cellCount),
          cellIds: fixture.input(
            'cells',
            Uint32Array.from({length: cellCount}, (_, index) => index),
            'uint32',
            cellCount
          ),
          fractions: fixture.input(
            'fractions',
            new Float32Array(cellCount).fill(1),
            'float32',
            cellCount
          ),
          zoneValidity: fixture.input('validity', new Uint32Array([1]), 'uint32', 1),
          status: {
            count: fixture.input('count', new Uint32Array([cellCount]), 'uint32', 1),
            requiredCount: fixture.input('required', new Uint32Array([cellCount]), 'uint32', 1),
            overflow: fixture.input('overflow', new Uint32Array([0]), 'uint32', 1)
          }
        }
      },
      sourceTotals: fixture.input('totals', new Float32Array([sourceTotal]), 'float32', 1),
      targetZones: fixture.input('targets', targets, 'uint32', cellCount),
      targetZoneCount: 2,
      ancillary: {
        id: 'density',
        format: 'float32',
        storage: {kind: 'buffer', values: fixture.input('density', density, 'float32', cellCount)}
      },
      denominator: 'zone',
      pairCapacity: 2
    });
    const target = publish(fixture, 'target', result.outputs.targetTotals);
    const allocated = publish(fixture, 'allocated', result.outputs.sourceAllocatedTotals);
    const residual = publish(fixture, 'residual', result.outputs.sourceUnallocatedTotals);
    const complete = publish(fixture, 'complete', result.status.complete);
    fixture.run();
    await expectClose(fixture, target, expectedTargets);
    const allocatedTotal = (await fixture.readFloat32(allocated, 1))[0];
    const residualTotal = (await fixture.readFloat32(residual, 1))[0];
    const targetTotal = (await fixture.readFloat32(target, 2)).reduce(
      (sum, value) => sum + value,
      0
    );
    const tolerance = 1e-4 + 1e-5 * sourceTotal;
    expect(Math.abs(allocatedTotal + residualTotal - sourceTotal)).toBeLessThanOrEqual(tolerance);
    expect(Math.abs(targetTotal + residualTotal - sourceTotal)).toBeLessThanOrEqual(tolerance);
    expect(residualTotal).toBeGreaterThanOrEqual(-tolerance);
    expect(await fixture.readUint32(complete, 1)).toEqual([1]);
  } finally {
    fixture.destroy();
  }
});
