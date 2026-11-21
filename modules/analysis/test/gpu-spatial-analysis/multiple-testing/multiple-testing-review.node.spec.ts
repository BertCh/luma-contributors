// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  CPUMultipleTestingAdjustmentOperation,
  computeMultipleTestingAdjustmentOnCPU
} from '../../../src/gpu-spatial-analysis/multiple-testing/cpu-multiple-testing-adjustment-operation';
import {
  GPUMultipleTestingAdjustmentOperation,
  type MultipleTestingAdjustmentOperationProps
} from '../../../src/gpu-spatial-analysis/multiple-testing/gpu-multiple-testing-adjustment-operation';
import {
  GPUMultipleTestingAdjustment,
  getMultipleTestingAdjustedPValues,
  resolveGPUMultipleTestingMethod
} from '../../../src/gpu-spatial-analysis/multiple-testing/gpu-multiple-testing-adjustment';
import {GPURasterFalseDiscoveryRateMask} from '../../../src/gpu-spatial-analysis/multiple-testing/gpu-raster-false-discovery-rate-mask';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

const OPERATION_PROPS: MultipleTestingAdjustmentOperationProps = {
  id: 'review-adjust',
  pValues: 'probabilities',
  rowCount: 2,
  method: 'bonferroni',
  output: {adjustedPValues: 'adjusted'}
};

function executeAdjustment(operation: CPUMultipleTestingAdjustmentOperation) {
  const values = Float32Array.of(0.125, 0.0625);
  return operation.execute({
    inputs: {
      pValues: {format: 'float32', length: 2, chunks: [{values, length: 2}], pack: () => values}
    }
  });
}

function createViews(graph: GPUCommandGraph) {
  return {
    pValues: createTransientView(graph, 'probabilities', 'float32', 2),
    adjustedPValues: createTransientView(graph, 'adjusted', 'float32', 2),
    parameters: createTransientView(graph, 'parameters', 'float32', 1),
    significant: createTransientView(graph, 'significant', 'uint32', 2),
    finiteTestCount: createTransientView(graph, 'observed', 'uint32', 1),
    effectiveFamilySize: createTransientView(graph, 'family', 'uint32', 1)
  };
}

it('B1 captures inherited/non-enumerable methods and destinations in CPU/GPU operations', async () => {
  for (const placement of ['inherited', 'non-enumerable'] as const) {
    const props = {...OPERATION_PROPS, output: {adjustedPValues: 'adjusted'}};
    let methodReads = 0;
    let countReads = 0;
    const methodDescriptor = {
      get: () => {
        methodReads++;
        return 'bonferroni';
      }
    };
    const countDescriptor = {
      get: () => {
        countReads++;
        return 'observed';
      }
    };
    delete (props as Partial<typeof props>).method;
    if (placement === 'inherited') {
      Object.setPrototypeOf(props, Object.defineProperty({}, 'method', methodDescriptor));
      Object.setPrototypeOf(
        props.output,
        Object.defineProperty({}, 'finiteTestCount', countDescriptor)
      );
    } else {
      Object.defineProperty(props, 'method', methodDescriptor);
      Object.defineProperty(props.output, 'finiteTestCount', countDescriptor);
    }
    const cpuOperation = new CPUMultipleTestingAdjustmentOperation(props);
    expect(await executeAdjustment(cpuOperation)).toEqual({
      adjusted: Float32Array.of(0.25, 0.125),
      observed: Uint32Array.of(2)
    });
    expect([methodReads, countReads]).toEqual([1, 1]);
    const gpuOperation = new GPUMultipleTestingAdjustmentOperation(props);
    expect(gpuOperation.contract.outputs).toEqual(cpuOperation.contract.outputs);
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const views = createViews(graph);
      const contributor = gpuOperation.createContributor(
        <Format extends GPUGraphVectorFormat>(name: string) =>
          views[name as keyof typeof views] as GraphDataView<Format>
      ) as GPUMultipleTestingAdjustment;
      expect(contributor.props.method).toBe('bonferroni');
      expect(contributor.props.output.finiteTestCount).toBe(views.finiteTestCount);
      expect([methodReads, countReads]).toEqual([2, 2]);
    } finally {
      device.destroy();
    }
  }
});

it('B1 captures inherited settings and count views on both GPU contributors', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const views = createViews(graph);
    const output = Object.assign(Object.create({finiteTestCount: views.finiteTestCount}), {
      adjustedPValues: views.adjustedPValues,
      significant: views.significant
    });
    const settings = Object.assign(
      Object.create({
        get method() {
          return 'bonferroni';
        }
      }),
      {id: 'inherited', pValues: views.pValues, cellCount: 2, parameters: views.parameters, output}
    );
    const adjustment = new GPUMultipleTestingAdjustment(settings);
    const raster = new GPURasterFalseDiscoveryRateMask(settings);
    expect(adjustment.props.method).toBe('bonferroni');
    expect(raster.method).toBe('bonferroni');
    expect(adjustment.props.output.finiteTestCount).toBe(views.finiteTestCount);
    expect(raster.props.output.finiteTestCount).toBe(views.finiteTestCount);
  } finally {
    device.destroy();
  }
});

it('B2 rejects retired count ports and validates independent operation destinations', async () => {
  for (const Operation of [
    CPUMultipleTestingAdjustmentOperation,
    GPUMultipleTestingAdjustmentOperation
  ]) {
    for (const [retiredPort, replacement] of [
      ['observedTestCount', 'finiteTestCount'],
      ['familySize', 'effectiveFamilySize'],
      ['testedCount', 'effectiveFamilySize']
    ]) {
      expect(
        () =>
          new Operation({
            ...OPERATION_PROPS,
            output: {adjustedPValues: 'adjusted', [retiredPort]: 'count'}
          })
      ).toThrow(new RegExp(`review-adjust.*${retiredPort} was removed.*${replacement}`));
    }
    const operation = new Operation({
      ...OPERATION_PROPS,
      output: {
        adjustedPValues: 'adjusted',
        effectiveFamilySize: 'family',
        finiteTestCount: 'observed'
      }
    });
    expect(Object.keys(operation.contract.outputs ?? {})).toEqual([
      'adjusted',
      'observed',
      'family'
    ]);
    expect(
      () =>
        new Operation({
          ...OPERATION_PROPS,
          output: {
            adjustedPValues: 'adjusted',
            finiteTestCount: 'same',
            effectiveFamilySize: 'same'
          }
        })
    ).toThrow(/review-adjust/);
    expect(
      () =>
        new Operation({
          ...OPERATION_PROPS,
          output: {
            adjustedPValues: 'same',
            effectiveFamilySize: 'same'
          }
        })
    ).toThrow(/review-adjust/);
  }
  const cpu = new CPUMultipleTestingAdjustmentOperation({
    ...OPERATION_PROPS,
    output: {
      adjustedPValues: 'adjusted',
      effectiveFamilySize: 'family',
      finiteTestCount: 'observed'
    }
  });
  expect(await executeAdjustment(cpu)).toEqual({
    adjusted: Float32Array.of(0.25, 0.125),
    observed: Uint32Array.of(2),
    family: Uint32Array.of(2)
  });
});

it('B2 rejects removed count ports on both GPU contributors', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const views = createViews(graph);
    const settings = {
      id: 'view-alias',
      pValues: views.pValues,
      cellCount: 2,
      method: 'bonferroni' as const,
      parameters: views.parameters,
      output: {adjustedPValues: views.adjustedPValues, significant: views.significant}
    };
    for (const Producer of [GPUMultipleTestingAdjustment, GPURasterFalseDiscoveryRateMask]) {
      expect(
        () =>
          new Producer({
            ...settings,
            output: {
              ...settings.output,
              ...{testedCount: views.finiteTestCount},
              effectiveFamilySize: views.effectiveFamilySize
            }
          })
      ).toThrow(
        /view-alias.*testedCount.*effectiveFamilySize|view-alias.*effectiveFamilySize.*testedCount/
      );
      const producer = new Producer({
        ...settings,
        output: {
          ...settings.output,
          effectiveFamilySize: views.effectiveFamilySize,
          finiteTestCount: views.finiteTestCount
        }
      });
      producer.getCommandNodes(graph);
      expect(
        () =>
          new Producer({
            ...settings,
            output: {
              ...settings.output,
              finiteTestCount: views.finiteTestCount,
              effectiveFamilySize: views.finiteTestCount
            }
          })
      ).toThrow(/view-alias/);
    }
  } finally {
    device.destroy();
  }
});

it('S1 rejects omitted required methods and keeps the optional resolver default explicit', () => {
  const missing = undefined as unknown as 'bonferroni';
  for (const Operation of [
    CPUMultipleTestingAdjustmentOperation,
    GPUMultipleTestingAdjustmentOperation
  ]) {
    expect(() => new Operation({...OPERATION_PROPS, method: missing})).toThrow(
      /review-adjust method.*required/
    );
    expect(
      () => new Operation({...OPERATION_PROPS, method: null as unknown as 'bonferroni'})
    ).toThrow(/review-adjust method.*required/);
  }
  expect(() => getMultipleTestingAdjustedPValues([0.0625, 0.25], missing)).toThrow(
    /method.*required/
  );
  expect(() =>
    computeMultipleTestingAdjustmentOnCPU(Float32Array.of(0.0625, 0.25), missing)
  ).toThrow(/method.*required/);
  expect(resolveGPUMultipleTestingMethod('optional', undefined)).toBe('none');
  expect(
    getMultipleTestingAdjustedPValues(
      [0.0625, 0.25],
      resolveGPUMultipleTestingMethod('optional', undefined)
    )
  ).toEqual(Float64Array.of(0.0625, 0.25));
  const device = createNullWebGPUDevice();
  try {
    const views = createViews(new GPUCommandGraph(device));
    expect(
      () =>
        new GPUMultipleTestingAdjustment({
          id: 'required-gpu',
          pValues: views.pValues,
          method: missing,
          output: {adjustedPValues: views.adjustedPValues}
        })
    ).toThrow(/required-gpu method.*required/);
  } finally {
    device.destroy();
  }
});

it('S2 enforces exactly one word for each count output', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const views = createViews(graph);
    const oversized = createTransientView(graph, 'oversized', 'uint32', 2);
    for (const port of ['finiteTestCount', 'effectiveFamilySize'] as const) {
      expect(
        () =>
          new GPUMultipleTestingAdjustment({
            id: 'count-layout',
            pValues: views.pValues,
            method: 'none',
            output: {adjustedPValues: views.adjustedPValues, [port]: oversized}
          })
      ).toThrow(/count-layout.*exactly one.*2/);
    }
  } finally {
    device.destroy();
  }
});

it('earlier-fix note rejects GPU operation size errors before lowering without capping CPU families', () => {
  for (const invalid of [
    {minimumFamilySize: -1},
    {minimumFamilySize: 2 ** 31 + 1},
    {rowCount: 2 ** 31 + 1}
  ]) {
    expect(
      () => new GPUMultipleTestingAdjustmentOperation({...OPERATION_PROPS, ...invalid})
    ).toThrow(/review-adjust.*(rowCount|minimumFamilySize)/);
  }
  expect(
    () =>
      new CPUMultipleTestingAdjustmentOperation({
        ...OPERATION_PROPS,
        minimumFamilySize: 2 ** 31 + 1
      })
  ).not.toThrow();
});

it('S5 documents the declared chunked source domain and default-only example', () => {
  const source = readFileSync(
    'modules/analysis/src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-validation.ts',
    'utf8'
  );
  const comment = source.slice(
    source.indexOf('/**\n * {@link GPUSpatialWeightsValidation}'),
    source.indexOf('export class GPUChunkedSpatialWeightsValidation')
  );
  expect(comment).toContain('columnCount');
  expect(comment).toMatch(/default.*policy|default.*columnCount/i);
});

it('B3 excludes negative subnormal bits before raster float arithmetic and preserves the valid adjustment', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const views = createViews(graph);
    new GPURasterFalseDiscoveryRateMask({
      id: 'bit-family',
      pValues: views.pValues,
      cellCount: 2,
      method: 'bonferroni',
      parameters: views.parameters,
      output: {
        adjustedPValues: views.adjustedPValues,
        significant: views.significant,
        finiteTestCount: views.finiteTestCount,
        effectiveFamilySize: views.effectiveFamilySize
      }
    }).getCommandNodes(graph);
    const memory = new Map<GraphDataView['buffer'], Float32Array | Uint32Array>([
      [views.pValues.buffer, new Float32Array(Uint32Array.of(0x80000001, 0x3e000000).buffer)],
      [views.parameters.buffer, Float32Array.of(0.2)]
    ]);
    for (const suffix of [
      'tested',
      'adjustment-count',
      'adjustment-family',
      'adjustment-publish-counts',
      'adjustment-apply',
      'significant'
    ]) {
      const record = getWGSLKernelNodeRecord(graph, `bit-family-${suffix}`);
      if (!record) throw new Error(`missing kernel ${suffix}`);
      const bindings: Record<number, Float32Array | Uint32Array> = {};
      record.bindings.forEach((binding, index) => {
        if (!memory.has(binding.view.buffer))
          memory.set(
            binding.view.buffer,
            binding.type === 'f32'
              ? new Float32Array(binding.view.buffer.byteLength / 4)
              : new Uint32Array(binding.view.buffer.byteLength / 4)
          );
        bindings[index] = memory.get(binding.view.buffer)!;
      });
      const source = record.source
        .replace(/_ = [^;]+;/g, '')
        .replace(
          /override (\w+): u32;/g,
          (_, name: string) => `const ${name}: u32 = ${record.constants[name]}u;`
        )
        // Reproduce a permitted FTZ float comparison route. Bit predicates have no such arithmetic.
        .replace(
          /let p = (pValues\[[^;]+);/g,
          'let originalP = $1; let p = select(originalP, 0.0, abs(originalP) < 1.17549435e-38);'
        );
      new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
        'main',
        [...record.workgroups],
        {0: bindings}
      );
    }
    // The negative nonzero bit pattern is outside [0,1]; exactly one .125 hypothesis remains.
    expect(memory.get(views.finiteTestCount.buffer)).toEqual(Uint32Array.of(1));
    expect(memory.get(views.effectiveFamilySize.buffer)).toEqual(Uint32Array.of(1));
    const adjusted = memory.get(views.adjustedPValues.buffer)!;
    expect(adjusted).toHaveLength(2);
    expect(adjusted[0]).toBeNaN();
    expect(Number.isFinite(adjusted[1])).toBe(true);
    expect(adjusted[1]).toBe(0.125);
    expect(memory.get(views.significant.buffer)).toEqual(Uint32Array.of(0, 1));
  } finally {
    device.destroy();
  }
});
