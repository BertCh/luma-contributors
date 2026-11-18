// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {entries} from '../../../../../website/src/catalog/entries/recipes-trajectories';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {computePopulationSupportAlignmentOnCPU} from '../../../src/gpu-spatial-analysis/landscape-ecology/cpu-population-support-alignment-operation';
import {GPUPopulationSupportAlignment} from '../../../src/gpu-spatial-analysis/landscape-ecology/gpu-population-support-alignment';
import {addPopulationSupportAlignmentRecipe} from '../../../src/gpu-spatial-analysis/landscape-ecology/segregation-counterfactual';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createProps(graph: GPUCommandGraph) {
  return {
    id: 'review-alignment',
    populations: createTransientView(graph, 'input-populations', 'float32x2', 1),
    sourceIds: createTransientView(graph, 'source-ids', 'uint32', 1),
    targetIds: createTransientView(graph, 'target-ids', 'uint32', 1),
    output: {
      populations: createTransientView(graph, 'output-populations', 'float32x2', 1),
      alignmentCode: createTransientView(graph, 'alignment-code', 'uint32', 1),
      matchedMask: createTransientView(graph, 'matched-mask', 'uint32', 1),
      status: createTransientView(graph, 'status', 'uint32x4', 1)
    }
  };
}

it.each([
  'length',
  'format',
  'ownership',
  'output-alias',
  'input-alias'
] as const)('final1 spatial-1:F13 preflights %s before allocating recipe outputs so retry succeeds', kind => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const {output, ...inputs} = createProps(graph);
    const props = {...inputs, id: 'retry'};
    const invalidOutput =
      kind === 'length'
        ? {matchedMask: createTransientView(graph, 'long-mask', 'uint32', 2)}
        : kind === 'format'
          ? {matchedMask: createTransientView(graph, 'float-mask', 'float32', 1) as never}
          : kind === 'ownership'
            ? {
                status: createTransientView(
                  new GPUCommandGraph(device),
                  'foreign-status',
                  'uint32x4',
                  1
                )
              }
            : kind === 'output-alias'
              ? {alignmentCode: output.alignmentCode, matchedMask: output.alignmentCode}
              : {populations: inputs.populations};
    expect(() =>
      addPopulationSupportAlignmentRecipe(graph, {...props, output: invalidOutput})
    ).toThrow();
    expect(() =>
      addPopulationSupportAlignmentRecipe(graph, {
        ...props,
        output: {matchedMask: output.matchedMask}
      })
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('final2 spatial-1:G4 accepts a metadata-bearing recipe override variable', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    const output = {matchedMask: props.output.matchedMask, label: 'trial mask'};
    const result = addPopulationSupportAlignmentRecipe(graph, {...props, output});
    expect(result.outputs.matchedMask).toBe(output.matchedMask);
    expect(Object.keys(result.outputs)).toEqual(['populations', 'alignmentCode', 'matchedMask']);
  } finally {
    device.destroy();
  }
});

it('final2 spatial-1:G4 ignores contributor output metadata when declaring commands', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    const output = {...props.output, label: 'trial mask'};
    const contributor = new GPUPopulationSupportAlignment({...props, output});
    expect(contributor.getCommandNodes(graph).length).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});

it('R1 rejects a legacy recipe override passed through a typed variable', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    // Structural typing accepts this variable even though validity is no longer a public port.
    const output = {
      populations: props.output.populations,
      validity: createTransientView(graph, 'legacy-validity', 'uint32', 1),
      status: props.output.status
    };
    expect(() => addPopulationSupportAlignmentRecipe(graph, {...props, output})).toThrow(
      'review-alignment output.validity is no longer supported; use output.alignmentCode for diagnostics and output.matchedMask for row selection'
    );
  } finally {
    device.destroy();
  }
});

for (const createAlignment of [
  (graph: GPUCommandGraph, props: ReturnType<typeof createProps>) =>
    new GPUPopulationSupportAlignment(props),
  addPopulationSupportAlignmentRecipe
]) {
  it.each([
    'imported',
    'transient'
  ] as const)(`R2 rejects shared row-output bytes at construction (${createAlignment.name}): %s`, kind => {
    const device = createNullWebGPUDevice();
    const buffer = device.createBuffer({byteLength: 4, usage: Buffer.STORAGE});
    try {
      const graph = new GPUCommandGraph(device);
      const props = createProps(graph);
      const alignmentCode =
        kind === 'imported'
          ? importGraphBuffer(graph, 'code-import', buffer, 'uint32', 1)
          : props.output.alignmentCode;
      const matchedMask =
        kind === 'imported'
          ? importGraphBuffer(graph, 'mask-import', buffer, 'uint32', 1)
          : alignmentCode;
      if (kind === 'imported') expect(alignmentCode.buffer).not.toBe(matchedMask.buffer);
      expect(() =>
        createAlignment(graph, {
          ...props,
          output: {...props.output, alignmentCode, matchedMask}
        })
      ).toThrow(
        'review-alignment output.alignmentCode and output.matchedMask must not share a buffer'
      );
    } finally {
      buffer.destroy();
      device.destroy();
    }
  });
}

it.each([
  'alignmentCode',
  'matchedMask'
] as const)('R3 names a missing required output.%s at contributor construction', name => {
  const device = createNullWebGPUDevice();
  try {
    const props = createProps(new GPUCommandGraph(device));
    const output: Partial<typeof props.output> = {...props.output};
    delete output[name];
    expect(
      () =>
        new GPUPopulationSupportAlignment({
          ...props,
          output: output as typeof props.output
        })
    ).toThrow(`review-alignment output.${name} is required`);
  } finally {
    device.destroy();
  }
});

it('R4 documents partial support results and downstream selection in the catalog example', () => {
  const result = computePopulationSupportAlignmentOnCPU({
    populations: Float32Array.of(10, 20),
    sourceIds: Uint32Array.of(7),
    targetIds: Uint32Array.of(7, 8)
  });
  expect(Array.from(result.populations)).toEqual([10, 20, NaN, NaN]);
  expect(Array.from(result.alignmentCode)).toEqual([0, 1]);
  expect(Array.from(result.matchedMask)).toEqual([1, 0]);
  expect(Array.from(result.status)).toEqual([0, 1, 0, 0]);
  const entry = entries.find(candidate => candidate.id === 'population-support-alignment-recipe')!;
  expect(entry.summary).not.toMatch(/IDs are errors/);
  expect(entry.explanation).not.toContain('fails loudly');
  expect(entry.example).toContain('aligned.outputs.matchedMask');
  expect(entry.outputs).toContainEqual({
    name: 'packedStatus',
    type: 'GraphDataView<"uint32x4">',
    description: '[complete, missingTargetRows, ambiguousTargetRows, excludedSourceRows].'
  });
  const stages = entry.outputs.find(output => output.name === 'status.stages');
  expect(stages?.type).toBe('readonly GPURecipeStageStatusPort[]');
  for (const name of ['missing-targets', 'ambiguous-targets', 'excluded-sources', 'invalidCount'])
    expect(stages?.description).toContain(name);
  expect(stages!.description.length).toBeLessThanOrEqual(80);
  expect(
    entry.outputs.some(output => output.name === 'status' && output.type.includes('GraphDataView'))
  ).toBe(false);
  expect(entry.example).toContain('read aligned.packedStatus');
  expect(entry.example).not.toContain('read aligned.status and require its first word');
});
