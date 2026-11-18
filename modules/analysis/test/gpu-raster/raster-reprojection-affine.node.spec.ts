// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {compileProjectionProgram} from '@luma.gl/graph/gpu-project';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  getGPURasterReprojectionParameterValues,
  GPURasterReprojection
} from '../../src/raster-analysis/gpu-raster-reprojection';
import {createNullWebGPUDevice, getParsedKernelSources} from '../utils/gpu-contributor-test-utils';
import {getGPUCommandNodeKernelSources} from '../utils/gpu-contributor-test-utils';

it('F7 bilinear renormalization preserves a lone sample with weight 1e-32', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const sourceGrid = {
    width: 2,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'point' as const
  };
  const destinationGrid = {...sourceGrid, width: 1, affine: [1, 0, 1e-32, 0, 1, 0] as const};
  const packed = getGPURasterReprojectionParameterValues({sourceGrid, destinationGrid});
  expect(packed[4]).toBeGreaterThan(0);
  const contributor = new GPURasterReprojection({
    sourceGrid,
    destinationGrid,
    method: 'bilinear',
    noDataPolicy: 'renormalize',
    parameters: createTransientView(graph, 'parameters', 'float32', 20),
    input: {
      id: 'source',
      format: 'float32',
      storage: {kind: 'buffer', values: createTransientView(graph, 'source', 'float32', 2)}
    },
    output: {
      values: createTransientView(graph, 'output', 'float32', 1),
      validity: createTransientView(graph, 'valid', 'uint32', 1)
    }
  });
  const source = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).find(kernel =>
    kernel.source.includes('weightSum>0.0')
  )!.source;
  const expression = /let result=([^;]+);/.exec(source)![1];
  const output = new Float32Array(1);
  new WgslExec(
    new WgslParser().parse(`@group(0) @binding(0) var<storage,read> weights:array<f32>; @group(0) @binding(1) var<storage,read_write> output:array<f32>;
  @compute @workgroup_size(1) fn main(){let weightSum=weights[0];let total=7.0*weightSum;let result=${expression};output[0]=result;}`)
  ).dispatchWorkgroups('main', [1, 1, 1], {0: {0: Float32Array.of(packed[4]), 1: output}});
  expect(output[0]).toBe(7);
});

// Emulate the packed sampler, rounding every arithmetic operation to float32. The legacy
// layout branch demonstrates the original caller-visible wrong nearest-neighbour selection.
function samplePackedColumn(parameters: Float32Array, column: number): number {
  const round = Math.fround;
  const relativeColumn = parameters[17] >= 3 ? round(column - parameters[18]) : column;
  const relativeRow = parameters[17] >= 3 ? -parameters[19] : 0;
  const positionX = round(
    round(round(parameters[0] * relativeColumn) + round(parameters[1] * relativeRow)) +
      parameters[4]
  );
  const positionY = round(
    round(round(parameters[2] * relativeColumn) + round(parameters[3] * relativeRow)) +
      parameters[5]
  );
  const sourceColumn =
    parameters[17] >= 2
      ? positionX
      : round(
          round(round(parameters[9] * positionX) - round(parameters[7] * positionY)) /
            parameters[10]
        );
  expect(Number.isFinite(sourceColumn)).toBe(true);
  if (parameters[17] >= 3) {
    const baseColumn = Math.floor(sourceColumn);
    return baseColumn + Number(round(sourceColumn - baseColumn) >= 0.5);
  }
  return Math.floor(round(sourceColumn + 0.5));
}

it.each([
  {width: 8, column: 4, shear: 1 + 0.55 * 2 ** -23},
  {width: 524292, column: 524288, shear: 1.047}
])('same-grid nearest keeps column $column on a $width-cell sheared strip (RT-04)', ({
  width,
  column,
  shear
}) => {
  const grid = {
    width,
    height: 1,
    affine: [1, 1, 0, 1, shear, 0] as const,
    pixelInterpretation: 'point' as const
  };
  const parameters = getGPURasterReprojectionParameterValues({
    sourceGrid: grid,
    destinationGrid: grid
  });
  expect(samplePackedColumn(parameters, column)).toBe(column);
});

it('preserves a source center after large scale/translation cancellation (R1)', () => {
  const sourceGrid = {
    width: 8,
    height: 1,
    affine: [5, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'point' as const
  };
  const destinationGrid = {
    ...sourceGrid,
    width: 4,
    affine: [16777232, 0, -50331676, 0, 1, 0] as const
  };
  const parameters = getGPURasterReprojectionParameterValues({sourceGrid, destinationGrid});
  // In exact integer world coordinates: 16777232 * 3 - 50331676 = 20 = 5 * 4.
  expect(samplePackedColumn(parameters, 3)).toBe(4);
});

it.each([
  {coordinate: 0.5 - 2 ** -25, expected: 0},
  {coordinate: 0.5, expected: 1},
  {coordinate: 0.5 + 2 ** -24, expected: 1}
])('selects nearest column $expected at the representable coordinate $coordinate (R4)', ({
  coordinate,
  expected
}) => {
  const sourceGrid = {
    width: 2,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'point' as const
  };
  const parameters = getGPURasterReprojectionParameterValues({
    sourceGrid,
    destinationGrid: {...sourceGrid, width: 1, affine: [1, 0, coordinate, 0, 1, 0]}
  });
  expect(samplePackedColumn(parameters, 0)).toBe(expected);
});

it('rejects a source inverse that becomes singular when packed, in the packer and constructor (R2)', () => {
  const grid = {
    width: 2,
    height: 1,
    affine: [1, 1 + 2 * 2 ** -27, 0, 1, 1 + 9 * 2 ** -27, 0] as const,
    pixelInterpretation: 'point' as const
  };
  const settings = {
    sourceGrid: grid,
    destinationGrid: grid,
    inverseProjection: compileProjectionProgram({precision: 'local-f32', operations: []})
  };
  expect(() => getGPURasterReprojectionParameterValues(settings)).toThrow(
    /sourceGrid.affine.*inverse.*invertible in float32/
  );
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    expect(
      () =>
        new GPURasterReprojection({
          ...settings,
          input: {
            id: 'source',
            format: 'float32',
            storage: {kind: 'buffer', values: createTransientView(graph, 'source', 'float32', 2)}
          },
          output: {
            values: createTransientView(graph, 'values', 'float32', 2),
            validity: createTransientView(graph, 'validity', 'uint32', 2)
          }
        })
    ).toThrow(/sourceGrid.affine.*inverse.*invertible in float32/);
  } finally {
    device.destroy();
  }
});

it('uses a fractional comparison for nearest sampling on both axes (R4)', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const grid = {
    width: 2,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'point' as const
  };
  const contributor = new GPURasterReprojection({
    sourceGrid: grid,
    destinationGrid: grid,
    method: 'nearest',
    input: {
      id: 'source',
      format: 'float32',
      storage: {kind: 'buffer', values: createTransientView(graph, 'source', 'float32', 2)}
    },
    output: {
      values: createTransientView(graph, 'values', 'float32', 2),
      validity: createTransientView(graph, 'validity', 'uint32', 2)
    }
  });
  try {
    const sampler = getParsedKernelSources(contributor.getCommandNodes(graph)).get(
      'raster-reprojection-sample'
    )!;
    expect(sampler).toContain('fract(column)>=0.5');
    expect(sampler).toContain('fract(row)>=0.5');
  } finally {
    contributor.destroy();
    device.destroy();
  }
});

for (const sourceInterpretation of ['point', 'area'] as const) {
  for (const destinationInterpretation of ['point', 'area'] as const) {
    it(`composes shear, large origin differences and ${sourceInterpretation}/${destinationInterpretation} shifts in source pixels`, () => {
      // Hand composition: inverse(source) * destination = [0.5, 0.25; -0.25, 1.5],
      // inverse(source) * (destinationOrigin - sourceOrigin) = [2, 3].
      const sourceGrid = {
        width: 17,
        height: 3,
        affine: [2, 1, 1000000, 1, 3, 2000000] as const,
        pixelInterpretation: sourceInterpretation
      };
      const destinationGrid = {
        ...sourceGrid,
        affine: [0.75, 2, 1000007, -0.25, 4.75, 2000011] as const,
        pixelInterpretation: destinationInterpretation
      };
      const sourceShift = sourceInterpretation === 'area' ? 0.5 : 0;
      const destinationShift = destinationInterpretation === 'area' ? 0.5 : 0;
      const parameters = getGPURasterReprojectionParameterValues({sourceGrid, destinationGrid});
      expect(Array.from(parameters.slice(0, 6))).toEqual([
        0.5,
        0.25,
        -0.25,
        1.5,
        2 + 0.75 * destinationShift - sourceShift,
        3 + 1.25 * destinationShift - sourceShift
      ]);
      expect(parameters[17]).toBe(3);
    });
  }
}

it('packs the source inverse directly for projected sampling, without a separately rounded divisor', () => {
  const sourceGrid = {
    width: 8,
    height: 1,
    affine: [1, 1, 0, 1, 1 + 0.55 * 2 ** -23, 0] as const,
    pixelInterpretation: 'point' as const
  };
  const parameters = getGPURasterReprojectionParameterValues({
    sourceGrid,
    destinationGrid: sourceGrid,
    inverseProjection: compileProjectionProgram({precision: 'local-f32', operations: []})
  });
  const determinant = sourceGrid.affine[4] - 1;
  expect(Array.from(parameters.slice(6, 10))).toEqual(
    Array.from(
      Float32Array.of(
        sourceGrid.affine[4] / determinant,
        -1 / determinant,
        -1 / determinant,
        1 / determinant
      )
    )
  );
  expect(parameters[10]).toBe(0);
});
