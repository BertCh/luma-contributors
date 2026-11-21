// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, expectTypeOf, it} from 'vitest';
import {
  GPUGaussianRandomField,
  getGPUGaussianRandomFieldParameterValues,
  type GPUGaussianRandomFieldProps
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import fixture from './fixtures/gaussian-random-field.json';
import {createFieldRun} from './gaussian-random-field-test-utils';

function createProps() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props: GPUGaussianRandomFieldProps = {
    parameters: createTransientView(graph, 'parameters', 'float32', 16),
    positions: createTransientView(graph, 'positions', 'float32x2', 5),
    output: {
      values: createTransientView(graph, 'values', 'float32', 7),
      validity: createTransientView(graph, 'validity', 'uint32', 7),
      status: {approximation: createTransientView(graph, 'approximation', 'uint32', 1)},
      clippedEigenvalueMass: createTransientView(graph, 'mass', 'float32', 1)
    }
  };
  return {device, graph, props};
}

it('packs full uint32 seeds and centered affine without float rounding of random keys', () => {
  const target = new Float32Array(new ArrayBuffer(80), 8, 16);
  const values = getGPUGaussianRandomFieldParameterValues(
    {
      variogram: {model: 'exponential', sill: 2, range: 7},
      seed: 0xfffffff3,
      realization: 0xffffffff,
      mean: 3,
      grid: {width: 4, height: 3, affine: [2, 0, 100, 0, -3, 200], pixelInterpretation: 'area'}
    },
    target
  );
  expect(values).toBe(target);
  expect(Array.from(values.slice(0, 9))).toEqual([2, 0, 0, 0, 1, 0, 2, 7, 3]);
  expect(Array.from(values.slice(11, 15))).toEqual([101, 198.5, 2, -3]);
  expect(Array.from(new Uint32Array(values.buffer, values.byteOffset, 16).slice(9, 11))).toEqual([
    0xfffffff3, 0xffffffff
  ]);
});

it('rejects unsupported shapes, limits, aliases and coordinate contexts at construction', () => {
  const {device, graph, props} = createProps();
  try {
    expect(() => new GPUGaussianRandomField({...props, method: 'fft'})).toThrow(/requires grid/);
    expect(() => new GPUGaussianRandomField({...props, modeCount: 4097})).toThrow(
      /modeCount.*4096/
    );
    expect(
      () =>
        new GPUGaussianRandomField({
          ...props,
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
        })
    ).toThrow();
    expect(
      () =>
        new GPUGaussianRandomField({
          ...props,
          output: {
            ...props.output,
            values: props.parameters,
            validity: createTransientView(graph, 'alias-validity', 'uint32', 16)
          }
        })
    ).toThrow(/outputs must not share buffers with inputs/);
    const missingDomain = {...props, positions: undefined};
    expectTypeOf(missingDomain).not.toExtend<GPUGaussianRandomFieldProps>();
    expect(() => Reflect.construct(GPUGaussianRandomField, [missingDomain])).toThrow(/exclusively/);
    const grid = {
      width: 1025,
      height: 1,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'point' as const
    };
    expect(
      () => new GPUGaussianRandomField({...props, grid, positions: undefined, method: 'fft'})
    ).toThrow(/grid.width.*1024/);
    expect(
      () =>
        new GPUGaussianRandomField({
          ...props,
          positions: {...props.positions!, length: 2 ** 24},
          modeCount: 4096
        })
    ).toThrow(/spectral|modeCount/);
    expect(() =>
      getGPUGaussianRandomFieldParameterValues({
        variogram: {model: 'gaussian', sill: 1, range: 1},
        seed: 2 ** 32
      })
    ).toThrow(/seed/);
  } finally {
    device.destroy();
  }
});

it('auto selects spectral beyond FFT limit and rejects a binding before declaring nodes', () => {
  const {device, graph, props} = createProps();
  try {
    const grid = {
      width: 1025,
      height: 1,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'point' as const
    };
    const output = {
      ...props.output,
      values: createTransientView(graph, 'large-values', 'float32', 1025),
      validity: createTransientView(graph, 'large-validity', 'uint32', 1025)
    };
    expect(new GPUGaussianRandomField({...props, grid, positions: undefined, output}).method).toBe(
      'spectral'
    );
    Object.defineProperty(device, 'limits', {
      value: {...device.limits, maxStorageBufferBindingSize: 16}
    });
    expect(() => new GPUGaussianRandomField(props).getCommandNodes(graph)).toThrow(
      /binding.*limit/
    );
  } finally {
    device.destroy();
  }
});

it('declared nodes remain within the published estimate on both methods and conditioning', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const scenario of [
      ...fixture.scenarios,
      {
        name: 'empty-points',
        method: 'spectral',
        positions: [],
        covariance: [2, 0, 1, 1],
        seed: 0,
        realization: 0,
        mean: 0
      }
    ]) {
      const run = createFieldRun(device, scenario, false);
      try {
        expect((run.graph as unknown as {nodes: unknown[]}).nodes.length).toBeLessThanOrEqual(
          run.contributor.estimatedCommandNodeCount
        );
      } finally {
        run.destroy();
      }
    }
  } finally {
    device.destroy();
  }
});

for (const [name, incorrectFormat, length, requiredFormat] of [
  ['parameters', 'uint32', 16, 'float32'],
  ['positions', 'uint32', 5, 'float32x2'],
  ['output.values', 'uint32', 7, 'float32'],
  ['output.validity', 'float32', 7, 'uint32'],
  ['output.status.approximation', 'float32', 1, 'uint32'],
  ['output.clippedEigenvalueMass', 'uint32', 1, 'float32'],
  ['conditioning.indices', 'float32', 2, 'uint32'],
  ['conditioning.values', 'uint32', 2, 'float32'],
  ['conditioning.mask', 'float32', 2, 'uint32'],
  ['conditioning.neighborhoods', 'float32', 10, 'uint32']
] as const) {
  it(`rejects ${name} with an incorrect runtime format before declaring nodes`, () => {
    const {device, graph, props} = createProps();
    try {
      const incorrectProps = {
        ...props,
        output: {...props.output, status: {...props.output.status}},
        conditioning: {
          indices: createTransientView(graph, 'observation-indices', 'uint32', 2),
          values: createTransientView(graph, 'observation-values', 'float32', 2),
          mask: createTransientView(graph, 'observation-mask', 'uint32', 2),
          neighborhoods: createTransientView(graph, 'observation-neighborhoods', 'uint32', 10),
          neighborCount: 2
        }
      };
      const segments = name.split('.');
      let owner = incorrectProps as unknown as Record<string, unknown>;
      for (const segment of segments.slice(0, -1)) {
        owner = owner[segment] as Record<string, unknown>;
      }
      owner[segments[segments.length - 1]!] = createTransientView(
        graph,
        'incorrect-format',
        incorrectFormat,
        length
      );
      expect(() => new GPUGaussianRandomField(incorrectProps)).toThrow(
        new RegExp(`^GPUGaussianRandomField ${name.replaceAll('.', '\\.')}.*${requiredFormat}`)
      );
      expect((graph as unknown as {nodes: unknown[]}).nodes).toHaveLength(0);
    } finally {
      device.destroy();
    }
  });
}

it('keeps nested approximation status in alias and graph ownership validation', () => {
  const {device, graph, props} = createProps();
  try {
    for (const aliasedView of [props.parameters, props.output.validity]) {
      const approximation = graph.createDataView(aliasedView.buffer, {
        format: 'uint32',
        length: 1
      });
      expect(
        () =>
          new GPUGaussianRandomField({
            ...props,
            output: {...props.output, status: {approximation}}
          })
      ).toThrow(/outputs must not share buffers/);
    }
    const otherGraph = new GPUCommandGraph(device);
    const approximation = createTransientView(otherGraph, 'foreign-status', 'uint32', 1);
    const contributor = new GPUGaussianRandomField({
      ...props,
      output: {...props.output, status: {approximation}}
    });
    expect(() => contributor.getCommandNodes(graph)).toThrow(/graph/);
    expect((graph as unknown as {nodes: unknown[]}).nodes).toHaveLength(0);
  } finally {
    device.destroy();
  }
});
