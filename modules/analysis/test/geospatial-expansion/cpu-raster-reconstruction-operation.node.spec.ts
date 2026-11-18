// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createKernelAuditContext} from '../analysis-core/kernel-audit/kernel-audit-context';
import type {GPURasterReconstruction} from '../../src/raster-analysis/gpu-raster-reconstruction';
import {
  computeRasterReconstructionOnCPU,
  CPURasterReconstructionOperation
} from '../../src/raster-analysis/raster-morphology/cpu-raster-reconstruction-operation';
import {
  GPURasterReconstructionOperation,
  type RasterReconstructionOperationProps
} from '../../src/raster-analysis/raster-morphology/gpu-raster-reconstruction-operation';
import {constructRasterOperationContributor} from '../utils/raster-operation-test-utils';
import fixtures from './raster-morphology-operation-fixtures.json';

const PROPS: RasterReconstructionOperationProps = {
  id: 'reconstruct',
  grid: {width: 5, height: 1, affine: [1, 0, 0, 0, -1, 1], pixelInterpretation: 'area'},
  marker: 'marker',
  mask: 'mask',
  output: {
    values: 'values',
    validity: 'validity',
    status: {converged: 'converged', iterationCount: 'rounds', iterationLimitReached: 'limit'}
  }
};

it('R2 both operation wrappers retain their construction budget and output names', async () => {
  const props = {
    ...PROPS,
    maximumIterations: 1,
    output: {...PROPS.output, status: {...PROPS.output.status}}
  };
  const cpu = new CPURasterReconstructionOperation(props);
  const gpu = new GPURasterReconstructionOperation(props);
  props.maximumIterations = 5;
  props.output.values = 'changed';
  const wrap = (values: Float32Array) => ({
    format: 'float32' as const,
    length: 5,
    chunks: [{values, length: 5}],
    pack: () => values
  });
  const result = await cpu.execute({
    inputs: {marker: wrap(Float32Array.of(1, 0, 0, 0, 0)), mask: wrap(new Float32Array(5).fill(1))}
  });
  expect(result['values']).toEqual(Float32Array.of(1, 1, 0, 0, 0));
  expect(result['rounds']).toEqual(Uint32Array.of(2));
  const context = createKernelAuditContext('operation-mutation');
  try {
    const views = Object.fromEntries(
      Object.entries({
        marker: {format: 'float32' as const, length: 5},
        mask: {format: 'float32' as const, length: 5},
        ...gpu.outputs
      }).map(([name, descriptor]) => [name, context.view(descriptor.format, descriptor.length)])
    );
    const contributor = gpu.createContributor(
      name => views[name] as never
    ) as GPURasterReconstruction;
    expect(contributor.props.maximumIterations).toBe(1);
    expect(contributor.getCommandNodes(context.graph)).toHaveLength(8);
  } finally {
    context.device.destroy();
    gpu.destroy();
  }
});

it('matches skimage reconstruction in dilation and erosion with four and eight neighbors', () => {
  for (const fixture of fixtures.reconstruction) {
    const result = computeRasterReconstructionOnCPU(
      Float32Array.from(fixture.marker),
      Float32Array.from(fixture.mask),
      {
        grid: {...PROPS.grid, width: fixture.width, height: fixture.height},
        method: fixture.method as 'dilation' | 'erosion',
        connectivity: fixture.connectivity as 4 | 8
      }
    );
    expect(Array.from(result.values)).toEqual(fixture.expected);
    expect(Array.from(result.validity)).toEqual(fixture.expected.map(() => 1));
    expect(Array.from(result.converged)).toEqual([1]);
    expect(Array.from(result.iterationLimitReached)).toEqual([0]);
  }
});

it('has the closed-form bottleneck profile and verifies the last changed round', () => {
  // The only path from the left passes through a height-3 bottleneck.
  const result = computeRasterReconstructionOnCPU(
    Float32Array.of(9, 0, 0, 0, 0),
    Float32Array.of(9, 7, 3, 8, 6),
    PROPS
  );
  expect(Array.from(result.values)).toEqual([9, 7, 3, 3, 3]);
  expect(result.iterationCount[0]).toBe(5);
  const inverse = computeRasterReconstructionOnCPU(
    Float32Array.of(1, 10, 10, 10, 10),
    Float32Array.of(1, 3, 7, 2, 4),
    {...PROPS, method: 'erosion'}
  );
  expect(Array.from(inverse.values)).toEqual([1, 3, 7, 7, 7]);
  expect(inverse.iterationCount[0]).toBe(5);
});

it('keeps invalid marker/mask pairs as barriers and rejects undersized arrays', () => {
  const result = computeRasterReconstructionOnCPU(
    Float32Array.of(9, 0, 2, NaN, Infinity),
    Float32Array.of(9, 9, 1, 9, 9),
    PROPS
  );
  expect(Array.from(result.values)).toEqual([9, 9, NaN, NaN, NaN]);
  expect(Array.from(result.validity)).toEqual([1, 1, 0, 0, 0]);
  expect(result.iterationCount[0]).toBe(2);
  expect(() =>
    computeRasterReconstructionOnCPU(new Float32Array(4), new Float32Array(5), PROPS)
  ).toThrow(/marker/);
  expect(() =>
    computeRasterReconstructionOnCPU(new Float32Array(5), new Float32Array(4), PROPS)
  ).toThrow(/mask/);
  expect(
    computeRasterReconstructionOnCPU(new Float32Array(5), new Float32Array(6), PROPS).values
  ).toEqual(new Float32Array(5));
});

it('validates both classes at construction, deferring GPU props until contributor creation', () => {
  const inputs = {
    marker: {format: 'float32' as const, length: 5},
    mask: {format: 'float32' as const, length: 5}
  };
  expect(() => new CPURasterReconstructionOperation({...PROPS, maximumIterations: 0})).toThrow(
    /^reconstruct .*maximumIterations/
  );
  expect(() =>
    constructRasterOperationContributor(
      new GPURasterReconstructionOperation({...PROPS, maximumIterations: 0}),
      inputs
    )
  ).toThrow(/^reconstruct .*maximumIterations/);
  for (const Operation of [CPURasterReconstructionOperation, GPURasterReconstructionOperation])
    expect(() => new Operation({...PROPS, grid: {...PROPS.grid, width: 0}})).toThrow(
      /^reconstruct /
    );
  expect(
    () => new CPURasterReconstructionOperation({...PROPS, maximumIterations: 30000})
  ).not.toThrow();
  expect(() =>
    constructRasterOperationContributor(
      new GPURasterReconstructionOperation({...PROPS, maximumIterations: 30000}),
      inputs
    )
  ).toThrow(/^reconstruct /);
  expect(() =>
    constructRasterOperationContributor(new GPURasterReconstructionOperation(PROPS), inputs)
  ).not.toThrow();
  expect(() =>
    constructRasterOperationContributor(new GPURasterReconstructionOperation(PROPS), {
      ...inputs,
      mask: {format: 'float32', length: 4}
    })
  ).toThrow(/^reconstruct .*mask/);
});

it('executes renamed outputs without reusing an input port', async () => {
  const wrap = (values: Float32Array) => ({
    format: 'float32' as const,
    length: 5,
    chunks: [{values, length: 5}],
    pack: () => values
  });
  const result = await new CPURasterReconstructionOperation({
    ...PROPS,
    maximumIterations: 1
  }).execute({
    inputs: {
      marker: wrap(Float32Array.of(9, 0, 0, 0, 0)),
      mask: wrap(Float32Array.of(9, 7, 3, 8, 6))
    }
  });
  expect(result['values']).toEqual(Float32Array.of(9, 7, 0, 0, 0));
  expect(result['validity']).toEqual(new Uint32Array(5).fill(1));
  expect(result['rounds']).toEqual(Uint32Array.of(2));
  expect(result['converged']).toEqual(Uint32Array.of(0));
  expect(result['limit']).toEqual(Uint32Array.of(1));
  expect(Object.keys(result).sort()).toEqual([
    'converged',
    'limit',
    'rounds',
    'validity',
    'values'
  ]);
});

it('publishes only budgeted dilation and erosion rounds while counting verification', () => {
  for (const method of ['dilation', 'erosion'] as const) {
    const dilation = method === 'dilation';
    const result = computeRasterReconstructionOnCPU(
      Float32Array.from(dilation ? [1, 0, 0, 0] : [0, 1, 1, 1]),
      new Float32Array(4).fill(dilation ? 1 : 0),
      {...PROPS, grid: {...PROPS.grid, width: 4}, method, maximumIterations: 1}
    );
    expect(Array.from(result.values)).toEqual(dilation ? [1, 1, 0, 0] : [0, 0, 1, 1]);
    expect(Array.from(result.validity)).toEqual([1, 1, 1, 1]);
    expect(result.iterationCount[0]).toBe(2);
    expect(result.converged[0]).toBe(0);
    expect(result.iterationLimitReached[0]).toBe(1);
  }
});

it('ignores surplus marker and mask rows independently on a unit grid', () => {
  const props = {...PROPS, grid: {...PROPS.grid, width: 1}};
  for (const marker of [Float32Array.of(2), Float32Array.of(2, NaN)]) {
    for (const mask of [Float32Array.of(3), Float32Array.of(3, NaN)]) {
      const result = computeRasterReconstructionOnCPU(marker, mask, props);
      // A singleton marker 2 already satisfies mask 3 and has no neighbor to raise it.
      expect(result.values).toEqual(Float32Array.of(2));
      expect(result.validity).toEqual(Uint32Array.of(1));
      expect(result.converged).toEqual(Uint32Array.of(1));
      expect(result.iterationCount).toEqual(Uint32Array.of(1));
      expect(result.iterationLimitReached).toEqual(Uint32Array.of(0));
      expect(() =>
        constructRasterOperationContributor(new GPURasterReconstructionOperation(props), {
          marker: {format: 'float32', length: marker.length},
          mask: {format: 'float32', length: mask.length}
        })
      ).not.toThrow();
    }
  }
});

for (const markerLayout of ['exact', 'padded', 'offset'] as const) {
  for (const maskLayout of ['exact', 'padded', 'offset'] as const) {
    it(`F8 calibrates the logical band prefix: marker ${markerLayout}, mask ${maskLayout}`, async () => {
      const samples = (layout: typeof markerLayout) =>
        layout === 'exact'
          ? Float32Array.of(1)
          : layout === 'padded'
            ? Float32Array.of(1, 99)
            : Float32Array.of(99, 1, 77).subarray(1);
      const flags = (layout: typeof markerLayout) =>
        layout === 'exact'
          ? Uint32Array.of(1)
          : layout === 'padded'
            ? Uint32Array.of(1, 0)
            : Uint32Array.of(0, 1, 0).subarray(1);
      const wrap = <Values extends Float32Array | Uint32Array>(values: Values) => ({
        format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
        length: values.length,
        chunks: [{values, length: values.length}],
        pack: () => values
      });
      const props: RasterReconstructionOperationProps = {
        ...PROPS,
        grid: {...PROPS.grid, width: 1},
        maximumIterations: 1,
        marker: {
          id: 'marker-band',
          format: 'float32',
          storage: {kind: 'buffer', values: 'marker'},
          validity: 'marker-validity',
          noDataValue: 99
        },
        mask: {
          id: 'mask-band',
          format: 'float32',
          storage: {kind: 'buffer', values: 'mask'},
          validity: 'mask-validity',
          noDataValue: 99
        }
      };
      const result = await new CPURasterReconstructionOperation(props).execute({
        inputs: {
          marker: wrap(samples(markerLayout)),
          mask: wrap(samples(maskLayout)),
          markerValidity: wrap(flags(markerLayout)),
          maskValidity: wrap(flags(maskLayout))
        }
      });
      expect(result['values']).toEqual(Float32Array.of(1));
      expect(result['validity']).toEqual(Uint32Array.of(1));
      expect(result['converged']).toEqual(Uint32Array.of(1));
      expect(result['rounds']).toEqual(Uint32Array.of(1));
      expect(result['limit']).toEqual(Uint32Array.of(0));
      expect(() =>
        constructRasterOperationContributor(new GPURasterReconstructionOperation(props), {
          marker: {format: 'float32', length: samples(markerLayout).length},
          mask: {format: 'float32', length: samples(maskLayout).length},
          markerValidity: {format: 'uint32', length: flags(markerLayout).length},
          maskValidity: {format: 'uint32', length: flags(maskLayout).length}
        })
      ).not.toThrow();
    });
  }
}

for (const shortPort of ['marker', 'mask', 'markerValidity', 'maskValidity'] as const) {
  it(`F8 rejects undersized ${shortPort} before calibrating a grid prefix`, async () => {
    const props: RasterReconstructionOperationProps = {
      ...PROPS,
      grid: {...PROPS.grid, width: 2},
      maximumIterations: 1,
      marker: {
        id: 'marker-band',
        format: 'float32',
        storage: {kind: 'buffer', values: 'marker'},
        validity: 'marker-validity'
      },
      mask: {
        id: 'mask-band',
        format: 'float32',
        storage: {kind: 'buffer', values: 'mask'},
        validity: 'mask-validity'
      }
    };
    const wrap = (values: Float32Array | Uint32Array) => ({
      format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
      length: values.length,
      chunks: [{values, length: values.length}],
      pack: () => values
    });
    const inputs = {
      marker: wrap(new Float32Array(shortPort === 'marker' ? 1 : 2).fill(1)),
      mask: wrap(new Float32Array(shortPort === 'mask' ? 1 : 2).fill(1)),
      markerValidity: wrap(new Uint32Array(shortPort === 'markerValidity' ? 1 : 2).fill(1)),
      maskValidity: wrap(new Uint32Array(shortPort === 'maskValidity' ? 1 : 2).fill(1))
    };
    expect(() => new CPURasterReconstructionOperation(props).execute({inputs})).toThrow(
      /input|marker|mask/
    );
    expect(() =>
      constructRasterOperationContributor(new GPURasterReconstructionOperation(props), {
        marker: {format: 'float32', length: inputs.marker.length},
        mask: {format: 'float32', length: inputs.mask.length},
        markerValidity: {format: 'uint32', length: inputs.markerValidity.length},
        maskValidity: {format: 'uint32', length: inputs.maskValidity.length}
      })
    ).toThrow(/marker|mask/);
  });
}
