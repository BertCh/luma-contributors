// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createCoordinateParticleState,
  createCoordinateOffsets,
  expectCoordinateStreamline
} from './flow-coordinate-fixtures';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUParticleAdvection} from '../../../src/raster-analysis/particle-advection/gpu-particle-advection';
import {
  getGPUParticleAdvectionParameterValues,
  getGPUParticleAdvectionUint32ParameterValues
} from '../../../src/raster-analysis/particle-advection/particle-advection-parameters';
import {
  GPULagrangianFlowMap,
  getGPULagrangianFlowMapParameterValues
} from '../../../src/raster-analysis/lagrangian-diagnostics/gpu-lagrangian-flow-map';
import {GPULineIntegralConvolution} from '../../../src/raster-analysis/flow-texture/gpu-line-integral-convolution';
import {
  getGPULineIntegralConvolutionParameterValues,
  getGPULineIntegralConvolutionUint32ParameterValues
} from '../../../src/raster-analysis/flow-texture/line-integral-convolution-parameters';
import {GPUStreamlines} from '../../../src/raster-analysis/flow-texture/gpu-streamlines';
import {getGPUStreamlinesParameterValues} from '../../../src/raster-analysis/flow-texture/streamlines-parameters';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';

it('FLT-07 particles retain local quarter-metre state across four encodings', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const origin of [0, 10_000_000]) {
    const rig = createParityGraph(device);
    const {positions, ages, generations} = createCoordinateParticleState(rig);
    const words = rig.input('words', [1, 0, 0, 0], 'uint32');
    const previous = rig.output('previous', 1, 'float32x2');
    const trails = rig.output('trails', 4, 'float32x2');
    rig.graph.add(
      new GPUParticleAdvection({
        fieldWidth: 1,
        fieldHeight: 1,
        velocities: rig.input('velocities', [1, 0], 'float32x2').view,
        parameters: rig.input(
          'parameters',
          Array.from(
            getGPUParticleAdvectionParameterValues(
              {
                fieldPlacement: {origin: [origin, 100], cellSize: [100, -100]},
                localOrigin: origin ? [origin, 0] : undefined,
                timeStep: 0.25,
                dropRate: 0
              },
              [1, 1]
            )
          ),
          'float32'
        ).view,
        uint32Parameters: words.view,
        state: {positions: positions.view, ages: ages.view, generations: generations.view},
        output: {previousPositions: previous.view, trails: {positions: trails.view, length: 4}}
      })
    );
    const compiled = rig.graph.compile();
    try {
      for (let frame = 0; frame < 4; frame++) {
        words.buffer.write(
          getGPUParticleAdvectionUint32ParameterValues({seed: 1, frame, maximumAge: 0})
        );
        submitGraph(device, compiled, undefined);
      }
      // Closed form x(t) = 50 + t in the local frame. State is never restored from world f32.
      expect(await readFloat32(positions.buffer, 2)).toEqual([51, 50]);
      expect(await readFloat32(previous.buffer, 2)).toEqual([50.75, 50]);
      expect(await readFloat32(trails.buffer, 8)).toEqual([50.25, 50, 50.5, 50, 50.75, 50, 51, 50]);
      expect(await readUint32(ages.buffer, 1)).toEqual([4]);
      expect(await readUint32(generations.buffer, 1)).toEqual([0]);
      expect(origin + (await readFloat32(positions.buffer, 2))[0]).toBe(origin + 51);
    } finally {
      compiled.destroy();
      rig.destroy();
    }
  }
});

for (const temporal of [false, true]) {
  it(`${temporal ? 'FLT-03' : 'FLT-07'} flow-map preserves ${temporal ? 'epoch slice interpolation' : 'four local quarter-metre steps'}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = createParityGraph(device);
    const origin = temporal ? 0 : 10_000_000;
    const epoch = 1_700_000_000;
    const positions = rig.output('positions', 1, 'float32x2');
    const validity = rig.output('validity', 1, 'uint32');
    rig.graph.add(
      new GPULagrangianFlowMap({
        fieldWidth: 1,
        fieldHeight: 1,
        stepCount: temporal ? 1 : 4,
        seeds: rig.input('seeds', [50, 50], 'float32x2').view,
        velocitiesStart: rig.input('start', [temporal ? 0 : 1, 0], 'float32x2').view,
        velocitiesEnd: rig.input('end', [temporal ? 2 : 1, 0], 'float32x2').view,
        parameters: rig.input(
          'parameters',
          Array.from(
            getGPULagrangianFlowMapParameterValues({
              fieldPlacement: {origin: [origin, 100], cellSize: [100, -100]},
              localOrigin: [origin, 0],
              startTime: epoch,
              timeOrigin: epoch,
              timeStep: temporal ? 10 : 0.25,
              sliceStartTime: epoch,
              sliceEndTime: epoch + 10
            })
          ),
          'float32'
        ).view,
        output: {positions: positions.view, validity: validity.view}
      })
    );
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      // Midpoint integration is exact for v(t)=t/5 (integral 10); v=1 for t=1 travels 1.
      expect(await readFloat32(positions.buffer, 2)).toEqual([temporal ? 60 : 51, 50]);
      expect(await readUint32(validity.buffer, 1)).toEqual([1]);
    } finally {
      compiled.destroy();
      rig.destroy();
    }
  });
}

it('FLT-10 LIC scalar values survive a ten-million-unit translation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const origin of [0, 10_000_000]) {
    const rig = createParityGraph(device);
    recordWGSLKernelNodes(rig.graph);
    const values = rig.output('values', 8, 'float32');
    const nodes = new GPULineIntegralConvolution({
      id: 'translation-lic',
      fieldWidth: 8,
      fieldHeight: 1,
      grid: makeGPURasterMetadata({width: 8, height: 1}),
      stepCount: 4,
      velocities: rig.input(
        'velocities',
        Array.from({length: 16}, (_, index) => (index % 2 ? 0 : 1)),
        'float32x2'
      ).view,
      parameters: rig.input(
        'parameters',
        Array.from(
          getGPULineIntegralConvolutionParameterValues({
            fieldPlacement: {origin: [origin, 0], cellSize: [1, 1]},
            outputPlacement: {origin: [origin, 0], cellSize: [1, 1]},
            localOrigin: [origin, 0],
            stepLength: 0.25,
            period: 0
          })
        ),
        'float32'
      ).view,
      uint32Parameters: rig.input(
        'words',
        Array.from(getGPULineIntegralConvolutionUint32ParameterValues()),
        'uint32'
      ).view,
      output: {values: values.view}
    }).getCommandNodes(rig.graph);
    const noiseBinding = getWGSLKernelNodeRecord(rig.graph, 'translation-lic-noise')!.bindings.find(
      binding => binding.name === 'noise'
    )!;
    rig.graph.add(
      createWGSLKernelNode(rig.graph, {
        id: 'fixed-noise',
        operation: 'LICTranslationFixture',
        invocationCount: 8,
        bindings: [noiseBinding],
        body: 'noise[noiseOffset + index] = select(0.0, 1.0, index == 2u || index == 4u);'
      })
    );
    for (const node of nodes) if (node.id !== 'translation-lic-noise') rig.graph.add(node);
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const actual = (await readFloat32(values.buffer, 8))[3];
      // Python: sum((1+cos(pi*k/5))/2 * noise[floor(3.5+k/4)] for k in range(-4,5)) / 5
      expect(Number.isFinite(actual)).toBe(true);
      expect(actual).toBeCloseTo(0.3072949016875158, 6);
    } finally {
      compiled.destroy();
      rig.destroy();
    }
  }
});

for (const translated of [false, true]) {
  it(`FLT-07 NAM-04 streamlines publish supported quarter-metre local offsets: translated ${translated}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const origin = translated ? 10_000_000 : 0;
    const rig = createParityGraph(device);
    const positions = rig.output('positions', 17, 'float32x2');
    const count = rig.output('count', 1, 'uint32');
    const positionCount = rig.output('position-count', 1, 'uint32');
    const overflow = rig.output('overflow', 1, 'uint32');
    rig.graph.add(
      new GPUStreamlines({
        fieldWidth: 1,
        fieldHeight: 1,
        gridWidth: 1,
        gridHeight: 1,
        seedColumns: 1,
        seedRows: 1,
        stepsPerDirection: 8,
        velocities: rig.input('velocities', [1, 0], 'float32x2').view,
        parameters: rig.input(
          'parameters',
          Array.from(
            getGPUStreamlinesParameterValues({
              fieldPlacement: {origin: [origin, 100], cellSize: [100, -100]},
              gridPlacement: {origin: [origin, 0], cellSize: [100, 100]},
              localOrigin: translated ? [origin, 0] : undefined,
              stepLength: 0.25
            })
          ),
          'float32'
        ).view,
        uint32Parameters: rig.input('words', [8, 2, 0, 0], 'uint32').view,
        output: {
          lines: {
            ids: rig.output('ids', 1, 'uint32').view,
            status: {count: count.view, overflow: overflow.view}
          },
          positions: positions.view,
          lineOffsets: rig.output('offsets', 2, 'uint32').view,
          positionCount: positionCount.view
        }
      })
    );
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(count.buffer, 1)).toEqual([1]);
      expect(await readUint32(positionCount.buffer, 1)).toEqual([17]);
      expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
      const actual = await readFloat32(positions.buffer, 34);
      expectCoordinateStreamline(actual);
    } finally {
      compiled.destroy();
      rig.destroy();
    }
  });
}

it('OUT-09 fills reused GPU offsets with the terminal position count', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = createParityGraph(device);
  const offsets = createCoordinateOffsets(rig);
  const count = rig.output('count', 1, 'uint32');
  const positionCount = rig.output('position-count', 1, 'uint32');
  rig.graph.add(
    new GPUStreamlines({
      fieldWidth: 1,
      fieldHeight: 1,
      gridWidth: 1,
      gridHeight: 1,
      seedColumns: 4,
      seedRows: 1,
      stepsPerDirection: 2,
      velocities: rig.input('velocities', [3, 0], 'float32x2').view,
      parameters: rig.input(
        'parameters',
        Array.from(
          getGPUStreamlinesParameterValues({
            fieldPlacement: {origin: [0, 0], cellSize: [10, 10]},
            gridPlacement: {origin: [0, 0], cellSize: [10, 10]},
            stepLength: 0.125
          })
        ),
        'float32'
      ).view,
      uint32Parameters: rig.input('words', [7, 2, 0, 0], 'uint32').view,
      output: {
        lines: {
          ids: rig.output('ids', 4, 'uint32').view,
          status: {count: count.view, overflow: rig.output('overflow', 1, 'uint32').view}
        },
        lineOffsets: offsets.view,
        positions: rig.output('positions', 20, 'float32x2').view,
        positionCount: positionCount.view
      }
    })
  );
  const compiled = rig.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(count.buffer, 1)).toEqual([1]);
    expect(await readUint32(positionCount.buffer, 1)).toEqual([5]);
    expect(await readUint32(offsets.buffer, 5)).toEqual([0, 5, 5, 5, 5]);
  } finally {
    compiled.destroy();
    rig.destroy();
  }
});

it('PCK-11 runs the valid vertical control and refuses a nonfinite reciprocal before upload', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const settings = {
    fieldPlacement: {origin: [0, 0], cellSize: [1, 100]} as const,
    gridPlacement: {origin: [0, 0], cellSize: [1, 100]} as const,
    stepLength: 0.25
  };
  expect(() =>
    getGPUStreamlinesParameterValues({
      ...settings,
      gridPlacement: {origin: [0, 0], cellSize: [1e-40, 100]}
    })
  ).toThrow(/reciprocal/);
  const rig = createParityGraph(device);
  const count = rig.output('count', 1, 'uint32');
  const positionCount = rig.output('position-count', 1, 'uint32');
  rig.graph.add(
    new GPUStreamlines({
      fieldWidth: 1,
      fieldHeight: 1,
      gridWidth: 1,
      gridHeight: 1,
      seedColumns: 1,
      seedRows: 1,
      stepsPerDirection: 1,
      velocities: rig.input('velocities', [0, 1], 'float32x2').view,
      parameters: rig.input(
        'parameters',
        Array.from(getGPUStreamlinesParameterValues(settings)),
        'float32'
      ).view,
      uint32Parameters: rig.input('words', [8, 1, 0, 0], 'uint32').view,
      output: {
        lines: {
          ids: rig.output('ids', 1, 'uint32').view,
          status: {count: count.view, overflow: rig.output('overflow', 1, 'uint32').view}
        },
        lineOffsets: rig.output('offsets', 2, 'uint32').view,
        positions: rig.output('positions', 3, 'float32x2').view,
        positionCount: positionCount.view
      }
    })
  );
  const compiled = rig.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(count.buffer, 1)).toEqual([1]);
    expect(await readUint32(positionCount.buffer, 1)).toEqual([3]);
  } finally {
    compiled.destroy();
    rig.destroy();
  }
});

it('B2 B3 LIC decimal cells land in output columns 3 and 4 for manual world and local packets', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const packets = [
    // Manually supplied legacy world-frame packet: no packer can hide shader-coordinate errors.
    Float32Array.of(13, -5, 0.1, 1, 13, -5, 0.1, 1, 0.5, 0, 0, 0),
    getGPULineIntegralConvolutionParameterValues({
      fieldPlacement: {origin: [13, -5], cellSize: [0.1, 1]},
      outputPlacement: {origin: [13, -5], cellSize: [0.1, 1]},
      localOrigin: [13, -5]
    })
  ];
  for (const parameters of packets) {
    const rig = createParityGraph(device);
    const values = rig.output('values', 7, 'float32');
    rig.graph.add(
      new GPULineIntegralConvolution({
        fieldWidth: 7,
        fieldHeight: 1,
        grid: makeGPURasterMetadata({width: 7, height: 1}),
        stepCount: 1,
        velocities: rig.input(
          'velocities',
          Array.from({length: 14}, (_, index) => (index % 2 ? 0 : 1)),
          'float32x2'
        ).view,
        parameters: rig.input('parameters', Array.from(parameters), 'float32').view,
        uint32Parameters: rig.input('words', [3, 0, 0, 0], 'uint32').view,
        output: {values: values.view}
      })
    );
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const actual = await readFloat32(values.buffer, 7);
      expect(actual).toHaveLength(7);
      expect(actual.every(Number.isFinite)).toBe(true);
      // Independent integer Philox gives noise[3]=.7811055183410645, noise[4]=.3611893653869629.
      // Landings 3.5 +/- .5 are columns 3 and 4; Hann weights are 1, .5, .5.
      expect(actual[3]).toBeCloseTo(0.6761264801025391, 6);
    } finally {
      compiled.destroy();
      rig.destroy();
    }
  }
});

it('review equal-time flow maps select the first slice at and after its timestamp', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const startTime of [4, 5]) {
    const rig = createParityGraph(device);
    const positions = rig.output('positions', 1, 'float32x2');
    const validity = rig.output('validity', 1, 'uint32');
    rig.graph.add(
      new GPULagrangianFlowMap({
        fieldWidth: 1,
        fieldHeight: 1,
        stepCount: 1,
        seeds: rig.input('seeds', [8, 8], 'float32x2').view,
        velocitiesStart: rig.input('start', [2, 0], 'float32x2').view,
        velocitiesEnd: rig.input('end', [6, 0], 'float32x2').view,
        parameters: rig.input(
          'parameters',
          Array.from(
            getGPULagrangianFlowMapParameterValues({
              fieldPlacement: {origin: [0, 0], cellSize: [100, 100]},
              startTime,
              sliceStartTime: 4,
              sliceEndTime: 4,
              timeStep: 1
            })
          ),
          'float32'
        ).view,
        output: {positions: positions.view, validity: validity.view}
      })
    );
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readFloat32(positions.buffer, 2)).toEqual([10, 8]);
      expect(await readUint32(validity.buffer, 1)).toEqual([1]);
    } finally {
      compiled.destroy();
      rig.destroy();
    }
  }
});
