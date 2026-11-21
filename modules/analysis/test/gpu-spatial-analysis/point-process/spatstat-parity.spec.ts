// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUPointPatternEnvelope} from '../../../src/gpu-spatial-analysis/point-process/gpu-point-pattern-envelope';
import {GPUInhomogeneousPointPattern} from '../../../src/gpu-spatial-analysis/point-process/gpu-inhomogeneous-point-pattern';
import {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window/gpu-study-window-preparation';
import {GPUNetworkKernelDensity} from '../../../src/gpu-network/gpu-network-kernel-density';
import {GPUNetworkInhomogeneousK} from '../../../src/gpu-network/gpu-network-inhomogeneous-k';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixtures from './spatstat-fixtures.json';

function harness(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32' | 'uint32x2'>(
    format: Format,
    values: number[]
  ) => {
    const buffer = device.createBuffer({
      data: format.startsWith('uint32') ? Uint32Array.from(values) : Float32Array.from(values),
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    return {
      view: importGraphBuffer(
        graph,
        `spatstat-${serial++}`,
        buffer,
        format,
        values.length / (format.endsWith('x2') ? 2 : 1)
      ),
      buffer
    };
  };
  return {
    graph,
    view,
    destroy() {
      buffers.forEach(buffer => buffer.destroy());
    }
  };
}
for (const fixture of fixtures.planar) {
  for (const supplied of [true, false]) {
    it(`spatstat3.8.2 polygon hole=${fixture.hole} supplied=${supplied}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const state = harness(device);
      const count = fixture.positions.length;
      const radiusCount = fixture.radii.length;
      const vertices = [
        0,
        0,
        10,
        0,
        10,
        10,
        0,
        10,
        0,
        0,
        ...(fixture.hole ? [4, 4, 4, 6, 6, 6, 6, 4, 4, 4] : [])
      ];
      const window = new GPUStudyWindowPreparation({
        positions: state.view('float32x2', vertices).view,
        ringOffsets: state.view('uint32', fixture.hole ? [0, 5, 10] : [0, 5]).view,
        polygonOffsets: state.view('uint32', [0, fixture.hole ? 2 : 1]).view,
        featureOffsets: state.view('uint32', [0, 1]).view,
        samplingGrid: [8, 8]
      });
      state.graph.add(window);
      const intensity = state.view('float32', new Array(count).fill(0));
      const values = state.view('float32', new Array(radiusCount).fill(0));
      const status = state.view('uint32', new Array(radiusCount).fill(0));
      state.graph.add(
        new GPUInhomogeneousPointPattern({
          positions: state.view('float32x2', fixture.positions.flat()).view,
          studyWindow: window,
          ...(supplied
            ? {suppliedIntensity: state.view('float32', fixture.intensity).view}
            : {bandwidth: fixture.bandwidth, selfContribution: 'include' as const}),
          radii: state.view('float32', fixture.radii).view,
          output: {
            intensity: intensity.view,
            k: values.view,
            pairCorrelation: state.view('float32', new Array(radiusCount).fill(0)).view,
            status: status.view
          }
        })
      );
      const compiled = state.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        if (supplied) {
          const actual = await readFloat32(values.buffer, radiusCount);
          fixture.k.forEach((value, row) => {
            if (value === null) expect(actual[row]).toBeNaN();
            else expect(actual[row]).toBeCloseTo(value, 5);
          });
        } else {
          const actual = await readFloat32(intensity.buffer, count);
          fixture.density.forEach((value, row) =>
            expect(Math.abs(actual[row] - value) / value).toBeLessThan(0.003)
          );
        }
        const flags = await readUint32(status.buffer, radiusCount);
        expect(flags.every(value => value === 0 || value === 1)).toBe(true);
      } finally {
        compiled.destroy();
        window.destroy();
        state.destroy();
      }
    });
  }
}
for (const fixture of fixtures.network) {
  it(`spatstat.linnet3.5.3 ${fixture.name} all6 correction/normalization combinations`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const expected of fixture.results) {
      const state = harness(device);
      const count = fixture.radii.length;
      const values = state.view('float32', new Array(count).fill(0));
      const status = state.view('uint32', new Array(count).fill(0));
      state.graph.add(
        new GPUNetworkInhomogeneousK({
          nodeCount: fixture.nodeCount,
          eventNodeIds: state.view('uint32', fixture.eventNodeIds).view,
          intensity: state.view('float32', fixture.intensity).view,
          nodeDistances: state.view(
            'float32',
            fixture.nodeDistances.map(value => (value === null ? Infinity : value))
          ).view,
          edgeNodes: state.view('uint32x2', fixture.edgeNodes.flat()).view,
          edgeLengths: state.view('float32', fixture.edgeLengths).view,
          radii: state.view('float32', fixture.radii).view,
          correction: expected.correction === 'ang' ? 'ang' : 'none',
          normalization:
            expected.normalization === 'network-length'
              ? 'network-length'
              : expected.normalization === 'inverse-intensity'
                ? 'inverse-intensity'
                : 'inverse-intensity-squared',
          output: {
            k: values.view,
            numerator: state.view('float32', new Array(count).fill(0)).view,
            normalization: state.view('float32', [0, 0, 0]).view,
            status: status.view
          }
        })
      );
      const compiled = state.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const actual = await readFloat32(values.buffer, count);
        expected.k.forEach((value, row) =>
          expect(
            actual[row],
            `${expected.correction}/${expected.normalization} radius${fixture.radii[row]}`
          ).toBeCloseTo(value, 5)
        );
        expect(await readUint32(status.buffer, count)).toEqual(new Array(count).fill(0));
      } finally {
        compiled.destroy();
        state.destroy();
      }
    }
  });
}

it('spatstat dclf.test supplied pooled reference and integration convention', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const state = harness(device);
  const fixture = fixtures.dclf;
  const count = fixture.observed.length;
  const statistics = state.view('float32', [0, 0, 0, 0]);
  const status = state.view('uint32', [0]);
  state.graph.add(
    new GPUPointPatternEnvelope({
      observed: state.view('float32', fixture.observed).view,
      simulations: state.view('float32', fixture.simulations).view,
      reference: state.view('float32', fixture.reference).view,
      integrationWeights: state.view('float32', fixture.integrationWeights).view,
      simulationCount: fixture.simulationCount,
      statistic: 'integrated-squared-deviation',
      nullModel:
        'Pinned spatstat supplied binomial CSR pattern curves; shared pooled reference leaveout0',
      fittedParameters: 'none',
      output: {
        lower: state.view('float32', new Array(count).fill(0)).view,
        upper: state.view('float32', new Array(count).fill(0)).view,
        statistics: statistics.view,
        status: status.view
      }
    })
  );
  const compiled = state.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const actual = await readFloat32(statistics.buffer, 4);
    expect(actual[0]).toBeCloseTo(fixture.statistic, 4);
    expect(actual[1]).toBeCloseTo(fixture.pValue, 6);
    expect(await readUint32(status.buffer, 1)).toEqual([0]);
  } finally {
    compiled.destroy();
    state.destroy();
  }
});

for (const fixture of fixtures.network) {
  it(`spatstat shortest paths and R continuous edge integrals validate ${fixture.name} radial KDE`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const state = harness(device);
    const expected = fixture.kde;
    const count = expected.quadratureLengths.length;
    const density = state.view('float32', new Array(count).fill(0));
    const masses = state.view('float32', new Array(expected.eventCount).fill(0));
    const validity = state.view('uint32', new Array(count).fill(0));
    state.graph.add(
      new GPUNetworkKernelDensity({
        distances: state.view(
          'float32',
          expected.distances.map(value => (value === null ? Infinity : value))
        ).view,
        eventCount: expected.eventCount,
        quadratureLengths: state.view('float32', expected.quadratureLengths).view,
        bandwidth: expected.bandwidth,
        output: {intensity: density.view, kernelMass: masses.view, validity: validity.view}
      })
    );
    const compiled = state.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const actual = await readFloat32(density.buffer, count);
      expected.continuousIntensity.forEach((value, row) =>
        expect(actual[row]).toBeCloseTo(value, 3)
      );
      const mass = await readFloat32(masses.buffer, expected.eventCount);
      expected.continuousMass.forEach((value, row) => expect(mass[row]).toBeCloseTo(value, 3));
      expect(
        actual.reduce((sum, value, row) => sum + value * expected.quadratureLengths[row], 0)
      ).toBeCloseTo(expected.eventCount, 5);
      expect(await readUint32(validity.buffer, count)).toEqual(new Array(count).fill(1));
    } finally {
      compiled.destroy();
      state.destroy();
    }
  });
}
