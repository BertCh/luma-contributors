// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUCrossVariogram} from '../../../src/dataframe-analysis/pair-statistics/gpu-cross-variogram';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const useTimes of [false, true]) {
  for (const constantSecond of [false, true]) {
    it(`legacy cross bins match independent pair enumeration (time ${useTimes}, constant ${constantSecond})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      // Coincident rows, both final upper edges, masked/NaN padding and an empty far bin.
      const positions = [
        [0, 0],
        [0, 0],
        [1, 0],
        [2, 0],
        [4, 0],
        [1, 0],
        [Number.NaN, 0]
      ];
      const values = [1, 2, 4, 3, 9, 100, 200];
      const secondValues = constantSecond ? values.map(() => 7) : [5, 4, 1, 2, -4, 50, 60];
      const times = [0, 0, 1, 2, 2, 0, 0];
      const mask = [1, 1, 1, 1, 1, 0, 1];
      const lagCount = 6;
      const temporalLagCount = useTimes ? 3 : 1;
      const bins = lagCount * temporalLagCount;
      for (const maximumDistance of [4, 0]) {
        const graph = new GPUCommandGraph(device);
        const buffers: ReturnType<typeof createInputBuffer>[] = [];
        let serial = 0;
        const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
          format: Format,
          values: number[]
        ) => {
          const buffer = createInputBuffer(
            device,
            format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
          );
          buffers.push(buffer);
          return importGraphBuffer(
            graph,
            `input-${serial++}`,
            buffer,
            format,
            values.length / (format === 'float32x2' ? 2 : 1)
          );
        };
        const semivariances = createOutputBuffer(device, bins);
        const pairCounts = createOutputBuffer(device, bins);
        const status = createOutputBuffer(device, bins);
        buffers.push(semivariances, pairCounts, status);
        graph.add(
          new GPUCrossVariogram({
            positions: input('float32x2', positions.flat()),
            values: input('float32', values),
            secondValues: input('float32', secondValues),
            times: useTimes ? input('float32', times) : undefined,
            mask: input('uint32', mask),
            parameters: input('float32', [maximumDistance, 2]),
            lagCount,
            temporalLagCount,
            output: {
              semivariances: importGraphBuffer(
                graph,
                'semivariances',
                semivariances,
                'float32',
                bins
              ),
              pairCounts: importGraphBuffer(graph, 'counts', pairCounts, 'uint32', bins),
              status: importGraphBuffer(graph, 'status', status, 'uint32', bins)
            }
          })
        );
        const compiled = graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          const actual = await readFloat32(semivariances, bins);
          const actualCounts = await readUint32(pairCounts, bins);
          const actualStatus = await readUint32(status, bins);
          const sums = new Array<number>(bins).fill(0);
          const counts = new Array<number>(bins).fill(0);
          for (let first = 0; first < values.length; first++) {
            for (let second = first + 1; second < values.length; second++) {
              if (
                !mask[first] ||
                !mask[second] ||
                !positions[first].every(Number.isFinite) ||
                !positions[second].every(Number.isFinite) ||
                maximumDistance <= 0
              )
                continue;
              const distance = Math.hypot(
                positions[first][0] - positions[second][0],
                positions[first][1] - positions[second][1]
              );
              const time = useTimes ? Math.abs(times[first] - times[second]) : 0;
              if (distance > maximumDistance || time > 2) continue;
              const spatialBin = Math.min(
                Math.floor((distance / maximumDistance) * lagCount),
                lagCount - 1
              );
              const temporalBin = Math.min(
                Math.floor((time / 2) * temporalLagCount),
                temporalLagCount - 1
              );
              const bin = spatialBin * temporalLagCount + temporalBin;
              sums[bin] +=
                ((values[first] - values[second]) * (secondValues[first] - secondValues[second])) /
                2;
              counts[bin]++;
            }
          }
          expect(actualCounts).toEqual(counts);
          counts.forEach((count, index) => {
            expect(actualStatus[index]).toBe(count ? 0 : 1);
            if (count) expect(Math.abs(actual[index] - sums[index] / count)).toBeLessThan(2e-5);
            else expect(Number.isNaN(actual[index])).toBe(true);
          });
        } finally {
          compiled.destroy();
          for (const buffer of buffers) buffer.destroy();
        }
      }
    });
  }
}

for (const secondScale of [1e-20, 1e20]) {
  it(`preserves legacy arithmetic when a variance unit is unrepresentable (second scale ${secondScale})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const positions = createInputBuffer(device, new Float32Array([0, 0, 1, 0]));
    const values = createInputBuffer(device, new Float32Array([1e20, 2e20]));
    const secondValues = createInputBuffer(
      device,
      new Float32Array([secondScale, 2 * secondScale])
    );
    const parameters = createInputBuffer(device, new Float32Array([2]));
    const semivariances = createOutputBuffer(device, 1);
    const pairCounts = createOutputBuffer(device, 1);
    const status = createOutputBuffer(device, 1);
    graph.add(
      new GPUCrossVariogram({
        positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', 2),
        values: importGraphBuffer(graph, 'values', values, 'float32', 2),
        secondValues: importGraphBuffer(graph, 'second', secondValues, 'float32', 2),
        parameters: importGraphBuffer(graph, 'parameters', parameters, 'float32', 1),
        lagCount: 1,
        output: {
          semivariances: importGraphBuffer(graph, 'semivariances', semivariances, 'float32', 1),
          pairCounts: importGraphBuffer(graph, 'counts', pairCounts, 'uint32', 1),
          status: importGraphBuffer(graph, 'status', status, 'uint32', 1)
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(pairCounts, 1)).toEqual([1]);
      const actual = (await readFloat32(semivariances, 1))[0];
      if (secondScale < 1) {
        expect(await readUint32(status, 1)).toEqual([0]);
        expect(Math.abs(actual - 0.5)).toBeLessThan(1e-6);
      } else {
        expect(await readUint32(status, 1)).toEqual([2]);
        expect(Number.isNaN(actual)).toBe(true);
      }
    } finally {
      compiled.destroy();
      for (const buffer of [
        positions,
        values,
        secondValues,
        parameters,
        semivariances,
        pairCounts,
        status
      ])
        buffer.destroy();
    }
  });
}

type LegacyVariogramScenario = {
  name: string;
  positions: number[];
  values: number[];
  secondValues?: number[];
  maximumDistance: number;
  count: number;
  semivariance: number;
};

for (const gridSize of [undefined, [3, 2] as const]) {
  const scenarios: LegacyVariogramScenario[] = [
    {
      name: 'small qualifying term beside a distant value outlier',
      positions: [0, 0, 1, 0, 100, 0],
      values: [0, 1, 100000],
      maximumDistance: 2,
      count: 1,
      semivariance: 0.5
    },
    {
      name: 'raw cross differences beside a centering outlier',
      positions: [0, 0, 1, 0, 10, 0, 20, 0, 30, 0],
      values: [0, 1, 0, 1, 1000000000],
      secondValues: [0, 1, 0, 1, 0],
      maximumDistance: 2,
      count: 1,
      semivariance: 0.5
    },
    {
      name: 'coincident pair inside overflowing horizontal bounds',
      positions: [-3e38, 0, -3e38, 0, 3e38, 0],
      values: [1, 3, 5],
      maximumDistance: 1,
      count: 1,
      semivariance: 2
    },
    {
      name: 'coincident pair inside overflowing vertical bounds',
      positions: [0, -3e38, 0, -3e38, 0, 3e38],
      values: [1, 3, 5],
      maximumDistance: 1,
      count: 1,
      semivariance: 2
    },
    {
      name: 'nonfinite fixed-point scale',
      positions: [0, 0, 1, 0, 2, 0, 3, 0, 4, 0],
      values: [0, 0, 0, 0, 1e-14],
      maximumDistance: 5,
      count: 10,
      semivariance: 2e-29
    },
    {
      name: 'finite radius with overflowing square',
      positions: [0, 0, 1, 0],
      values: [1, 3],
      maximumDistance: 1e20,
      count: 1,
      semivariance: 2
    },
    {
      name: 'largest finite radius with overflowing cell margin',
      positions: [0, 0, 1, 0],
      values: [1, 3],
      maximumDistance: Math.fround(3.4028234663852886e38),
      count: 1,
      semivariance: 2
    },
    {
      name: 'large radius and nonfinite scale together',
      positions: [0, 0, 1, 0, 2, 0, 3, 0, 4, 0],
      values: [0, 0, 0, 0, 1e-14],
      maximumDistance: 1e20,
      count: 10,
      semivariance: 2e-29
    },
    {
      name: 'nonfinite pair distance remains excluded',
      positions: [0, 0, 1e20, 0],
      values: [1, 3],
      maximumDistance: 1e20,
      count: 0,
      semivariance: Number.NaN
    }
  ];
  for (const scenario of scenarios) {
    it(`preserves legacy ${scenario.name} (${gridSize ? 'explicit' : 'default'} grid) across frames`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const graph = new GPUCommandGraph(device);
      const positions = createInputBuffer(device, Float32Array.from(scenario.positions));
      const values = createInputBuffer(device, Float32Array.from(scenario.values));
      const secondValues = scenario.secondValues
        ? createInputBuffer(device, Float32Array.from(scenario.secondValues))
        : undefined;
      const parameters = createInputBuffer(device, Float32Array.of(scenario.maximumDistance));
      const semivariances = createOutputBuffer(device, 1);
      const pairCounts = createOutputBuffer(device, 1);
      const status = createOutputBuffer(device, 1);
      graph.add(
        new GPUCrossVariogram({
          positions: importGraphBuffer(
            graph,
            'positions',
            positions,
            'float32x2',
            scenario.values.length
          ),
          values: importGraphBuffer(graph, 'values', values, 'float32', scenario.values.length),
          secondValues: secondValues
            ? importGraphBuffer(
                graph,
                'secondValues',
                secondValues,
                'float32',
                scenario.values.length
              )
            : undefined,
          parameters: importGraphBuffer(graph, 'parameters', parameters, 'float32', 1),
          lagCount: 1,
          gridSize,
          output: {
            semivariances: importGraphBuffer(graph, 'semivariances', semivariances, 'float32', 1),
            pairCounts: importGraphBuffer(graph, 'counts', pairCounts, 'uint32', 1),
            status: importGraphBuffer(graph, 'status', status, 'uint32', 1)
          }
        })
      );
      const compiled = graph.compile();
      try {
        for (const maximumDistance of [scenario.maximumDistance, 0, scenario.maximumDistance]) {
          parameters.write(Float32Array.of(maximumDistance));
          submitGraph(device, compiled, undefined);
          const expectedCount = maximumDistance > 0 ? scenario.count : 0;
          expect(await readUint32(pairCounts, 1)).toEqual([expectedCount]);
          expect(await readUint32(status, 1)).toEqual([expectedCount ? 0 : 1]);
          const actual = (await readFloat32(semivariances, 1))[0];
          if (expectedCount) {
            // Hand sums: four terms of 0.5e-28 among ten pairs, or one qualifying
            // term of 0.5 (unit differences) or 2 (difference of two).
            // Relative tolerance also checks the tiny result rather than accepting zero.
            expect(Math.abs(actual / scenario.semivariance - 1)).toBeLessThan(2e-5);
          } else {
            expect(Number.isNaN(actual)).toBe(true);
          }
        }
      } finally {
        compiled.destroy();
        secondValues?.destroy();
        for (const buffer of [positions, values, parameters, semivariances, pairCounts, status])
          buffer.destroy();
      }
    });
  }
}

it('legacy signed cancellation agrees between the default lattice and a single cell', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const estimates: number[] = [];
  for (const gridSize of [undefined, [1, 1] as const, [3, 2] as const]) {
    const graph = new GPUCommandGraph(device);
    const positions = createInputBuffer(
      device,
      new Float32Array([0, 0, 1, 0, -0.5, 0.8660254, -0.5, -0.8660254])
    );
    const values = createInputBuffer(device, new Float32Array([0, 1, 1, 1]));
    const secondValues = createInputBuffer(device, new Float32Array([0, 2e10, -2e10, 2]));
    const parameters = createInputBuffer(device, new Float32Array([1.01]));
    const semivariances = createOutputBuffer(device, 1);
    const pairCounts = createOutputBuffer(device, 1);
    const status = createOutputBuffer(device, 1);
    graph.add(
      new GPUCrossVariogram({
        positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', 4),
        values: importGraphBuffer(graph, 'values', values, 'float32', 4),
        secondValues: importGraphBuffer(graph, 'secondValues', secondValues, 'float32', 4),
        parameters: importGraphBuffer(graph, 'parameters', parameters, 'float32', 1),
        lagCount: 1,
        gridSize,
        output: {
          semivariances: importGraphBuffer(graph, 'semivariances', semivariances, 'float32', 1),
          pairCounts: importGraphBuffer(graph, 'counts', pairCounts, 'uint32', 1),
          status: importGraphBuffer(graph, 'status', status, 'uint32', 1)
        }
      })
    );
    const compiled = graph.compile();
    try {
      for (const maximumDistance of [1.01, 0, 1.01]) {
        parameters.write(Float32Array.of(maximumDistance));
        submitGraph(device, compiled, undefined);
        expect(await readUint32(pairCounts, 1)).toEqual([maximumDistance ? 3 : 0]);
        expect(await readUint32(status, 1)).toEqual([maximumDistance ? 0 : 1]);
        const estimate = (await readFloat32(semivariances, 1))[0];
        if (maximumDistance) {
          // Only center-to-neighbor pairs qualify. Input row order gives
          // float32 terms [1e10, -1e10, 1], whose sum is exactly one.
          expect(Math.abs(estimate - 1 / 3)).toBeLessThan(1e-7);
          estimates.push(estimate);
        } else {
          expect(Number.isNaN(estimate)).toBe(true);
        }
      }
    } finally {
      compiled.destroy();
      for (const buffer of [
        positions,
        values,
        secondValues,
        parameters,
        semivariances,
        pairCounts,
        status
      ])
        buffer.destroy();
    }
  }
  expect(new Set(estimates).size).toBe(1);
});
