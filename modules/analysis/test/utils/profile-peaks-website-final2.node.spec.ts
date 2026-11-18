// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, test, vi} from 'vitest';
import {computeProfilePeaksOnCPU} from '../../src/gpu-terrain/terrain-features/cpu-profile-peaks-operation';
import entry from '../../../../website/src/gpu/runs/entries/terrain-relief-light/profile-peaks';

vi.mock('../../../../website/src/gpu/runs/define-run', () => ({
  defineOperationRun: (definition: unknown) => definition
}));

function assertPeakDots(mask: Uint32Array, dots: Float32Array, count: number): void {
  expect(dots).toHaveLength(mask.length * 2);
  let kept = 0;
  for (let sample = 0; sample < mask.length; sample++) {
    const point = Array.from(dots.subarray(sample * 2, sample * 2 + 2));
    if (mask[sample] === 1) {
      kept++;
      expect(Number.isFinite(point[0]), `peak ${sample} x`).toBe(true);
      expect(Number.isFinite(point[1]), `peak ${sample} y`).toBe(true);
    } else expect(point, `unused sample ${sample}`).toEqual([-10, -10]);
  }
  expect(kept).toBe(count);
}

test.each([16, 32])('final3 H5 rejects an all-NaN peak-coordinate mutation (%s rounds)', rounds => {
  const values = Float32Array.from({length: 96}, (_, index) => (index % 4 === 0 ? index : 0));
  const result = computeProfilePeaksOnCPU(values, new Uint32Array([0, 96]), new Float32Array([5]), {
    window: 4,
    suppressionRadius: 4,
    maximumSuppressionIterations: rounds,
    capacity: 96
  });
  const dots = new Float32Array(values.length * 2).fill(-10);
  for (let sample = 0; sample < values.length; sample++)
    if (result.peakMask[sample] === 1) dots.set([NaN, NaN], sample * 2);
  expect(() => assertPeakDots(result.peakMask, dots, result.count[0])).toThrow();
});

test('final2 G4 website counts only kept peaks and qualifies unfinished suppression', () => {
  const values = Float32Array.from({length: 96}, (_, index) => (index % 4 === 0 ? index : 0));
  const definition = entry as unknown as {
    interaction: {
      resolve: (
        pointer: {unitY: number; y: number},
        context: {
          inputs: {width: number; ranges: [number, number][]};
          outputs: {
            peakMask: object;
            series: object;
            peakDots: {write: (dots: Float32Array) => void};
          };
          values: {noise: number};
          world: {width: number};
          peek: (output: object) => Uint32Array | Float32Array;
        }
      ) => {readout: string};
    };
  };
  for (const [rounds, count, incomplete] of [
    [16, 8, true],
    [32, 11, false]
  ] as const) {
    const result = computeProfilePeaksOnCPU(
      values,
      new Uint32Array([0, 96]),
      new Float32Array([5]),
      {window: 4, suppressionRadius: 4, maximumSuppressionIterations: rounds, capacity: 96}
    );
    expect(result.count[0]).toBe(count);
    const mask = {},
      series = {};
    let drawn = 0;
    let displayedDots = new Float32Array();
    const rendered = definition.interaction.resolve(
      // The slider reserves 5% at each end: 1 - (0.05 + 0.9 * 5 / 250).
      {unitY: 0.932, y: 0.5},
      {
        inputs: {width: 1, ranges: [[0, 100]]},
        outputs: {
          peakMask: mask,
          series,
          peakDots: {
            write: dots => {
              displayedDots = dots.slice();
              drawn = Array.from(dots).filter(
                (value, index) => index % 2 === 0 && value !== -10
              ).length;
            }
          }
        },
        values: {noise: 0},
        world: {width: 1},
        peek: output => (output === mask ? result.peakMask : values)
      }
    );
    expect(rendered.readout).toContain('minimumProminence 5 m');
    expect(rendered.readout).toContain(`${count} peaks`);
    expect(drawn).toBe(count);
    assertPeakDots(result.peakMask, displayedDots, count);
    expect(rendered.readout.includes('incomplete')).toBe(incomplete);
  }
});
