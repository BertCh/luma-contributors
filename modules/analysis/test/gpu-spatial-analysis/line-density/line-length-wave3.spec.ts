// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPULineDensity,
  getGPULineDensityParameterValues,
  getGPULineDensityRasterOutputMapping
} from '../../../src/gpu-spatial-analysis/line-density/gpu-line-density';
import {GPULineLengthPerPolygon} from '../../../src/gpu-spatial-analysis/line-density/gpu-line-length-per-polygon';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';

it('UGE-07 clips a half-unit polygon in both directions at 2^28', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    for (const positions of [
      [-(2 ** 28), 0.5, 1, 0.5],
      [1, 0.5, -(2 ** 28), 0.5]
    ]) {
      const rig = createParityGraph(device);
      const lengths = rig.output('length', 1, 'float32'),
        weighted = rig.output('weighted', 1, 'float32');
      const counts = rig.output('count', 1, 'uint32'),
        overflow = rig.output('overflow', 1, 'uint32'),
        candidateOverflow = rig.output('candidate-overflow', 1, 'uint32');
      rig.graph.add(
        new GPULineLengthPerPolygon({
          positions: rig.input('positions', positions, 'float32x2').view,
          lineOffsets: rig.input('offsets', [0, 2], 'uint32').view,
          lineWeights: rig.input('weights', [3], 'float32').view,
          polygons: {
            kind: 'polygons',
            positions: rig.input('polygon-positions', [0, 0, 0.5, 0, 0.5, 1, 0, 1], 'float32x2')
              .view,
            ringOffsets: rig.input('rings', [0, 4], 'uint32').view,
            polygonOffsets: rig.input('polygons', [0, 1], 'uint32').view,
            featureOffsets: rig.input('features', [0, 1], 'uint32').view
          },
          pairCapacity: 4,
          output: {
            clippedLengths: lengths.view,
            weightedLengths: weighted.view,
            segmentCounts: counts.view,
            status: {overflow: overflow.view, candidateOverflow: candidateOverflow.view}
          }
        })
      );
      const compiled = rig.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        // The horizontal intersection is [0,.5]; weighted by 3, independent of direction.
        expect(await readFloat32(lengths.buffer, 1)).toEqual([0.5]);
        expect(await readFloat32(weighted.buffer, 1)).toEqual([1.5]);
        expect(await readUint32(counts.buffer, 1)).toEqual([1]);
        expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
        expect(await readUint32(candidateOverflow.buffer, 1)).toEqual([0]);
      } finally {
        compiled.destroy();
        rig.destroy();
      }
    }
  } finally {
    device.destroy();
  }
});

it('UPT-12 places a northern line in raster row zero on a 1 by 2 north-up grid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = createParityGraph(device);
  try {
    const lengths = rig.output('lengths', 2, 'float32'),
      densities = rig.output('densities', 2, 'float32'),
      overflow = rig.output('overflow', 1, 'uint32');
    rig.graph.add(
      new GPULineDensity({
        positions: rig.input('positions', [0.25, 1.5, 0.75, 1.5], 'float32x2').view,
        lineOffsets: rig.input('offsets', [0, 2], 'uint32').view,
        columns: 1,
        rows: 2,
        ...getGPULineDensityRasterOutputMapping({
          width: 1,
          height: 2,
          affine: [1, 0, 0, 0, -1, 2],
          pixelInterpretation: 'area'
        }),
        parameters: rig.input(
          'parameters',
          [
            ...getGPULineDensityParameterValues({
              width: 1,
              height: 2,
              affine: [1, 0, 0, 0, -1, 2],
              pixelInterpretation: 'area'
            })
          ],
          'float32'
        ).view,
        output: {
          lengths: lengths.view,
          densities: densities.view,
          status: {overflow: overflow.view}
        }
      })
    );
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      // A horizontal .5-unit segment wholly in northern cell, each cell has unit area.
      expect(await readFloat32(lengths.buffer, 2)).toEqual([0.5, 0]);
      expect(await readFloat32(densities.buffer, 2)).toEqual([0.5, 0]);
      expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
    device.destroy();
  }
});

it('B3 keeps physical north/east boundary ownership under raster row and column reversal', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    for (const vertical of [false, true])
      for (const reversed of [false, true]) {
        const width = vertical ? 2 : 1,
          height = vertical ? 1 : 2;
        const grid = {
          width,
          height,
          affine: (vertical
            ? [reversed ? -1 : 1, 0, reversed ? 2 : 0, 0, 1, 0]
            : [1, 0, 0, 0, reversed ? -1 : 1, reversed ? 2 : 0]) as [
            number,
            number,
            number,
            number,
            number,
            number
          ],
          pixelInterpretation: 'area' as const
        };
        const rig = createParityGraph(device),
          lengths = rig.output('lengths', 2, 'float32'),
          densities = rig.output('densities', 2, 'float32'),
          overflow = rig.output('overflow', 1, 'uint32');
        try {
          rig.graph.add(
            new GPULineDensity({
              positions: rig.input(
                'positions',
                vertical ? [1, 0.25, 1, 0.75] : [0.25, 1, 0.75, 1],
                'float32x2'
              ).view,
              lineOffsets: rig.input('offsets', [0, 2], 'uint32').view,
              columns: width,
              rows: height,
              ...getGPULineDensityRasterOutputMapping(grid),
              parameters: rig.input(
                'parameters',
                [...getGPULineDensityParameterValues(grid)],
                'float32'
              ).view,
              output: {
                lengths: lengths.view,
                densities: densities.view,
                status: {overflow: overflow.view}
              }
            })
          );
          const compiled = rig.graph.compile();
          try {
            submitGraph(device, compiled, undefined);
            // The physical northern/eastern cell owns the half-unit boundary segment.
            expect(await readFloat32(lengths.buffer, 2)).toEqual(reversed ? [0.5, 0] : [0, 0.5]);
            expect(await readFloat32(densities.buffer, 2)).toEqual(reversed ? [0.5, 0] : [0, 0.5]);
            expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
          } finally {
            compiled.destroy();
          }
        } finally {
          rig.destroy();
        }
      }
  } finally {
    device.destroy();
  }
});

it('S1 retains a narrow polygon with both endpoints distant and B1 preserves tangent vertices', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    for (const tangent of [false, true])
      for (const reverse of [false, true]) {
        const endpoints = tangent ? [-5, 1.2, 15, 1.2] : [-65536, 0.5, 65536, 0.5];
        const positions = reverse ? [...endpoints.slice(2), ...endpoints.slice(0, 2)] : endpoints;
        const polygon = tangent
          ? [0.2, 1.2, -0.8, 2.2, 2.3, 3.3]
          : [0, 0, 2 ** -10, 0, 2 ** -10, 1, 0, 1];
        const expected = tangent ? 0 : 2 ** -10;
        const rig = createParityGraph(device),
          lengths = rig.output('lengths', 1, 'float32'),
          weighted = rig.output('weighted', 1, 'float32'),
          count = rig.output('count', 1, 'uint32'),
          overflow = rig.output('overflow', 1, 'uint32'),
          candidateOverflow = rig.output('candidate', 1, 'uint32');
        try {
          rig.graph.add(
            new GPULineLengthPerPolygon({
              positions: rig.input('positions', positions, 'float32x2').view,
              lineOffsets: rig.input('offsets', [0, 2], 'uint32').view,
              lineWeights: rig.input('weights', [3], 'float32').view,
              pairCapacity: 4,
              polygons: {
                kind: 'polygons',
                positions: rig.input('polygon', polygon, 'float32x2').view,
                ringOffsets: rig.input('rings', [0, polygon.length / 2], 'uint32').view,
                polygonOffsets: rig.input('parts', [0, 1], 'uint32').view,
                featureOffsets: rig.input('features', [0, 1], 'uint32').view
              },
              output: {
                clippedLengths: lengths.view,
                weightedLengths: weighted.view,
                segmentCounts: count.view,
                status: {overflow: overflow.view, candidateOverflow: candidateOverflow.view}
              }
            })
          );
          const compiled = rig.graph.compile();
          try {
            submitGraph(device, compiled, undefined);
            expect(await readFloat32(lengths.buffer, 1)).toEqual([expected]);
            expect(await readFloat32(weighted.buffer, 1)).toEqual([3 * expected]);
            expect(await readUint32(count.buffer, 1)).toEqual([Number(!tangent)]);
            expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
            expect(await readUint32(candidateOverflow.buffer, 1)).toEqual([0]);
          } finally {
            compiled.destroy();
          }
        } finally {
          rig.destroy();
        }
      }
  } finally {
    device.destroy();
  }
});
