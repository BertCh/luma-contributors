// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {expect, it} from 'vitest';
import {
  getGPURasterProfileParameterValues,
  getGPURasterSamplingParameterValues,
  GPURasterProfile,
  GPURasterSampling
} from '../../../src/raster-analysis/raster-sampling/index';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '../../../src/raster-analysis/raster-grid/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  getCubicWeightsOnCPU,
  makeExtentGrid,
  profileRasterOnCPU,
  sampleRasterOnCPU,
  type OracleRaster
} from './raster-sampling-oracle';

let serial = 0;

function view<Format extends 'uint32' | 'float32' | 'float32x2'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) {
  return createTransientView(graph, `view-${serial++}`, format, length);
}

function withGraph(callback: (graph: GPUCommandGraph) => void): void {
  const device = createNullWebGPUDevice();
  callback(new GPUCommandGraph(device));
  device.destroy();
}

/** 8x8 raster over [0, 8) x [0, 8) whose cell (column, row) holds `columnScale * column + rowScale * row`. */
function createPlane(columnScale: number, rowScale: number): OracleRaster {
  const values = new Float32Array(64);
  for (let row = 0; row < 8; row++) {
    for (let column = 0; column < 8; column++) {
      values[row * 8 + column] = columnScale * column + rowScale * row;
    }
  }
  return {width: 8, height: 8, values};
}

const EXTENT = [0, 0, 8, 8] as const;

function pack(
  method: 'nearest' | 'bilinear' | 'bicubic',
  noDataPolicy: 'propagate' | 'renormalize' = 'propagate'
) {
  return getGPURasterSamplingParameterValues({method, noDataPolicy}, makeExtentGrid(8, 8, EXTENT));
}

it('raster-sampling parameter helpers pack layouts and validate', () => {
  const values = getGPURasterSamplingParameterValues(
    {method: 'bicubic', noDataPolicy: 'renormalize'},
    makeExtentGrid(4, 2, [10, 20, 18, 24])
  );
  expect(Array.from(values)).toEqual([10, 20, 18, 24, 2, 2, 0.5, 0.5, 2, 1, 0, 20]);
  const profile = getGPURasterProfileParameterValues(
    {spacing: 1.5},
    makeExtentGrid(4, 2, [10, 20, 18, 24])
  );
  expect(profile[10]).toBe(1.5);
  expect(profile[8]).toBe(1);
  expect(profile[9]).toBe(0);
  // A north-up grid has the same extent, a negative row reciprocal and the row origin at maxY.
  const northUp = getGPURasterSamplingParameterValues(
    {localOrigin: [10, 20]},
    makeGPURasterMetadata({width: 4, height: 2, cellSize: 2, origin: [10, 24]})
  );
  expect(Array.from(northUp)).toEqual([0, 0, 8, 4, 2, 2, 0.5, -0.5, 1, 0, 0, 4]);
  const unit = makeExtentGrid(2, 2, [0, 0, 1, 1]);
  expect(() =>
    getGPURasterProfileParameterValues({spacing: 0}, makeExtentGrid(4, 2, [0, 0, 1, 1]))
  ).toThrow(/spacing/);
  expect(() =>
    getGPURasterSamplingParameterValues({}, makeGPURasterMetadata({width: 0, height: 2}))
  ).toThrow(/grid dimensions/);
  expect(() =>
    getGPURasterSamplingParameterValues({}, {...unit, affine: [0, 0, 0, 0, 1, 0]})
  ).toThrow(/cell size/);
  expect(() =>
    getGPURasterSamplingParameterValues({}, {...unit, affine: [1, 0.5, 0, 0, 1, 0]})
  ).toThrow(/axis-aligned/);
  expect(() => getGPURasterSamplingParameterValues({}, unit, new Float32Array(3))).toThrow(
    /12 values/
  );
});

it('raster-sampling oracle: nearest, outside, edges and clamping', () => {
  const plane = createPlane(1, 10);
  const nearest = pack('nearest');
  // The cell containing the point.
  expect(sampleRasterOnCPU(plane, nearest, 0.25, 0.25)).toBe(0);
  expect(sampleRasterOnCPU(plane, nearest, 3.75, 2.0)).toBe(3 + 20);
  // Cell boundary belongs to the upper cell; the max edge belongs to the last cell.
  expect(sampleRasterOnCPU(plane, nearest, 1, 1)).toBe(11);
  expect(sampleRasterOnCPU(plane, nearest, 8, 8)).toBe(7 + 70);
  // Outside the extent and NaN coordinates.
  expect(sampleRasterOnCPU(plane, nearest, -0.125, 3)).toBeNaN();
  expect(sampleRasterOnCPU(plane, nearest, 3, 8.5)).toBeNaN();
  expect(sampleRasterOnCPU(plane, nearest, NaN, 3)).toBeNaN();
  // Bilinear on a plane is the plane; edge clamping holds the edge centre value.
  const bilinear = pack('bilinear');
  expect(sampleRasterOnCPU(plane, bilinear, 3, 4)).toBe(2.5 + 10 * 3.5);
  expect(sampleRasterOnCPU(plane, bilinear, 2.5, 4.5)).toBe(2 + 40);
  expect(sampleRasterOnCPU(plane, bilinear, 0, 0)).toBe(0);
  expect(sampleRasterOnCPU(plane, bilinear, 8, 0.25)).toBe(7);
  expect(sampleRasterOnCPU(plane, bilinear, 0.1875, 7.9375)).toBe(0 + 70);
});

it('raster-sampling oracle: bicubic reproduces planes inside and clamps at edges', () => {
  const plane = createPlane(2, 3);
  const bicubic = pack('bicubic');
  expect(sampleRasterOnCPU(plane, bicubic, 4.25, 3.5)).toBeCloseTo(2 * 3.75 + 3 * 3, 6);
  expect(sampleRasterOnCPU(plane, bicubic, 3.5, 3.5)).toBe(2 * 3 + 3 * 3);
  // Within half a cell of the edge the clamped support flattens the plane: not equal to the plane.
  expect(sampleRasterOnCPU(plane, bicubic, 0.5, 3.5)).toBe(3 * 3);
  const clamped = sampleRasterOnCPU(plane, bicubic, 0.25, 3.5);
  expect(Math.abs(clamped - (2 * -0.25 + 3 * 3))).toBeGreaterThan(0.05);
  expect(clamped).toBeGreaterThan(0);
  for (const t of [0, 0.125, 0.5, 0.875]) {
    const weights = getCubicWeightsOnCPU(t);
    expect(weights.reduce((sum, weight) => sum + weight, 0)).toBeCloseTo(1, 6);
  }
  expect(getCubicWeightsOnCPU(0).map(weight => weight + 0)).toEqual([0, 1, 0, 0]);
});

it('raster-sampling oracle: nodata policies', () => {
  const raster = createPlane(1, 0);
  raster.values[3 * 8 + 3] = NaN;
  const propagate = pack('bilinear', 'propagate');
  const renormalize = pack('bilinear', 'renormalize');
  // Point at the corner between cells (3,3) (NaN), (4,3), (3,4), (4,4).
  expect(sampleRasterOnCPU(raster, propagate, 4, 4)).toBeNaN();
  expect(sampleRasterOnCPU(raster, renormalize, 4, 4)).toBeCloseTo(11 / 3, 6);
  // On the centre of a valid cell next to the hole the hole carries zero weight.
  expect(sampleRasterOnCPU(raster, propagate, 4.5, 3.5)).toBe(4);
  // On the hole's own centre: propagate and renormalize both give NaN.
  expect(sampleRasterOnCPU(raster, propagate, 3.5, 3.5)).toBeNaN();
  expect(sampleRasterOnCPU(raster, renormalize, 3.5, 3.5)).toBeNaN();
  // Bicubic with nodata in its support falls back to the bilinear rule.
  expect(sampleRasterOnCPU(raster, pack('bicubic', 'propagate'), 4, 4)).toBeNaN();
  expect(sampleRasterOnCPU(raster, pack('bicubic', 'renormalize'), 4, 4)).toBeCloseTo(11 / 3, 6);
  // Nearest on a nodata cell is NaN under both policies; sentinel and validity count as nodata.
  expect(sampleRasterOnCPU(raster, pack('nearest', 'renormalize'), 3.5, 3.5)).toBeNaN();
  const flagged: OracleRaster = {
    ...createPlane(1, 0),
    validity: new Uint32Array(64).fill(1),
    noDataValue: 5
  };
  flagged.validity![2] = 0;
  expect(sampleRasterOnCPU(flagged, pack('nearest'), 2.5, 0.5)).toBeNaN();
  expect(sampleRasterOnCPU(flagged, pack('nearest'), 5.5, 0.5)).toBeNaN();
  expect(sampleRasterOnCPU(flagged, pack('nearest'), 6.5, 0.5)).toBe(6);
});

it('raster-profile oracle: ramp, counts and zero-length lines', () => {
  const ramp = createPlane(10, 0);
  const parameters = getGPURasterProfileParameterValues({spacing: 2}, makeExtentGrid(8, 8, EXTENT));
  // Line 0 rises along the ramp, line 1 is zero length, line 2 has no vertices, line 3 descends.
  const linePositions = new Float32Array([0.5, 4, 6.5, 4, 3, 3, 6.5, 4, 0.5, 4]);
  const lineOffsets = Uint32Array.from([0, 2, 3, 3, 5]);
  const profile = profileRasterOnCPU(ramp, parameters, linePositions, lineOffsets);
  expect(Array.from(profile.sampleOffsets)).toEqual([0, 4, 5, 5, 9]);
  expect(Array.from(profile.lineLengths)).toEqual([6, 0, 0, 6]);
  expect(Array.from(profile.sampleDistances.slice(0, 4))).toEqual([0, 2, 4, 6]);
  expect(Array.from(profile.sampleValues.slice(0, 4))).toEqual([0, 20, 40, 60]);
  expect(Array.from(profile.lineGains)).toEqual([60, 0, 0, 0]);
  expect(Array.from(profile.lineLosses)).toEqual([0, 0, 0, 60]);
  expect(Array.from(profile.sampleCumulativeGain.slice(0, 4))).toEqual([0, 20, 40, 60]);
  expect(Array.from(profile.lineMinimums)).toEqual([0, 25, NaN, 0]);
  expect(Array.from(profile.lineMaximums)).toEqual([60, 25, NaN, 60]);
  expect(Array.from(profile.sampleLineIds)).toEqual([0, 0, 0, 0, 1, 3, 3, 3, 3]);
  // Spacing that does not divide the length: samples at 0, 4 and the final vertex at 6.
  const wide = getGPURasterProfileParameterValues({spacing: 4}, makeExtentGrid(8, 8, EXTENT));
  const wideProfile = profileRasterOnCPU(ramp, wide, linePositions, lineOffsets);
  expect(Array.from(wideProfile.sampleOffsets)).toEqual([0, 3, 4, 4, 7]);
  expect(Array.from(wideProfile.sampleDistances.slice(0, 3))).toEqual([0, 4, 6]);
});

it('raster-profile oracle: NaN-aware gain and loss', () => {
  const raster = createPlane(10, 0);
  raster.values[4] = NaN;
  const parameters = getGPURasterProfileParameterValues(
    {method: 'nearest', spacing: 1},
    makeExtentGrid(8, 8, EXTENT)
  );
  const profile = profileRasterOnCPU(
    raster,
    parameters,
    new Float32Array([0.5, 0.5, 7.5, 0.5]),
    Uint32Array.from([0, 2])
  );
  // Samples at x = 0.5 ... 7.5: cell 4 is NaN and skipped; consecutive valid samples are 10 apart.
  expect(profile.sampleValues[4]).toBeNaN();
  expect(profile.lineGains[0]).toBe(70);
  expect(profile.lineLosses[0]).toBe(0);
  expect(profile.sampleCumulativeGain[4]).toBe(30);
  expect(profile.lineMinimums[0]).toBe(0);
  expect(profile.lineMaximums[0]).toBe(70);
});

it('GPURasterSampling and GPURasterProfile validate their inputs', () => {
  withGraph(graph => {
    const parameters = view(graph, 'float32', 12);
    const sampling = (overrides: Record<string, unknown> = {}) =>
      new GPURasterSampling({
        grid: makeGPURasterMetadata({width: 4, height: 4}),
        values: view(graph, 'float32', 16),
        positions: view(graph, 'float32x2', 5),
        parameters,
        output: {values: view(graph, 'float32', 5)},
        ...overrides
      } as ConstructorParameters<typeof GPURasterSampling>[0]);
    expect(sampling({id: 'custom'}).id).toBe('custom');
    expect(() => sampling({grid: makeGPURasterMetadata({width: 0, height: 4})})).toThrow(
      /grid dimensions/
    );
    expect(() => sampling({noDataValue: NaN})).toThrow(/noDataValue/);
    expect(() => sampling({values: view(graph, 'float32', 15)})).toThrow(/values/);
    expect(() => sampling({values: view(graph, 'uint32', 16)})).toThrow(/float32/);
    expect(() => sampling({mask: view(graph, 'uint32', 15)})).toThrow(/mask/);
    expect(() => sampling({positions: view(graph, 'float32', 5)})).toThrow(/float32x2/);
    expect(() => sampling({parameters: view(graph, 'float32', 11)})).toThrow(/parameters/);
    expect(() => sampling({pointCount: view(graph, 'float32', 1)})).toThrow(/uint32/);
    expect(() => sampling({output: {}})).toThrow(/output.values/);
    expect(() => sampling({output: {values: view(graph, 'float32', 4)}})).toThrow(/at least 5/);

    const profile = (overrides: Record<string, unknown> = {}) =>
      new GPURasterProfile({
        grid: makeGPURasterMetadata({width: 4, height: 4}),
        values: view(graph, 'float32', 16),
        positions: view(graph, 'float32x2', 6),
        lineOffsets: view(graph, 'uint32', 3),
        parameters,
        output: {
          status: {count: view(graph, 'uint32', 1), overflow: view(graph, 'uint32', 1)},
          sampleValues: view(graph, 'float32', 10)
        },
        ...overrides
      } as ConstructorParameters<typeof GPURasterProfile>[0]);
    const created = profile();
    expect(created.lineCount).toBe(2);
    expect(created.sampleCapacity).toBe(10);
    expect(() => profile({lineOffsets: view(graph, 'uint32', 1)})).toThrow(/lineOffsets/);
    expect(() => profile({output: {status: {count: view(graph, 'uint32', 1)}}})).toThrow(
      /overflow/
    );
    expect(() =>
      profile({
        output: {status: {count: view(graph, 'uint32', 1), overflow: view(graph, 'uint32', 1)}}
      })
    ).toThrow(/sampleCapacity/);
    expect(() =>
      profile({
        output: {
          status: {count: view(graph, 'uint32', 1), overflow: view(graph, 'uint32', 1)},
          sampleValues: view(graph, 'float32', 10),
          lineGains: view(graph, 'float32', 1)
        }
      })
    ).toThrow(/lineGains/);
    expect(() =>
      profile({
        output: {
          status: {count: view(graph, 'uint32', 1), overflow: view(graph, 'uint32', 1)},
          sampleValues: view(graph, 'float32', 10),
          lineSampleOffsets: view(graph, 'uint32', 2)
        }
      })
    ).toThrow(/lineSampleOffsets/);
    expect(() => profile({sampleCapacity: 12})).toThrow(/sampleValues/);
    const shared = view(graph, 'float32', 10);
    expect(() =>
      profile({
        output: {
          status: {count: view(graph, 'uint32', 1), overflow: view(graph, 'uint32', 1)},
          sampleValues: shared,
          sampleDistances: shared
        }
      })
    ).toThrow(/share/);
  });
});

it.each([
  'bare values',
  'band values',
  'band validity'
] as const)('sampling and profile alias errors name the exact ports for %s', inputKind => {
  withGraph(graph => {
    const values = view(graph, 'float32', 4);
    const validity = view(graph, 'uint32', 4);
    const grid = makeGPURasterMetadata({width: 2, height: 2});
    const input =
      inputKind === 'bare values' ? values : makeGPURasterBand(values, grid, {validity});
    const inputPort =
      inputKind === 'bare values'
        ? 'values'
        : inputKind === 'band values'
          ? 'values.storage.values'
          : 'values.validity';
    const positions = view(graph, 'float32x2', 2);
    const parameters = view(graph, 'float32', 12);
    const sampleOutputPort =
      inputKind === 'band validity' ? 'output.outputValidity' : 'output.values';
    expect(
      () =>
        new GPURasterSampling({
          id: 'sampling-pair',
          grid,
          values: input,
          positions,
          parameters,
          output: {
            values: inputKind === 'band validity' ? view(graph, 'float32', 2) : values,
            outputValidity: inputKind === 'band validity' ? validity : undefined
          }
        })
    ).toThrow(
      `sampling-pair outputs must not share buffers with inputs: ${sampleOutputPort} conflicts with ${inputPort}`
    );
    const profileOutputPort =
      inputKind === 'band validity' ? 'output.status.count' : 'output.sampleValues';
    const count =
      inputKind === 'band validity'
        ? graph.createDataView(validity.buffer, {format: 'uint32', length: 1})
        : view(graph, 'uint32', 1);
    expect(
      () =>
        new GPURasterProfile({
          id: 'profile-pair',
          grid,
          values: input,
          positions,
          parameters,
          lineOffsets: view(graph, 'uint32', 2),
          sampleCapacity: 4,
          output: {
            status: {count, overflow: view(graph, 'uint32', 1)},
            sampleValues: inputKind === 'band validity' ? view(graph, 'float32', 4) : values
          }
        })
    ).toThrow(
      `profile-pair outputs must not share buffers with inputs: ${profileOutputPort} conflicts with ${inputPort}`
    );
  });
});

it('raster alias diagnostics recognise separate handles for one physical buffer', () => {
  withGraph(graph => {
    const buffer = graph.device.createBuffer({byteLength: 64, usage: Buffer.STORAGE});
    try {
      const values = importGraphBuffer(graph, 'raster-input', buffer, 'float32', 4);
      const output = importGraphBuffer(graph, 'raster-output', buffer, 'float32', 4);
      const props = {
        grid: makeGPURasterMetadata({width: 2, height: 2}),
        values,
        positions: view(graph, 'float32x2', 2),
        parameters: view(graph, 'float32', 12)
      };
      expect
        .soft(
          () => new GPURasterSampling({id: 'sampling-physical', ...props, output: {values: output}})
        )
        .toThrow('output.values conflicts with values');
      expect
        .soft(
          () =>
            new GPURasterProfile({
              id: 'profile-physical',
              ...props,
              lineOffsets: view(graph, 'uint32', 2),
              sampleCapacity: 4,
              output: {
                sampleValues: output,
                status: {count: view(graph, 'uint32', 1), overflow: view(graph, 'uint32', 1)}
              }
            })
        )
        .toThrow('output.sampleValues conflicts with values');
    } finally {
      buffer.destroy();
    }
  });
});

it('GPURasterSampling and GPURasterProfile create deterministic nodes', () => {
  withGraph(graph => {
    const parameters = view(graph, 'float32', 12);
    const sampling = new GPURasterSampling({
      id: 's',
      grid: makeGPURasterMetadata({width: 4, height: 4}),
      values: view(graph, 'float32', 16),
      positions: view(graph, 'float32x2', 5),
      parameters,
      output: {values: view(graph, 'float32', 5), outputValidity: view(graph, 'uint32', 5)}
    });
    expect(sampling.getCommandNodes(graph).map(node => node.id)).toEqual(['s-sample']);
    const profile = new GPURasterProfile({
      id: 'p',
      grid: makeGPURasterMetadata({width: 4, height: 4}),
      values: view(graph, 'float32', 16),
      positions: view(graph, 'float32x2', 6),
      lineOffsets: view(graph, 'uint32', 3),
      sampleCapacity: 8,
      parameters,
      output: {
        status: {count: view(graph, 'uint32', 1), overflow: view(graph, 'uint32', 1)},
        sampleValues: view(graph, 'float32', 8)
      }
    });
    const ids = profile.getCommandNodes(graph).map(node => node.id);
    expect(ids[0]).toBe('p-truncated-fill');
    expect(ids[1]).toBe('p-measure');
    expect(ids).toContain('p-locate');
    expect(ids).toContain('p-sample');
    // No cumulative column or line summary is requested, so the cumulative scan is skipped.
    expect(ids.filter(id => id.startsWith('p-cumulative'))).toEqual([]);
    expect(ids.at(-1)).toBe('p-finish-counts');
    expect(ids).not.toContain('p-finish-lines');

    // WRT-11: geometric counts need the distance/count scans, but no sample columns.
    const statusProfile = new GPURasterProfile({
      ...profile.props,
      id: 'status-only',
      output: {
        status: {count: view(graph, 'uint32', 1), overflow: view(graph, 'uint32', 1)}
      }
    });
    const statusIds = statusProfile.getCommandNodes(graph).map(node => node.id);
    expect(statusIds.slice(0, 5)).toEqual([
      'status-only-truncated-fill',
      'status-only-measure',
      'status-only-measure-carries',
      'status-only-measure-distances',
      'status-only-sample-counts'
    ]);
    expect(statusIds.at(-1)).toBe('status-only-finish-counts');
    expect(statusIds).not.toContain('status-only-locate');
    expect(statusIds).not.toContain('status-only-sample');
    expect(statusIds.filter(id => id.startsWith('status-only-cumulative'))).toEqual([]);
    expect(statusIds).not.toContain('status-only-finish-lines');
  });
});

it('GPURasterSampling and GPURasterProfile take the contract mask', () => {
  withGraph(graph => {
    const parameters = view(graph, 'float32', 12);
    const values = view(graph, 'float32', 16);
    const positions = view(graph, 'float32x2', 5);
    const sampling = (overrides: Record<string, unknown>) =>
      new GPURasterSampling({
        id: 'sampling',
        grid: makeGPURasterMetadata({width: 4, height: 4}),
        values,
        positions,
        parameters,
        output: {values: view(graph, 'float32', 5)},
        ...overrides
      } as ConstructorParameters<typeof GPURasterSampling>[0]);
    const mask = view(graph, 'uint32', 16);
    expect(sampling({mask}).props.mask).toBe(mask);
    expect(() => sampling({mask: view(graph, 'uint32', 15)})).toThrow(/^sampling mask/);
    const profile = (overrides: Record<string, unknown>) =>
      new GPURasterProfile({
        id: 'profile',
        grid: makeGPURasterMetadata({width: 4, height: 4}),
        values,
        positions: view(graph, 'float32x2', 6),
        lineOffsets: view(graph, 'uint32', 3),
        parameters,
        sampleCapacity: 10,
        output: {
          status: {count: view(graph, 'uint32', 1), overflow: view(graph, 'uint32', 1)},
          sampleValues: view(graph, 'float32', 10)
        },
        ...overrides
      } as ConstructorParameters<typeof GPURasterProfile>[0]);
    expect(profile({mask}).props.mask).toBe(mask);
    expect(() => profile({mask: view(graph, 'uint32', 15)})).toThrow(/^profile mask/);
  });
});
