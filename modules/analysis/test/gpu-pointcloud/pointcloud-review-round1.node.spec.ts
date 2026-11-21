// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  GPUPointCloudSurfaces,
  getRasterMetadataFromPointCloudGrid
} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-surfaces';
import {
  GPUPointCloudChange,
  decodeGPUPointCloudChangeStatus,
  decodeGPUPointCloudChangeRow,
  type GPUPointCloudChangeReadback
} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-change';
import {decodeGPUPointCloudNormalScaleStatus} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-normal-scale';
import {
  GPUPointCloudICP,
  type GPUPointCloudICPProps
} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-icp';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../utils/gpu-contributor-test-utils';

function makeDeclarations() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let sequence = 0;
  const view = <Format extends 'float32' | 'float32x3' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `review-${sequence++}`, format, length);
  const change = {
    reference: view('float32x3', 3),
    comparison: view('float32x3', 3),
    cores: view('float32x3', 1),
    normals: view('float32x3', 1),
    alignment: 'already-aligned' as const,
    radius: 1,
    halfDepth: 1,
    output: {
      changes: view('float32x3', 1),
      roughness: view('float32x3', 1),
      counts: view('uint32', 2),
      status: view('uint32', 1)
    }
  };
  const icp: GPUPointCloudICPProps = {
    source: view('float32x3', 3),
    target: view('float32x3', 3),
    maximumDistance: 1,
    maximumIterations: 1,
    output: {
      transform: view('float32', 16),
      history: view('float32x3', 1),
      inlierMask: view('uint32', 3),
      status: view('uint32', 1)
    }
  };
  return {device, graph, view, change, icp};
}

it('B1: raster metadata uses the float32 origin and step of the surface producer', () => {
  const grid = {origin: [1000000.03, 200.03] as const, cellSize: 0.1, width: 2, height: 2};
  // IEEE float32 literals, independently derived: originX=1000000,
  // originY=200.029998779296875, cellSize=0.10000000149011612.
  for (const rowDirection of ['south', 'north'] as const) {
    const affine = getRasterMetadataFromPointCloudGrid({...grid, rowDirection}).affine;
    expect(affine).toEqual([
      0.10000000149011612,
      0,
      1000000,
      0,
      rowDirection === 'south' ? -0.10000000149011612 : 0.10000000149011612,
      rowDirection === 'south' ? 200.2299987822771 : 200.02999877929688
    ]);
    const column = (1000000.125 - affine[2]) / affine[0];
    expect(Number.isFinite(column)).toBe(true);
    expect(column).toBeCloseTo(1.2499999813735487, 12);
    expect(Math.floor(column)).toBe(1);
  }
});

it('B3: status decoders reject other typed element formats before numeric conversion', () => {
  for (const decode of [decodeGPUPointCloudChangeStatus, decodeGPUPointCloudNormalScaleStatus]) {
    for (const value of [
      new Int8Array([0]),
      new Uint8ClampedArray(4),
      Float32Array.of(NaN),
      Float64Array.of(0),
      [0],
      new DataView(new ArrayBuffer(4))
    ])
      expect(() => decode(value as unknown as Uint32Array)).toThrow(/Uint32Array/);
    expect(decode(new Uint8Array(4)).word).toBe(0);
    expect(decode(new ArrayBuffer(4)).word).toBe(0);
  }
});

it('B3: row decoding refuses wrong float and integer port element types', () => {
  const output: GPUPointCloudChangeReadback = {
    changes: Float32Array.of(-2, -3, -1),
    roughness: Float32Array.of(0.5, 0.75, 1),
    counts: Uint32Array.of(4, 5),
    status: Uint32Array.of(256)
  };
  for (const port of ['changes', 'roughness', 'counts', 'status'] as const) {
    const wrong =
      port === 'changes' || port === 'roughness'
        ? Uint32Array.of(0, 0, 0)
        : Float32Array.of(2.75, -1);
    expect(() =>
      decodeGPUPointCloudChangeRow({...output, [port]: wrong} as GPUPointCloudChangeReadback, 0)
    ).toThrow(new RegExp(port));
  }
});

it('S1: decoding one row allocates no readback-sized typed arrays', () => {
  const output: GPUPointCloudChangeReadback = {
    changes: new Float32Array(15000),
    roughness: new Float32Array(15000),
    counts: new Uint32Array(10000),
    status: new Uint32Array(5000)
  };
  const raw = Object.fromEntries(
    Object.entries(output).map(([name, value]) => {
      const typed = value as Float32Array | Uint32Array;
      const bytes = new Uint8Array(typed.byteLength + 1);
      bytes.set(new Uint8Array(typed.buffer), 1);
      return [name, bytes.subarray(1)];
    })
  ) as GPUPointCloudChangeReadback;
  const floats = vi.spyOn(globalThis, 'Float32Array');
  const words = vi.spyOn(globalThis, 'Uint32Array');
  try {
    expect(decodeGPUPointCloudChangeRow(output, 4999).referenceCount).toBe(0);
    expect(decodeGPUPointCloudChangeRow(raw, 4999).referenceCount).toBe(0);
    expect(floats).not.toHaveBeenCalled();
    expect(words).not.toHaveBeenCalled();
  } finally {
    floats.mockRestore();
    words.mockRestore();
  }
});

it('S3: complete row plus partial word and final-lane truncation are distinct refusals', () => {
  const output: GPUPointCloudChangeReadback = {
    changes: new Float32Array(6),
    roughness: new Float32Array(6),
    counts: new Uint32Array(4),
    status: new Uint32Array(2)
  };
  for (const port of ['changes', 'roughness', 'counts', 'status'] as const) {
    const value = output[port] as Float32Array | Uint32Array;
    const partial = new Uint8Array(value.byteLength + 1);
    expect(() => decodeGPUPointCloudChangeRow({...output, [port]: partial}, 1)).toThrow(
      new RegExp(port)
    );
    const missingLane = new Uint8Array(value.buffer, 0, value.byteLength - 4);
    expect(() => decodeGPUPointCloudChangeRow({...output, [port]: missingLane}, 1)).toThrow(
      new RegExp(port)
    );
  }
});

it('B4: ICP rejects unknown metric and robust method at construction', () => {
  const {device, icp} = makeDeclarations();
  try {
    expect(
      () => new GPUPointCloudICP({...icp, metric: 'plane' as GPUPointCloudICPProps['metric']})
    ).toThrow(/metric.*point-to-point.*point-to-plane.*plane/);
    expect(
      () => new GPUPointCloudICP({...icp, robust: {method: 'bisquare' as 'tukey', scale: 1}})
    ).toThrow(/robust.method.*huber.*tukey.*bisquare/);
  } finally {
    device.destroy();
  }
});

it('N1: change captures its nested alignment descriptor and keeps the validated handle', () => {
  const {device, graph, view, change} = makeDeclarations();
  try {
    const transform = view('float32', 16);
    const alignment = {transform};
    const contributor = new GPUPointCloudChange({...change, alignment});
    alignment.transform = view('uint32', 16) as unknown as typeof transform;
    expect(contributor.props.alignment).not.toBe(alignment);
    expect(
      contributor.props.alignment !== 'already-aligned' && contributor.props.alignment.transform
    ).toBe(transform);
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('N2: runtime coverage uses exponent bits to distinguish missing values from finite zero', () => {
  const {device, graph, view} = makeDeclarations();
  try {
    const contributor = new GPUPointCloudSurfaces({
      positions: view('float32x3', 3),
      classifications: view('uint32', 3),
      grid: {origin: [0, 0], width: 2, height: 2, cellSize: 1},
      gridParameters: view('float32', 4),
      output: {
        terrain: view('float32', 4),
        surface: view('float32', 4),
        canopy: view('float32', 4),
        coverage: view('uint32', 4)
      }
    });
    const finalizer = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).find(
      kernel => kernel.id === 'gpu-pointcloud-surfaces-finalize'
    )!;
    expect(finalizer.source).toContain('(bitcast<u32>(minimum) & 0x7f800000u) != 0x7f800000u');
    expect(finalizer.source).toContain('(bitcast<u32>(maximum) & 0x7f800000u) != 0x7f800000u');
  } finally {
    device.destroy();
  }
});

it('N3: surface and change search-window builders emit float literals at large grid origins', () => {
  const {device, graph, view, change} = makeDeclarations();
  try {
    const surfaces = new GPUPointCloudSurfaces({
      positions: view('float32x3', 3),
      classifications: view('uint32', 3),
      grid: {origin: [1e19, -1e19], width: 2, height: 2, cellSize: 1},
      output: {
        terrain: view('float32', 4),
        surface: view('float32', 4),
        canopy: view('float32', 4),
        coverage: view('uint32', 4)
      }
    });
    const surface = getGPUCommandNodeKernelSources(surfaces.getCommandNodes(graph)).find(
      kernel => kernel.id === 'gpu-pointcloud-surfaces'
    )!;
    expect(surface.source).toContain('vec2f(9999999980506448000.0,-9999999980506448000.0)');
    const contributor = new GPUPointCloudChange({
      ...change,
      grid: {origin: [1e19, -1e19, 0], dimensions: [2, 2, 2], cellSize: 1}
    });
    const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    for (const id of [
      'gpu-pointcloud-change-reference',
      'gpu-pointcloud-change-comparison',
      'gpu-pointcloud-change'
    ]) {
      const kernel = sources.find(kernel => kernel.id === id)!;
      expect(kernel.source).toContain(
        'core - vec3f(9999999980506448000.0,-9999999980506448000.0,0.0)'
      );
    }
  } finally {
    device.destroy();
  }
});

it('B4: ICP retains the validated robust option values after descriptor edits', () => {
  const {device, graph, icp} = makeDeclarations();
  try {
    const robust = {method: 'huber' as 'huber' | 'tukey', scale: 1};
    const contributor = new GPUPointCloudICP({...icp, robust});
    robust.method = 'tukey';
    robust.scale = 0;
    expect(contributor.props.robust).toEqual({method: 'huber', scale: 1});
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});
