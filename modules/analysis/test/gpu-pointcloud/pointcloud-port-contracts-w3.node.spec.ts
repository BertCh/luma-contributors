// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {expect, it} from 'vitest';
import {GPUCommandGraph, GraphVectorView, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  GPUPointCloudChange,
  type GPUPointCloudChangeProps
} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-change';
import {
  GPUPointCloudGround,
  type GPUPointCloudGroundProps
} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-ground';
import {
  GPUPointCloudICP,
  type GPUPointCloudICPProps
} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-icp';
import {
  GPUPointCloudNormalScale,
  type GPUPointCloudNormalScaleProps
} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-normal-scale';
import {GPUPointCloudVoxelMerge} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-voxel-merge';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

it('VAL-01: rejects wrong element formats and missing declared ports before dereferencing', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let sequence = 0;
    const view = <Format extends 'uint32' | 'float32' | 'float32x3'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `ports-w3-${sequence++}`, format, length);
    const change: GPUPointCloudChangeProps = {
      reference: view('float32x3', 3),
      comparison: view('float32x3', 3),
      cores: view('float32x3', 3),
      normals: view('float32x3', 3),
      alignment: 'already-aligned',
      radius: 1,
      halfDepth: 1,
      output: {
        changes: view('float32x3', 3),
        roughness: view('float32x3', 3),
        counts: view('uint32', 6),
        status: view('uint32', 3)
      }
    };
    const ground: GPUPointCloudGroundProps = {
      positions: view('float32x3', 3),
      classifications: view('uint32', 3),
      grid: {origin: [0, 0], width: 2, height: 2, cellSize: 1},
      minimumSurface: view('float32', 4),
      coverage: view('uint32', 4),
      radii: [1],
      initialThreshold: 1,
      slope: 1,
      maximumThreshold: 2,
      output: {
        groundMask: view('uint32', 3),
        outputClassifications: view('uint32', 3),
        status: view('uint32', 3)
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
    const normal: GPUPointCloudNormalScaleProps = {
      positions: view('float32x3', 3),
      radii: [1],
      capacity: 3,
      parameters: view('float32', 5),
      output: {
        normals: view('float32x3', 3),
        normalScales: view('float32', 3),
        normalVariances: view('float32', 3),
        status: view('uint32', 3)
      }
    };
    // S3: optional ports need explicit wrong-format coverage even when omitted by the base fixtures.
    expect(
      () =>
        new GPUPointCloudChange({
          ...change,
          alignment: {transform: view('uint32', 16)}
        } as unknown as GPUPointCloudChangeProps)
    ).toThrow(/alignment.transform.*format/);
    expect(
      () =>
        new GPUPointCloudChange({
          ...change,
          parameters: view('float32', 12)
        } as unknown as GPUPointCloudChangeProps)
    ).toThrow(/parameters.*format/);
    expect(
      () =>
        new GPUPointCloudGround({
          ...ground,
          gridParameters: view('uint32', 4)
        } as unknown as GPUPointCloudGroundProps)
    ).toThrow(/gridParameters.*format/);
    expect(
      () =>
        new GPUPointCloudICP({
          ...icp,
          targetNormals: view('uint32', 3)
        } as unknown as GPUPointCloudICPProps)
    ).toThrow(/targetNormals.*format/);
    expect(
      () =>
        new GPUPointCloudICP({
          ...icp,
          parameters: view('uint32', 22)
        } as unknown as GPUPointCloudICPProps)
    ).toThrow(/parameters.*format/);
    expect(
      () =>
        new GPUPointCloudNormalScale({
          ...normal,
          queryRows: view('float32', 3)
        } as unknown as GPUPointCloudNormalScaleProps)
    ).toThrow(/queryRows.*format/);
    expect(
      () =>
        new GPUPointCloudNormalScale({
          ...normal,
          gridParameters: view('uint32', 4)
        } as unknown as GPUPointCloudNormalScaleProps)
    ).toThrow(/gridParameters.*format/);
    const cases = [
      {
        props: change,
        create: (props: unknown) => new GPUPointCloudChange(props as GPUPointCloudChangeProps)
      },
      {
        props: ground,
        create: (props: unknown) => new GPUPointCloudGround(props as GPUPointCloudGroundProps)
      },
      {
        props: icp,
        create: (props: unknown) => new GPUPointCloudICP(props as GPUPointCloudICPProps)
      },
      {
        props: normal,
        create: (props: unknown) =>
          new GPUPointCloudNormalScale(props as GPUPointCloudNormalScaleProps)
      }
    ];
    for (const {props, create} of cases) {
      expect(() => create(props)).not.toThrow();
      for (const [name, original] of Object.entries(props)) {
        if (!original || typeof original !== 'object' || !('format' in original)) continue;
        const wrong = view(original.format === 'uint32' ? 'float32' : 'uint32', original.length);
        expect(() => create({...props, [name]: wrong}), name).toThrow(
          new RegExp(`${name}.*format`)
        );
        expect(() => create({...props, [name]: undefined}), name).toThrow(new RegExp(name));
      }
      for (const [name, original] of Object.entries(props.output)) {
        const wrong = view(original.format === 'uint32' ? 'float32' : 'uint32', original.length);
        expect(() => create({...props, output: {...props.output, [name]: wrong}}), name).toThrow(
          new RegExp(`output.${name}.*format`)
        );
        expect(
          () => create({...props, output: {...props.output, [name]: undefined}}),
          name
        ).toThrow(new RegExp(`output.${name}`));
      }
    }
  } finally {
    device.destroy();
  }
});

it('ENT-02: VoxelMerge accepts chunked vectors from the CommonJS module condition', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const {GraphVectorView: ForeignVector} = createRequire(import.meta.url)(
      '@luma.gl/graph/gpu-core'
    ) as {
      GraphVectorView: new (
        props: ConstructorParameters<typeof GraphVectorView>[0]
      ) => GraphVectorView;
    };
    let sequence = 0;
    const vector = <Format extends 'uint32' | 'float32x3'>(format: Format) => {
      const data = [1, 2].map(length =>
        createTransientView(graph, `foreign-${sequence++}`, format, length)
      );
      const stride = format === 'uint32' ? 1 : 3;
      return new ForeignVector({
        id: `vector-${sequence++}`,
        name: 'voxels',
        format,
        length: 3,
        valueLength: 3 * stride,
        stride,
        byteStride: stride * 4,
        rowByteLength: stride * 4,
        data
      }) as GraphVectorView<Format>;
    };
    const counts = vector('uint32');
    expect(counts instanceof GraphVectorView).toBe(false);
    const contributor = new GPUPointCloudVoxelMerge({
      partitionPolicy: 'disjoint-owned-rows',
      parts: [
        {
          counts,
          centroids: vector('float32x3'),
          representatives: vector('uint32'),
          sourceRowOffset: 0
        }
      ],
      output: {
        counts: vector('uint32'),
        centroids: vector('float32x3'),
        representatives: vector('uint32'),
        status: vector('uint32')
      }
    });
    expect(contributor.getCommandNodes(graph).length).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});
