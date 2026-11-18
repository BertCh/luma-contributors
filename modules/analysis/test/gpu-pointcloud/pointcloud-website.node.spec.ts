// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUPointCloudVoxelDownsample} from '@luma.gl/graph/gpu-pointcloud';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';
import {computePointCloudVoxelMeans} from '../../../../website/src/gpu/runs/entries/dataframe-pointcloud-records/_lib/pointcloud-voxel-means';
import {
  getPointCloudFeatureClass,
  getPointCloudFeatureSupportText,
  getPointCloudChangeSupportText,
  summarizePointCloudChanges
} from '../../../../website/src/gpu/runs/entries/dataframe-pointcloud-records/_lib/pointcloud-display';

it('prepares arithmetic voxel means rather than selected point positions', () => {
  const means = computePointCloudVoxelMeans(
    new Float32Array([0.03125, 0.05, 0.05, 0.15625, 0.05, 0.05]),
    0.2,
    [1, 1, 1]
  );
  expect(means[0]).toBe(0.09375);
  expect([...computePointCloudVoxelMeans(new Float32Array(), 0.2, [1, 1, 1])]).toEqual([0, 0, 0]);
});

it('G6 rejects partial XYZ rows before voxel accumulation', () => {
  expect(() =>
    computePointCloudVoxelMeans(new Float32Array([0.25, 0.5, 0.75, 9]), 1, [1, 1, 1])
  ).toThrow(/positions.*XYZ/);
});

it('H5 rejects fractional voxel dimensions before allocating a partial XYZ output', () => {
  expect(() =>
    computePointCloudVoxelMeans(new Float32Array([0.25, 0.5, 0.75]), 1, [1.5, 1, 1])
  ).toThrow(/gridSize/);
});

it('H5 rejects a nonfinite voxel origin rather than returning an empty-voxel value', () => {
  expect(() =>
    computePointCloudVoxelMeans(new Float32Array([0.25, 0.5, 0.75]), 1, [1, 1, 1], [NaN, 0, 0])
  ).toThrow(/origin.*finite.*float32/);
});

it('H5 preserves complete XYZ subviews without reading spare backing rows', () => {
  const positions = new Float32Array([99, 99, 99, 0.25, 0.5, 0.75, 88, 88, 88]).subarray(3, 6);
  expect([...computePointCloudVoxelMeans(positions, 1, [1, 1, 1])]).toEqual([0.25, 0.5, 0.75]);
});

it.each([
  {gridSize: [1.5, 1, 1] as const, origin: [0, 0, 0] as const},
  {gridSize: [1, 1, 1] as const, origin: [NaN, 0, 0] as const},
  {gridSize: [0, 1, 1] as const, origin: [0, 0, 0] as const},
  {gridSize: [1, 1, 1] as const, origin: [1e100, 0, 0] as const},
  {gridSize: [1, 1, 1] as const, origin: [0, 0, 0] as const}
])('H5 helper and downsampler agree on grid/origin validity: %j', ({gridSize, origin}) => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const construct = () =>
    new GPUPointCloudVoxelDownsample({
      positions: createTransientView(graph, 'positions', 'float32x3', 1),
      voxelSize: 1,
      gridSize,
      origin,
      counts: createTransientView(graph, 'counts', 'uint32', 1),
      representatives: createTransientView(graph, 'rows', 'uint32', 1),
      output: createTransientView(graph, 'selected', 'float32x3', 1)
    });
  const compute = () =>
    computePointCloudVoxelMeans(new Float32Array([0.25, 0.5, 0.75]), 1, gridSize, origin);
  if (
    gridSize[0] === 1 &&
    origin.every(Number.isFinite) &&
    origin.every(value => Number.isFinite(Math.fround(value)))
  ) {
    expect(construct).not.toThrow();
    expect([...compute()]).toEqual([0.25, 0.5, 0.75]);
  } else {
    expect(construct).toThrow();
    expect(compute).toThrow();
  }
});

it.each([
  {gridSize: [0xffffffff, 1, 1] as const},
  {gridSize: [65536, 65536, 1] as const},
  {gridSize: [Number.MAX_SAFE_INTEGER + 1, 1, 1] as const}
])('H5 rejects unsafe voxel capacity before allocation: %j', ({gridSize}) => {
  const allocation = vi.spyOn(globalThis, 'Float64Array').mockImplementation(function () {
    throw new Error('unexpected allocation');
  });
  try {
    expect(() =>
      computePointCloudVoxelMeans(new Float32Array([0.25, 0.5, 0.75]), 1, gridSize)
    ).toThrow(/gridSize/);
    expect(allocation).not.toHaveBeenCalled();
  } finally {
    allocation.mockRestore();
  }
});

it.each([
  0,
  -1,
  NaN,
  Infinity,
  1e100,
  1e-100
])('G6 rejects unusable float32 voxel size %s', voxelSize => {
  expect(() =>
    computePointCloudVoxelMeans(new Float32Array([0.25, 0.5, 0.75]), voxelSize, [1, 1, 1])
  ).toThrow(/voxelSize.*positive.*finite.*float32/);
});

it('never assigns a shape class to an isolated or unconverged feature', () => {
  expect(Number.isNaN(getPointCloudFeatureClass(0, 0, 0, 1))).toBe(true);
  expect(Number.isNaN(getPointCloudFeatureClass(0, 1, 0, 4))).toBe(true);
  expect(getPointCloudFeatureClass(0, 1, 0, 2)).toBe(1);
  expect(getPointCloudFeatureSupportText(2)).toBe('truncated support');
  expect(getPointCloudFeatureSupportText(1)).toBe('unusable shape');
});

it('suppresses unsupported change and reports the reason instead of stable zero', () => {
  expect(summarizePointCloudChanges([0], [2])).toEqual({
    strongest: undefined,
    changed: 0,
    supported: 0
  });
  expect(getPointCloudChangeSupportText(2)).toBe('insufficiently supported');
  expect(summarizePointCloudChanges([0, 0.125, 10], [2, 0, 0x100])).toEqual({
    strongest: 0.125,
    changed: 1,
    supported: 1
  });
  expect(getPointCloudChangeSupportText(0x100)).toBe('incomplete grid support');
});
