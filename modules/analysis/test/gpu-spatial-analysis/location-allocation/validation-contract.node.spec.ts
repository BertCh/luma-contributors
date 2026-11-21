// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {computeLocationAllocationOnCPU} from '../../../src/gpu-spatial-analysis/location-allocation/cpu-location-allocation-operation';
import {GPULocationAllocation} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-location-allocation';
import {GPUSpatialKMeans} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-spatial-kmeans';
import {getGPUDBSCANParameterValues} from '../../../src/gpu-spatial-analysis/spatial-clustering/spatial-clustering-parameters';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('prefixes CPU facility shape failures with the contributor ID and named property constraint', () => {
  expect(() =>
    computeLocationAllocationOnCPU(
      {costs: Float32Array.of(1)},
      {
        facilityCount: 2,
        demandCount: 2,
        maximumFacilityCount: 1,
        operation: 'p-median'
      }
    )
  ).toThrow(/^location-allocation costs length must equal demandCount \* facilityCount/);
});

it('names the DBSCAN parameter and float32 requirement', () => {
  expect(() =>
    getGPUDBSCANParameterValues({
      bounds: [0, 0, 1, 1],
      epsilon: Infinity,
      minimumSamples: 1
    })
  ).toThrow(/^GPUDBSCAN epsilon must be finite in float32/);
});

it('keeps custom IDs and actual topology constraints in facility and clustering failures', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let sequence = 0;
    const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `view-${sequence++}`, format, length);
    expect(
      () =>
        new GPULocationAllocation({
          id: 'custom-facility',
          operation: 'p-median',
          facilityCount: 2,
          maximumFacilityCount: 1,
          costs: view('float32', 3),
          output: {
            selectedFacilities: view('uint32', 2),
            assignments: view('uint32', 2),
            objective: view('float32', 1),
            status: view('uint32', 7)
          }
        })
    ).toThrow(/^custom-facility costs length must be a positive multiple of facilityCount/);
    expect(
      () =>
        new GPUSpatialKMeans({
          id: 'custom-cluster',
          clusterCount: 257,
          maximumIterations: 1,
          positions: view('float32x2', 2),
          parameters: view('uint32', 4),
          output: {labels: view('uint32', 2), centers: view('float32x2', 257)}
        })
    ).toThrow(/^custom-cluster clusterCount must be.*256/);
  } finally {
    device.destroy();
  }
});
