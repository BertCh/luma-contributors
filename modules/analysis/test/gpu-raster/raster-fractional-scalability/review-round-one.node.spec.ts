// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {runInNewContext} from 'node:vm';
import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {GPURasterFractionalCategoricalStatistics} from '../../../src/raster-analysis/gpu-raster-fractional-categorical-statistics';
import {getGPURasterFractionalCoverageParameterValues} from '../../../src/raster-analysis/gpu-raster-fractional-coverage';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/gpu-raster-grid';
import {
  getRasterRowAreaScale,
  getRasterRowEdgeWGSL
} from '../../../src/raster-analysis/raster-grid/raster-cell-area-scale';
import {GPURasterZonalStatistics} from '../../../src/raster-analysis/raster-zonal-statistics/gpu-raster-zonal-statistics';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createScalabilityProps} from './raster-fractional-scalability-fixture';

it('B1 rejects undersized destinations and invalid cell modes in planning as in construction', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const {categorical} = createScalabilityProps(graph, 2, 2);
  const props = {
    ...categorical,
    output: {
      ...categorical.output,
      classAreas: createTransientView(graph, 'short-class-areas', 'float32', 2)
    }
  };
  const plan = GPURasterFractionalCategoricalStatistics.getExecutionPlan(props, device.limits);
  expect(plan.feasible).toBe(false);
  expect(
    plan.constraints.some(
      constraint => !constraint.satisfied && constraint.quantity.includes('output.classAreas')
    )
  ).toBe(true);
  expect(() => new GPURasterFractionalCategoricalStatistics(props)).toThrow(/output.classAreas/);
  expect(() =>
    GPURasterFractionalCategoricalStatistics.getExecutionPlan(
      {...categorical, cellSizeMode: 'metres' as never},
      device.limits
    )
  ).toThrow(/uniform.*web-mercator.*geographic/);
});

it('B1 preflights actual storage-binding counts before graph resources are declared', () => {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxStorageBuffersPerShaderStage: 4}
  });
  const graph = new GPUCommandGraph(device);
  const {categorical} = createScalabilityProps(graph, 2, 2);
  const props = {...categorical, parameters: undefined};
  const plan = GPURasterFractionalCategoricalStatistics.getExecutionPlan(props, device.limits);
  expect(plan.feasible).toBe(false);
  expect(
    plan.constraints.some(
      constraint =>
        !constraint.satisfied &&
        constraint.quantity.includes('summarize') &&
        constraint.received === 7
    )
  ).toBe(true);
  const resources = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => new GPURasterFractionalCategoricalStatistics(props).getCommandNodes(graph)).toThrow(
    /storage.*binding/
  );
  expect(resources).not.toHaveBeenCalled();
});

it('B2 preserves inherited candidate loss and suppresses every categorical result', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  recordWGSLKernelNodes(graph);
  const {categorical} = createScalabilityProps(graph, 2, 2);
  const loss = categorical.coverage.status.candidateOverflow!;
  const inheritedStatus = Object.assign(
    Object.create({
      get candidateOverflow() {
        return loss;
      }
    }),
    {
      count: categorical.coverage.status.count,
      requiredCount: categorical.coverage.status.requiredCount,
      overflow: categorical.coverage.status.overflow
    }
  );
  const contributor = new GPURasterFractionalCategoricalStatistics({
    ...categorical,
    coverage: {...categorical.coverage, status: inheritedStatus}
  });
  expect(contributor.props.coverage.status.candidateOverflow).toBe(loss);
  const nodes = contributor.getCommandNodes(graph);
  const values = new Map<GraphDataView, Uint32Array | Float32Array>([
    [loss, new Uint32Array([1])],
    [categorical.coverage.status.count, new Uint32Array([1])],
    [categorical.coverage.status.requiredCount, new Uint32Array([1])],
    [categorical.coverage.status.overflow, new Uint32Array([0])],
    [categorical.coverage.zoneValidity, new Uint32Array([1, 1])]
  ]);
  for (const suffix of ['reset-status', 'summarize', 'publish-classes', 'publish-areas']) {
    const node = nodes.find(candidate => candidate.id.endsWith(`-${suffix}`))!;
    const kernel = getWGSLKernelNodeRecord(graph, node.id)!;
    const bindings: Record<number, Uint32Array | Float32Array> = {};
    kernel.bindings.forEach((binding, index) => {
      if (!values.has(binding.view))
        values.set(
          binding.view,
          binding.type === 'f32'
            ? new Float32Array((binding.view.length * binding.view.rowByteLength) / 4)
            : new Uint32Array(binding.view.length)
        );
      bindings[index] = values.get(binding.view)!;
      if (binding.name === 'sums') bindings[index].fill(1);
    });
    const source = kernel.source
      .replace(
        /override (\w+): u32;/g,
        (_, name: string) => `const ${name}: u32 = ${kernel.constants[name]}u;`
      )
      .replace(/^\s*_ = .*;$/gm, '');
    new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', 1, {0: bindings});
  }
  expect(Array.from(values.get(categorical.output.validity)!)).toEqual([0, 0]);
  for (const output of [
    categorical.output.classAreas,
    categorical.output.coverageSums,
    categorical.output.proportions,
    categorical.output.coveredArea,
    categorical.output.validArea,
    categorical.output.missingArea
  ])
    expect(Array.from(values.get(output)!).every(Number.isNaN)).toBe(true);
});

it('B3 snapshots inherited calibration, nodata, grid and output leaves by name', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const {categorical} = createScalabilityProps(graph, 2, 2);
  const inputValues = createTransientView(graph, 'raw-values', 'float32', 16);
  const means = createTransientView(graph, 'means', 'float32', 2);
  const incomplete = createTransientView(graph, 'incomplete', 'uint32', 1);
  const band = Object.assign(
    Object.create({
      get scale() {
        return 3;
      },
      get noDataValue() {
        return -9999;
      }
    }),
    {
      id: 'values',
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: inputValues}
    }
  );
  const output = Object.create({
    get means() {
      return means;
    },
    get status() {
      return Object.create({
        get incomplete() {
          return incomplete;
        }
      });
    }
  });
  const inheritedGrid = Object.create({
    get width() {
      return 4;
    },
    get height() {
      return 4;
    },
    get affine() {
      return categorical.grid.affine;
    },
    get pixelInterpretation() {
      return 'area' as const;
    }
  });
  const props = {
    grid: inheritedGrid,
    values: band,
    zones: categorical.coverage.zoneIds,
    zoneCapacity: 2,
    output
  };
  const zonal = new GPURasterZonalStatistics({
    ...props,
    zones: createTransientView(graph, 'all-zones', 'uint32', 16)
  });
  expect(zonal.props.values.scale).toBe(3);
  expect(zonal.props.values.noDataValue).toBe(-9999);
  expect(zonal.props.output.means).toBe(means);
  expect(zonal.props.output.status?.incomplete).toBe(incomplete);
});

it('B5 ignores legacy padding and discriminates rebased latitude metadata on CPU and WGSL', () => {
  const grid = {
    width: 1,
    height: 1,
    affine: [1, 0, 0, 0, 1, 25] as const,
    pixelInterpretation: 'area' as const
  };
  const legacy = Float32Array.of(0, 25, 1, 0, 0, 1, 1, 31);
  const rebased = getGPURasterFractionalCoverageParameterValues(grid, {localOrigin: [0, 25]});
  for (const parameters of [legacy, rebased]) {
    const area = getRasterRowAreaScale(grid, 'geographic', 0, parameters);
    expect(Number.isFinite(area)).toBe(true);
    // WGS84 midpoint N*M*cos(25.5 degrees)*(pi/180)^2, calculated independently with Python.
    expect(area / 11137607818.7).toBeCloseTo(1, 10);
    const source = `@group(0) @binding(0) var<storage, read> settings: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
const settingsOffset: u32 = 0u;
@compute @workgroup_size(1) fn main() { let row = 0u;
${getRasterRowEdgeWGSL(grid, 'settings')}
result[0] = edge; }`;
    const result = new Float32Array(1);
    new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', 1, {
      0: {0: parameters, 1: result}
    });
    expect(result[0]).toBe(25.5);
  }
});

it('B6 writes and returns a Float32Array destination from another realm', () => {
  const grid = {
    width: 1,
    height: 1,
    affine: [1, 0, 2, 0, 1, 3] as const,
    pixelInterpretation: 'area' as const
  };
  const target = runInNewContext('new Float32Array(12).fill(88)') as Float32Array;
  const result = getGPURasterFractionalCoverageParameterValues(grid, target);
  expect(result).toBe(target);
  // Word 7 is legacy padding. Clear the version-2 discriminator and absolute Y offset
  // in words 8 and 9. Words 10 and 11 are beyond the packet.
  expect(Array.from(target)).toEqual([2, 3, 1, 0, 0, 1, 1, 0, 0, 0, 88, 88]);
});

it('reports why a singular affine cannot be packed', () => {
  const grid = makeGPURasterMetadata({width: 2, height: 1, origin: [-120, -30]});
  // The second axis row is twice the first, so 1 * 4 - 2 * 2 gives a determinant of zero.
  expect(() =>
    getGPURasterFractionalCoverageParameterValues({...grid, affine: [1, 2, 0, 2, 4, 0]})
  ).toThrow(
    /^getGPURasterFractionalCoverageParameterValues grid\.affine must be invertible \(finite nonzero determinant\), received determinant 0$/
  );
});

it('B7 exact-route diagnostics identify the route, enforced limit and tiled alternative', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const {categorical} = createScalabilityProps(graph, 30000, 100);
  expect(
    () => new GPURasterFractionalCategoricalStatistics({...categorical, mode: 'exact'})
  ).toThrow(/exact mode.*90000.*65535.*tiled/);
});
