// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUNetworkServiceAreas} from '../../../src/gpu-network/network-analysis/index';
import {
  GPUNetworkIsochrones,
  type GPUNetworkIsochronesProps,
  type GPUNetworkIsochronesRaster
} from '../../../src/gpu-network/network-isochrones/index';
import {
  addDriveTimeCatchmentRecipe,
  type GPUDriveTimeCatchmentRecipeProps
} from '../../../src/gpu-network/recipes/drive-time-catchment-recipe';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createRecipeCellReviewProps} from './recipe-cell-review-fixture';
import {getResourceBuffer} from '../network-resource-buffers';
import {
  createOracleRingOutput,
  createIsochroneOracleHarness
} from './isochrones-output-oracle-utils';

function createFixture() {
  const device = createNullWebGPUDevice();
  const harness = createIsochroneOracleHarness(device);
  const props: GPUDriveTimeCatchmentRecipeProps = {
    id: 'final-wave-drive-time',
    topology: {
      offsets: harness.input('uint32', [0, 1, 2]),
      neighbors: harness.input('uint32', [1, 0]),
      weights: harness.input('float32', [1, 1]),
      nodePositions: harness.input('float32x2', [0, 0, 1, 0])
    },
    facilities: harness.input('float32x2', [0, 0]),
    demand: harness.input('float32x2', [1, 0]),
    costBreaks: harness.input('float32', [2])
  };
  return {device, harness, props};
}

it('UCT-10 retains output destinations in the service-area writer', () => {
  const {device, harness, props} = createFixture();
  const serviceNodes = vi.spyOn(GPUNetworkServiceAreas.prototype, 'getCommandNodes');
  try {
    props.bandCapacity = 16;
    const assignments = harness.output('assignments', 'uint32', 2);
    const converged = harness.output('converged', 'uint32', 1);
    const nodeCosts = harness.output('nodeCosts', 'float32', 2);
    const keys = harness.output('keys', 'uint32', 16);
    const counts = harness.output('counts', 'uint32', 16);
    props.output = {
      assignments,
      nodeCosts,
      status: {serviceAreaConverged: converged},
      bands: {keys, counts}
    };
    const recipe = addDriveTimeCatchmentRecipe(harness.graph, props);
    expect(recipe.assignments === assignments).toBe(true);
    expect(recipe.outputs['assignments'] === assignments).toBe(true);
    const writer = recipe.contributors.find(
      (contributor): contributor is GPUNetworkServiceAreas =>
        contributor instanceof GPUNetworkServiceAreas
    )!;
    expect(writer.props.output.assignments === assignments).toBe(true);
    expect(
      serviceNodes.mock.results.some(
        result =>
          result.type === 'return' &&
          result.value.some(node =>
            node.resources?.some(
              resource =>
                resource.usage === 'storage-read-write' &&
                getResourceBuffer(resource) === assignments
            )
          )
      )
    ).toBe(true);
    expect(writer.props.output.status?.converged === converged).toBe(true);
    expect(recipe.bands.keys === keys).toBe(true);
    expect(writer.props.output.costs === nodeCosts).toBe(true);
    expect(recipe.bands.counts === counts).toBe(true);
  } finally {
    serviceNodes.mockRestore();
    harness.destroy();
    device.destroy();
  }
});

it('UCT-10 rejects the retired output prop before mutation and permits a same-ID retry', () => {
  const {device, harness, props} = createFixture();
  const add = vi.spyOn(GPUCommandGraph.prototype, 'add');
  try {
    const converged = harness.output('converged', 'uint32', 1);
    props.output = {status: {serviceAreaConverged: converged}};
    const retired = {...props, outputs: props.output};
    expect(() => addDriveTimeCatchmentRecipe(harness.graph, retired)).toThrow(
      /outputs was removed; use output/
    );
    expect(add).not.toHaveBeenCalled();
    expect(
      addDriveTimeCatchmentRecipe(harness.graph, props).status.serviceAreaConverged === converged
    ).toBe(true);
  } finally {
    add.mockRestore();
    harness.destroy();
    device.destroy();
  }
});

it('STA-08 rejects unsupported prefilled join diagnostics', () => {
  for (const diagnostic of ['unresolvedCount', 'complete'] as const) {
    const {device, harness, props} = createFixture();
    const add = vi.spyOn(GPUCommandGraph.prototype, 'add');
    try {
      // A stale nonzero diagnostic must never be accepted as a writable destination.
      const stale = harness.input('uint32', [7]);
      props.output = {
        isochroneDemand: {
          status: {overflow: harness.output('overflow', 'uint32', 1), [diagnostic]: stale}
        }
      };
      expect(() => addDriveTimeCatchmentRecipe(harness.graph, props)).toThrow(
        new RegExp(`final-wave-drive-time.*isochroneDemand.status.${diagnostic}.*not supported`)
      );
      expect(add).not.toHaveBeenCalled();
    } finally {
      add.mockRestore();
      harness.destroy();
      device.destroy();
    }
  }
});

it.each([
  'proxy',
  'inherited',
  'omitted'
] as const)('R4a captures the raster mode once (%s)', route => {
  const {device, harness, props} = createFixture();
  const add = vi.spyOn(GPUCommandGraph.prototype, 'add');
  let reads = 0;
  try {
    const raster: GPUNetworkIsochronesRaster = {
      width: 2,
      height: 2,
      output: {bandClasses: harness.output('classes', 'uint32', 4)}
    };
    if (route === 'inherited') {
      Object.setPrototypeOf(raster, {
        get mode() {
          reads++;
          return 'min';
        }
      });
    }
    const capturedRaster =
      route === 'proxy'
        ? new Proxy(raster, {
            get(target, key, receiver) {
              if (key === 'mode') {
                reads++;
                return 'max';
              }
              return Reflect.get(target, key, receiver);
            }
          })
        : raster;
    props.isochrones = {
      costBreaks: props.costBreaks,
      parameters: harness.input('float32', new Array(12).fill(0)),
      raster: capturedRaster
    };
    if (route === 'proxy') {
      expect(() => addDriveTimeCatchmentRecipe(harness.graph, props)).toThrow(
        /final-wave-drive-time.*isochrones.raster.mode.*min.*max/
      );
      expect(add).not.toHaveBeenCalled();
    } else {
      const recipe = addDriveTimeCatchmentRecipe(harness.graph, props);
      const contributor = recipe.contributors.find(
        (entry): entry is GPUNetworkIsochrones => entry instanceof GPUNetworkIsochrones
      )!;
      expect(contributor.props.raster?.mode ?? 'min').toBe('min');
    }
    expect(reads).toBe(route === 'omitted' ? 0 : 1);
  } finally {
    add.mockRestore();
    harness.destroy();
    device.destroy();
  }
});

it.each([
  'startCosts',
  'costLimit',
  'startCount'
] as const)('TSC-01/02/03 keeps packed validation and structured identity for %s', path => {
  const {device, harness, props} = createFixture();
  try {
    const options: GPUNetworkIsochronesProps = {
      id: 'typed-isochrones',
      topology: {...props.topology, nodePositions: props.topology.nodePositions!},
      costs: harness.output('costs', 'float32', 2),
      startNodes: harness.input('uint32', [0, 1]),
      costBreaks: props.costBreaks,
      parameters: harness.input('float32', new Array(12).fill(0)),
      raster: {width: 2, height: 2, output: {bandClasses: harness.output('classes', 'uint32', 4)}}
    };
    const buffer = harness.graph.createTransientBuffer({
      id: 'strided',
      byteLength: 16,
      usage: Buffer.STORAGE
    });
    if (path === 'startCount')
      options.startCount = harness.graph.createDataView(buffer, {
        format: 'uint32',
        length: 1,
        byteStride: 8
      });
    else
      options[path] = harness.graph.createDataView(buffer, {
        format: 'float32',
        length: path === 'startCosts' ? 2 : 1,
        byteStride: 8
      });
    expect(() => new GPUNetworkIsochrones(options)).toThrowError(
      expect.objectContaining({
        id: 'typed-isochrones',
        contributor: 'GPUNetworkIsochrones',
        path,
        code: 'VIEW_FORMAT'
      })
    );
    if (path === 'startCount') options.startCount = harness.input('uint32', [2, 2]);
    else options[path] = harness.input('float32', path === 'startCosts' ? [0] : [2, 2]);
    expect(() => new GPUNetworkIsochrones(options)).toThrow(
      new RegExp(
        `${path}.*${path === 'startCosts' ? 'equal startNodes.length' : 'exactly one row'}`
      )
    );
    if (path === 'startCount') {
      options.startCount = 3;
      expect(() => new GPUNetworkIsochrones(options)).toThrow(/startCount/);
      options.startCount = 2;
    } else options[path] = harness.input('float32', path === 'startCosts' ? [0, 0] : [2]);
    expect(() => new GPUNetworkIsochrones(options)).not.toThrow();
  } finally {
    harness.destroy();
    device.destroy();
  }
});

it('TSC-05 retains disjoint nested cell views and their byte ranges', () => {
  const {device, harness} = createFixture();
  try {
    const props = createRecipeCellReviewProps(harness);
    const buffer = harness.graph.createTransientBuffer({
      id: 'shared-cell-status',
      byteLength: 512,
      usage: Buffer.STORAGE
    });
    const overflow = harness.graph.createDataView(buffer, {format: 'uint32', length: 1});
    const candidateOverflow = harness.graph.createDataView(buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 256
    });
    props.isochrones!.cellOutline!.table!.status.overflow = overflow;
    props.isochrones!.cellOutline!.output.status.candidateOverflow = candidateOverflow;
    const recipe = addDriveTimeCatchmentRecipe(harness.graph, props);
    const contributor = recipe.contributors.find(
      (entry): entry is GPUNetworkIsochrones => entry instanceof GPUNetworkIsochrones
    )!;
    expect(contributor.props.cellOutline!.table!.status.overflow === overflow).toBe(true);
    expect(
      contributor.props.cellOutline!.output.status.candidateOverflow === candidateOverflow
    ).toBe(true);
    expect(overflow.byteOffset).toBe(0);
    expect(candidateOverflow.byteOffset).toBe(256);
    expect(overflow.length).toBe(1);
    expect(candidateOverflow.length).toBe(1);
  } finally {
    harness.destroy();
    device.destroy();
  }
});

it('B2 rejects destinations for disabled stages', () => {
  for (const leaf of [
    'matches',
    'overflow',
    'uncertainCount',
    'sumValues',
    'means',
    'facilitiesOverflow',
    'demandOverflow'
  ] as const) {
    const {device, harness, props} = createFixture();
    const add = vi.spyOn(GPUCommandGraph.prototype, 'add');
    try {
      const integer = harness.output('disabled', 'uint32', 1);
      const real = harness.output('disabled-real', 'float32', 16);
      props.output =
        leaf === 'matches'
          ? {isochroneDemand: {pointFeatureIds: integer}}
          : leaf === 'overflow' || leaf === 'uncertainCount'
            ? {isochroneDemand: {status: {overflow: integer, [leaf]: integer}}}
            : leaf === 'sumValues' || leaf === 'means'
              ? {bands: {[leaf]: real}}
              : {snapping: {status: {[leaf]: integer}}};
      expect(() => addDriveTimeCatchmentRecipe(harness.graph, props), leaf).toThrow(/requires/);
      expect(add).not.toHaveBeenCalled();
      props.output = undefined;
      expect(() => addDriveTimeCatchmentRecipe(harness.graph, props)).not.toThrow();
    } finally {
      add.mockRestore();
      harness.destroy();
      device.destroy();
    }
  }
});

it('B2 rejects the retired join destination prop before declarations', () => {
  const {device, harness, props} = createFixture();
  const add = vi.spyOn(GPUCommandGraph.prototype, 'add');
  try {
    props.output = {isochroneDemand: {pointFeatureIds: harness.output('one', 'uint32', 1)}};
    const retired = {...props, outputs: {isochroneDemand: props.output.isochroneDemand}};
    expect(() => addDriveTimeCatchmentRecipe(harness.graph, retired)).toThrow(
      /outputs was removed; use output/
    );
    expect(add).not.toHaveBeenCalled();
  } finally {
    add.mockRestore();
    harness.destroy();
    device.destroy();
  }
});

it.each([false, true])('B4 retains inherited ring polygonCount with join=%s', join => {
  const {device, harness} = createFixture();
  const writers = vi.spyOn(GPUNetworkIsochrones.prototype, 'getCommandNodes');
  try {
    const props = createRecipeCellReviewProps(harness);
    if (join) props.isochrones!.joinDemand = {candidateCapacity: 8};
    const ring = props.isochrones!.cellOutline!.rings!.output;
    const polygonCount = harness.output('inherited-polygon-count', 'uint32', 1);
    const ringAreas = ring.ringAreas!;
    delete ring.ringAreas;
    Object.setPrototypeOf(ring, {
      get ringAreas() {
        return ringAreas;
      }
    });
    delete ring.status.polygonCount;
    Object.setPrototypeOf(ring.status, {
      get polygonCount() {
        return polygonCount;
      }
    });
    const recipe = addDriveTimeCatchmentRecipe(harness.graph, props);
    const producer = recipe.contributors.find(
      (entry): entry is GPUNetworkIsochrones => entry instanceof GPUNetworkIsochrones
    )!;
    expect(producer.props.cellOutline!.rings!.output.status.polygonCount === polygonCount).toBe(
      true
    );
    expect(recipe.isochroneRings === producer.props.cellOutline!.rings!.output).toBe(true);
    expect(producer.props.cellOutline!.rings!.output.ringAreas === ringAreas).toBe(true);
    expect(
      writers.mock.results.some(
        result =>
          result.type === 'return' &&
          result.value.some(node =>
            node.resources?.some(
              resource =>
                resource.usage === 'storage-read-write' &&
                getResourceBuffer(resource) === polygonCount
            )
          )
      )
    ).toBe(true);
  } finally {
    writers.mockRestore();
    harness.destroy();
    device.destroy();
  }
});

it('earlier-fix note forwards raster-ring complete independently of edge loss', () => {
  const {device, harness, props} = createFixture();
  try {
    const complete = harness.output('raster-complete', 'uint32', 1);
    const edgeOverflow = harness.output('raster-edge-loss', 'uint32', 1);
    const output = createOracleRingOutput(harness, 'all', 4);
    props.isochrones = {
      costBreaks: props.costBreaks,
      parameters: harness.input('float32', new Array(12).fill(0)),
      raster: {
        width: 2,
        height: 2,
        output: {bandClasses: harness.output('classes', 'uint32', 4)},
        rings: {
          edgeCapacity: 4,
          output: {...output, status: {...output.status, complete, edgeOverflow}}
        }
      }
    };
    const recipe = addDriveTimeCatchmentRecipe(harness.graph, props);
    const stage = recipe.status.stages.find(entry => entry.stage === 'isochrone-band-rings')!;
    expect(stage.status.complete === complete).toBe(true);
    expect(stage.status.candidateOverflow === edgeOverflow).toBe(true);
  } finally {
    harness.destroy();
    device.destroy();
  }
});

it('S3 accepts partial join status maps', () => {
  const {device, harness} = createFixture();
  try {
    const props = createRecipeCellReviewProps(harness);
    props.isochrones!.joinDemand = {candidateCapacity: 8};
    const uncertainCount = harness.output('only-uncertain', 'uint32', 1);
    const candidateOverflow = harness.output('only-candidate', 'uint32', 1);
    props.output = {isochroneDemand: {status: {uncertainCount, candidateOverflow}}};
    const result = addDriveTimeCatchmentRecipe(harness.graph, props);
    expect(result.isochroneDemand!.status.uncertainCount === uncertainCount).toBe(true);
    expect(result.isochroneDemand!.status.candidateOverflow === candidateOverflow).toBe(true);
    expect(result.isochroneDemand!.status.overflow.format).toBe('uint32');
  } finally {
    harness.destroy();
    device.destroy();
  }
});
