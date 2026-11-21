// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterAll, describe, expect, it} from 'vitest';
import {
  GPUGeometryMeasures,
  type GPUGeometryMeasuresProps
} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geometry-measures';
import {GPUGeodesicPairs} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs';
import {GEOMETRY_GROUP_INTERVAL_WGSL} from '../../../src/gpu-spatial-analysis/geometry-measures/geometry-measures-kernels';
import {GEODESIC_WGSL} from '../../../src/gpu-spatial-analysis/geometry-measures/geodesic-wgsl';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';

const device = createNullWebGPUDevice();
afterAll(() => device.destroy());

describe('geometry wave three contracts', () => {
  useCPUExecutorTestHooks();

  it('retains round-even half-turn signs and short seams after endpoint canonicalization', () => {
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    const deltas = createTransientView(graph, 'deltas', 'float32', 7);
    executeCommandNodesOnCPU(
      [
        createWGSLKernelNode(graph, {
          id: 'deltas',
          operation: 'GPUGeodesicPairs',
          invocationCount: 1,
          bindings: [{name: 'deltas', view: deltas, type: 'f32', access: 'read_write'}],
          declarations: GEODESIC_WGSL,
          body: `deltas[deltasOffset] = geodesicLongitudeDelta(0.0, 180.0);
      deltas[deltasOffset + 1u] = geodesicLongitudeDelta(360.0, 180.0);
      deltas[deltasOffset + 2u] = geodesicLongitudeDelta(180.0, 0.0);
      deltas[deltasOffset + 3u] = geodesicLongitudeDelta(540.0, 0.0);
      deltas[deltasOffset + 4u] = geodesicLongitudeDelta(0.0, -180.0);
      deltas[deltasOffset + 5u] = geodesicLongitudeDelta(360.0, -180.0);
      deltas[deltasOffset + 6u] = geodesicLongitudeDelta(539.5, -179.5);`
        })
      ],
      memory
    );
    // Whole-turn parity selects the sign at a round-even tie; 539.5 = 179.5 modulo 360.
    expect(Array.from(memory.readFloat32(deltas))).toEqual([180, -180, -180, 180, -180, 180, 1]);
  });

  it('aligns whole geographic intervals without displacing an already-contained endpoint', () => {
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    const shifts = createTransientView(graph, 'shifts', 'float32', 3);
    executeCommandNodesOnCPU(
      [
        createWGSLKernelNode(graph, {
          id: 'intervals',
          operation: 'GPUGeometryMeasures',
          invocationCount: 1,
          bindings: [{name: 'shifts', view: shifts, type: 'f32', access: 'read_write'}],
          declarations: GEOMETRY_GROUP_INTERVAL_WGSL,
          body: `shifts[shiftsOffset] = getGeometryGroupIntervalTurn(vec2<f32>(-100.0, 100.0), vec2<f32>(100.0));
      shifts[shiftsOffset + 1u] = getGeometryGroupIntervalTurn(vec2<f32>(179.0, 181.0), vec2<f32>(-179.5));
      shifts[shiftsOffset + 2u] = getGeometryGroupIntervalTurn(vec2<f32>(179.0, 181.0), vec2<f32>(-178.5));`
        })
      ],
      memory
    );
    // An interval already inside [-100,100] stays put. Both seam controls shift one turn.
    expect(Array.from(memory.readFloat32(shifts))).toEqual([0, 360, 360]);
  });

  it('initializes only the scalar invalidCount word', () => {
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    const positions = createTransientView(graph, 'positions', 'float32x2', 1);
    const invalidCount = createTransientView(graph, 'invalid-count', 'uint32', 2);
    const vertexCounts = createTransientView(graph, 'vertex-counts', 'uint32', 1);
    memory.write(positions, Float32Array.of(0, 0));
    memory.write(invalidCount, Uint32Array.of(99, 0x12345678));
    executeCommandNodesOnCPU(
      new GPUGeometryMeasures({
        geometryType: 'points',
        positions,
        output: {vertexCounts, status: {invalidCount}}
      }).getCommandNodes(graph),
      memory
    );
    expect(Array.from(memory.readUint32(invalidCount))).toEqual([0, 0x12345678]);
  });

  it('rejects contradictory planar frame units before deriving length and area labels', () => {
    const graph = new GPUCommandGraph(device);
    const positions = createTransientView(graph, 'positions', 'float32x2', 1);
    for (const geometryType of ['lines', 'polygons'] as const) {
      const lineOffsets = createTransientView(graph, `${geometryType}-offsets`, 'uint32', 2);
      const measurement = createTransientView(graph, `${geometryType}-measurement`, 'float32', 1);
      // Preserve the discriminated multipart source alternatives in this fixture.
      const geometry: GPUGeometryMeasuresProps =
        geometryType === 'lines'
          ? {geometryType, positions, lineOffsets, output: {lengths: measurement}}
          : {
              geometryType,
              positions,
              ringOffsets: lineOffsets,
              polygonOffsets: createTransientView(graph, 'polygon-offsets', 'uint32', 2),
              featureOffsets: createTransientView(graph, 'feature-offsets', 'uint32', 2),
              output: {areas: measurement}
            };
      const compatible = {
        coordinateSpace: 'planar' as const,
        metric: 'native' as const,
        units: 'meters' as const,
        frame: {id: 'meter-frame', kind: 'crs' as const, unit: {name: 'meters', metersPerUnit: 1}}
      };
      expect(
        () =>
          new GPUGeometryMeasures({
            ...geometry,
            spatialContext: compatible
          })
      ).not.toThrow();
      for (const unit of [
        {name: 'degrees', degreesPerUnit: 1},
        {name: 'US survey feet', metersPerUnit: 1200 / 3937}
      ]) {
        expect(
          () =>
            new GPUGeometryMeasures({
              ...geometry,
              spatialContext: {
                coordinateSpace: 'planar',
                metric: 'native',
                units: 'meters',
                frame: {id: 'known-frame', kind: 'crs', unit}
              }
            })
        ).toThrow(/frame.*angular|units meters contradict frame.unit/);
      }
    }
  });

  it('reduces huge GPU inverse longitudes without certifying coincidence in the host interpreter', () => {
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    const origins = createTransientView(graph, 'origins', 'float32x2', 2);
    const targets = createTransientView(graph, 'targets', 'float32x2', 2);
    const distances = createTransientView(graph, 'distances', 'float32', 2);
    memory.write(origins, Float32Array.of(2 ** 62, 0, -176, 0));
    memory.write(targets, new Float32Array(4));
    executeCommandNodesOnCPU(
      new GPUGeodesicPairs({
        origins,
        targets,
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          sphereRadius: 1
        },
        output: {distances}
      }).getCommandNodes(graph),
      memory
    );
    const actual = memory.readFloat32(distances);
    expect(actual).toHaveLength(2);
    for (const distance of actual) {
      expect(Number.isFinite(distance)).toBe(true);
      expect(Math.abs(distance - (176 * Math.PI) / 180)).toBeLessThan(1e-6);
    }
  });

  it.each([
    'great-circle',
    'rhumb',
    'ellipsoidal'
  ] as const)('retains a tiny equatorial trip from equivalent 0 and 360 degree origins (%s)', metric => {
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    const origins = createTransientView(graph, 'origins', 'float32x2', 2);
    const targets = createTransientView(graph, 'targets', 'float32x2', 2);
    const distances = createTransientView(graph, 'distances', 'float32', 2);
    const approximation = createTransientView(graph, 'approximation', 'uint32', 2);
    memory.write(origins, Float32Array.of(0, 0, 360, 0));
    memory.write(targets, Float32Array.of(0.000001, 0, 0.000001, 0));
    executeCommandNodesOnCPU(
      new GPUGeodesicPairs({
        origins,
        targets,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric, units: 'meters'},
        output: {distances, status: {approximation}}
      }).getCommandNodes(graph),
      memory
    );
    const actual = memory.readFloat32(distances);
    expect(actual).toHaveLength(2);
    // Equatorial arc length R*pi/180*f32(1e-6), R=6378137 for WGS84, 6371008.8 otherwise.
    const expected = Math.fround(
      (((metric === 'ellipsoidal' ? 6378137 : 6371008.8) * Math.PI) / 180) * Math.fround(1e-6)
    );
    for (const distance of actual) {
      expect(Number.isFinite(distance)).toBe(true);
      expect(Math.abs(distance - expected)).toBeLessThan(0.000001);
    }
    expect(Array.from(memory.readUint32(approximation))).toEqual([0, 0]);
  });

  it('measures equivalent extreme-longitude multipoint bounds on the feature branch', () => {
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    const positions = createTransientView(graph, 'positions', 'float32x2', 4);
    const featureOffsets = createTransientView(graph, 'feature-offsets', 'uint32', 3);
    const bounds = createTransientView(graph, 'bounds', 'float32x4', 2);
    const centroids = createTransientView(graph, 'centroids', 'float32x2', 2);
    memory.write(positions, Float32Array.of(0, 0, 2 ** 62, 0, 0, 0, -176, 0));
    memory.write(featureOffsets, Uint32Array.of(0, 2, 4));
    executeCommandNodesOnCPU(
      new GPUGeometryMeasures({
        geometryType: 'points',
        positions,
        featureOffsets,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
        output: {bounds, centroids}
      }).getCommandNodes(graph),
      memory
    );
    expect(Array.from(memory.readFloat32(bounds))).toEqual([-176, 0, 0, 0, -176, 0, 0, 0]);
    expect(Array.from(memory.readFloat32(centroids))).toEqual([-88, 0, -88, 0]);
  });
});
