// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUIsolines} from '../../../src/raster-analysis/isolines/gpu-isolines';
import {GPUIsolinesOperation} from '../../../src/raster-analysis/isolines/gpu-isolines-operation';
import {CPUIsolinesOperation} from '../../../src/raster-analysis/isolines/cpu-isolines-operation';
import {getGPUIsolinesParameterValues} from '../../../src/raster-analysis/isolines/isolines-parameters';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {GPUIsobandRings} from '../../../src/raster-analysis/isolines/gpu-isoband-rings';
import {GPUIsobands} from '../../../src/raster-analysis/isolines/gpu-isobands';
import {getGPUIsobandsParameterValues} from '../../../src/raster-analysis/isolines/isobands-parameters';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createCensusRasterHarness, expectCensusValues} from '../census-raster-oracle-utils';

it('isoband square writes every zero diagnostic over poison', async () => {
  const device = await getWebGPUTestDevice('core');
  requireTestDevice(device);
  const harness = createCensusRasterHarness(device);
  const grid = {
    width: 2,
    height: 2,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const result = await harness.run(
    new GPUIsobandRings({
      grid,
      edgeCapacity: 4,
      values: harness.input('float32', [1, 1, 1, 1]),
      breaks: harness.input('float32', [0.5]),
      parameters: harness.input(
        'float32',
        Array.from(
          getGPUIsobandsParameterValues(
            {breakCount: 1, firstBand: 1, lastBand: 1, extent: [0, 0, 2, 2]},
            grid
          )
        )
      ),
      output: {
        ringOffsets: harness.output('offsets', 'uint32', 3),
        positions: harness.output('positions', 'float32x2', 5),
        segmentFlags: harness.output('flags', 'uint32', 4),
        status: {
          count: harness.output('count', 'uint32', 1),
          requiredCount: harness.output('required', 'uint32', 1),
          overflow: harness.output('overflow', 'uint32', 1),
          openSegmentCount: harness.output('open', 'uint32', 1),
          touchingSegmentCount: harness.output('touching', 'uint32', 1),
          edgeCount: harness.output('edges', 'uint32', 1),
          edgeOverflow: harness.output('edgeOverflow', 'uint32', 1)
        }
      }
    })
  );
  // Four sample centers bound one square in a single band: four edges, one closed ring,
  // no open/touching segment and enough ring/vertex/edge capacity. Every word was poisoned.
  for (const [name, expected] of Object.entries({
    count: [1],
    required: [1],
    overflow: [0],
    open: [0],
    touching: [0],
    edges: [4],
    edgeOverflow: [0],
    flags: [0, 0, 0, 0]
  })) {
    expectCensusValues(result[name], expected);
  }
});

for (const scene of [
  {
    triangleCapacity: 2,
    edgeCapacity: 4,
    expected: {
      count: 2,
      required: 2,
      vertices: 6,
      overflow: 0,
      edges: 4,
      edgeRequired: 4,
      edgeOverflow: 0
    }
  },
  {
    triangleCapacity: 1,
    edgeCapacity: 3,
    expected: {
      count: 1,
      required: 2,
      vertices: 3,
      overflow: 1,
      edges: 3,
      edgeRequired: 4,
      edgeOverflow: 1
    }
  },
  {
    triangleCapacity: 1,
    edgeCapacity: 4,
    expected: {
      count: 1,
      required: 2,
      vertices: 3,
      overflow: 1,
      edges: 4,
      edgeRequired: 4,
      edgeOverflow: 0
    }
  },
  {
    triangleCapacity: 2,
    edgeCapacity: 3,
    expected: {
      count: 2,
      required: 2,
      vertices: 6,
      overflow: 0,
      edges: 3,
      edgeRequired: 4,
      edgeOverflow: 1
    }
  }
]) {
  it(`isoband square derives triangle and boundary counts independently (triangles=${scene.triangleCapacity}, edges=${scene.edgeCapacity})`, async () => {
    const device = await getWebGPUTestDevice('core');
    requireTestDevice(device);
    const harness = createCensusRasterHarness(device);
    const grid = {
      width: 2,
      height: 2,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'area' as const
    };
    const {triangleCapacity, edgeCapacity} = scene;
    const result = await harness.run(
      new GPUIsobands({
        grid,
        values: harness.input('float32', [1, 1, 1, 1]),
        breaks: harness.input('float32', [0.5]),
        parameters: harness.input(
          'float32',
          Array.from(
            getGPUIsobandsParameterValues(
              {breakCount: 1, firstBand: 1, lastBand: 1, extent: [0, 0, 2, 2]},
              grid
            )
          )
        ),
        output: {
          triangles: harness.output('triangles', 'float32x2', triangleCapacity * 3),
          triangleBands: harness.output('triangleBands', 'uint32', triangleCapacity),
          edges: harness.output('edgePositions', 'float32x4', edgeCapacity),
          edgeBands: harness.output('edgeBands', 'uint32', edgeCapacity),
          status: {
            count: harness.output('count', 'uint32', 1),
            requiredCount: harness.output('required', 'uint32', 1),
            vertexCount: harness.output('vertices', 'uint32', 1),
            overflow: harness.output('overflow', 'uint32', 1),
            edgeCount: harness.output('edges', 'uint32', 1),
            edgeRequiredCount: harness.output('edgeRequired', 'uint32', 1),
            edgeOverflow: harness.output('edgeOverflow', 'uint32', 1)
          }
        }
      })
    );
    // A convex unit square has four distinct boundary edges and two triangles (each half
    // of its fixed diagonal). Counts follow this geometry, never another GPU output port.
    // Independently short capacities distinguish triangle overflow from boundary overflow.
    for (const [name, expected] of Object.entries(scene.expected)) {
      expectCensusValues(result[name], [expected]);
    }
  });
}

it('R6 a single retained square edge is open and cannot form a ring', async () => {
  const device = await getWebGPUTestDevice('core');
  requireTestDevice(device);
  const harness = createCensusRasterHarness(device);
  const grid = {
    width: 2,
    height: 2,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const result = await harness.run(
    new GPUIsobandRings({
      grid,
      edgeCapacity: 1,
      values: harness.input('float32', [1, 1, 1, 1]),
      breaks: harness.input('float32', [0.5]),
      parameters: harness.input(
        'float32',
        Array.from(
          getGPUIsobandsParameterValues(
            {breakCount: 1, firstBand: 1, lastBand: 1, extent: [0, 0, 2, 2]},
            grid
          )
        )
      ),
      output: {
        ringOffsets: harness.output('offsets', 'uint32', 2),
        positions: harness.output('positions', 'float32x2', 2),
        status: {
          count: harness.output('count', 'uint32', 1),
          overflow: harness.output('overflow', 'uint32', 1),
          openSegmentCount: harness.output('open', 'uint32', 1),
          edgeCount: harness.output('edges', 'uint32', 1),
          edgeOverflow: harness.output('edgeOverflow', 'uint32', 1)
        }
      }
    })
  );
  // The square needs four edges. Capacity one retains an isolated directed segment with
  // distinct endpoints: exactly one open segment, no closed ring, and dropped edges.
  expectCensusValues(result['edges'], [1]);
  expectCensusValues(result['open'], [1]);
  expectCensusValues(result['edgeOverflow'], [1]);
  expectCensusValues(result['count'], [0]);
});

for (const scene of [
  {segments: 2, vertices: 3, expected: [2, 2, 0, 0, 1, 3]},
  {segments: 2, vertices: 2, expected: [2, 2, 0, 1, 0, 0]},
  {segments: 1, vertices: 3, expected: [1, 2, 1, 1, 0, 0]}
]) {
  it(`isoline forms derive every ramp count and overflow (segments=${scene.segments}, vertices=${scene.vertices})`, async () => {
    const device = await getWebGPUTestDevice('core');
    requireTestDevice(device);
    const grid = {
      width: 2,
      height: 3,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'area' as const
    };
    const parameters = getGPUIsolinesParameterValues({levelCount: 1, extent: [0, 0, 2, 3]}, grid);
    const harness = createCensusRasterHarness(device);
    const result = await harness.run(
      new GPUIsolines({
        grid,
        values: harness.input('float32', [0, 2, 0, 2, 0, 2]),
        levels: harness.input('float32', [1]),
        parameters: harness.input('float32', Array.from(parameters)),
        output: {
          segments: harness.output('segments', 'float32x4', scene.segments),
          segmentLevels: harness.output('levels', 'uint32', scene.segments),
          status: {
            count: harness.output('count', 'uint32', 1),
            requiredCount: harness.output('required', 'uint32', 1),
            overflow: harness.output('overflow', 'uint32', 1),
            polylineOverflow: harness.output('polyOverflow', 'uint32', 1)
          },
          polylines: {
            vertices: harness.output('vertices', 'float32x2', scene.vertices),
            polylineOffsets: harness.output('offsets', 'uint32', scene.segments + 1),
            polylineLevels: harness.output('polyLevels', 'uint32', scene.segments),
            polylineClosed: harness.output('closed', 'uint32', scene.segments),
            polylineCount: harness.output('polylines', 'uint32', 1),
            vertexCount: harness.output('vertexCount', 'uint32', 1)
          }
        }
      })
    );
    // Two level crossings join into one three-vertex chain; capacities alone decide each
    // overflow. Poison protects every expected zero, including suppressed chain counts.
    const names = ['count', 'required', 'overflow', 'polyOverflow', 'polylines', 'vertexCount'];
    expect(names.map(name => result[name][0])).toEqual(scene.expected);
    for (const Operation of [GPUIsolinesOperation, CPUIsolinesOperation]) {
      const rows = await runOperationGraph(device, {
        inputs: {
          values: {format: 'float32', values: Float32Array.of(0, 2, 0, 2, 0, 2)},
          levels: {format: 'float32', values: Float32Array.of(1)},
          parameters: {format: 'float32', values: parameters}
        },
        operations: [
          new Operation({
            grid,
            segmentCapacity: scene.segments,
            values: 'values',
            levels: 'levels',
            parameters: 'parameters',
            output: {
              segments: 'segments',
              segmentLevels: 'segmentLevels',
              status: {
                count: 'count',
                requiredCount: 'required',
                overflow: 'overflow',
                polylineOverflow: 'polyOverflow'
              },
              polylines: {
                vertexCapacity: scene.vertices,
                vertices: 'vertices',
                polylineOffsets: 'offsets',
                polylineLevels: 'polyLevels',
                polylineClosed: 'closed',
                polylineCount: 'polylines',
                vertexCount: 'vertexCount'
              }
            }
          })
        ],
        outputs: names
      });
      expect(names.map(name => rows[name][0])).toEqual(scene.expected);
    }
  });
}
