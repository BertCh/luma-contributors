// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, Texture} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand} from '@luma.gl/graph/gpu-raster';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUIsolines,
  GPUIsobands,
  getGPUIsolinesParameterValues,
  getGPUIsobandsParameterValues
} from '../../../src/raster-analysis/isolines/index';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const SAMPLE_COUNT = 15;
const RAW = [0, 1, 2, 3, 99, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4];
const VALIDITY = [1, 7, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1];
const MASK = [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1];
const MISSING = 0xffffffff;
const EXPECTED_CLASSES = [0, 2, 2, 2, MISSING, 0, 2, MISSING, 2, MISSING, MISSING, 2, 2, 2, 2];

it.each([
  'uint32-buffer',
  'sint32-texture',
  'float32-buffer'
] as const)('%s contours honor calibration, native nodata, validity, masks and padding on repeated encodings', async variant => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const grid = {
    width: 5,
    height: 3,
    affine: [1, 0, 0, 0, -1, 3] as const,
    pixelInterpretation: 'area' as const
  };
  const extent = [0, 0, 5, 3] as const;
  const buffers: Buffer[] = [];
  let sequence = 0;
  const input = <Format extends 'float32' | 'uint32' | 'float32x2' | 'float32x4' | 'uint32x2'>(
    format: Format,
    data: Float32Array | Uint32Array,
    length = data.length
  ) => {
    const buffer = device.createBuffer({
      data,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, `buffer-${sequence++}`, buffer, format, length)};
  };
  const integer = (length: number) => input('uint32', new Uint32Array(length).fill(777));
  const floating = <Format extends 'float32x2' | 'float32x4'>(
    format: Format,
    length: number,
    components: number
  ) => input(format, new Float32Array(length * components).fill(777), length);
  const validity = input('uint32', Uint32Array.from([777, ...VALIDITY, 777, 777]));
  const bandMetadata = {
    id: 'native',
    scale: 2,
    offset: 10,
    noDataValue: 99,
    validity: graph.createDataView(validity.view.buffer, {
      format: 'uint32',
      byteOffset: 4,
      length: SAMPLE_COUNT + 2
    })
  };
  const texture =
    variant === 'sint32-texture'
      ? device.createTexture({
          format: 'r32sint',
          width: 5,
          height: 3,
          usage: Texture.SAMPLE | Texture.COPY_DST
        })
      : undefined;
  let band: GPURasterBand;
  if (texture) {
    texture.writeData(Int32Array.from(RAW));
    band = {
      ...bandMetadata,
      format: 'sint32',
      storage: {
        kind: 'texture',
        view: graph.createTextureView(
          graph.importTexture(
            {
              id: 'native-texture',
              format: 'r32sint',
              width: 5,
              height: 3,
              usage: texture.props.usage
            },
            texture
          )
        )
      }
    };
  } else if (variant === 'uint32-buffer') {
    const native = input('uint32', Uint32Array.from([777, ...RAW, 777, 777]));
    band = {
      ...bandMetadata,
      format: 'uint32',
      storage: {
        kind: 'buffer',
        values: graph.createDataView(native.view.buffer, {
          format: 'uint32',
          byteOffset: 4,
          length: SAMPLE_COUNT + 2
        })
      }
    };
  } else {
    const native = input(
      'float32',
      Float32Array.from([777, ...RAW.map((value, index) => (index === 4 ? NaN : value)), 777, 777])
    );
    band = {
      ...bandMetadata,
      format: 'float32',
      storage: {
        kind: 'buffer',
        values: graph.createDataView(native.view.buffer, {
          format: 'float32',
          byteOffset: 4,
          length: SAMPLE_COUNT + 2
        })
      }
    };
  }
  const mask = input('uint32', Uint32Array.from([777, ...MASK, 777]));
  const maskView = graph.createDataView(mask.view.buffer, {
    format: 'uint32',
    byteOffset: 4,
    length: SAMPLE_COUNT + 1
  });
  const levels = input('float32', Float32Array.of(11, 12)).view;
  const lineCount = integer(1),
    lineRequired = integer(1),
    lineOverflow = integer(1);
  const segments = floating('float32x4', 4, 4);
  const segmentLevels = integer(4);
  const segmentEdges = input('uint32x2', new Uint32Array(8).fill(777), 4);
  const vertices = floating('float32x2', 8, 2);
  const polylineOffsets = integer(5),
    polylineLevels = integer(4),
    polylineClosed = integer(4);
  const polylineCount = integer(1),
    vertexCount = integer(1),
    polylineOverflow = integer(1);
  graph.add(
    new GPUIsolines({
      id: 'band-lines',
      grid,
      values: band,
      mask: maskView,
      noDataValue: 10,
      levels,
      parameters: input('float32', getGPUIsolinesParameterValues({levelCount: 2, extent}, grid))
        .view,
      output: {
        segments: segments.view,
        segmentLevels: segmentLevels.view,
        segmentEdges: segmentEdges.view,
        status: {
          count: lineCount.view,
          requiredCount: lineRequired.view,
          overflow: lineOverflow.view,
          polylineOverflow: polylineOverflow.view
        },
        polylines: {
          vertices: vertices.view,
          polylineOffsets: polylineOffsets.view,
          polylineLevels: polylineLevels.view,
          polylineClosed: polylineClosed.view,
          polylineCount: polylineCount.view,
          vertexCount: vertexCount.view
        }
      }
    })
  );
  const classes = integer(SAMPLE_COUNT + 2);
  const triangles = floating('float32x2', 48, 2),
    triangleBands = integer(16);
  const edges = floating('float32x4', 16, 4),
    edgeBands = integer(16);
  const triangleCount = integer(1),
    triangleRequired = integer(1),
    triangleOverflow = integer(1);
  const edgeCount = integer(1),
    edgeRequired = integer(1),
    edgeOverflow = integer(1);
  graph.add(
    new GPUIsobands({
      id: 'band-bands',
      grid,
      values: band,
      mask: maskView,
      noDataValue: 10,
      breaks: levels,
      parameters: input('float32', getGPUIsobandsParameterValues({breakCount: 2, extent}, grid))
        .view,
      output: {
        bandClasses: classes.view,
        triangles: triangles.view,
        triangleBands: triangleBands.view,
        edges: edges.view,
        edgeBands: edgeBands.view,
        status: {
          count: triangleCount.view,
          requiredCount: triangleRequired.view,
          overflow: triangleOverflow.view,
          edgeCount: edgeCount.view,
          edgeRequiredCount: edgeRequired.view,
          edgeOverflow: edgeOverflow.view
        }
      }
    })
  );
  const compiled = graph.compile();
  try {
    for (const empty of [false, true, false]) {
      validity.buffer.write(
        Uint32Array.from([777, ...(empty ? new Array(SAMPLE_COUNT).fill(0) : VALIDITY), 777, 777])
      );
      submitGraph(device, compiled, undefined);
      expect(await readUint32(classes.buffer, SAMPLE_COUNT + 2)).toEqual([
        ...(empty ? new Array(SAMPLE_COUNT).fill(MISSING) : EXPECTED_CLASSES),
        777,
        777
      ]);
      for (const scalar of [lineCount, lineRequired, polylineCount])
        expect(await readUint32(scalar.buffer, 1)).toEqual([empty ? 0 : 2]);
      expect(await readUint32(vertexCount.buffer, 1)).toEqual([empty ? 0 : 4]);
      for (const scalar of [lineOverflow, polylineOverflow, triangleOverflow, edgeOverflow])
        expect(await readUint32(scalar.buffer, 1)).toEqual([0]);
      for (const scalar of [triangleCount, triangleRequired])
        expect(await readUint32(scalar.buffer, 1)).toEqual([empty ? 0 : 6]);
      for (const scalar of [edgeCount, edgeRequired])
        expect(await readUint32(scalar.buffer, 1)).toEqual([empty ? 0 : 8]);
      if (empty) continue;
      // Only the lower-left marching cell has four valid corners. Calibrated values 10/12
      // cross level 11 at x=1 and equal level 12 at x=1.5; high east means travel south.
      expect(await readFloat32(segments.buffer, 16)).toEqual([
        1,
        1.5,
        1,
        0.5,
        1.5,
        1.5,
        1.5,
        0.5,
        ...new Array(8).fill(777)
      ]);
      expect(await readUint32(segmentLevels.buffer, 4)).toEqual([0, 1, 777, 777]);
      expect(await readUint32(segmentEdges.buffer, 8)).toEqual([
        4,
        0,
        4,
        0,
        ...new Array(4).fill(777)
      ]);
      expect((await readFloat32(vertices.buffer, 16)).slice(0, 8)).toEqual([
        1, 1.5, 1, 0.5, 1.5, 1.5, 1.5, 0.5
      ]);
      expect((await readUint32(polylineOffsets.buffer, 5)).slice(0, 3)).toEqual([0, 2, 4]);
      expect((await readUint32(polylineLevels.buffer, 4)).slice(0, 2)).toEqual([0, 1]);
      expect((await readUint32(polylineClosed.buffer, 4)).slice(0, 2)).toEqual([0, 0]);
      // Two half-width rectangles tile this one valid cell: area 0.5 for bands 0 and 1.
      // Band 2 lies exactly on x=1.5. Its retained corner/crossing events fan into two
      // zero-area triangle records, while its zero-area boundary edges are suppressed.
      const triangleValues = await readFloat32(triangles.buffer, 96);
      const bands = await readUint32(triangleBands.buffer, 16);
      expect(bands).toEqual([0, 0, 1, 1, 2, 2, ...new Array(10).fill(777)]);
      const areas = [0, 0, 0];
      const triangleAreas: number[] = [];
      for (let triangle = 0; triangle < 6; triangle++) {
        const [firstX, firstY, secondX, secondY, thirdX, thirdY] = triangleValues.slice(
          triangle * 6,
          triangle * 6 + 6
        );
        const area =
          ((secondX - firstX) * (thirdY - firstY) - (secondY - firstY) * (thirdX - firstX)) / 2;
        triangleAreas.push(area);
        areas[bands[triangle]] += area;
      }
      expect(triangleAreas).toEqual([0.25, 0.25, 0.25, 0.25, 0, 0]);
      expect(areas).toEqual([0.5, 0.5, 0]);
      expect(triangleValues.slice(24, 36)).toEqual([
        1.5, 0.5, 1.5, 0.5, 1.5, 1.5, 1.5, 0.5, 1.5, 1.5, 1.5, 1.5
      ]);
      expect(triangleValues.slice(36).every(value => value === 777)).toBe(true);
      const edgeValues = await readFloat32(edges.buffer, 64);
      const groups = await readUint32(edgeBands.buffer, 16);
      const actualEdges = Array.from(
        {length: 8},
        (_, edge) => `${groups[edge]}:${edgeValues.slice(4 * edge, 4 * edge + 4).join(',')}`
      ).sort();
      const expectedEdges = [0, 1]
        .flatMap(group => {
          const left = 0.5 + group * 0.5,
            right = left + 0.5;
          const corners = [
            [left, 0.5],
            [right, 0.5],
            [right, 1.5],
            [left, 1.5]
          ];
          return corners.map(
            (point, index) => `${group}:${[...point, ...corners[(index + 1) % 4]].join(',')}`
          );
        })
        .sort();
      expect(actualEdges).toEqual(expectedEdges);
      expect(edgeValues.slice(32).every(value => value === 777)).toBe(true);
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
    texture?.destroy();
  }
});
