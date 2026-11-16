// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPURandomPointsOnLine,
  type GPURandomPointsOnLineProps
} from '../../../src/gpu-spatial-analysis/dot-density/gpu-random-points-on-line';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

function createProps(graph: GPUCommandGraph): GPURandomPointsOnLineProps {
  return {
    positions: createTransientView(graph, 'positions', 'float32x2', 4),
    lineOffsets: createTransientView(graph, 'offsets', 'uint32', 3),
    counts: createTransientView(graph, 'counts', 'uint32', 2),
    parameters: createTransientView(graph, 'parameters', 'uint32', 8),
    output: {
      positions: createTransientView(graph, 'point-positions', 'float32x2', 1),
      points: {
        ids: createTransientView(graph, 'ids', 'uint32', 1),
        status: {
          count: createTransientView(graph, 'count', 'uint32', 1),
          overflow: createTransientView(graph, 'overflow', 'uint32', 1)
        }
      }
    }
  };
}

it('uses atomic shared clamping, accepts separate-pass packing and rejects co-bound overlap', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  const packed = createTransientView(graph, 'packed', 'uint32', 4);
  props.output.positions = graph.createDataView(packed.buffer, {format: 'float32x2', length: 1});
  props.output.points.ids = graph.createDataView(packed.buffer, {
    format: 'uint32',
    byteOffset: 8,
    length: 1
  });
  const contributor = new GPURandomPointsOnLine(props);
  const nodes = contributor.getCommandNodes(graph);
  expect(() => graph.add(nodes)).not.toThrow();
  expect(
    () =>
      new GPURandomPointsOnLine({
        ...props,
        output: {
          ...props.output,
          fractions: graph.createDataView(packed.buffer, {
            format: 'float32',
            byteOffset: 8,
            length: 1
          })
        }
      })
  ).toThrow(/must not overlap/);
  const sources = getGPUCommandNodeKernelSources(nodes);
  expect(sources.find(entry => entry.source.includes('const LINE_LIMIT:'))!.source).toContain(
    'atomicStore(&clampFlag[clampFlagOffset], 1u)'
  );
  for (const {source} of sources)
    expect(
      (
        source.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '').match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []
      ).filter(word => WGSL_RESERVED_WORDS.has(word))
    ).toEqual([]);
  device.destroy();
});

it('preflights a non-aligned oversized source without declaring resources', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  const packed = createTransientView(graph, 'large-positions', 'float32x2', 16_777_217);
  props.positions = graph.createDataView(packed.buffer, {
    format: 'float32x2',
    byteOffset: 4,
    length: 16_777_216
  });
  const contributor = new GPURandomPointsOnLine(props);
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/needs 134217732 bytes/);
  expect(allocate).not.toHaveBeenCalled();
  vi.restoreAllMocks();
  device.destroy();
});
