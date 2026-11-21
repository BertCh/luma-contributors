// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getGPUVectorFormatInfo} from '@luma.gl/graph/gpu-data';
import {expect, it, vi} from 'vitest';
import {GPUCellSetOutline} from '../../../src/gpu-spatial-analysis/cell-set-outline/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createCPUGraphMemory} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {createCellLossBoundaryFixture} from './cell-loss-boundary-fixture';

it.each([
  false,
  true
])('wires the full cell status into outline and ring outputs (candidateOnly=%s, CPL-12)', candidateOnly => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  let serial = 0;
  const outline = vi.spyOn(GPUCellSetOutline.prototype, 'getCommandNodes');
  const fixture = createCellLossBoundaryFixture(
    {
      graph,
      input(format, values) {
        const view = createTransientView(
          graph,
          `input-${serial++}`,
          format,
          (values.length * 4) / getGPUVectorFormatInfo(format).byteLength
        );
        memory.write(
          view,
          format.startsWith('uint32') ? Uint32Array.from(values) : Float32Array.from(values)
        );
        return view;
      },
      output(name, format, length) {
        return createTransientView(graph, name, format, length);
      }
    },
    candidateOnly
  );
  try {
    const consumer = outline.mock.contexts[0] as GPUCellSetOutline;
    expect(consumer.props.source?.cells).toBe(fixture.table.cells);
    expect(consumer.props.source?.status.count === fixture.table.status.count).toBe(true);
    expect(
      consumer.props.source?.status.overflow.buffer === fixture.table.status.overflow.buffer
    ).toBe(false);
    expect(
      consumer.props.source?.status.candidateOverflow?.buffer ===
        fixture.table.status.candidateOverflow.buffer
    ).toBe(false);
    expect(fixture.nodes.some(node => node.id.endsWith('outline-capture-overflow'))).toBe(true);
    expect(consumer.props.output.status.candidateOverflow).toBe(
      fixture.outlineStatus.candidateOverflow
    );
    expect(consumer.props.rings?.output.status.candidateOverflow).toBe(
      fixture.ringStatus.candidateOverflow
    );
    expect(consumer.props.rings?.output.status.complete).toBe(fixture.ringStatus.complete);
  } finally {
    outline.mockRestore();
    device.destroy();
  }
});
