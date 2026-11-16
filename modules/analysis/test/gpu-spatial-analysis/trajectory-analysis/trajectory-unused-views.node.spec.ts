// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUTrajectoryResample} from '../../../src/gpu-spatial-analysis/trajectory-interpolation/gpu-trajectory-resample';
import {GPUTrajectoryPlayhead} from '../../../src/gpu-spatial-analysis/trajectory-interpolation/gpu-trajectory-playhead';
import {GPUSpaceTimeCrossCorrelation} from '../../../src/gpu-spatial-analysis/space-time-statistics/gpu-space-time-cross-correlation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPUModalSplit} from '../../../src/gpu-spatial-analysis/trajectory-mobility/gpu-modal-split';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it('does not preflight timestamps or elevations unused by spatial-only resampling', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const buffer = device.createBuffer({byteLength: 8, usage: Buffer.COPY_DST});
  try {
    const unused = importGraphBuffer(graph, 'unused', buffer, 'float32', 2);
    const contributor = new GPUTrajectoryResample({
      positions: createTransientView(graph, 'positions', 'float32x2', 2),
      trackOffsets: createTransientView(graph, 'offsets', 'uint32', 2),
      timestamps: unused,
      elevations: unused,
      spacing: 'arc-length',
      sampleCount: 2,
      output: {samples: createTransientView(graph, 'samples', 'float32x2', 2)}
    });
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it('does not preflight elevations when playhead elevation output is absent', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const buffer = device.createBuffer({byteLength: 8, usage: Buffer.COPY_DST});
  try {
    const contributor = new GPUTrajectoryPlayhead({
      positions: createTransientView(graph, 'positions', 'float32x2', 2),
      trackOffsets: createTransientView(graph, 'offsets', 'uint32', 2),
      timestamps: createTransientView(graph, 'times', 'float32', 2),
      parameters: createTransientView(graph, 'parameters', 'float32', 4),
      elevations: importGraphBuffer(graph, 'unused', buffer, 'float32', 2),
      output: {currentPositions: createTransientView(graph, 'current', 'float32x2', 1)}
    });
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it('ignores an unused tested-count view when no multiple-testing output is requested', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const buffer = device.createBuffer({byteLength: 4, usage: Buffer.COPY_DST});
  try {
    const first = createTransientView(graph, 'first', 'float32', 4);
    const correlations = createTransientView(graph, 'correlations', 'float32', 1);
    const props = {
      first,
      second: createTransientView(graph, 'second', 'float32', 4),
      layout: 'cell-major' as const,
      cellCount: 1,
      periodCount: 4,
      lags: [0],
      output: {
        correlations,
        pValues: createTransientView(graph, 'p-values', 'float32', 1)
      }
    };
    for (const testedCount of [
      importGraphBuffer(graph, 'unused', buffer, 'uint32', 1),
      graph.createDataView(correlations.buffer, {format: 'uint32', length: 1}),
      graph.createDataView(first.buffer, {format: 'uint32', length: 1})
    ]) {
      expect(() =>
        new GPUSpaceTimeCrossCorrelation({
          ...props,
          output: {...props.output, testedCount}
        }).getCommandNodes(graph)
      ).not.toThrow();
    }
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

function createModalProps(graph: GPUCommandGraph) {
  return {
    positions: createTransientView(graph, 'positions', 'float32x2', 2),
    timestamps: createTransientView(graph, 'times', 'float32', 2),
    trackOffsets: createTransientView(graph, 'tracks', 'uint32', 2),
    triplegs: {
      tracks: createTransientView(graph, 'trip-tracks', 'uint32', 1),
      startRows: createTransientView(graph, 'starts', 'uint32', 1),
      endRows: createTransientView(graph, 'ends', 'uint32', 1),
      status: {count: createTransientView(graph, 'count', 'uint32', 1)}
    },
    modes: createTransientView(graph, 'modes', 'uint32', 1),
    metric: 'count' as const,
    modeCount: 2,
    output: {
      modalSplit: createTransientView(graph, 'split', 'float32', 2),
      predictedModes: createTransientView(graph, 'predicted', 'uint32', 1)
    }
  };
}

it('publishes predicted modes without overwriting a supplied modes input', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createModalProps(graph);
    const nodes = new GPUModalSplit({...props, modeSpeedBreaks: [1, Infinity]}).getCommandNodes(
      graph
    );
    const memory = createCPUGraphMemory();
    memory.write(props.positions, new Float32Array([0, 0, 10, 0]));
    memory.write(props.timestamps, new Float32Array([0, 1]));
    memory.write(props.triplegs.startRows, new Uint32Array([0]));
    memory.write(props.triplegs.endRows, new Uint32Array([2]));
    memory.write(props.triplegs.status.count, new Uint32Array([1]));
    memory.write(props.modes, new Uint32Array([42]));
    memory.write(props.output.predictedModes, new Uint32Array([99]));
    executeCommandNodesOnCPU([nodes.find(node => node.id === 'modal-split-predict')!], memory);
    expect(Array.from(memory.readUint32(props.output.predictedModes))).toEqual([1]);
    expect(Array.from(memory.readUint32(props.modes))).toEqual([42]);
  } finally {
    device.destroy();
  }
});

it('ignores an unused predicted-modes output when using supplied modes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const buffer = device.createBuffer({byteLength: 4, usage: Buffer.COPY_DST});
  try {
    const props = createModalProps(graph);
    const predictedModes = importGraphBuffer(graph, 'unused', buffer, 'uint32', 1);
    expect(() =>
      new GPUModalSplit({...props, output: {...props.output, predictedModes}}).getCommandNodes(
        graph
      )
    ).not.toThrow();
  } finally {
    buffer.destroy();
    device.destroy();
  }
});
