// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUTrajectorySegments} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-segments';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it('publishes trip candidate loss from triplegs and stays, and clears it on a later frame', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(format: Format, length = 1) =>
      createTransientView(graph, `status-view-${serial++}`, format, length);
    const stayOverflow = view('uint32');
    const stayCandidateOverflow = view('uint32');
    const triplegOverflow = view('uint32');
    const triplegCandidateOverflow = view('uint32');
    const tripOverflow = view('uint32');
    const tripCandidateOverflow = view('uint32');
    recordWGSLKernelNodes(graph);
    const props = {
      id: 'loss-probe',
      positions: view('float32x2', 4),
      timestamps: view('float32', 4),
      trackOffsets: view('uint32', 3),
      staypointIds: view('uint32', 4),
      parameters: view('float32', 4),
      staypoints: {
        tracks: view('uint32'),
        startRows: view('uint32'),
        finishRows: view('uint32'),
        durations: view('float32'),
        status: {
          count: view('uint32'),
          overflow: stayOverflow,
          candidateOverflow: stayCandidateOverflow
        }
      },
      output: {
        triplegs: {
          ids: view('uint32'),
          status: {
            count: view('uint32'),
            overflow: triplegOverflow,
            candidateOverflow: triplegCandidateOverflow
          }
        },
        trips: {
          ids: view('uint32', 2),
          status: {
            count: view('uint32'),
            overflow: tripOverflow,
            candidateOverflow: tripCandidateOverflow
          }
        }
      }
    };
    expect(
      () =>
        new GPUTrajectorySegments({
          ...props,
          output: {
            ...props.output,
            trips: {
              ...props.output.trips,
              status: {...props.output.trips.status, candidateOverflow: undefined}
            }
          }
        })
    ).toThrow(/loss-probe output.trips.status.candidateOverflow/);
    expect(
      () =>
        new GPUTrajectorySegments({
          ...props,
          output: {
            ...props.output,
            triplegs: {
              ...props.output.triplegs,
              status: {...props.output.triplegs.status, candidateOverflow: undefined}
            }
          }
        })
    ).toThrow(/loss-probe output.triplegs.status.candidateOverflow/);
    const contributor = new GPUTrajectorySegments(props);
    props.staypoints.status.candidateOverflow = view('uint32');
    props.output.trips.status.candidateOverflow = view('uint32');
    props.output.triplegs.status.candidateOverflow = view('uint32');
    const nodes = contributor.getCommandNodes(graph);
    const legPublisher = nodes.find(node => node.id === 'loss-probe-tripleg-publish')!;
    const lossNode = nodes.find(node => node.id === 'loss-probe-trip-input-loss')!;
    const tripPublisher = nodes.find(node => node.id === 'loss-probe-trip-publish')!;
    const legTotal = getWGSLKernelNodeRecord(graph, legPublisher.id)!.bindings.find(
      binding => binding.name === 'totalIn'
    )!.view;
    const tripTotal = getWGSLKernelNodeRecord(graph, tripPublisher.id)!.bindings.find(
      binding => binding.name === 'totalIn'
    )!.view;
    const memory = createCPUGraphMemory();
    memory.write(tripTotal, Uint32Array.of(1));
    for (const [required, upstreamFinal, upstreamCandidate, expected] of [
      [2, 0, 0, 1],
      [1, 1, 0, 1],
      [1, 0, 1, 1],
      [1, 0, 0, 0]
    ]) {
      memory.write(legTotal, Uint32Array.of(required));
      memory.write(stayOverflow, Uint32Array.of(upstreamFinal));
      memory.write(stayCandidateOverflow, Uint32Array.of(upstreamCandidate));
      executeCommandNodesOnCPU([legPublisher, lossNode, tripPublisher], memory);
      expect(memory.readUint32(tripCandidateOverflow)[0]).toBe(expected);
      expect(memory.readUint32(tripOverflow)[0]).toBe(0);
      expect(memory.readUint32(triplegOverflow)[0]).toBe(Number(required > 1));
      expect(memory.readUint32(triplegCandidateOverflow)[0]).toBe(
        Number(Boolean(upstreamFinal || upstreamCandidate))
      );
    }
  } finally {
    device.destroy();
  }
});
