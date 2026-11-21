// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUModalSplit} from '../../../src/gpu-spatial-analysis/trajectory-mobility/gpu-modal-split';
import {GPUModalSplitOperation} from '../../../src/gpu-spatial-analysis/trajectory-mobility/gpu-modal-split-operation';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {getModalProps} from './modal-split-operation-fixture';
import {GPUMobilityIndicators} from '../../../src/gpu-spatial-analysis/trajectory-mobility/gpu-mobility-indicators';
import {findConstantNonFiniteBitcasts} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

it('suppresses every staypoint-dependent kernel output and recovers on a later frame', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(format: Format, length = 1) =>
      createTransientView(graph, `stay-loss-${serial++}`, format, length);
    const overflow = view('uint32');
    const candidateOverflow = view('uint32');
    let reads = 0;
    const status = Object.create({
      overflow,
      get candidateOverflow() {
        reads++;
        return candidateOverflow;
      }
    }) as {
      count: GraphDataView<'uint32'>;
      overflow: GraphDataView<'uint32'>;
      candidateOverflow: GraphDataView<'uint32'>;
    };
    status.count = view('uint32');
    const stays = {
      tracks: view('uint32', 3),
      centroids: view('float32x2', 3),
      startRows: view('uint32', 3),
      durations: view('float32', 3),
      locationIds: view('uint32', 3),
      status
    };
    const timestamps = view('float32', 6);
    const output = {
      radiusOfGyration: view('float32', 2),
      jumpLengths: view('float32', 3),
      waitingTimes: view('float32', 3),
      randomEntropy: view('float32', 2),
      uncorrelatedEntropy: view('float32', 2),
      realEntropy: view('float32', 2),
      homeLocations: view('uint32', 2),
      workLocations: view('uint32', 2),
      status: {candidateOverflow: view('uint32')}
    };
    const contributor = new GPUMobilityIndicators({
      trackCount: 2,
      timestamps,
      staypoints: stays,
      output
    });
    expect(reads).toBe(1);
    Object.defineProperty(status, 'candidateOverflow', {value: view('uint32')});
    status.overflow = view('uint32');
    recordWGSLKernelNodes(graph);
    const nodes = contributor.getCommandNodes(graph);
    for (const node of nodes) {
      const record = getWGSLKernelNodeRecord(graph, node.id);
      if (record) expect(findConstantNonFiniteBitcasts(record.source)).toEqual([]);
    }
    const memory = createCPUGraphMemory();
    memory.write(stays.tracks, Uint32Array.of(0, 0, 0));
    memory.write(stays.centroids, Float32Array.of(0, 0, 10, 0, 20, 0));
    memory.write(stays.startRows, Uint32Array.of(0, 2, 4));
    memory.write(stays.durations, Float32Array.of(1, 1, 1));
    memory.write(stays.locationIds, Uint32Array.of(0, 1, 2));
    memory.write(status.count, Uint32Array.of(3));
    memory.write(timestamps, Float32Array.of(0, 1, 2, 3, 4, 5));
    for (const [finalLoss, candidateLoss] of [
      [1, 0],
      [0, 1],
      [0, 0]
    ]) {
      memory.write(overflow, Uint32Array.of(finalLoss));
      memory.write(candidateOverflow, Uint32Array.of(candidateLoss));
      executeCommandNodesOnCPU(nodes, memory);
      expect(memory.readUint32(output.status.candidateOverflow)[0]).toBe(
        Number(Boolean(finalLoss || candidateLoss))
      );
      if (finalLoss || candidateLoss) {
        for (const values of [
          output.radiusOfGyration,
          output.jumpLengths,
          output.waitingTimes,
          output.randomEntropy,
          output.uncorrelatedEntropy,
          output.realEntropy
        ]) {
          expect(Array.from(memory.readFloat32(values))).toEqual(
            new Array(values.length).fill(NaN)
          );
        }
        expect(Array.from(memory.readUint32(output.homeLocations))).toEqual([
          0xffffffff, 0xffffffff
        ]);
        expect(Array.from(memory.readUint32(output.workLocations))).toEqual([
          0xffffffff, 0xffffffff
        ]);
      } else {
        expect(memory.readFloat32(output.radiusOfGyration)[0]).toBeCloseTo(Math.sqrt(200 / 3), 5);
        expect(memory.readFloat32(output.randomEntropy)[0]).toBeCloseTo(Math.log2(3), 6);
        expect(memory.readUint32(output.homeLocations)[0]).toBe(0);
      }
    }
    for (const exposeStatus of [false, true]) {
      const radius = view('float32', 2);
      const candidate = view('uint32');
      const numericOnly = new GPUMobilityIndicators({
        id: `numeric-only-${exposeStatus}`,
        trackCount: 2,
        timestamps,
        staypoints: {...stays, status: {count: status.count, overflow, candidateOverflow}},
        output: {
          radiusOfGyration: radius,
          ...(exposeStatus ? {status: {candidateOverflow: candidate}} : {})
        }
      }).getCommandNodes(graph);
      for (const loss of [1, 0]) {
        memory.write(overflow, Uint32Array.of(loss));
        memory.write(candidateOverflow, Uint32Array.of(0));
        memory.write(candidate, Uint32Array.of(99));
        executeCommandNodesOnCPU(numericOnly, memory);
        if (loss) expect(Array.from(memory.readFloat32(radius))).toEqual([NaN, NaN]);
        else expect(memory.readFloat32(radius)[0]).toBeCloseTo(Math.sqrt(200 / 3), 5);
        if (exposeStatus) expect(memory.readUint32(candidate)[0]).toBe(loss);
      }
    }
  } finally {
    device.destroy();
  }
});

it('suppresses the tracking-quality shader output on either captured loss input', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(format: Format, length = 1) =>
      createTransientView(graph, `quality-loss-${serial++}`, format, length);
    const overflow = view('uint32');
    const candidateOverflow = view('uint32');
    const trackingQuality = view('float32', 2);
    const status = {count: view('uint32'), overflow, candidateOverflow};
    const contributor = new GPUMobilityIndicators({
      trackCount: 2,
      timestamps: view('float32', 4),
      staypoints: {
        tracks: view('uint32'),
        centroids: view('float32x2'),
        startRows: view('uint32'),
        durations: view('float32'),
        locationIds: view('uint32'),
        status: {count: view('uint32')}
      },
      triplegs: {
        tracks: view('uint32'),
        startRows: view('uint32'),
        endRows: view('uint32'),
        staypointFinishRows: view('uint32'),
        status
      },
      output: {trackingQuality}
    });
    status.overflow = view('uint32');
    status.candidateOverflow = view('uint32');
    recordWGSLKernelNodes(graph);
    const nodes = contributor
      .getCommandNodes(graph)
      .filter(node => node.id.endsWith('-suppress-incomplete-quality'));
    expect(nodes).toHaveLength(1);
    expect(
      findConstantNonFiniteBitcasts(getWGSLKernelNodeRecord(graph, nodes[0].id)!.source)
    ).toEqual([]);
    const memory = createCPUGraphMemory();
    for (const [finalLoss, candidateLoss] of [
      [1, 0],
      [0, 1],
      [0, 0]
    ]) {
      memory.write(overflow, Uint32Array.of(finalLoss));
      memory.write(candidateOverflow, Uint32Array.of(candidateLoss));
      memory.write(trackingQuality, Float32Array.of(1, 0.5));
      executeCommandNodesOnCPU(nodes, memory);
      expect(Array.from(memory.readFloat32(trackingQuality))).toEqual(
        finalLoss || candidateLoss ? [NaN, NaN] : [1, 0.5]
      );
    }
  } finally {
    device.destroy();
  }
});

it('snapshots loss maps once and suppresses actual modal kernels on either flag', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(format: Format, length = 1) =>
      createTransientView(graph, `loss-${serial++}`, format, length);
    const overflow = view('uint32');
    const candidate = view('uint32');
    let reads = 0;
    const status = {
      count: view('uint32'),
      overflow,
      get candidateOverflow() {
        reads++;
        return candidate;
      }
    };
    const output = {modalSplit: view('float32', 2), predictedModes: view('uint32')};
    const props = {
      positions: view('float32x2', 4),
      timestamps: view('float32', 4),
      trackOffsets: view('uint32', 3),
      triplegs: {
        tracks: view('uint32'),
        startRows: view('uint32'),
        endRows: view('uint32'),
        status
      },
      modeCount: 1,
      modeSpeedUpperBoundsFloat32: [Infinity],
      metric: 'count' as const,
      output
    };
    const contributor = new GPUModalSplit(props);
    expect(reads).toBe(1);
    status.overflow = view('uint32');
    props.modeSpeedUpperBoundsFloat32[0] = 1;
    const originalOutput = output.modalSplit;
    output.modalSplit = view('float32', 2);
    recordWGSLKernelNodes(graph);
    const nodes = contributor
      .getCommandNodes(graph)
      .filter(node => /-(predict|aggregate|suppress-incomplete)$/.test(node.id));
    expect(reads).toBe(1);
    const memory = createCPUGraphMemory();
    const suppression = nodes.find(node => node.id.endsWith('-suppress-incomplete'))!;
    expect(
      findConstantNonFiniteBitcasts(getWGSLKernelNodeRecord(graph, suppression.id)!.source)
    ).toEqual([]);
    const aggregate = getWGSLKernelNodeRecord(graph, 'modal-split-aggregate')!;
    const offsets = aggregate.bindings.find(binding => binding.name === 'offsets')!.view;
    memory.write(offsets, Uint32Array.of(0, 1, 1));
    memory.write(status.count, Uint32Array.of(1));
    memory.write(props.triplegs.startRows, Uint32Array.of(0));
    memory.write(props.triplegs.endRows, Uint32Array.of(2));
    memory.write(props.positions, Float32Array.of(0, 0, 1, 0, 0, 0, 1, 0));
    memory.write(props.timestamps, Float32Array.of(0, 1, 0, 1));
    for (const [finalLoss, candidateLoss] of [
      [1, 0],
      [0, 1],
      [0, 0]
    ]) {
      memory.write(overflow, Uint32Array.of(finalLoss));
      memory.write(candidate, Uint32Array.of(candidateLoss));
      executeCommandNodesOnCPU(nodes, memory);
      expect(Array.from(memory.readFloat32(originalOutput))).toEqual(
        finalLoss || candidateLoss ? [NaN, NaN] : [1, 0]
      );
      expect(memory.readUint32(output.predictedModes)[0]).toBe(
        finalLoss || candidateLoss ? 0xffffffff : 0
      );
    }
    const operationProps = getModalProps('count', true);
    operationProps.triplegs.status.candidateOverflow = 'candidate';
    const operation = new GPUModalSplitOperation(operationProps);
    delete operationProps.triplegs.status.candidateOverflow;
    operationProps.modeCount = 1;
    const views: Record<string, GraphDataView> = {
      positions: props.positions,
      timestamps: props.timestamps,
      trackOffsets: props.trackOffsets,
      triplegTracks: props.triplegs.tracks,
      triplegStartRows: props.triplegs.startRows,
      triplegEndRows: props.triplegs.endRows,
      triplegCount: status.count,
      triplegCandidateOverflow: candidate,
      modalSplit: view('float32', 6),
      predictedModes: output.predictedModes
    };
    const captured = operation.createContributor(port => views[port] as never) as GPUModalSplit;
    expect(captured.props.modeCount).toBe(3);
    expect(captured.props.triplegs.status.candidateOverflow).toBe(candidate);
  } finally {
    device.destroy();
  }
});
