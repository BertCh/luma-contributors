// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUPolygonOverlay} from '../../../src/gpu-spatial-analysis/polygon-overlay/gpu-polygon-overlay';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {UNWRITTEN_STATUS} from '../polygon-buffer/buffer-output-oracles';
import {createPolygonOverlayOutputOracle} from './polygon-overlay-output-oracles';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {
  expectPolygonOverlayBoundaryOracle,
  expectPolygonOverlayNodingDiagnostics
} from './polygon-overlay-output-oracles.expect';

useCPUExecutorTestHooks();

it('keeps inherited nonzero winding on a twice-traced square', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const origin of ['own', 'inherited'] as const) {
      const fixture = createPolygonOverlayOutputOracle(device, false, origin);
      try {
        const memory = createCPUGraphMemory();
        for (const entry of fixture.columns) memory.write(entry.view, entry.values);
        executeCommandNodesOnCPU(
          fixture.nodes.slice(
            0,
            fixture.nodes.findIndex(node => node.id.endsWith('-assemble-ring-stats'))
          ),
          memory
        );
        expect(
          Array.from(memory.readUint32(fixture.props.output.boundary.status.count)),
          origin
        ).toEqual([4]);
        expect(
          Array.from(memory.readUint32(fixture.props.output.boundary.status.candidateOverflow))
        ).toEqual([0]);
      } finally {
        fixture.destroy();
      }
    }
  } finally {
    device.destroy();
  }
});

it('reads the overlay output and boundary descriptors once', () => {
  const device = createNullWebGPUDevice();
  const fixture = createPolygonOverlayOutputOracle(device);
  try {
    let outputReads = 0;
    let boundaryReads = 0;
    const output = {
      ...fixture.props.output,
      get boundary() {
        boundaryReads++;
        return fixture.props.output.boundary;
      }
    };
    const contributor = new GPUPolygonOverlay({
      ...fixture.props,
      get output() {
        outputReads++;
        return output;
      }
    });
    expect(outputReads).toBe(1);
    expect(boundaryReads).toBe(1);
    expect(contributor.props.output.boundary.endpoints).toBe(
      fixture.props.output.boundary.endpoints
    );
  } finally {
    fixture.destroy();
    device.destroy();
  }
});

it('refuses every hidden result-only boundary counter', () => {
  const device = createNullWebGPUDevice();
  const fixture = createPolygonOverlayOutputOracle(device);
  try {
    for (const name of ['ringCount', 'requiredRingCount', 'nodedSegmentRequiredCount'])
      for (const inherited of [true, false]) {
        const view = fixture.output(`${name}-${inherited}`, 'uint32', 1);
        const status = inherited
          ? Object.assign(
              Object.create({
                get [name]() {
                  return view;
                }
              }),
              fixture.props.output.boundary.status
            )
          : Object.defineProperty({...fixture.props.output.boundary.status}, name, {value: view});
        expect(
          () =>
            new GPUPolygonOverlay({
              ...fixture.props,
              output: {
                ...fixture.props.output,
                boundary: {...fixture.props.output.boundary, status}
              }
            })
        ).toThrow(new RegExp(`output\\.boundary\\.status\\.${name}.*unsupported`));
      }
  } finally {
    fixture.destroy();
    device.destroy();
  }
});

it('refuses hidden status views and snapshots deferred supported views once', () => {
  const device = createNullWebGPUDevice();
  const fixture = createPolygonOverlayOutputOracle(device, true);
  try {
    const invalidCount = fixture.output('hidden-invalid', 'uint32', 1);
    for (const boundary of [false, true]) {
      for (const inherited of [false, true]) {
        const original = boundary
          ? fixture.props.output.boundary.status
          : fixture.props.output.status;
        const status = inherited
          ? Object.assign(
              Object.create({
                get invalidCount() {
                  return invalidCount;
                }
              }),
              original
            )
          : Object.defineProperty({...original}, 'invalidCount', {value: invalidCount});
        const output = boundary
          ? {...fixture.props.output, boundary: {...fixture.props.output.boundary, status}}
          : {...fixture.props.output, status};
        expect(() => new GPUPolygonOverlay({...fixture.props, output})).toThrow(
          /invalidCount.*unsupported/
        );
      }
    }
    let reads = 0;
    const uncertainCount = fixture.output('captured-uncertain', 'uint32', 1);
    const status = {
      ...fixture.props.output.status,
      get uncertainCount() {
        reads++;
        return uncertainCount;
      }
    };
    const contributor = new GPUPolygonOverlay({
      ...fixture.props,
      id: 'captured-overlay',
      output: {...fixture.props.output, status}
    });
    Object.defineProperty(status, 'uncertainCount', {value: undefined});
    Object.defineProperty(status, 'invalidCount', {value: invalidCount, enumerable: true});
    expect(contributor.props.output.status.uncertainCount).toBe(uncertainCount);
    expect(reads).toBe(1);
    expect('invalidCount' in contributor.props.output.status).toBe(false);
    const nodes = contributor.getCommandNodes(fixture.graph);
    const memory = createCPUGraphMemory();
    for (const entry of fixture.columns) memory.write(entry.view, entry.values);
    const end = nodes.findIndex(node => node.id.endsWith('-assemble-ring-stats'));
    executeCommandNodesOnCPU(nodes.slice(0, end), memory);
    expect(Array.from(memory.readUint32(uncertainCount))).toEqual([0]);
    expect(reads).toBe(1);
  } finally {
    fixture.destroy();
    device.destroy();
  }
});

it('keeps rejected vertices separate from uncertainty and clears candidate loss on recovery', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const counter of [true, false]) {
      const fixture = createPolygonOverlayOutputOracle(device, counter);
      try {
        const memory = createCPUGraphMemory();
        for (const entry of fixture.columns) memory.write(entry.view, entry.values);
        // Execute all real overlay noding/selection/boundary publishers. Assembly ring stats
        // require workgroup barriers, which the shared CPU interpreter cannot execute.
        const statisticsIndex = fixture.nodes.findIndex(
          node => node.id === 'polygon-overlay-assemble-ring-stats'
        );
        expect(statisticsIndex).toBeGreaterThan(0);
        const nodes = fixture.nodes.slice(0, statisticsIndex);
        for (const usable of [true, false, true]) {
          for (const entry of fixture.columns.filter(
            entry => entry.name === 'left' || entry.name === 'right'
          ))
            memory.write(
              entry.view,
              usable ? entry.values : new Float32Array(entry.values.length).fill(Number.NaN)
            );
          for (const entry of fixture.columns.filter(
            entry =>
              entry.output && (entry.name.startsWith('boundary') || entry.name === 'uncertainCount')
          ))
            memory.write(entry.view, entry.values);
          executeCommandNodesOnCPU(nodes, memory);
          const actual = Object.fromEntries(
            fixture.columns
              .filter(
                entry =>
                  entry.output &&
                  (entry.name.startsWith('boundary') ||
                    ['uncertainCount', 'endpoints', 'operandIds', 'sourceFeatureIds'].includes(
                      entry.name
                    ))
              )
              .map(entry => [
                entry.name,
                Array.from(
                  entry.values instanceof Uint32Array
                    ? memory.readUint32(entry.view)
                    : memory.readFloat32(entry.view)
                )
              ])
          );
          expectPolygonOverlayNodingDiagnostics(actual, usable);
          if (usable) expectPolygonOverlayBoundaryOracle(actual);
        }
      } finally {
        fixture.destroy();
      }
    }
  } finally {
    device.destroy();
  }
});

it('refuses status views the overlay pipeline cannot publish', () => {
  const device = createNullWebGPUDevice();
  const fixture = createPolygonOverlayOutputOracle(device);
  try {
    for (const name of UNWRITTEN_STATUS) {
      const scalar = fixture.output(`unsupported-${name}`, 'uint32', 1);
      for (const boundary of [false, true]) {
        const output = boundary
          ? {
              ...fixture.props.output,
              boundary: {
                ...fixture.props.output.boundary,
                status: {...fixture.props.output.boundary.status, [name]: scalar}
              }
            }
          : {...fixture.props.output, status: {...fixture.props.output.status, [name]: scalar}};
        expect(() => new GPUPolygonOverlay({...fixture.props, output}), name).toThrow(
          new RegExp(`output\\.${boundary ? 'boundary\\.status' : 'status'}\\.${name}`)
        );
      }
    }
    const output = {
      ...fixture.props.output,
      boundary: {
        ...fixture.props.output.boundary,
        status: {
          ...fixture.props.output.boundary.status,
          uncertainCount: fixture.output('unsupported-uncertain', 'uint32', 1)
        }
      }
    };
    expect(() => new GPUPolygonOverlay({...fixture.props, output})).toThrow(
      /output.boundary.status.uncertainCount/
    );
  } finally {
    fixture.destroy();
    device.destroy();
  }
});
