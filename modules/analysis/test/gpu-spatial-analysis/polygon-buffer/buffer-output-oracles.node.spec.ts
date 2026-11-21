// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUBufferSurface} from '../../../src/gpu-spatial-analysis/polygon-buffer/gpu-buffer-surface';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createBufferOutputOracle, UNWRITTEN_STATUS} from './buffer-output-oracles';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {
  expectBufferBoundaryOracle,
  expectBufferNodingDiagnostics
} from './buffer-output-oracles.expect';

useCPUExecutorTestHooks();

it('retains square and round caps of a finite collapsed stored line but no flat cap', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const capStyle of ['square', 'round', 'flat'] as const) {
      const fixture = createBufferOutputOracle(device, undefined, {
        positions: [3, 7, 3, 7],
        lineOffsets: [0, 0, 2, 2],
        capStyle,
        distance: 1,
        vertexTolerance: capStyle === 'round' ? 1 / 65536 : 0.25,
        boundaryCapacity: 32,
        vertexCapacity: 33,
        intersections: 128,
        nodedSegments: 64
      });
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
        const expected = capStyle === 'square' ? 4 : capStyle === 'round' ? 32 : 0;
        expect(
          Array.from(memory.readUint32(fixture.props.output.boundary.status.count)),
          capStyle
        ).toEqual([expected]);
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

it('captures a changing boundary descriptor coherently with one getter read', () => {
  const device = createNullWebGPUDevice();
  const fixture = createBufferOutputOracle(device);
  try {
    let outputReads = 0;
    let boundaryReads = 0;
    const first = fixture.props.output.boundary;
    const second = {...first, endpoints: fixture.output('second-endpoints', 'float32x4', 4)};
    const output = {
      ...fixture.props.output,
      get boundary() {
        return ++boundaryReads === 1 ? first : second;
      }
    };
    const contributor = new GPUBufferSurface({
      ...fixture.props,
      get output() {
        outputReads++;
        return output;
      }
    });
    expect(contributor.props.output.boundary.endpoints).toBe(first.endpoints);
    expect(contributor.props.output.boundary.status.count).toBe(first.status.count);
    expect(outputReads).toBe(1);
    expect(boundaryReads).toBe(1);
  } finally {
    fixture.destroy();
    device.destroy();
  }
});

it('refuses every hidden result-only boundary counter', () => {
  const device = createNullWebGPUDevice();
  const fixture = createBufferOutputOracle(device);
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
            new GPUBufferSurface({
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

it('keeps written counts distinct from retry requirements at both buffer capacities', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const boundaryCapacity of [3, 4]) {
      const fixture = createBufferOutputOracle(device, undefined, {
        boundaryCapacity,
        vertexCapacity: 4,
        ringCounts: true
      });
      try {
        const memory = createCPUGraphMemory();
        for (const entry of fixture.columns) memory.write(entry.view, entry.values);
        const end = fixture.nodes.findIndex(node => node.id.endsWith('-assemble-ring-stats'));
        executeCommandNodesOnCPU(fixture.nodes.slice(0, end), memory);
        executeCommandNodesOnCPU(
          fixture.nodes.filter(
            node =>
              node.id.endsWith('-assemble-publish') ||
              node.id.includes('-ring-status-') ||
              node.id === 'buffer-surface-repair-publish'
          ),
          memory
        );
        const read = (name: string) =>
          Array.from(memory.readUint32(fixture.columns.find(entry => entry.name === name)!.view));
        if (boundaryCapacity === 3) {
          expect(read('boundaryCount')).toEqual([3]);
          expect(read('boundaryRequiredCount')).toEqual([4]);
          expect(read('boundaryOverflow')).toEqual([1]);
        } else {
          expect(read('ringCount')).toEqual([0]);
          expect(read('requiredRingCount')).toEqual([1]);
          expect(read('overflow')).toEqual([1]);
        }
      } finally {
        fixture.destroy();
      }
    }
  } finally {
    device.destroy();
  }
});

it('refuses hidden buffer status views and captures supported getters once', () => {
  const device = createNullWebGPUDevice();
  const fixture = createBufferOutputOracle(device);
  try {
    const invalidCount = fixture.output('hidden-invalid', 'uint32', 1);
    for (const boundary of [false, true])
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
        expect(() => new GPUBufferSurface({...fixture.props, output})).toThrow(
          /invalidCount.*unsupported/
        );
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
    const contributor = new GPUBufferSurface({
      ...fixture.props,
      id: 'captured-buffer',
      output: {...fixture.props.output, status}
    });
    Object.defineProperty(status, 'uncertainCount', {value: undefined});
    Object.defineProperty(status, 'invalidCount', {value: invalidCount, enumerable: true});
    expect(contributor.props.output.status.uncertainCount).toBe(uncertainCount);
    expect(reads).toBe(1);
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

it('preserves invalid owned vertices in short parts and clears the flag after correction', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const shortPart of [
      [Number.NaN, 0],
      [Infinity, 0, Infinity, 0]
    ]) {
      const fixture = createBufferOutputOracle(device, 'uncertainCount', {
        positions: [...shortPart, 0, 0, 4, 0],
        lineOffsets: [0, shortPart.length / 2, shortPart.length / 2 + 2]
      });
      try {
        const memory = createCPUGraphMemory();
        for (const entry of fixture.columns) memory.write(entry.view, entry.values);
        const end = fixture.nodes.findIndex(node => node.id.endsWith('-assemble-ring-stats'));
        for (const invalid of [true, false]) {
          memory.write(
            fixture.props.geometry.positions,
            new Float32Array([...(invalid ? shortPart : shortPart.map(() => 0)), 0, 0, 4, 0])
          );
          executeCommandNodesOnCPU(fixture.nodes.slice(0, end), memory);
          // The buffer's final status merge does not depend on ring geometry.
          executeCommandNodesOnCPU(
            fixture.nodes.filter(
              node =>
                node.id === 'buffer-surface-repair-publish' || node.id.endsWith('-input-status')
            ),
            memory
          );
          for (const [name, expected] of [
            ['boundaryCount', 4],
            ['boundaryCandidateOverflow', invalid ? 1 : 0],
            ['candidateOverflow', invalid ? 1 : 0],
            ['uncertainCount', 0]
          ] as const) {
            const entry = fixture.columns.find(entry => entry.name === name)!;
            expect(Array.from(memory.readUint32(entry.view)), `${shortPart}: ${name}`).toEqual([
              expected
            ]);
          }
        }
      } finally {
        fixture.destroy();
      }
    }
  } finally {
    device.destroy();
  }
});

it('keeps translated square-cap boundaries when leading and trailing parts are empty', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const lineOffsets of [
      [0, 2],
      [0, 0, 2],
      [0, 2, 2],
      [0, 0, 2, 2]
    ]) {
      const fixture = createBufferOutputOracle(device, undefined, {
        positions: [0, 5000000, 0, 5000004],
        lineOffsets,
        capStyle: 'square',
        distance: 0.5,
        vertexTolerance: 0.5,
        boundaryCapacity: 8,
        vertexCapacity: 9
      });
      try {
        const memory = createCPUGraphMemory();
        for (const entry of fixture.columns) memory.write(entry.view, entry.values);
        const end = fixture.nodes.findIndex(node => node.id.endsWith('-assemble-ring-stats'));
        executeCommandNodesOnCPU(fixture.nodes.slice(0, end), memory);
        for (const [name, expected] of [
          ['boundaryCount', 8],
          ['boundaryRequiredCount', 8],
          ['boundaryOverflow', 0],
          ['boundaryCandidateOverflow', 0]
        ] as const) {
          const entry = fixture.columns.find(entry => entry.name === name)!;
          expect(Array.from(memory.readUint32(entry.view)), `${lineOffsets}: ${name}`).toEqual([
            expected
          ]);
        }
      } finally {
        fixture.destroy();
      }
    }
  } finally {
    device.destroy();
  }
});

it('uses the resolved contributor ID in early provenance validation', () => {
  const device = createNullWebGPUDevice();
  const fixture = createBufferOutputOracle(device);
  try {
    // These branches preceded the constructor's local id declaration in the lead's failing
    // revision. A temporal-dead-zone ReferenceError must not replace either named diagnostic.
    const sourceFeatureRows = fixture.output('short-source-rows', 'uint32', 3);
    expect(
      () =>
        new GPUBufferSurface({
          ...fixture.props,
          id: 'oracle-polygons-id',
          output: {
            ...fixture.props.output,
            boundary: {...fixture.props.output.boundary, sourceFeatureRows}
          }
        })
    ).toThrow(/^oracle-polygons-id .*output.boundary.sourceFeatureRows.length/);
    const sourceIds = fixture.column('short-source-ids', 'uint32', new Uint32Array([99]));
    expect(
      () =>
        new GPUBufferSurface({
          ...fixture.props,
          id: 'oracle-polygons-id',
          // Polygon inputs bypass line snapshot validation, reaching the constructor branch
          // that used its local id before declaration in the lead's failing revision.
          geometry: {
            kind: 'polygons',
            positions: fixture.props.geometry.positions,
            featureOffsets: fixture.column('id-features', 'uint32', new Uint32Array([0, 0, 1])),
            polygonOffsets: fixture.column('id-polygons', 'uint32', new Uint32Array([0, 1])),
            ringOffsets: fixture.column('id-rings', 'uint32', new Uint32Array([0, 2])),
            sourceIds
          }
        })
    ).toThrow(/^oracle-polygons-id geometry.sourceIds length must equal logical feature count$/);
  } finally {
    fixture.destroy();
    device.destroy();
  }
});

it('separates nonfinite input loss from predicate uncertainty and clears both on recovery', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const counter of [true, false]) {
      const fixture = createBufferOutputOracle(device, counter ? 'uncertainCount' : undefined);
      try {
        const memory = createCPUGraphMemory();
        for (const entry of fixture.columns) memory.write(entry.view, entry.values);
        // All buffer generation, snap rounding, intersections and boundary publication run.
        // The following ring-statistics kernel has a workgroup barrier unsupported by this
        // interpreter; no assembly or top-level publication result is claimed by this test.
        const statisticsIndex = fixture.nodes.findIndex(
          node => node.id === 'buffer-surface-repair-assemble-ring-stats'
        );
        expect(statisticsIndex).toBeGreaterThan(0);
        const nodes = fixture.nodes.slice(0, statisticsIndex);
        for (const usable of [true, false, true]) {
          memory.write(
            fixture.props.geometry.positions,
            new Float32Array(usable ? [0, 0, 4, 0] : [Number.NaN, 0, Number.NaN, 0])
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
          expectBufferNodingDiagnostics(actual, usable);
          if (usable) expectBufferBoundaryOracle(actual);
        }
      } finally {
        fixture.destroy();
      }
    }
  } finally {
    device.destroy();
  }
});

it('refuses status views the buffer pipeline cannot publish', () => {
  const device = createNullWebGPUDevice();
  const fixture = createBufferOutputOracle(device);
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
        expect(() => new GPUBufferSurface({...fixture.props, output}), name).toThrow(
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
    expect(() => new GPUBufferSurface({...fixture.props, output})).toThrow(
      /output.boundary.status.uncertainCount/
    );
  } finally {
    fixture.destroy();
    device.destroy();
  }
});
