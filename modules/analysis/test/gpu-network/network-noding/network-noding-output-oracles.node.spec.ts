// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  GPUNetworkNoding,
  type GPUNetworkNodingProps
} from '../../../src/gpu-network/network-noding/index';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {createOracleContext} from '../adjacency-matrix/network-output-oracle-test-utils';

function createNodingBindingFixture() {
  const device = createNullWebGPUDevice();
  const context = createOracleContext(device);
  const {input, output} = context;
  const props: GPUNetworkNodingProps = {
    lines: {
      kind: 'lines',
      positions: input('positions', 'float32x2', [0, 1, 2, 1, 1, 0, 1, 2]),
      lineOffsets: input('lines', 'uint32', [0, 2, 4])
    },
    tolerance: input('tolerance', 'float32', [0]),
    intersectionCapacity: 8,
    output: {
      pieces: {
        geometry: {
          kind: 'lines',
          positions: output('piece-positions', 'float32x2', 8),
          lineOffsets: output('piece-offsets', 'uint32', 5)
        },
        sourceIds: output('piece-rows', 'uint32', 4),
        status: {count: output('pieces', 'uint32'), overflow: output('piece-overflow', 'uint32')}
      },
      nodes: {
        positions: output('nodes', 'float32x2', 5),
        status: {
          count: output('node-count', 'uint32'),
          requiredCount: output('node-required', 'uint32')
        }
      },
      edges: {
        fromNodes: output('from', 'uint32', 4),
        toNodes: output('to', 'uint32', 4),
        lengths: output('lengths', 'float32', 4)
      },
      topology: {
        offsets: output('offsets', 'uint32', 6),
        neighbors: output('neighbors', 'uint32', 8),
        weights: output('weights', 'float32', 8)
      },
      status: {overflow: output('whole-overflow', 'uint32')}
    }
  };
  return {
    ...context,
    props,
    destroy() {
      context.destroy();
      device.destroy();
    }
  };
}

for (const route of ['candidate', 'aggregate', 'both', 'neither', 'unrestricted'] as const) {
  it(`W5-02 validates restricted leaf loss at the noding boundary (${route})`, () => {
    const fixture = createNodingBindingFixture();
    fixture.props.id = 'noding-boundary';
    fixture.props.leafCapacity = route === 'unrestricted' ? 4 : 2;
    if (route !== 'aggregate' && route !== 'both') fixture.props.output.status = undefined;
    if (route === 'candidate' || route === 'both')
      fixture.props.output.pieces.status.candidateOverflow = fixture.output('candidate', 'uint32');
    const state = fixture.graph as unknown as {nodes: unknown[]; buffers: Map<string, unknown>};
    const bufferCount = state.buffers.size;
    try {
      const construct = () => new GPUNetworkNoding(fixture.props);
      if (route === 'neither') {
        expect(construct).toThrow(
          /^noding-boundary.*leafCapacity.*2.*positions.length.*4.*candidateOverflow.*output.status.overflow/
        );
        expect(state.nodes).toHaveLength(0);
        expect(state.buffers.size).toBe(bufferCount);
      } else expect(construct).not.toThrow();
    } finally {
      fixture.destroy();
    }
  });
}

it('network noding accepts omitted and explicitly undefined inherited output fields', () => {
  for (const field of [
    'sourceIds',
    'complete',
    'unresolvedCount',
    'approximation',
    'converged',
    'invalidCount',
    'iterationCount',
    'iterationLimitReached',
    'requiredRingCount',
    'ringCount',
    'uncertainCount'
  ])
    for (const mode of ['omitted', 'undefined', 'inherited-undefined']) {
      const fixture = createNodingBindingFixture();
      const target =
        field === 'sourceIds'
          ? fixture.props.output.pieces.geometry
          : fixture.props.output.pieces.status;
      if (mode === 'undefined') Object.assign(target, {[field]: undefined});
      if (mode === 'inherited-undefined') Object.setPrototypeOf(target, {[field]: undefined});
      try {
        expect(
          () => new GPUNetworkNoding(fixture.props).getCommandNodes(fixture.graph),
          `${field}: ${mode}`
        ).not.toThrow();
      } finally {
        fixture.destroy();
      }
    }
});

it('network noding includes split candidate loss in whole-network overflow with or without its caller view', () => {
  for (const withDiagnostic of [true, false]) {
    const fixture = createNodingBindingFixture();
    if (withDiagnostic)
      fixture.props.output.pieces.status.candidateOverflow = fixture.output('candidate', 'uint32');
    try {
      const nodes = new GPUNetworkNoding(fixture.props).getCommandNodes(fixture.graph);
      const scalar = nodes.find(node => node.id === 'network-noding-scalars')!;
      const source = getParsedKernelSources([scalar]).get(scalar.id)!;
      // Node and piece capacity suffice in the headless three-pair fixture; the
      // independent incompleteness condition must still read the pair-loss word.
      expect(source).toContain('splitCandidateOverflow[splitCandidateOverflowOffset] != 0u');
    } finally {
      fixture.destroy();
    }
  }
});

it('network noding snapshots the same piece count for the split writer and endpoint reader', () => {
  const fixture = createNodingBindingFixture();
  const originalCount = fixture.props.output.pieces.status.count;
  const originalOffsets = fixture.props.output.pieces.geometry.lineOffsets;
  const originalFromNodes = fixture.props.output.edges.fromNodes;
  const contributor = new GPUNetworkNoding(fixture.props);
  fixture.props.output.pieces.status.count = fixture.output('replacement-count', 'uint32');
  fixture.props.output.pieces.geometry.lineOffsets = fixture.output(
    'replacement-offsets',
    'uint32',
    5
  );
  fixture.props.output.edges.fromNodes = fixture.output('replacement-from', 'uint32', 4);
  try {
    const nodes = contributor.getCommandNodes(fixture.graph);
    const split = nodes.find(node => node.id === 'network-noding-split-scalars')!;
    const endpoints = nodes.find(node => node.id === 'network-noding-endpoints')!;
    const edges = nodes.find(node => node.id === 'network-noding-edges')!;
    expect(
      split.resources?.some(
        resource =>
          'buffer' in resource &&
          resource.buffer === originalCount &&
          resource.usage === 'storage-read-write'
      )
    ).toBe(true);
    expect(
      endpoints.resources?.some(
        resource =>
          'buffer' in resource &&
          resource.buffer === originalCount &&
          resource.usage === 'storage-read'
      )
    ).toBe(true);
    expect(
      endpoints.resources?.some(
        resource => 'buffer' in resource && resource.buffer === originalOffsets
      )
    ).toBe(true);
    expect(
      edges.resources?.some(
        resource =>
          'buffer' in resource &&
          resource.buffer === originalFromNodes &&
          resource.usage === 'storage-read-write'
      )
    ).toBe(true);
  } finally {
    fixture.destroy();
  }
});

it('network noding refuses inherited ports that its line-split stage never writes', () => {
  const device = createNullWebGPUDevice();
  for (const inherited of [false, true])
    for (const field of [
      'sourceIds',
      'complete',
      'unresolvedCount',
      'approximation',
      'converged',
      'invalidCount',
      'iterationCount',
      'iterationLimitReached',
      'requiredRingCount',
      'ringCount',
      'uncertainCount'
    ]) {
      const context = createOracleContext(device);
      const {input, output} = context;
      const props: GPUNetworkNodingProps = {
        lines: {
          kind: 'lines',
          positions: input('positions', 'float32x2', [0, 0, 2, 0, 1, 0, 1, 2]),
          lineOffsets: input('lines', 'uint32', [0, 2, 4])
        },
        tolerance: input('tolerance', 'float32', [0]),
        intersectionCapacity: 8,
        output: {
          pieces: {
            geometry: {
              kind: 'lines',
              positions: output('piece-positions', 'float32x2', 8),
              lineOffsets: output('piece-offsets', 'uint32', 5)
            },
            sourceIds: output('piece-rows', 'uint32', 4),
            status: {
              count: output('pieces', 'uint32'),
              overflow: output('piece-overflow', 'uint32')
            }
          },
          nodes: {
            positions: output('node-positions', 'float32x2', 5),
            status: {count: output('nodes', 'uint32')}
          },
          edges: {
            fromNodes: output('from', 'uint32', 4),
            toNodes: output('to', 'uint32', 4),
            lengths: output('lengths', 'float32', 4)
          },
          topology: {
            offsets: output('offsets', 'uint32', 6),
            neighbors: output('neighbors', 'uint32', 8),
            weights: output('weights', 'float32', 8)
          }
        }
      };
      // Object spreads and JS callers bypass excess-property checks: refuse instead of silently
      // accepting caller buffers that no command writes.
      const target =
        field === 'sourceIds' ? props.output.pieces.geometry : props.output.pieces.status;
      const unsupported = {[field]: output('unsupported', 'uint32', field === 'sourceIds' ? 4 : 1)};
      if (inherited) Object.setPrototypeOf(target, unsupported);
      else Object.assign(target, unsupported);
      try {
        expect(() => new GPUNetworkNoding(props), field).toThrow(
          new RegExp('network-noding.*' + field + '.*use')
        );
      } finally {
        context.destroy();
      }
    }
  device.destroy();
});
