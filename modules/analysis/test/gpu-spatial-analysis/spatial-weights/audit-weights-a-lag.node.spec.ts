// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterEach, beforeEach, expect, it, vi} from 'vitest';
import * as loadBalanced from '../../../src/analysis-core/kernels/load-balanced-for-each';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {
  createLagAuditCase,
  LAG_AUDIT_EXPECTED,
  type LagAuditOptions
} from './audit-weights-a-lag-fixture';

beforeEach(() => {
  // Replace only the analysis-owned cooperative CSR reducer, which the serial interpreter
  // explicitly cannot execute. Contribution functions/bindings, conversion, normalization,
  // chunk output views and loss suppression all come unchanged from GPUSpatialLag.
  // This proves their mathematics/wiring against NumPy; it does not prove merge-path execution.
  vi.spyOn(loadBalanced, 'createLoadBalancedSegmentSumNodes').mockImplementation((graph, props) => {
    if (props.accumulatorType !== 'f32') throw new Error('Lag audit supports only scalar f32 sums');
    const lanes = props.laneCount ?? 1;
    return [
      createWGSLKernelNode(graph, {
        id: `${props.id}-audit-serial-sum`,
        operation: props.operation,
        invocationCount: (props.offsets.length - 1) * lanes,
        bindings: [
          {name: 'auditOffsets', view: props.offsets, type: 'u32', access: 'read'},
          ...props.bindings,
          {name: 'auditOutput', view: props.output, type: 'f32', access: 'read_write'}
        ],
        declarations: props.declarations,
        body: `let row = index / ${lanes}u;
let lane = index % ${lanes}u;
var sum = 0.0;
for (var item = min(auditOffsets[auditOffsetsOffset + row], ${props.itemCapacity}u);
     item < min(auditOffsets[auditOffsetsOffset + row + 1u], ${props.itemCapacity}u); item++) {
  sum += getLoadBalancedContribution(row, item, lane);
}
auditOutput[auditOutputOffset + index] = sum;`
      })
    ];
  });
});
afterEach(() => vi.restoreAllMocks());

function executeCase(options: LagAuditOptions) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  let serial = 0;
  const fixture = createLagAuditCase(
    {
      input: (data, format) => {
        const view = createTransientView(graph, `lag-input-${serial++}`, format, data.length);
        memory.write(view, data);
        return view;
      },
      output: length => {
        const backing = createTransientView(graph, `lag-output-${serial++}`, 'float32', length + 2);
        memory.write(backing, new Float32Array(length + 2).fill(-999));
        return graph.createDataView(backing.buffer, {format: 'float32', byteOffset: 4, length});
      }
    },
    options
  );
  return {device, graph, memory, fixture};
}

for (const scenario of Object.keys(LAG_AUDIT_EXPECTED) as LagAuditOptions['scenario'][]) {
  it(`lag ${scenario}: production contribution and normalization shaders match dense oracle in every form`, () => {
    let maximumError = 0;
    for (const chunked of [false, true])
      for (const normalize of [false, true])
        for (const columnCount of [1, 2] as const)
          for (const valueFormat of ['float32', 'uint32'] as const) {
            const {device, graph, memory, fixture} = executeCase({
              scenario,
              chunked,
              normalize,
              columnCount,
              valueFormat
            });
            try {
              executeCommandNodesOnCPU(fixture.contributor.getCommandNodes(graph), memory);
              const actual = Array.from(memory.readFloat32(fixture.output));
              expect(actual.length).toBe(fixture.expected.length);
              for (const [row, value] of actual.entries()) {
                expect(Number.isFinite(value)).toBe(true);
                maximumError = Math.max(maximumError, Math.abs(value - fixture.expected[row]));
                // <=2e-6 covers one final float32 division, including the affine copy up to 41.
                expect(Math.abs(value - fixture.expected[row])).toBeLessThanOrEqual(2e-6);
              }
              const backing = new Float32Array(memory.getBuffer(fixture.output.buffer).backing);
              expect(backing[0]).toBe(-999);
              expect(backing[fixture.output.length + 1]).toBe(-999);
            } finally {
              device.destroy();
            }
          }
    process.stdout.write(`lag ${scenario} maximum absolute output error: ${maximumError}\n`);
  });
}

it('whole/chunked lag suppresses every column for either loss cause and recovers on the same graph', () => {
  for (const chunked of [false, true])
    for (const lossKind of ['overflow', 'candidateOverflow'] as const)
      for (const lossLocation of chunked ? (['outer', 'chunk'] as const) : (['outer'] as const)) {
        const {device, graph, memory, fixture} = executeCase({
          scenario: 'masked',
          chunked,
          normalize: true,
          columnCount: 2,
          valueFormat: 'uint32',
          lossKind,
          lossLocation
        });
        try {
          const nodes = fixture.contributor.getCommandNodes(graph);
          for (const loss of [1, 0]) {
            memory.write(fixture.loss!, Uint32Array.of(loss));
            executeCommandNodesOnCPU(nodes, memory);
            const actual = Array.from(memory.readFloat32(fixture.output));
            for (const [row, value] of actual.entries()) {
              if (loss) expect(value).toBeNaN();
              else {
                expect(Number.isFinite(value)).toBe(true);
                expect(Math.abs(value - fixture.expected[row])).toBeLessThanOrEqual(2e-6);
              }
            }
          }
        } finally {
          device.destroy();
        }
      }
});
