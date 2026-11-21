// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {getUndeclaredCalls} from '../segment-intersection/wgsl-call-sites';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {
  createMultipartPolygonizeCase,
  type MultipartTestFormat
} from './multipart-line-forms-fixtures';
import {
  createNodingOracleCase,
  createPolygonizeOracleCase,
  getOracleView,
  UNSUPPORTED_TOPOLOGY_STATUS
} from './noding-polygonize-output-oracles-fixtures';

useCPUExecutorTestHooks();

it('snapped projection comparison accepts the float32 reciprocal route', () => {
  // The shader's lattice projection is 52/160. A float32 reciprocal multiply
  // and a rounded division differ by one representable float32 value.
  const reciprocalRoute = Math.fround(52 * Math.fround(1 / 160));
  const expected = [0, 0.32499998807907104, 0, 0.25];
  expect(matchesSnappedReviewParameters(expected, expected)).toBe(true);
  expect(matchesSnappedReviewParameters([0, reciprocalRoute, 0, 0.25], expected)).toBe(true);
  expect(
    matchesSnappedReviewParameters([0, Math.fround(reciprocalRoute + 2 ** -25), 0, 0.25], expected)
  ).toBe(false);
  for (const wrong of [
    0,
    1,
    0.25,
    Math.fround(1 / 3),
    Number.NaN,
    Number.POSITIVE_INFINITY,
    Number.NEGATIVE_INFINITY
  ]) {
    expect(matchesSnappedReviewParameters([0, wrong, 0, 0.25], expected), String(wrong)).toBe(
      false
    );
  }
  expect(
    matchesSnappedReviewParameters([0, reciprocalRoute, 0, Math.fround(1 / 3)], expected)
  ).toBe(false);
  expect(
    matchesSnappedReviewParameters([0, reciprocalRoute, 0, 0.2500000298023224], expected)
  ).toBe(false);
  expect(matchesSnappedReviewParameters([0, reciprocalRoute, 0], expected)).toBe(false);
});

import {
  createReviewNodingCase,
  REVIEW_NODING_CASES,
  matchesSnappedReviewParameters
} from './noding-review-output-oracles-fixtures';
import {
  createReviewPolygonizeCase,
  createReviewPolygonizeProps
} from './polygonize-review-output-oracles-fixtures';
import {GPUPolygonize} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-polygonize';

it('review dangle-pruning kernels parse and stay within core storage bindings', () => {
  const {device, graph, allocate} = createHarness();
  try {
    const fixture = createReviewPolygonizeCase(graph, allocate, 'spur');
    const pruning = fixture.nodes.filter(node => node.id.startsWith('polygonize-dangles-'));
    expect(pruning.length).toBeGreaterThan(8);
    for (const {id, source} of getParsedKernelSources(pruning).configurations) {
      expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
      // The call-site utility predates this standard WGSL synchronization builtin.
      expect(
        getUndeclaredCalls(source).filter(name => name !== 'workgroupUniformLoad'),
        id
      ).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});

it.each([
  'spur',
  'chain',
  'short-spur',
  'overlap-square-first',
  'overlap-exterior-first'
] as const)('review polygonize recursively removes dangles: %s', subject => {
  const {device, graph, allocate, memory} = createHarness();
  try {
    const fixture = createReviewPolygonizeCase(graph, allocate, subject);
    const placement = fixture.nodes.findIndex(
      node => node.id === 'polygonize-faces-propagate-duplicate-owners'
    );
    expect(placement).toBeGreaterThan(0);
    executeCommandNodesOnCPU(fixture.nodes.slice(0, placement + 1), memory);
    for (const path of ['directedEdges.ringIds', 'directedEdges.flags']) {
      const expected = fixture.expected[path];
      expect(
        Array.from(memory.readUint32(getOracleView(fixture.output, path), expected.length)),
        path
      ).toEqual(expected);
    }
  } finally {
    device.destroy();
  }
});

it('review polygonize rejects obsolete requiredCount directly at construction', () => {
  const {device, allocate} = createHarness();
  try {
    const props = createReviewPolygonizeProps(allocate);
    Object.assign(props.output.status, {requiredCount: allocate('obsolete-required', 'uint32', 1)});
    expect(() => new GPUPolygonize(props)).toThrow('requiredRingCount');
  } finally {
    device.destroy();
  }
});

it.each([
  'top',
  'noded',
  'directed'
] as const)('review polygonize validates %s uncertainty scalar at construction', destination => {
  const {device, allocate} = createHarness();
  try {
    const props = createReviewPolygonizeProps(allocate);
    const status =
      destination === 'top'
        ? props.output.status
        : destination === 'noded'
          ? props.output.nodedSegments.status
          : props.output.directedEdges.status;
    status.uncertainCount = allocate('zero-uncertain', 'uint32', 0);
    expect(() => new GPUPolygonize(props)).toThrow('uncertainCount');
  } finally {
    device.destroy();
  }
});

it.each([
  'top',
  'noded',
  'directed'
] as const)('review polygonize validates %s uncertainty format at construction', destination => {
  const {device, allocate} = createHarness();
  try {
    const props = createReviewPolygonizeProps(allocate);
    const status =
      destination === 'top'
        ? props.output.status
        : destination === 'noded'
          ? props.output.nodedSegments.status
          : props.output.directedEdges.status;
    Object.assign(status, {uncertainCount: allocate('float-uncertain', 'float32', 1)});
    expect(() => new GPUPolygonize(props)).toThrow('uncertainCount');
  } finally {
    device.destroy();
  }
});

it('review bridge expectations cover every defined padding row', () => {
  const {device, graph, allocate} = createHarness();
  try {
    const fixture = createPolygonizeOracleCase(graph, allocate);
    for (const path of [
      'nodedSegments.endpoints',
      'nodedSegments.sourceFeatureIds',
      'nodedSegments.sourceRingIds',
      'nodedSegments.sourceEdgeIds',
      'nodedSegments.sourceStartParameters',
      'nodedSegments.sourceEndParameters',
      'directedEdges.endpoints',
      'directedEdges.sourceSegmentIds',
      'directedEdges.sourceFeatureIds',
      'directedEdges.twinIds',
      'directedEdges.ringIds',
      'directedEdges.flags',
      'diagnostics.edgeClasses'
    ]) {
      const view = getOracleView(fixture.output, path);
      expect(fixture.expected[path].length, path).toBe((view.length * view.rowByteLength) / 4);
    }
  } finally {
    device.destroy();
  }
});

it.each(REVIEW_NODING_CASES)('review noding oracle %s', subject => {
  const {device, graph, allocate, memory} = createHarness();
  try {
    const fixture = createReviewNodingCase(graph, allocate, subject);
    executeCommandNodesOnCPU(fixture.nodes, memory);
    for (const [path, expected] of Object.entries(fixture.expected)) {
      const view = getOracleView(fixture.output, path);
      expect(
        Array.from(
          view.format === 'uint32'
            ? memory.readUint32(view, expected.length)
            : memory.readFloat32(view, expected.length)
        ),
        path
      ).toEqual(expected);
    }
  } finally {
    device.destroy();
  }
});

it('review uncertain counter omission still propagates incompleteness', () => {
  const {device, graph, allocate, memory} = createHarness();
  try {
    const fixture = createReviewNodingCase(graph, allocate, 'uncertain', false);
    executeCommandNodesOnCPU(fixture.nodes, memory);
    expect(Array.from(memory.readUint32(fixture.output.status.candidateOverflow))).toEqual([1]);
  } finally {
    device.destroy();
  }
});

function createHarness() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const allocate = <Format extends MultipartTestFormat>(
    name: string,
    format: Format,
    length: number,
    values?: readonly number[]
  ) => {
    const view = createTransientView(graph, name, format, length);
    const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
    memory.write(
      view,
      format.startsWith('uint')
        ? Uint32Array.from(values ?? new Array(length * width).fill(777))
        : Float32Array.from(values ?? new Array(length * width).fill(-777))
    );
    return view;
  };
  return {device, graph, memory, allocate};
}

it.each(UNSUPPORTED_TOPOLOGY_STATUS)('GPULineNoding refuses unwritten output.status.%s', field => {
  const {device, graph, allocate} = createHarness();
  try {
    expect(() => createNodingOracleCase(graph, allocate, false, field)).toThrow(
      `line-noding output.status.${field}`
    );
  } finally {
    device.destroy();
  }
});

it.each([
  {grouped: false, snapped: false},
  {grouped: true, snapped: false},
  {grouped: false, snapped: true},
  {grouped: true, snapped: true}
])('multipart triangle fixture retains all directed rings with groups=$grouped snap=$snapped', ({
  grouped,
  snapped
}) => {
  const {device, graph, allocate, memory} = createHarness();
  try {
    const fixture = createMultipartPolygonizeCase(graph, allocate, grouped, snapped);
    const placement = fixture.nodes.findIndex(
      node => node.id === 'polygonize-faces-propagate-duplicate-owners'
    );
    expect(placement).toBeGreaterThan(0);
    executeCommandNodesOnCPU(
      [
        ...fixture.nodes.slice(0, placement + 1),
        ...fixture.nodes.filter(node =>
          ['polygonize-faces-publish', 'polygonize-publish-status'].includes(node.id)
        )
      ],
      memory
    );
    // multipart-line-forms.oracle.py: Shapely produces two area-2 triangles.
    // Each has a CCW bounded walk and a CW unbounded walk, three vertices plus
    // closure apiece. The shared assembly scratch needs 4 * 4 = 16 vertices,
    // although the final polygon port publishes only 2 * 3 = 6 vertices.
    expect(Array.from(memory.readUint32(fixture.output.directedEdges.ringIds))).toEqual([
      0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3
    ]);
    expect(Array.from(memory.readUint32(fixture.output.directedEdges.flags))).toEqual([
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]);
    expect(Array.from(memory.readUint32(fixture.output.status.requiredRingCount))).toEqual([4]);
    expect(Array.from(memory.readUint32(fixture.output.status.overflow!))).toEqual([0]);
  } finally {
    device.destroy();
  }
});

it('multipart triangle scratch truncation reports the lost unbounded walk', () => {
  const {device, graph, allocate, memory} = createHarness();
  try {
    const fixture = createMultipartPolygonizeCase(graph, (name, format, length, values) =>
      allocate(name, format, name === 'polygon-positions' ? 12 : length, values)
    );
    const placement = fixture.nodes.findIndex(
      node => node.id === 'polygonize-faces-propagate-duplicate-owners'
    );
    expect(placement).toBeGreaterThan(0);
    executeCommandNodesOnCPU(
      [
        ...fixture.nodes.slice(0, placement + 1),
        ...fixture.nodes.filter(node =>
          ['polygonize-faces-publish', 'polygonize-publish-status'].includes(node.id)
        )
      ],
      memory
    );
    // Four four-vertex walks need 16 rows. Twelve rows hold only three walks;
    // source rows 7,9,11 therefore have no stored fourth ring (GPU_TOPOLOGY_NONE).
    expect(Array.from(memory.readUint32(fixture.output.directedEdges.ringIds))).toEqual([
      0, 1, 0, 1, 0, 1, 2, 0xffffffff, 2, 0xffffffff, 2, 0xffffffff
    ]);
    expect(Array.from(memory.readUint32(fixture.output.status.requiredRingCount))).toEqual([4]);
    expect(Array.from(memory.readUint32(fixture.output.status.overflow!))).toEqual([1]);
  } finally {
    device.destroy();
  }
});

it.each(
  UNSUPPORTED_TOPOLOGY_STATUS
)('GPUSnapRoundedNoding refuses unwritten output.status.%s', field => {
  const {device, graph, allocate} = createHarness();
  try {
    expect(() => createNodingOracleCase(graph, allocate, false, field, 'snap')).toThrow(
      `snap-rounded-noding output.status.${field}`
    );
  } finally {
    device.destroy();
  }
});

it.each(
  (['status', 'nodedSegments.status', 'directedEdges.status'] as const).flatMap(path =>
    UNSUPPORTED_TOPOLOGY_STATUS.map(field => ({path, field}))
  )
)('GPUPolygonize refuses unwritten $path scalar $field', ({path, field}) => {
  const {device, graph, allocate} = createHarness();
  try {
    expect(() => createPolygonizeOracleCase(graph, allocate, 'none', false, {path, field})).toThrow(
      `polygonize output.${path}.${field}`
    );
  } finally {
    device.destroy();
  }
});

it.each([
  false,
  true
])('polygonize noding and directed provenance match bridge literals with snap=%s', snapped => {
  const {device, graph, allocate, memory} = createHarness();
  try {
    const fixture = createPolygonizeOracleCase(graph, allocate, 'all', snapped);
    const placement = fixture.nodes.findIndex(
      node => node.id === 'polygonize-faces-propagate-duplicate-owners'
    );
    expect(placement).toBeGreaterThan(0);
    // The Node interpreter cannot synchronize the later cooperative area kernel.
    // Execute the full noding, matching and face traversal through ring placement;
    // headless additionally covers polygon publication and diagnostics.
    const nodes = [
      ...fixture.nodes.slice(0, placement + 1),
      ...fixture.nodes.filter(node => node.id.startsWith('polygonize-publish-uncertain-'))
    ];
    executeCommandNodesOnCPU(nodes, memory);
    for (const [path, expected] of Object.entries(fixture.expected)) {
      if (
        !path.startsWith('nodedSegments.') &&
        !path.startsWith('directedEdges.') &&
        path !== 'lattice' &&
        path !== 'status.uncertainCount'
      )
        continue;
      const view = getOracleView(fixture.output, path);
      expect(
        Array.from(
          view.format === 'uint32'
            ? memory.readUint32(view, expected.length)
            : memory.readFloat32(view, expected.length)
        ),
        path
      ).toEqual(expected);
    }
  } finally {
    device.destroy();
  }
});

it.each([
  {mode: 'line', uncertainty: false},
  {mode: 'line', uncertainty: true},
  {mode: 'snap', uncertainty: false},
  {mode: 'snap', uncertainty: true}
] as const)('$mode noding pins quarter-crossing provenance with uncertainty=$uncertainty', ({
  mode,
  uncertainty
}) => {
  const {device, graph, allocate, memory} = createHarness();
  try {
    const fixture = createNodingOracleCase(graph, allocate, uncertainty, undefined, mode);
    for (let submission = 0; submission < 2; submission++) {
      // Poison again on the same nodes to reject stale values on resubmission.
      for (const path of Object.keys(fixture.expected)) {
        const view = getOracleView(fixture.output, path);
        const words = (view.length * view.rowByteLength) / 4;
        memory.write(
          view,
          view.format === 'uint32'
            ? new Uint32Array(words).fill(777)
            : new Float32Array(words).fill(-777)
        );
      }
      executeCommandNodesOnCPU(fixture.nodes, memory);
      for (const [path, expected] of Object.entries(fixture.expected)) {
        const view = getOracleView(fixture.output, path);
        expect(
          Array.from(view.format === 'uint32' ? memory.readUint32(view) : memory.readFloat32(view)),
          path
        ).toEqual(expected);
      }
    }
  } finally {
    device.destroy();
  }
});

it('w5 dangle controls preserve both square permutations on replay', () => {
  for (const subject of ['overlap-square-first', 'overlap-exterior-first'] as const) {
    const {device, graph, allocate, memory} = createHarness();
    try {
      const fixture = createReviewPolygonizeCase(graph, allocate, subject);
      const placement = fixture.nodes.findIndex(
        node => node.id === 'polygonize-faces-propagate-duplicate-owners'
      );
      expect(placement).toBeGreaterThan(0);
      for (let replay = 0; replay < 2; replay++) {
        executeCommandNodesOnCPU(fixture.nodes.slice(0, placement + 1), memory);
        const expected = fixture.expected['directedEdges.ringIds'];
        expect(
          Array.from(memory.readUint32(fixture.output.directedEdges.ringIds, expected.length))
        ).toEqual(expected);
      }
    } finally {
      device.destroy();
    }
  }
});
