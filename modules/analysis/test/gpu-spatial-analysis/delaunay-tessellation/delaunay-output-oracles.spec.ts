// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {expectFiniteCoordinatePrefix} from './coordinate-readback-expectations';
import {GPUDelaunayTessellationOperation} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation-operation';
import {GPUDelaunayTessellation} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation';
import {GPUVoronoiDiagram} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-voronoi-diagram';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {CPUDelaunayTessellationOperation} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/cpu-delaunay-tessellation-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

for (const scenario of ['capacity', 'exhausted', 'complete'] as const) {
  it(`propagates Delaunay completeness into composed Voronoi: ${scenario}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createOutputOracleFixture(device),
      {graph, input, output} = fixture;
    const scalar = (name: string) => output(name, 'uint32', 1);
    const positions = input('positions', 'float32x2', new Float32Array([0, 0, 4, 0, 3, 2, 0, 3]));
    const triangles = output('triangles', 'uint32x3', scenario === 'capacity' ? 1 : 2);
    const upstream = {
      count: scalar('triangle-count'),
      requiredCount: scalar('triangle-required'),
      overflow: scalar('triangle-overflow'),
      converged: scalar('triangle-converged')
    };
    let nodes = new GPUDelaunayTessellation({
      positions,
      output: {triangles, status: upstream}
    }).getCommandNodes(graph);
    if (scenario === 'exhausted')
      nodes = nodes.filter(
        node =>
          !node.id.includes('-continue-') &&
          (!/-step-(\d+)-/.test(node.id) || Number(/-step-(\d+)-/.exec(node.id)![1]) < 3)
      );
    graph.add(nodes);
    const status = {
      count: scalar('count'),
      requiredCount: scalar('required'),
      overflow: scalar('overflow'),
      candidateOverflow: scalar('candidate'),
      converged: scalar('converged')
    };
    const segments = output('segments', 'float32x4', 6);
    graph.add(
      new GPUVoronoiDiagram({
        positions,
        triangles,
        triangleCount: upstream.count,
        triangleStatus: upstream,
        clipBounds: input('bounds', 'float32x4', new Float32Array([-2, -2, 6, 5])),
        output: {segments, siteIds: output('sites', 'uint32x2', 6), status}
      })
    );
    const compiled = graph.compile();
    try {
      const actual = await fixture.submitTwice(compiled, async () => ({
        status: await fixture.readRows([
          status.count,
          status.requiredCount,
          status.overflow,
          status.candidateOverflow,
          status.converged
        ]),
        segments: (await fixture.readRows([segments])).slice(0, scenario === 'complete' ? 20 : 0)
      }));
      expect(actual.status).toEqual(
        scenario === 'complete'
          ? [5, 5, 0, 0, 1]
          : scenario === 'capacity'
            ? [0, 0, 1, 1, 0]
            : [0, 0, 0, 0, 0]
      );
      // Hand circumcentres (2,1/4),(7/6,3/2), four clipped hull rays and their connecting diagonal.
      const expected = [
        2,
        0.25,
        2,
        -2,
        2,
        0.25,
        6,
        2.25,
        2,
        0.25,
        7 / 6,
        1.5,
        7 / 6,
        1.5,
        7 / 3,
        5,
        7 / 6,
        1.5,
        -2,
        1.5
      ];
      if (scenario === 'complete') {
        expectFiniteCoordinatePrefix(actual.segments, actual.status[0], 4, expected, 2e-5);
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}

it('both Delaunay operation forms expose unclamped group ranges and one stored triangle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPUDelaunayTessellationOperation, CPUDelaunayTessellationOperation]) {
    const actual = await runOperationGraph(device, {
      inputs: {
        positions: {
          format: 'float32x2',
          values: new Float32Array([0, 0, 4, 0, 3, 2, 0, 3, 10, 0, 14, 0, 13, 2, 10, 3])
        },
        groupIds: {format: 'uint32', values: new Uint32Array([0, 0, 0, 0, 1, 1, 1, 1])}
      },
      operations: [
        new Operation({
          positions: 'positions',
          capacity: 1,
          groups: {ids: 'groupIds', count: 3},
          output: {
            triangles: 'triangles',
            status: {count: 'count', requiredCount: 'required', overflow: 'overflow'},
            groups: {
              triangleCounts: 'counts',
              triangleOffsets: 'offsets',
              status: {converged: 'converged'}
            }
          }
        })
      ],
      outputs: ['triangles', 'count', 'required', 'overflow', 'counts', 'offsets', 'converged']
    });
    expect(Array.from(actual['triangles'])).toEqual([0, 1, 2]);
    expect(Array.from(actual['count'])).toEqual([1]);
    expect(Array.from(actual['required'])).toEqual([4]);
    expect(Array.from(actual['overflow'])).toEqual([1]);
    expect(Array.from(actual['counts'])).toEqual([2, 2, 0]);
    expect(Array.from(actual['offsets'])).toEqual([0, 2, 4]);
    expect(
      Array.from(actual['offsets'], (offset, group) =>
        Math.max(0, Math.min(actual['counts'][group], actual['count'][0] - offset))
      )
    ).toEqual([1, 0, 0]);
  }
});

it('reports a retained seed simplex as approximate after forced exhaustion on WebGPU', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOutputOracleFixture(device);
  const {graph, input, output} = fixture;
  const triangles = output('triangles', 'uint32x3', 3),
    count = output('count', 'uint32', 1),
    requiredCount = output('required', 'uint32', 1),
    overflow = output('overflow', 'uint32', 1),
    converged = output('converged', 'uint32', 1),
    iterationLimitReached = output('limit', 'uint32', 1),
    approximation = output('approximation', 'uint32', 1);
  const positions = input(
    'positions',
    'float32x2',
    new Float32Array([0, 0, 1, 0, 0, 1, 0.25, 0.25])
  );
  const nodes = new GPUDelaunayTessellation({
    positions,
    output: {
      triangles,
      status: {count, requiredCount, overflow, converged, iterationLimitReached, approximation}
    }
  }).getCommandNodes(graph);
  graph.add(
    nodes.filter(
      node =>
        !node.id.includes('-continue-') &&
        (!/-step-(\d+)-/.test(node.id) || Number(/-step-(\d+)-/.exec(node.id)![1]) < 3)
    )
  );
  const compiled = graph.compile();
  try {
    const actual = await fixture.submitTwice(compiled, async () => ({
      triangles: (await fixture.readRows([triangles])).slice(0, 3),
      count: await fixture.readRows([count]),
      converged: await fixture.readRows([converged]),
      limit: await fixture.readRows([iterationLimitReached]),
      approximation: await fixture.readRows([approximation])
    }));
    expect(actual).toEqual({
      triangles: [0, 1, 2],
      count: [1],
      converged: [0],
      limit: [1],
      approximation: [1]
    });
    // Same compiled three-step schedule can finish a coincident batch without real triangles.
    fixture.writeInput(positions, new Float32Array(8));
    const complete = await fixture.submitTwice(compiled, async () => ({
      count: await fixture.readRows([count]),
      converged: await fixture.readRows([converged]),
      limit: await fixture.readRows([iterationLimitReached]),
      approximation: await fixture.readRows([approximation])
    }));
    expect(complete).toEqual({count: [0], converged: [1], limit: [0], approximation: [0]});
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('flags oversized Voronoi input and changed validity on the same compiled graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOutputOracleFixture(device);
  const {graph, input, output} = fixture;
  const positions = input('positions', 'float32x2', new Float32Array([0, 0, 2, 0, 0, 2]));
  const triangleCount = input('triangle-count', 'uint32', new Uint32Array([1]));
  const count = output('count', 'uint32', 1),
    requiredCount = output('required', 'uint32', 1),
    overflow = output('overflow', 'uint32', 1),
    candidateOverflow = output('candidate', 'uint32', 1),
    invalidCount = output('invalid', 'uint32', 1),
    converged = output('converged', 'uint32', 1);
  graph.add(
    new GPUVoronoiDiagram({
      positions,
      triangles: input('triangles', 'uint32x3', new Uint32Array([0, 1, 2])),
      triangleCount,
      clipBounds: input('bounds', 'float32x4', new Float32Array([-2, -2, 3, 3])),
      output: {
        segments: output('segments', 'float32x4', 3),
        siteIds: output('sites', 'uint32x2', 3),
        status: {count, requiredCount, overflow, candidateOverflow, invalidCount, converged}
      }
    })
  );
  const compiled = graph.compile();
  try {
    for (const requested of [1, 2, 1]) {
      fixture.writeInput(triangleCount, new Uint32Array([requested]));
      const actual = await fixture.submitTwice(compiled, async () => ({
        count: await fixture.readRows([count]),
        candidate: await fixture.readRows([candidateOverflow]),
        converged: await fixture.readRows([converged])
      }));
      expect(actual).toEqual({
        count: [3],
        candidate: [requested === 2 ? 1 : 0],
        converged: [requested === 2 ? 0 : 1]
      });
    }
    for (const invalid of [false, true, false]) {
      fixture.writeInput(
        positions,
        new Float32Array(invalid ? [0, 0, 1, 0, 2, 0] : [0, 0, 2, 0, 0, 2])
      );
      const actual = await fixture.submitTwice(compiled, async () => ({
        count: await fixture.readRows([count]),
        invalid: await fixture.readRows([invalidCount]),
        converged: await fixture.readRows([converged])
      }));
      expect(actual).toEqual({
        count: [invalid ? 0 : 3],
        invalid: [invalid ? 1 : 0],
        converged: [invalid ? 0 : 1]
      });
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('rejects shared invalid/convergence buffers at offsets zero and 256 on a real device', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOutputOracleFixture(device);
  const {graph, input, output} = fixture;
  const packed = output('packed', 'uint32', 65);
  const invalidCount = graph.createDataView(packed.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 0
  });
  const converged = graph.createDataView(packed.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 256
  });
  try {
    expect(
      () =>
        new GPUVoronoiDiagram({
          positions: input('positions', 'float32x2', new Float32Array([0, 0, 2, 0, 0, 2])),
          triangles: input('triangles', 'uint32x3', new Uint32Array([0, 1, 2])),
          triangleCount: input('triangle-count', 'uint32', new Uint32Array([1])),
          clipBounds: input('bounds', 'float32x4', new Float32Array([-2, -2, 3, 3])),
          output: {
            segments: output('segments', 'float32x4', 3),
            siteIds: output('sites', 'uint32x2', 3),
            status: {
              count: output('count', 'uint32', 1),
              requiredCount: output('required', 'uint32', 1),
              overflow: output('overflow', 'uint32', 1),
              invalidCount,
              converged
            }
          }
        })
    ).toThrow(/converged.*invalidCount|invalidCount.*converged/);
  } finally {
    fixture.destroy();
  }
});

for (const field of ['invalidCount', 'duplicateCount', 'converged'] as const) {
  it(`writes nonconstant ungrouped ${field} from a duplicate and nonfinite site`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createOutputOracleFixture(device);
    const {graph, input, output} = fixture;
    const selected = output('selected', 'uint32', 1),
      requiredCount = output('required', 'uint32', 1);
    const invalid = field !== 'duplicateCount';
    const positions = input(
      'positions',
      'float32x2',
      new Float32Array([0, 0, 3, 0, 0, 4, 0, 0, ...(invalid ? [NaN, 0] : [])])
    );
    const views = {
      positions,
      triangles: output('triangles', 'uint32x3', 1),
      count: output('count', 'uint32', 1),
      requiredCount,
      overflow: output('overflow', 'uint32', 1),
      [field]: selected
    };
    const operation = new GPUDelaunayTessellationOperation({
      positions: 'positions',
      capacity: 1,
      output: {
        triangles: 'triangles',
        status: {
          count: 'count',
          requiredCount: 'required',
          overflow: 'overflow',
          [field]: 'selected'
        }
      }
    });
    graph.add(operation.createContributor(port => views[port as keyof typeof views] as never));
    const compiled = graph.compile();
    try {
      const actual = await fixture.submitTwice(compiled, () => fixture.readRows([selected]));
      expect(actual).toEqual([field === 'converged' ? 0 : 1]);
      expect(await fixture.readRows([requiredCount])).toEqual([invalid ? 0 : 1]);
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}

for (const operationForm of [true, false]) {
  for (const optional of [
    undefined,
    'invalidCount',
    'duplicateCount',
    'converged',
    'iterationLimitReached',
    'triangleOffsets',
    'candidateOverflow',
    'uncertainCount',
    'approximation'
  ] as const) {
    it(`writes SciPy grouped quadrilaterals with only ${optional ?? 'required outputs'}, operation ${operationForm}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = createOutputOracleFixture(device);
      const {graph, input, output} = fixture;
      const positions = input(
        'positions',
        'float32x2',
        new Float32Array([0, 0, 4, 0, 3, 2, 0, 3, 10, 0, 14, 0, 13, 2, 10, 3])
      );
      const groupIds = input('groups', 'uint32', new Uint32Array([0, 0, 0, 0, 1, 1, 1, 1]));
      const triangles = output('triangles', 'uint32x3', 4);
      const count = output('count', 'uint32', 1),
        requiredCount = output('required', 'uint32', 1),
        overflow = output('overflow', 'uint32', 1);
      const triangleCounts = output('group-counts', 'uint32', 3),
        groupConverged = output('group-converged', 'uint32', 3);
      const selected = optional
        ? output('selected', 'uint32', optional === 'triangleOffsets' ? 3 : 1)
        : undefined;
      const status = {
        count,
        requiredCount,
        overflow,
        ...(optional && optional !== 'triangleOffsets' ? {[optional]: selected} : {})
      };
      const groups = {
        triangleCounts,
        triangleOffsets: optional === 'triangleOffsets' ? selected : undefined,
        status: {converged: groupConverged}
      };
      const options = {
        groups: {ids: groupIds, count: 3},
        positions,
        output: {triangles, status, groups}
      };
      const operation = new GPUDelaunayTessellationOperation({
        positions: 'positions',
        capacity: 4,
        groups: {ids: 'groups', count: 3},
        output: {
          triangles: 'triangles',
          status: {
            count: 'count',
            requiredCount: 'required',
            overflow: 'overflow',
            ...(optional && optional !== 'triangleOffsets' ? {[optional]: 'selected'} : {})
          },
          groups: {
            triangleCounts: 'counts',
            triangleOffsets: optional === 'triangleOffsets' ? 'offsets' : undefined,
            status: {converged: 'converged'}
          }
        }
      });
      const views = {
        positions,
        groupIds,
        triangles,
        count,
        requiredCount,
        overflow,
        groupTriangleCounts: triangleCounts,
        groupConverged,
        groupTriangleOffsets: selected,
        ...(optional ? {[optional]: selected} : {})
      };
      graph.add(
        operationForm
          ? operation.createContributor(port => views[port as keyof typeof views] as never)
          : new GPUDelaunayTessellation(options)
      );
      const compiled = graph.compile();
      try {
        const actual = await fixture.submitTwice(compiled, async () => ({
          triangles: await fixture.readRows([triangles]),
          count: await fixture.readRows([count]),
          required: await fixture.readRows([requiredCount]),
          overflow: await fixture.readRows([overflow]),
          counts: await fixture.readRows([triangleCounts]),
          converged: await fixture.readRows([groupConverged]),
          selected: selected ? await fixture.readRows([selected]) : []
        }));
        // SciPy script beside this spec supplies the diagonals; translation preserves topology.
        expect(actual.triangles).toEqual([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]);
        expect(actual.count).toEqual([4]);
        expect(actual.required).toEqual([4]);
        expect(actual.overflow).toEqual([0]);
        expect(actual.counts).toEqual([2, 2, 0]);
        expect(actual.converged).toEqual([1, 1, 1]);
        expect(actual.selected).toEqual(
          optional === 'triangleOffsets'
            ? [0, 2, 4]
            : optional
              ? [optional === 'converged' ? 1 : 0]
              : []
        );
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    });
  }
}

for (const optional of [
  undefined,
  'candidateOverflow',
  'uncertainCount',
  'iterationCount',
  'converged',
  'iterationLimitReached',
  'approximation'
] as const) {
  for (const invalid of optional === 'converged' ? [false, true] : [false]) {
    it(`writes analytic clipped triangle bisectors and ${optional ?? 'no optional diagnostics'}, invalid ${invalid}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = createOutputOracleFixture(device);
      const {graph, input, output} = fixture;
      const segments = output('segments', 'float32x4', 3),
        siteIds = output('site-ids', 'uint32x2', 3),
        count = output('count', 'uint32', 1),
        requiredCount = output('required', 'uint32', 1),
        overflow = output('overflow', 'uint32', 1);
      const selected = optional ? output('selected', 'uint32', 1) : undefined;
      graph.add(
        new GPUVoronoiDiagram({
          positions: input(
            'positions',
            'float32x2',
            new Float32Array(invalid ? [0, 0, 1, 0, 2, 0] : [0, 0, 2, 0, 0, 2])
          ),
          triangles: input('triangles', 'uint32x3', new Uint32Array([0, 1, 2])),
          triangleCount: input('triangle-count', 'uint32', new Uint32Array([1])),
          clipBounds: input('bounds', 'float32x4', new Float32Array([-2, -2, 3, 3])),
          output: {
            segments,
            siteIds,
            status: {count, requiredCount, overflow, ...(optional ? {[optional]: selected} : {})}
          }
        })
      );
      const compiled = graph.compile();
      try {
        const actual = await fixture.submitTwice(compiled, async () => ({
          segments: await fixture.readRows([segments]),
          sites: await fixture.readRows([siteIds]),
          count: await fixture.readRows([count]),
          required: await fixture.readRows([requiredCount]),
          selected: selected ? await fixture.readRows([selected]) : []
        }));
        expect(actual.count).toEqual([invalid ? 0 : 3]);
        expect(actual.required).toEqual([invalid ? 0 : 3]);
        expect(actual.selected).toEqual(
          optional ? [optional === 'converged' && !invalid ? 1 : 0] : []
        );
        if (!invalid) {
          // Circumcentre (1,1); outward rays are south, northeast, west.
          const expected = [1, 1, 1, -2, 1, 1, 3, 3, 1, 1, -2, 1];
          actual.segments.forEach((value, index) => {
            expect(Number.isFinite(value)).toBe(true);
            expect(Math.abs(value - expected[index])).toBeLessThan(2e-6);
          });
          expect(actual.sites).toEqual([0, 1, 1, 2, 0, 2]);
        }
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    });
  }
}

for (const capacity of [2, 3]) {
  it(`rewrites Voronoi solver, capacity and completeness destinations on clean reuse (capacity=${capacity})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createOutputOracleFixture(device);
    const {graph, input, output} = fixture;
    const upstream = {
      overflow: input('upstream-overflow', 'uint32', Uint32Array.of(0)),
      converged: input('upstream-converged', 'uint32', Uint32Array.of(0))
    };
    const status = {
      count: output('count', 'uint32', 1),
      requiredCount: output('required', 'uint32', 1),
      overflow: output('overflow', 'uint32', 1),
      candidateOverflow: output('candidate', 'uint32', 1),
      converged: output('converged', 'uint32', 1),
      complete: output('complete', 'uint32', 1)
    };
    let reads = 0;
    graph.add(
      new GPUVoronoiDiagram({
        positions: input('positions', 'float32x2', Float32Array.of(0, 0, 2, 0, 0, 2)),
        triangles: input('triangles', 'uint32x3', Uint32Array.of(0, 1, 2)),
        triangleCount: input('triangle-count', 'uint32', Uint32Array.of(1)),
        triangleStatus: upstream,
        clipBounds: input('bounds', 'float32x4', Float32Array.of(-2, -2, 3, 3)),
        output: {
          segments: output('segments', 'float32x4', capacity),
          siteIds: output('sites', 'uint32x2', capacity),
          status: {
            ...status,
            get complete() {
              reads++;
              return status.complete;
            }
          }
        }
      })
    );
    expect(reads).toBe(1);
    const compiled = graph.compile();
    try {
      // The right triangle has three rays. Exhaustion suppresses them without a capacity
      // flag; upstream capacity loss sets both flags; only the two-slot local output truncates.
      for (const [overflow, converged, expected] of [
        [0, 0, [0, 0, 0, 0, 0, 0]],
        [1, 1, [0, 0, 1, 1, 0, 0]],
        [0, 1, capacity === 2 ? [2, 3, 1, 0, 1, 0] : [3, 3, 0, 0, 1, 1]],
        [0, 0, [0, 0, 0, 0, 0, 0]],
        [0, 1, capacity === 2 ? [2, 3, 1, 0, 1, 0] : [3, 3, 0, 0, 1, 1]]
      ] as const) {
        fixture.writeInput(upstream.overflow, Uint32Array.of(overflow));
        fixture.writeInput(upstream.converged, Uint32Array.of(converged));
        const actual = await fixture.submitTwice(compiled, () =>
          fixture.readRows(Object.values(status))
        );
        expect(actual).toEqual(expected);
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}

it('S2 rejects invalid-triangle completeness and recovers on the same compiled Voronoi graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOutputOracleFixture(device);
  const {graph, input, output} = fixture;
  const triangleCount = input('requested', 'uint32', Uint32Array.of(2));
  const status = {
    count: output('count', 'uint32', 1),
    requiredCount: output('required', 'uint32', 1),
    overflow: output('overflow', 'uint32', 1),
    invalidCount: output('invalid', 'uint32', 1),
    complete: output('complete', 'uint32', 1)
  };
  graph.add(
    new GPUVoronoiDiagram({
      positions: input('positions', 'float32x2', Float32Array.of(0, 0, 2, 0, 0, 2)),
      triangles: input('triangles', 'uint32x3', Uint32Array.of(0, 1, 2, 99, 100, 101)),
      triangleCount,
      clipBounds: input('bounds', 'float32x4', Float32Array.of(-2, -2, 3, 3)),
      output: {
        segments: output('segments', 'float32x4', 6),
        siteIds: output('sites', 'uint32x2', 6),
        status
      }
    })
  );
  const compiled = graph.compile();
  try {
    for (const requested of [2, 1, 2, 1]) {
      fixture.writeInput(triangleCount, Uint32Array.of(requested));
      const actual = await fixture.submitTwice(compiled, () =>
        fixture.readRows(Object.values(status))
      );
      // Three hull rays from the valid triangle; the invalid triple is disjoint and rejected.
      expect(actual).toEqual([3, 3, 0, requested === 2 ? 1 : 0, requested === 2 ? 0 : 1]);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
