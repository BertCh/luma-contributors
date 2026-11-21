// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUTrajectoryLocations} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-locations';
import {computeTrajectoryLocationsOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-locations-operation';
import {createKernelAuditContext} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {GPUTrajectoryLocationsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-locations-operation';
import {CPUTrajectoryLocationsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-locations-operation';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

it('locations range discovery and relabeling dispatch parallel work over the capacity', () => {
  const context = createKernelAuditContext('locations-parallel-contract');
  try {
    const contributor = new GPUTrajectoryLocations({
      maximumIterations: 1,
      centroids: context.view('float32x2', 5),
      tracks: context.view('uint32', 5),
      count: context.view('uint32', 1),
      parameters: context.view('float32', 4),
      output: {
        locationIds: context.view('uint32', 5),
        centers: context.view('float32x2', 5),
        status: {count: context.view('uint32', 1), iterationLimitReached: context.view('uint32', 1)}
      }
    });
    recordWGSLKernelNodes(context.graph);
    contributor.getCommandNodes(context.graph);
    expect(
      getWGSLKernelNodeRecord(context.graph, 'trajectory-locations-groups')?.invocationCount
    ).toBe(5);
    expect(
      getWGSLKernelNodeRecord(context.graph, 'trajectory-locations-relabel')?.invocationCount
    ).toBe(5);
  } finally {
    context.device.destroy();
  }
});
function makeCPUInput(
  format: 'float32x2' | 'float32' | 'uint32',
  values: Float32Array | Uint32Array
) {
  const length = values.length / (format === 'float32x2' ? 2 : 1);
  return {format, length, chunks: [{values, length}], pack: () => values};
}

it('parallel locations preserve converged first-visit order with inactive tails', () => {
  // Two rounds settle {0,1,2,3,4} before {100} in first-visit order.
  for (const active of [0, 3, 6]) {
    const result = computeTrajectoryLocationsOnCPU(
      {
        centroids: Float32Array.of(0, 0, 1, 0, 2, 0, 100, 0, 3, 0, 4, 0, 900, 0, 901, 0),
        tracks: new Uint32Array(8),
        count: Uint32Array.of(active),
        parameters: Float32Array.of(1, 0, 0, 0)
      },
      {maximumIterations: 2}
    );
    const expectedIds = active === 0 ? [] : active === 3 ? [0, 0, 0] : [0, 0, 0, 1, 0, 0];
    expect(Array.from(result.locationIds)).toEqual([
      ...expectedIds,
      ...new Array(8 - active).fill(0xffffffff)
    ]);
    expect(Array.from(result.locationCount)).toEqual([active === 0 ? 0 : active === 3 ? 1 : 2]);
    expect(Array.from(result.centers).slice(0, active === 6 ? 4 : active === 3 ? 2 : 0)).toEqual(
      active === 0 ? [] : active === 3 ? [1, 0] : [2, 0, 100, 0]
    );
  }
});

it('R4 CPU operation executes the requested two rounds and defaults to 32 when counted', async () => {
  for (const maximumIterations of [2, undefined]) {
    const operation = new CPUTrajectoryLocationsOperation({
      centroids: 'points',
      tracks: 'tracks',
      count: 'active',
      parameters: 'parameters',
      capacity: 1,
      maximumIterations,
      output: {
        locationIds: 'ids',
        status: {
          count: 'count',
          iterationLimitReached: 'iterationLimitReached',
          iterationCount: 'rounds'
        }
      }
    });
    const result = await operation.execute({
      inputs: {
        centroids: makeCPUInput('float32x2', Float32Array.of(7, 9)),
        tracks: makeCPUInput('uint32', Uint32Array.of(0)),
        count: makeCPUInput('uint32', Uint32Array.of(1)),
        parameters: makeCPUInput('float32', Float32Array.of(1, 0, 0, 0))
      }
    });
    expect(Array.from(result['rounds'] as Uint32Array)).toEqual([maximumIterations ?? 32]);
  }
});

it('R6 graph factory retains nested status names and validated settings after caller mutation', () => {
  const context = createKernelAuditContext('locations-operation-snapshot');
  try {
    const props = {
      centroids: 'points',
      tracks: 'tracks',
      count: 'active',
      parameters: 'parameters',
      capacity: 1,
      maximumIterations: 2,
      output: {
        locationIds: 'ids',
        status: {
          count: 'count',
          iterationLimitReached: 'limit',
          iterationCount: 'rounds' as string | undefined
        }
      }
    };
    const operation = new GPUTrajectoryLocationsOperation(props);
    props.maximumIterations = 1.5;
    props.output.status.iterationCount = undefined;
    const views = {
      centroids: context.view('float32x2', 1),
      tracks: context.view('uint32', 1),
      count: context.view('uint32', 1),
      parameters: context.view('float32', 4),
      locationIds: context.view('uint32', 1),
      locationCount: context.view('uint32', 1),
      iterationCount: context.view('uint32', 1),
      iterationLimitReached: context.view('uint32', 1)
    };
    const contributor = operation.createContributor(
      port => views[port as keyof typeof views] as never
    ) as GPUTrajectoryLocations;
    expect(contributor.maximumIterations).toBe(2);
    expect(
      contributor
        .getCommandNodes(context.graph)
        .flatMap(node => node.resources ?? [])
        .filter(
          resource =>
            'buffer' in resource &&
            resource.buffer === views.iterationCount &&
            resource.usage === 'storage-read-write'
        )
    ).toHaveLength(1);
  } finally {
    context.device.destroy();
  }
});

it('R1 CPU graph operation preserves the two-track converged centers and IDs', async () => {
  const operation = new CPUTrajectoryLocationsOperation({
    centroids: 'points',
    tracks: 'tracks',
    count: 'active',
    parameters: 'parameters',
    capacity: 5,
    maximumIterations: 2,
    output: {
      locationIds: 'ids',
      centers: 'centers',
      status: {count: 'count', iterationLimitReached: 'limit'}
    }
  });
  const result = await operation.execute({
    inputs: {
      centroids: makeCPUInput('float32x2', Float32Array.of(50, 0, 0, 0, 1, 0, 2, 0, 4, 0)),
      tracks: makeCPUInput('uint32', Uint32Array.of(0, 1, 1, 1, 1)),
      count: makeCPUInput('uint32', Uint32Array.of(5)),
      parameters: makeCPUInput('float32', Float32Array.of(1, 0, 0, 0))
    }
  });
  expect(Array.from(result['ids'] as Uint32Array)).toEqual([0, 1, 1, 1, 2]);
  expect(Array.from(result['centers'] as Float32Array).slice(0, 6)).toEqual([50, 0, 1, 0, 4, 0]);
  expect(Array.from(result['count'] as Uint32Array)).toEqual([3]);
  expect(Array.from(result['limit'] as Uint32Array)).toEqual([0]);
});

it('R5 bounded singleton work never scans unrelated track rows', () => {
  const total = 64;
  let trackReads = 0;
  const tracks = new Proxy(
    Uint32Array.from({length: total}, (_, row) => row),
    {
      get(target, key) {
        if (typeof key === 'string' && /^\d+$/.test(key)) trackReads++;
        return Reflect.get(target, key, target);
      }
    }
  );
  const result = computeTrajectoryLocationsOnCPU(
    {
      centroids: new Float32Array(total * 2),
      tracks,
      count: Uint32Array.of(total),
      parameters: Float32Array.of(1, 0, 0, 0)
    },
    {maximumIterations: 4}
  );
  // Singleton ranges have no cross-row neighbors. Linear group discovery plus four
  // within-range rounds must stay comfortably below this linear bound, independent of time.
  expect(trackReads).toBeLessThanOrEqual(total * 40);
  expect(Array.from(result.locationIds)).toEqual(Array.from({length: total}, (_, row) => row));
});

it('R4 both operation descriptors retain a requested iterationCount output', () => {
  const props = {
    centroids: 'points',
    tracks: 'tracks',
    count: 'active',
    parameters: 'parameters',
    capacity: 1,
    maximumIterations: 2,
    output: {
      locationIds: 'ids',
      status: {count: 'count', iterationLimitReached: 'limit', iterationCount: 'rounds'}
    }
  };
  for (const Operation of [GPUTrajectoryLocationsOperation, CPUTrajectoryLocationsOperation])
    expect(Object.keys(new Operation(props).contract.outputs ?? {})).toContain('rounds');
});

for (const [format, length] of [
  ['uint32', 0],
  ['float32', 1]
] as const)
  it(`R3 rejects iterationCount destination ${format} length ${length}`, () => {
    const context = createKernelAuditContext('locations-invalid-status');
    try {
      expect(
        () =>
          new GPUTrajectoryLocations({
            centroids: context.view('float32x2', 1),
            tracks: context.view('uint32', 1),
            count: context.view('uint32', 1),
            parameters: context.view('float32', 4),
            output: {
              locationIds: context.view('uint32', 1),
              status: {
                count: context.view('uint32', 1),
                iterationLimitReached: context.view('uint32', 1),
                iterationCount: context.view(format, length) as never
              }
            }
          })
      ).toThrow(/iterationCount.*uint32/);
    } finally {
      context.device.destroy();
    }
  });

for (const invalidMaximumIterations of [-1, 1.5, null])
  it(`R2 rejects invalid iteration budget ${invalidMaximumIterations} in every form`, () => {
    // Exercise JavaScript runtime input outside the TypeScript prop contract as well.
    const maximumIterations = invalidMaximumIterations as number;
    const context = createKernelAuditContext('locations-invalid-budget');
    try {
      expect(
        () =>
          new GPUTrajectoryLocations({
            centroids: context.view('float32x2', 1),
            tracks: context.view('uint32', 1),
            count: context.view('uint32', 1),
            parameters: context.view('float32', 4),
            maximumIterations,
            output: {
              locationIds: context.view('uint32', 1),
              status: {
                count: context.view('uint32', 1),
                iterationLimitReached: context.view('uint32', 1),
                iterationCount: context.view('uint32', 1)
              }
            }
          })
      ).toThrow(/maximumIterations/);
      expect(() =>
        computeTrajectoryLocationsOnCPU(
          {
            centroids: Float32Array.of(7, 9),
            tracks: Uint32Array.of(0),
            count: Uint32Array.of(1),
            parameters: Float32Array.of(1, 0, 0, 0)
          },
          {maximumIterations}
        )
      ).toThrow(/maximumIterations/);
      for (const Operation of [GPUTrajectoryLocationsOperation, CPUTrajectoryLocationsOperation])
        expect(
          () =>
            new Operation({
              centroids: 'points',
              tracks: 'tracks',
              count: 'active',
              parameters: 'parameters',
              capacity: 1,
              maximumIterations,
              output: {locationIds: 'ids', status: {count: 'count', iterationLimitReached: 'limit'}}
            })
        ).toThrow(/maximumIterations/);
    } finally {
      context.device.destroy();
    }
  });

it('R9 pure CPU captures a changing budget getter once before validation and execution', () => {
  let budgetReads = 0;
  const options = {
    get maximumIterations() {
      budgetReads++;
      return budgetReads <= 2 ? 1 : 1.5;
    }
  };
  const result = computeTrajectoryLocationsOnCPU(
    {
      centroids: Float32Array.of(50, 0, 0, 0, 1, 0, 2, 0, 3, 0),
      tracks: Uint32Array.of(0, 1, 1, 1, 1),
      count: Uint32Array.of(5),
      parameters: Float32Array.of(1, 0, 0, 0)
    },
    options
  );
  // One round is unsettled, so no provisional identities or means are readable.
  expect({
    budgetReads,
    ids: Array.from(result.locationIds),
    centers: Array.from(result.centers).slice(0, 6),
    iterationCount: Array.from(result.iterationCount),
    limit: Array.from(result.iterationLimitReached)
  }).toEqual({
    budgetReads: 1,
    ids: new Array(5).fill(0xffffffff),
    centers: new Array(6).fill(0),
    iterationCount: [1],
    limit: [1]
  });
});

it('R9 contributor and graph forms capture budget getters once before lowering or execution', async () => {
  const context = createKernelAuditContext('locations-getter-forms');
  try {
    const views = {
      centroids: context.view('float32x2', 5),
      tracks: context.view('uint32', 5),
      count: context.view('uint32', 1),
      parameters: context.view('float32', 4),
      locationIds: context.view('uint32', 5),
      locationCount: context.view('uint32', 1),
      iterationCount: context.view('uint32', 1),
      iterationLimitReached: context.view('uint32', 1)
    };
    let directReads = 0;
    const direct = new GPUTrajectoryLocations({
      centroids: views.centroids,
      tracks: views.tracks,
      count: views.count,
      parameters: views.parameters,
      get maximumIterations() {
        directReads++;
        return directReads === 1 ? 1 : 1.5;
      },
      output: {
        locationIds: views.locationIds,
        status: {
          count: views.locationCount,
          iterationLimitReached: views.iterationLimitReached,
          iterationCount: views.iterationCount
        }
      }
    });
    direct.getCommandNodes(context.graph);
    expect(directReads).toBe(1);
    expect(direct.maximumIterations).toBe(1);
    for (const Operation of [GPUTrajectoryLocationsOperation, CPUTrajectoryLocationsOperation]) {
      let reads = 0;
      const operation = new Operation({
        centroids: 'points',
        tracks: 'tracks',
        count: 'active',
        parameters: 'parameters',
        capacity: 5,
        get maximumIterations() {
          reads++;
          return reads === 1 ? 1 : 1.5;
        },
        output: {
          locationIds: 'ids',
          status: {count: 'count', iterationLimitReached: 'limit', iterationCount: 'rounds'}
        }
      });
      if (operation instanceof GPUTrajectoryLocationsOperation) {
        const contributor = operation.createContributor(
          port => views[port as keyof typeof views] as never
        ) as GPUTrajectoryLocations;
        expect(contributor.maximumIterations).toBe(1);
      } else {
        const result = await operation.execute({
          inputs: {
            centroids: makeCPUInput('float32x2', Float32Array.of(50, 0, 0, 0, 1, 0, 2, 0, 3, 0)),
            tracks: makeCPUInput('uint32', Uint32Array.of(0, 1, 1, 1, 1)),
            count: makeCPUInput('uint32', Uint32Array.of(5)),
            parameters: makeCPUInput('float32', Float32Array.of(1, 0, 0, 0))
          }
        });
        expect(Array.from(result['ids'] as Uint32Array)).toEqual(new Array(5).fill(0xffffffff));
        expect(Array.from(result['rounds'] as Uint32Array)).toEqual([1]);
        expect(result['limit']).toEqual(Uint32Array.of(1));
      }
      expect(reads).toBe(1);
    }
  } finally {
    context.device.destroy();
  }
});

it('R1 compacts every converged label without borrowing another track identity', () => {
  // Two rounds settle the components. First visits are x=50, then {0,1,2}, then x=4.
  const result = computeTrajectoryLocationsOnCPU(
    {
      centroids: Float32Array.of(50, 0, 0, 0, 1, 0, 2, 0, 4, 0),
      tracks: Uint32Array.of(0, 1, 1, 1, 1),
      count: Uint32Array.of(5),
      parameters: Float32Array.of(1, 0, 0, 0)
    },
    {maximumIterations: 2}
  );
  expect(Array.from(result.locationIds)).toEqual([0, 1, 1, 1, 2]);
  expect(Array.from(result.centers).slice(0, 6)).toEqual([50, 0, 1, 0, 4, 0]);
  expect(Array.from(result.locationCount)).toEqual([3]);
  expect(Array.from(result.iterationLimitReached)).toEqual([0]);
});

it('locations declares a writer for an iterationCount supplied without other optional outputs', () => {
  const context = createKernelAuditContext('locations-iteration-oracle');
  try {
    const iterationCount = context.view('uint32', 1);
    const props = {
      centroids: context.view('float32x2', 1),
      tracks: context.view('uint32', 1),
      count: context.view('uint32', 1),
      parameters: context.view('float32', 4),
      maximumIterations: 2,
      output: {
        locationIds: context.view('uint32', 1),
        status: {
          count: context.view('uint32', 1),
          iterationLimitReached: context.view('uint32', 1),
          iterationCount
        }
      }
    };
    const contributor = new GPUTrajectoryLocations(props);
    const replacement = context.view('uint32', 1);
    props.output.status.iterationCount = replacement;
    const nodes = contributor.getCommandNodes(context.graph);
    const writers = nodes
      .flatMap(node => node.resources ?? [])
      .filter(
        resource =>
          'buffer' in resource &&
          resource.buffer === iterationCount &&
          resource.usage === 'storage-read-write'
      );
    expect(writers).toHaveLength(1);
    const redirected = nodes
      .flatMap(node => node.resources ?? [])
      .filter(
        resource =>
          'buffer' in resource &&
          resource.buffer === replacement &&
          resource.usage === 'storage-read-write'
      );
    expect(redirected).toHaveLength(0);
  } finally {
    context.device.destroy();
  }
});

it('locations distinct component centers and convergence have hand-derived literals', () => {
  // Components {0,2,4,0}, {10}, {0,2} on a separate track: duplicate 0 votes once.
  const result = computeTrajectoryLocationsOnCPU({
    centroids: Float32Array.of(0, 0, 2, 0, 4, 0, 0, 0, 10, 0, 0, 0, 2, 0),
    tracks: Uint32Array.of(0, 0, 0, 0, 0, 1, 1),
    count: Uint32Array.of(7),
    parameters: Float32Array.of(2, 0, 0, 0)
  });
  expect(Array.from(result.centers).slice(0, 6)).toEqual([2, 0, 10, 0, 1, 0]);
  expect(Array.from(result.converged)).toEqual([1]);
  expect(Array.from(result.iterationLimitReached)).toEqual([0]);
});

it('locations CPU honors an explicit one-round budget on a five-point chain', () => {
  // Hook labels [0,0,1,2,3], then one compression [0,0,0,1,2]. Another hook
  // can still lower rows 3,4, so this round is incomplete. Root 0 contains 0,1,2.
  const result = computeTrajectoryLocationsOnCPU(
    {
      centroids: Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0, 4, 0),
      tracks: new Uint32Array(5),
      count: Uint32Array.of(5),
      parameters: Float32Array.of(1, 0, 0, 0)
    },
    {maximumIterations: 1}
  );
  expect(Array.from(result.converged)).toEqual([0]);
  expect(Array.from(result.iterationLimitReached)).toEqual([1]);
  expect(Array.from(result.centers).slice(0, 2)).toEqual([0, 0]);
  const complete = computeTrajectoryLocationsOnCPU(
    {
      centroids: Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0, 4, 0),
      tracks: new Uint32Array(5),
      count: Uint32Array.of(5),
      parameters: Float32Array.of(1, 0, 0, 0)
    },
    {maximumIterations: 2}
  );
  expect(Array.from(complete.converged)).toEqual([1]);
  expect(Array.from(complete.iterationLimitReached)).toEqual([0]);
  expect(Array.from(complete.centers).slice(0, 2)).toEqual([2, 0]);
});
it('review keeps unsettled location tables suppressed rather than provisional', () => {
  const result = computeTrajectoryLocationsOnCPU(
    {
      centroids: Float32Array.of(50, 0, 0, 0, 1, 0, 2, 0, 3, 0),
      tracks: Uint32Array.of(0, 1, 1, 1, 1),
      count: Uint32Array.of(5),
      parameters: Float32Array.of(1, 0, 0, 0)
    },
    {maximumIterations: 1}
  );
  expect(result.locationIds).toEqual(new Uint32Array(5).fill(0xffffffff));
  expect(result.locationCount).toEqual(Uint32Array.of(0));
  expect(result.iterationLimitReached).toEqual(Uint32Array.of(1));
});
