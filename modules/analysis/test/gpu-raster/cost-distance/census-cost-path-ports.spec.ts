// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {Buffer} from '@luma.gl/core';
import {it} from 'vitest';
import {
  GPUCostDistancePath,
  type GPUCostDistancePathProps
} from '../../../src/raster-analysis/cost-distance/gpu-cost-distance-path';
import {GPUCostDistancePathOperation} from '../../../src/raster-analysis/cost-distance/gpu-cost-distance-path-operation';
import {CPUCostDistancePathOperation} from '../../../src/raster-analysis/cost-distance/cpu-cost-distance-path-operation';
import {GPU_COST_DISTANCE_UNRESOLVED} from '../../../src/raster-analysis/cost-distance/gpu-cost-distance';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {createCensusRasterHarness, expectCensusValues} from '../census-raster-oracle-utils';

it('census truncated path clears candidate overflow with requiredCount absent', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createCensusRasterHarness(device);
  const result = await harness.run(
    new GPUCostDistancePath({
      grid: makeGPURasterMetadata({width: 3, height: 1}),
      backLinks: harness.input('uint32', [0, 16, 16]),
      target: harness.input('uint32', [2]),
      output: {
        path: {
          ids: harness.output('ids', 'uint32', 2),
          status: {
            count: harness.output('count', 'uint32', 1),
            overflow: harness.output('overflow', 'uint32', 1),
            candidateOverflow: harness.output('candidateOverflow', 'uint32', 1)
          }
        }
      }
    })
  );
  // The westward walk is 2 -> 1 -> 0. Capacity two truncates its prefix, but there is
  // no candidate search to truncate. Poisoned scalar outputs distinguish overflow meanings.
  expectCensusValues(result['ids'], [2, 1]);
  expectCensusValues(result['count'], [2]);
  expectCensusValues(result['overflow'], [1]);
  expectCensusValues(result['candidateOverflow'], [0]);
});

it('census path retains original candidate output after caller replacement', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createCensusRasterHarness(device);
  const ids = harness.output('ids', 'uint32', 2);
  const props: GPUCostDistancePathProps = {
    grid: makeGPURasterMetadata({width: 3, height: 1}),
    backLinks: harness.input('uint32', [0, 16, 16]),
    target: harness.input('uint32', [2]),
    output: {
      path: {
        ids,
        status: {
          count: harness.output('count', 'uint32', 1),
          overflow: harness.output('overflow', 'uint32', 1),
          candidateOverflow: harness.output('candidateOverflow', 'uint32', 1)
        }
      }
    }
  };
  const contributor = new GPUCostDistancePath(props);
  props.output.path.status.candidateOverflow = harness.graph.createDataView(ids.buffer, {
    format: 'uint32',
    length: 1
  });
  const result = await harness.run(contributor);
  // A captured destination must survive replacement with ids[0]. Otherwise the publisher's
  // zero clears target ID 2 and leaves the original candidate poison (12345) untouched.
  expectCensusValues(result['ids'], [2, 1]);
  expectCensusValues(result['count'], [2]);
  expectCensusValues(result['overflow'], [1]);
  expectCensusValues(result['candidateOverflow'], [0]);
});

it('census path preserves back-links sharing disjoint storage with candidate output', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createCensusRasterHarness(device);
  const words = new Uint32Array(128).fill(12345);
  words.set([0, 16, 16]);
  const buffer = device.createBuffer({
    data: words,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  try {
    const storage = importGraphBuffer(
      harness.graph,
      'shared-links-candidate',
      buffer,
      'uint32',
      128
    );
    const backLinks = harness.graph.createDataView(storage.buffer, {format: 'uint32', length: 3});
    const candidateOverflow = harness.graph.createDataView(storage.buffer, {
      format: 'uint32',
      byteOffset: 256,
      length: 1
    });
    const result = await harness.run(
      new GPUCostDistancePath({
        grid: makeGPURasterMetadata({width: 3, height: 1}),
        backLinks,
        target: harness.input('uint32', [2]),
        output: {
          path: {
            ids: harness.output('ids', 'uint32', 2),
            status: {
              count: harness.output('count', 'uint32', 1),
              overflow: harness.output('overflow', 'uint32', 1),
              candidateOverflow
            }
          }
        }
      })
    );
    const shared = await readUint32(buffer, 65);
    // The walk reads [0,12); only its later publisher writes [256,260). The path is the
    // capacity-two prefix of 2 -> 1 -> 0, and the original predecessor codes must survive.
    expectCensusValues(shared.slice(0, 3), [0, 16, 16]);
    expectCensusValues(shared.slice(64, 65), [0]);
    expectCensusValues(result['ids'], [2, 1]);
    expectCensusValues(result['count'], [2]);
    expectCensusValues(result['overflow'], [1]);
  } finally {
    buffer.destroy();
  }
});

for (const Operation of [GPUCostDistancePathOperation, CPUCostDistancePathOperation]) {
  it(`census ${Operation.name} publishes candidate zero with requiredCount absent at capacity two`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const result = await runOperationGraph(device, {
      inputs: {
        backLinks: {format: 'uint32', values: Uint32Array.of(0, 16, 16)},
        target: {format: 'uint32', values: Uint32Array.of(2)}
      },
      operations: [
        new Operation({
          grid: makeGPURasterMetadata({width: 3, height: 1}),
          backLinks: 'backLinks',
          target: 'target',
          capacity: 2,
          output: {
            path: {
              ids: 'ids',
              status: {count: 'count', overflow: 'overflow', candidateOverflow: 'candidate'}
            }
          }
        })
      ],
      outputs: ['ids', 'count', 'overflow', 'candidate']
    });
    // Independent 2 -> 1 -> 0 walk, with capacity two and no bounded candidate search.
    expectCensusValues(result['ids'], [2, 1]);
    expectCensusValues(result['count'], [2]);
    expectCensusValues(result['overflow'], [1]);
    expectCensusValues(result['candidate'], [0]);
  });
}

it('census GPU path operation forwards a poisoned candidate view without requiredCount', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createCensusRasterHarness(device);
  const operation = new GPUCostDistancePathOperation({
    grid: makeGPURasterMetadata({width: 3, height: 1}),
    backLinks: 'links',
    target: 'target',
    capacity: 2,
    output: {
      path: {
        ids: 'ids',
        status: {count: 'count', overflow: 'overflow', candidateOverflow: 'candidate'}
      }
    }
  });
  const views = {
    backLinks: harness.input('uint32', [0, 16, 16]),
    target: harness.input('uint32', [2]),
    ids: harness.output('ids', 'uint32', 2),
    count: harness.output('count', 'uint32', 1),
    overflow: harness.output('overflow', 'uint32', 1),
    candidateOverflow: harness.output('candidate', 'uint32', 1)
  };
  const contributor = operation.createContributor(
    port => views[port as keyof typeof views] as never
  ) as GPUCostDistancePath;
  const result = await harness.run(contributor);
  expectCensusValues(result['ids'], [2, 1]);
  expectCensusValues(result['count'], [2]);
  expectCensusValues(result['overflow'], [1]);
  expectCensusValues(result['candidate'], [0]);
});

for (const unresolved of [false, true]) {
  for (const withCandidateOverflow of [false, true]) {
    it(`census path candidate status is rewritten (unresolved=${unresolved}, present=${withCandidateOverflow})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = createCensusRasterHarness(device);
      const result = await harness.run(
        new GPUCostDistancePath({
          grid: makeGPURasterMetadata({width: 3, height: 1}),
          backLinks: harness.input('uint32', [
            0,
            16,
            unresolved ? GPU_COST_DISTANCE_UNRESOLVED : 16
          ]),
          target: harness.input('uint32', [2]),
          output: {
            path: {
              ids: harness.output('ids', 'uint32', 3),
              status: {
                count: harness.output('count', 'uint32', 1),
                overflow: harness.output('overflow', 'uint32', 1),
                ...(withCandidateOverflow
                  ? {
                      candidateOverflow: harness.output('candidateOverflow', 'uint32', 1),
                      requiredCount: harness.output('requiredCount', 'uint32', 1)
                    }
                  : {})
              }
            },
            status: {iterationLimitReached: harness.output('limit', 'uint32', 1)}
          }
        })
      );
      // Direction 16 points west, so the only complete walk is 2 -> 1 -> 0.
      // An unresolved first link suppresses the walk; neither case loses search candidates.
      if (!unresolved) expectCensusValues(result['ids'], [2, 1, 0]);
      expectCensusValues(result['count'], [unresolved ? 0 : 3]);
      if (withCandidateOverflow) expectCensusValues(result['requiredCount'], [unresolved ? 0 : 3]);
      expectCensusValues(result['overflow'], [0]);
      expectCensusValues(result['limit'], [unresolved ? 1 : 0]);
      if (withCandidateOverflow) expectCensusValues(result['candidateOverflow'], [0]);
    });
  }

  for (const withLimit of [false, true])
    for (const Operation of [GPUCostDistancePathOperation, CPUCostDistancePathOperation]) {
      it(`census ${Operation.name} has explicit path IDs and limit status (unresolved=${unresolved}, limit=${withLimit})`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const result = await runOperationGraph(device, {
          inputs: {
            backLinks: {
              format: 'uint32',
              values: Uint32Array.of(0, 16, unresolved ? GPU_COST_DISTANCE_UNRESOLVED : 16)
            },
            target: {format: 'uint32', values: Uint32Array.of(2)}
          },
          operations: [
            new Operation({
              grid: makeGPURasterMetadata({width: 3, height: 1}),
              backLinks: 'backLinks',
              target: 'target',
              capacity: 3,
              output: {
                path: {
                  ids: 'ids',
                  status: {count: 'count', requiredCount: 'requiredCount', overflow: 'overflow'}
                },
                ...(withLimit ? {status: {iterationLimitReached: 'limit'}} : {})
              }
            })
          ],
          outputs: ['ids', 'count', 'requiredCount', 'overflow', ...(withLimit ? ['limit'] : [])]
        });
        if (!unresolved) expectCensusValues(result['ids'], [2, 1, 0]);
        expectCensusValues(result['count'], [unresolved ? 0 : 3]);
        expectCensusValues(result['requiredCount'], [unresolved ? 0 : 3]);
        expectCensusValues(result['overflow'], [0]);
        if (withLimit) expectCensusValues(result['limit'], [unresolved ? 1 : 0]);
      });
    }
}

it('R4 GPU path operation writes inherited compact destinations including full required length', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const status = new (class {
    get count() {
      return 'stored';
    }
    get overflow() {
      return 'truncated';
    }
    get requiredCount() {
      return 'full-length';
    }
    get candidateOverflow() {
      return 'candidate-loss';
    }
  })();
  const path = new (class {
    get ids() {
      return 'path-ids';
    }
    get status() {
      return status;
    }
  })();
  const result = await runOperationGraph(device, {
    inputs: {
      backLinks: {format: 'uint32', values: Uint32Array.of(0, 16, 16)},
      target: {format: 'uint32', values: Uint32Array.of(2)}
    },
    operations: [
      new GPUCostDistancePathOperation({
        grid: makeGPURasterMetadata({width: 3, height: 1}),
        backLinks: 'backLinks',
        target: 'target',
        capacity: 2,
        output: {path}
      })
    ],
    outputs: ['path-ids', 'stored', 'truncated', 'full-length', 'candidate-loss']
  });
  // Hand walk 2 -> 1 -> 0: capacity two truncates the prefix, full length remains three.
  expectCensusValues(result['path-ids'], [2, 1]);
  expectCensusValues(result['stored'], [2]);
  expectCensusValues(result['truncated'], [1]);
  expectCensusValues(result['full-length'], [3]);
  expectCensusValues(result['candidate-loss'], [0]);
});
