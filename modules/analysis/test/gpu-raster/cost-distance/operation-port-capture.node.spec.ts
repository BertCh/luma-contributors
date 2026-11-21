// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCostDistancePathOperation} from '../../../src/raster-analysis/cost-distance/gpu-cost-distance-path-operation';
import {CPUCostDistancePathOperation} from '../../../src/raster-analysis/cost-distance/cpu-cost-distance-path-operation';
import {GPUCostDistancePath} from '../../../src/raster-analysis/cost-distance/gpu-cost-distance-path';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid';
import {createKernelAuditContext} from '../../analysis-core/kernel-audit/kernel-audit-context';

for (const Operation of [CPUCostDistancePathOperation, GPUCostDistancePathOperation]) {
  it(`R4 ${Operation.name} captures inherited path and status fields once`, async () => {
    let destination = 'required';
    let reads = 0;
    const status = new (class {
      get count() {
        return 'count';
      }
      get overflow() {
        return 'overflow';
      }
      get requiredCount() {
        reads++;
        return destination;
      }
      get candidateOverflow() {
        return 'candidate';
      }
    })();
    const path = new (class {
      get ids() {
        return 'ids';
      }
      get status() {
        return status;
      }
    })();
    const operation = new Operation({
      grid: Object.create(makeGPURasterMetadata({width: 3, height: 1})),
      backLinks: 'links',
      target: 'target',
      capacity: 2,
      output: {path}
    });
    destination = 'changed';
    expect(reads).toBe(1);
    const declared = operation.contract.outputs;
    if (!declared || typeof declared === 'function') throw new Error('static outputs required');
    expect(Object.keys(declared).sort()).toEqual([
      'candidate',
      'count',
      'ids',
      'overflow',
      'required'
    ]);
    if (operation instanceof CPUCostDistancePathOperation) {
      const rows = (values: Uint32Array) => ({
        format: 'uint32' as const,
        length: values.length,
        chunks: [{values, length: values.length}],
        pack: () => values
      });
      const result = await operation.execute({
        inputs: {
          backLinks: rows(Uint32Array.of(0, 16, 16)),
          target: rows(Uint32Array.of(2))
        }
      });
      // Hand walk 2 -> 1 -> 0: two stored IDs, full length three, no candidate loss.
      expect(result['ids']).toEqual(Uint32Array.of(2, 1));
      expect(result['count']).toEqual(Uint32Array.of(2));
      expect(result['overflow']).toEqual(Uint32Array.of(1));
      expect(result['required']).toEqual(Uint32Array.of(3));
      expect(result['candidate']).toEqual(Uint32Array.of(0));
    } else {
      const context = createKernelAuditContext('R4-capture');
      try {
        const views = Object.fromEntries(
          Object.entries(operation.outputs).map(([port, descriptor]) => [
            port,
            context.view(descriptor.format, descriptor.length)
          ])
        );
        views['backLinks'] = context.view('uint32', 3);
        views['target'] = context.view('uint32', 1);
        const contributor = operation.createContributor(
          port => views[port] as never
        ) as GPUCostDistancePath;
        expect(contributor.props.output.path.ids).toBe(views['ids']);
        expect(contributor.props.output.path.status.requiredCount).toBe(views['requiredCount']);
        expect(contributor.props.output.path.status.candidateOverflow).toBe(
          views['candidateOverflow']
        );
        expect(() => contributor.getCommandNodes(context.graph)).not.toThrow();
      } finally {
        context.device.destroy();
        operation.destroy();
      }
    }
    expect(reads).toBe(1);
  });

  it(`R4 ${Operation.name} rejects colliding inherited status destinations`, () => {
    const status = new (class {
      get count() {
        return 'count';
      }
      get overflow() {
        return 'overflow';
      }
      get candidateOverflow() {
        return 'overflow';
      }
    })();
    expect(
      () =>
        new Operation({
          grid: makeGPURasterMetadata({width: 3, height: 1}),
          backLinks: 'links',
          target: 'target',
          capacity: 2,
          output: {path: {ids: 'ids', status}}
        })
    ).toThrow(/no other port uses/);
  });
}
