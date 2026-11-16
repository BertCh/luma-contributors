// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createAZPObjectiveReductionNode} from '../../../src/gpu-spatial-analysis/spatial-regionalization/azp-objective-reduction';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {requireTestDevice} from '../../utils/require-test-device';

for (const rows of [2, 3, 5, 257, 4097]) {
  it(`sums ${rows} objective terms with exact level bindings and a fixed pair tree`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    try {
      // Integer terms have an exactly representable total, independent of the summation tree.
      const terms = Float32Array.from({length: rows}, (_, index) => index % 17);
      let source = rig.input(terms, 'float32');
      let destination = createTransientView(rig.graph, 'objective-scratch', 'float32', rows);
      const result = rig.output('float32', 1);
      rig.run({
        getCommandNodes: graph => {
          const nodes = [];
          let remaining = rows;
          let level = 0;
          while (remaining > 1) {
            nodes.push(
              createAZPObjectiveReductionNode(graph, {
                id: `objective-${level++}`,
                input: source,
                output: destination,
                inputCount: remaining
              })
            );
            remaining = Math.ceil(remaining / 2);
            [source, destination] = [destination, source];
          }
          nodes.push(
            createWGSLKernelNode(graph, {
              id: 'publish-objective',
              operation: 'AZPObjectiveTest',
              invocationCount: 1,
              bindings: [
                {name: 'objective', view: source, type: 'f32', access: 'read'},
                {name: 'result', view: result, type: 'f32', access: 'read_write'}
              ],
              body: 'result[resultOffset] = objective[objectiveOffset];'
            })
          );
          return nodes;
        }
      });
      expect((await rig.readFloat(result))[0]).toBe(terms.reduce((sum, value) => sum + value, 0));
    } finally {
      rig.destroy();
    }
  });
}
