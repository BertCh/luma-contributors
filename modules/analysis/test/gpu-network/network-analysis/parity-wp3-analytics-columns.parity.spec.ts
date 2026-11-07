// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPUNetworkAnalyticsColumns} from '../../../src/gpu-network/network-analysis/gpu-network-analytics-columns';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {compareParity, EXACT} from '../parity-wp3-comparison';
import {createParityResources} from '../parity-wp3-round2-resources';
import {analysisFixtures} from './parity-wp3-network-analysis-fixture';

for (const fixture of analysisFixtures) {
  it(`external analytics columns ${fixture.input.case}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const resources = createParityResources(device);
    const {input, reverse} = fixture;
    const topology = {
      offsets: resources.input('offsets', 'uint32', input.offsets),
      neighbors: resources.input('neighbors', 'uint32', input.neighbors),
      // These nonunit/zero weights must have no effect on the unweighted API.
      weights: resources.input(
        'weights',
        'float32',
        input.neighbors.map((_, index) => index % 7)
      )
    };
    function column<Format extends 'uint32' | 'float32'>(name: string, format: Format) {
      return {
        output: resources.output(name, format, input.nodeCount),
        normalized: resources.output(`${name}-normalized`, 'float32', input.nodeCount),
        extent: resources.output(`${name}-extent`, format, 2)
      };
    }
    resources.graph.add(
      new GPUNetworkAnalyticsColumns({
        topology,
        reverseTopology: input.directed
          ? {
              offsets: resources.input('reverse-offsets', 'uint32', reverse.offsets),
              neighbors: resources.input('reverse-neighbors', 'uint32', reverse.neighbors)
            }
          : undefined,
        degree: column('degree', 'uint32'),
        inDegree: input.directed ? column('inDegree', 'uint32') : undefined,
        components: {
          output: resources.output('components', 'uint32', input.nodeCount),
          maximumIterations: 512,
          status: {converged: resources.output('components-converged', 'uint32', 1)}
        }
      })
    );
    const compiled = resources.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      for (const metric of fixture.metrics) {
        const name = metric.metric;
        if (name === 'inDegree' && !input.directed) continue;
        const actual = await resources.readUint(name);
        compareParity(
          'GPUNetworkAnalyticsColumns',
          metric,
          `${name}: every original vertex row`,
          actual,
          metric.values,
          EXACT
        );
        if (name === 'components') continue;
        const extent = await resources.readUint(`${name}-extent`);
        compareParity(
          'GPUNetworkAnalyticsColumns',
          metric,
          `${name}: min/max`,
          extent,
          metric.extent,
          EXACT
        );
        // S profile at normalized output scale one. Constant columns are exactly zero.
        compareParity(
          'GPUNetworkAnalyticsColumns',
          metric,
          `${name}: every normalized row`,
          await resources.readFloat(`${name}-normalized`),
          metric.normalized,
          metric.extent[0] === metric.extent[1] ? EXACT : {absolute: 1e-6, relative: 1e-4}
        );
      }
      compareParity(
        'GPUNetworkAnalyticsColumns',
        input,
        'component convergence',
        await resources.readUint('components-converged'),
        [1],
        EXACT
      );
    } finally {
      compiled.destroy();
      resources.destroy();
    }
  }, 180000);
}
