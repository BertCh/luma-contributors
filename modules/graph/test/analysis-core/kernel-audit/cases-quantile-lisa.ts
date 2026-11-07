// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUQuantileLISA} from '../../../src/gpu-spatial-analysis/recipes/quantile-lisa-recipe';
import type {KernelAuditCase} from './kernel-audit-context';

/** Exercise all optional views, the mask and the largest inference neighbor capacity. */
export const QUANTILE_LISA_AUDIT_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUQuantileLISA',
    name: 'masked selected quantile with exact conditional inference',
    build: ({graph, view}) => {
      graph.add(
        new GPUQuantileLISA({
          values: view('float32', 300, 'values'),
          mask: view('uint32', 300, 'mask'),
          quantileParameters: view('uint32', 2, 'quantile-parameters'),
          parameters: view('uint32', 4, 'permutation-parameters'),
          weights: {
            offsets: view('uint32', 301, 'offsets'),
            neighbors: view('uint32', 1800, 'neighbors'),
            weights: view('float32', 1800, 'weights'),
            status: {overflow: view('uint32', 1, 'weights-overflow')}
          },
          maximumPermutations: 999,
          maximumNeighbors: 64,
          output: {
            membership: view('uint32', 300, 'membership'),
            boundaries: view('float32', 2, 'boundaries'),
            validCount: view('uint32', 1, 'valid-count'),
            invalidQuantile: view('uint32', 1, 'invalid-quantile'),
            classes: view('uint32', 300, 'classes'),
            observed: view('float32', 300, 'observed'),
            significant: view('uint32', 300, 'significant'),
            exceedances: view('uint32', 300, 'exceedances'),
            pseudoPValues: view('float32', 300, 'pseudo-p-values'),
            islandCount: view('uint32', 1, 'island-count'),
            status: {overflow: view('uint32', 1, 'overflow')}
          }
        })
      );
    }
  }
];
