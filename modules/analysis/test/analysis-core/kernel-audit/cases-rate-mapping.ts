// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  addExcessRiskRecipe,
  addSpatialRateRecipe
} from '../../../src/gpu-spatial-analysis/recipes/rate-mapping-recipes';
import type {KernelAuditCase} from './kernel-audit-context';

/** Inspect the maximal masked uint32 compositions on core WebGPU limits. */
export const RATE_MAPPING_KERNEL_AUDIT_CASES: KernelAuditCase[] = [
  {
    exportName: 'addExcessRiskRecipe',
    name: 'masked uint32 counts and pooled expected counts',
    build: ({graph, view}) => {
      addExcessRiskRecipe(graph, {
        events: view('uint32', 300, 'events'),
        populations: view('uint32', 300, 'populations'),
        mask: view('uint32', 300, 'mask')
      });
    }
  },
  {
    exportName: 'addSpatialRateRecipe',
    name: 'masked uint32 spatial rates with island exclusion and scale',
    build: ({graph, view}) => {
      addSpatialRateRecipe(graph, {
        events: view('uint32', 300, 'events'),
        populations: view('uint32', 300, 'populations'),
        mask: view('uint32', 300, 'mask'),
        scale: 100000,
        weights: {
          offsets: view('uint32', 301, 'offsets'),
          neighbors: view('uint32', 1800, 'neighbors'),
          weights: view('float32', 1800, 'weights')
        }
      });
    }
  }
];
