// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationFormulaRef} from '@luma-contributors/analysis/computation-record';
import type {ViewPackData} from '../../computation-views/registry/view-registration';
import {NETWORK_CONSTRUCTION_PACK} from './network-pack-construction';
import {NETWORK_SEARCH_PACK} from './network-pack-search';
import {NETWORK_STRUCTURE_PACK} from './network-pack-structure';
import {NETWORK_OPTIMIZATION_PACK} from './network-optimization-metadata';

/** Curated formulas keyed by the ids recorded on reachability kernels. */
export const NETWORK_FORMULA_REFS: Record<string, GPUComputationFormulaRef> = {
  ...NETWORK_OPTIMIZATION_PACK.formulaRefs,
  ...NETWORK_SEARCH_PACK.formulaRefs,
  ...NETWORK_STRUCTURE_PACK.formulaRefs,
  ...NETWORK_CONSTRUCTION_PACK.formulaRefs,
  'gpu-network.reachability.relax': {
    id: 'gpu-network.reachability.relax',
    derivation: 'curated',
    tex: String.raw`\term{costs}{c_v}\leftarrow\min(\term{costs}{c_v},\operatorname{f32}(\term{source}{c_u}+\term{weight}{w_{uv}})),\quad v<\term{count}{N},\ w_{uv}\ge0,\ c_u+w_{uv}\le L;\qquad c_v^*=\min_s(c_s^0+d(s,v))\le L`,
    ascii:
      'Atomic minimum of non-negative f32 cost bits: c[v] <- min(c[v], f32(c[u] + w[u,v])). Negative/NaN weights are impassable; out-of-range neighbors and non-finite candidates are ignored. With cutoff L, candidates above L stay unreached (+Infinity); without a cutoff L = +Infinity. At convergence c[v] = min_s(c0[s] + d(s,v)), restricted to <= L, with path sums evaluated in f32. Chaotic: a visit may read costs improved earlier in the same round; rounds are not synchronous Bellman-Ford sweeps.',
    symbols: [
      {id: 'costs', role: 'output', label: 'minimum cost (f32 bits)', binding: 'wgsl:costBits'},
      {
        id: 'source',
        role: 'intermediate',
        label: 'cost read by this visit',
        binding: 'wgsl:sourceCost'
      },
      {id: 'weight', role: 'input', label: 'directed edge weight', binding: 'wgsl:weight'},
      {id: 'count', role: 'parameter', label: 'node count', binding: 'wgsl:NODE_COUNT'}
    ]
  },
  'gpu-network.reachability.bands': {
    id: 'gpu-network.reachability.bands',
    derivation: 'curated',
    tex: String.raw`\term{band}{b_v}=\begin{cases}\#\{i:\term{thresholds}{t_i}<\term{cost}{c_v}\}&c_v<\infty,\ c_v\le t_{m-1}\\\term{none}{\mathrm{NONE}}&\text{otherwise}\end{cases}`,
    ascii:
      'For finite cost <= the last ascending threshold, band = count(threshold < cost); otherwise NONE (0xffffffff). Band 0 includes the first threshold; band i is (threshold[i-1], threshold[i]].',
    symbols: [
      {id: 'band', role: 'output', label: 'band id', binding: 'wgsl:bands'},
      {id: 'thresholds', role: 'input', label: 'ascending cost breaks', binding: 'wgsl:thresholds'},
      {id: 'cost', role: 'input', label: 'final cost', binding: 'wgsl:cost'},
      {id: 'none', role: 'constant', label: 'no band', binding: 'wgsl:NONE'}
    ]
  }
};

/** Per-element primary formula; later network tasks append their own element keys. */
export const NETWORK_FORMULAS: NonNullable<ViewPackData['formulas']> = {
  ...NETWORK_OPTIMIZATION_PACK.formulas,
  GPUNetworkInhomogeneousK: {
    formula: 'none',
    reason:
      'Intensity-reweighted node-event pairs use explicit metric-network shell edge correction and caller-selected normalization.'
  },
  GPUNetworkKernelDensity: {
    formula: 'none',
    reason:
      'Caller-prepared network distances and edge-length quadrature normalize each event kernel over its reachable support; missing support remains explicit.'
  },
  GPUNetworkStronglyConnectedComponents: {
    formula: 'none',
    reason:
      'GPU-gated trim, minimum-colour propagation and backward marking publish minimum-ID directed component labels; mandatory convergence status certifies the labels.'
  },
  GPUNetworkStructure: {
    formula: 'none',
    reason:
      'Connectivity and low-link traversal produce component labels, bridge flags and edge blocks rather than one scalar estimator.'
  },
  GPUKShortestPaths: {
    formula: 'none',
    reason:
      'Yen rounds compose parallel CSR spur searches with a persistent candidate pool to rank simple edge paths. Per-query counts retain certified routes; batch convergence, iteration limits and candidate overflow remain separate.'
  },
  GPUAlternativeRoutes: {
    formula: 'none',
    reason:
      'Bounded heap-frontier enumeration ranks simple edge paths by accumulated nonnegative cost; completeness is a separate mandatory status.'
  },
  addNetworkDisruptionRecipe: {
    formula: 'none',
    reason:
      'A workflow compares two directed searches, convergence flags, reachability and accessibility; no single formula represents all outputs.'
  },
  addTrackQualityRecipe: {
    formula: 'none',
    reason:
      'Raw trajectory metrics, temporal quality flags and optional route comparison/map matching retain distinct output and status contracts.'
  },
  addFacilityScenarioRecipe: {
    formula: 'none',
    reason:
      'A workflow combines masked supply, gravity accessibility, nearest-facility assignment and covered demand with different output units.'
  },
  ...NETWORK_SEARCH_PACK.formulas,
  ...NETWORK_STRUCTURE_PACK.formulas,
  ...NETWORK_CONSTRUCTION_PACK.formulas,
  GPUNetworkReachability: NETWORK_FORMULA_REFS['gpu-network.reachability.relax']
};
