// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {OperatorDescriptor} from '../../computation-views/types';
import type {NetworkPackPart} from './network-pack-part';

export const NETWORK_OPTIMIZATION_OPERATORS: Record<string, OperatorDescriptor> = {
  GPUNetworkPointRouting: {scope: 'solver', scopeSecondary: 'pairwise', reduction: 'argmin'},
  GPUNetworkFlow: {scope: 'solver', reduction: 'select'},
  GPUNetworkLouvain: {scope: 'solver', scopeSecondary: 'zonal', reduction: 'select'},
  GPUNetworkFacilityOptimization: {scope: 'solver', reduction: 'select'},
  GPUNetworkTimeRouting: {scope: 'solver', reduction: 'argmin'},
  GPUNetworkRationalAccess: {scope: 'solver', reduction: 'weighted-mean'}
};

export const NETWORK_OPTIMIZATION_PACK: NetworkPackPart = {
  formulaRefs: {},
  formulas: {
    GPUNetworkPointRouting: {
      formula: 'none',
      reason:
        'Partial-edge initialization, legal turn-state relaxation and partial target evaluation form a procedural shortest-path composition.'
    },
    GPUNetworkFlow: {
      formula: 'none',
      reason:
        'Bounded shortest augmenting paths mutate explicit forward/reverse residual capacities, preserving integral flow and producing a cut certificate.'
    },
    GPUNetworkLouvain: {
      formula: 'none',
      reason:
        'Ascending or seeded local moves reevaluate directed/undirected weighted modularity and contract immutable vertex groups; the objective trace documents accepted sweeps.'
    },
    GPUNetworkFacilityOptimization: {
      formula: 'none',
      reason:
        'Exhaustive subset/assignment enumeration implements distinct backup, dispersion and unsplit capacitated objectives with explicit completion certificates.'
    },
    GPUNetworkTimeRouting: {
      formula: 'none',
      reason:
        'FIFO affine or piecewise arrival curves, including first-sample waiting, are evaluated at each expanded-state arrival, followed by bounded relaxation and target-state selection.'
    },
    GPUNetworkRationalAccess: {
      formula: 'none',
      reason:
        'Integral or fractional pairwise demand shifts with fractional/absolute caps combine congestion equalization, travel/supply normalization, step decay and initial inflow limits; final cost gaps distinguish stalls from equilibrium.'
    }
  },
  captureStages: {
    flow: 'assignment',
    cut: 'summary',
    communities: 'assignment',
    objectiveTrace: 'summary',
    selectedFacilities: 'assignment',
    arrivals: 'search',
    predecessorTransitions: 'search',
    predecessorEdges: 'search',
    unservedDemand: 'summary',
    scores: 'summary'
  },
  variableKinds: {
    flow: {level: 'ratio', extent: 'extensive'},
    cut: {level: 'nominal'},
    communities: {level: 'nominal'},
    objectiveTrace: {level: 'interval', extent: 'intensive'},
    selectedFacilities: {level: 'nominal'},
    arrivals: {level: 'interval', extent: 'intensive'},
    predecessorTransitions: {level: 'nominal'},
    predecessorEdges: {level: 'nominal'},
    unservedDemand: {level: 'ratio', extent: 'extensive'},
    scores: {level: 'ratio', extent: 'intensive'}
  }
};
