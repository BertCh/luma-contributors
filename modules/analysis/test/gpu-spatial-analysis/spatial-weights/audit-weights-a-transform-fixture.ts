// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUSpatialWeightsTransformProps} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transform';

/** Direct dense-row formulas from audit-weights-a-helpers.oracle.py (not shader outputs). */
export const AUDIT_TRANSFORM_CASES: {
  options: Pick<
    GPUSpatialWeightsTransformProps,
    'operation' | 'kernel' | 'bandwidth' | 'doubleSum'
  >;
  expected: number[];
}[] = [
  {
    options: {operation: 'row'},
    expected: [
      0.6, 0.13333333333333333, 0.26666666666666666, 0.375, 0.625, 0.46153846153846156,
      0.5384615384615384, 1, 1, 1
    ]
  },
  {options: {operation: 'binary'}, expected: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},
  {
    options: {operation: 'double'},
    expected: [
      0.1836734693877551, 0.04081632653061224, 0.08163265306122448, 0.061224489795918366,
      0.10204081632653061, 0.12244897959183673, 0.14285714285714285, 0.16326530612244897,
      0.04081632653061224, 0.061224489795918366
    ]
  },
  {
    options: {operation: 'double', doubleSum: 'rows'},
    expected: [
      1.469387755102041, 0.32653061224489793, 0.6530612244897959, 0.4897959183673469,
      0.8163265306122449, 0.9795918367346939, 1.1428571428571428, 1.3061224489795917,
      0.32653061224489793, 0.4897959183673469
    ]
  },
  {
    options: {operation: 'variance'},
    expected: [
      0.984834246916261, 0.2188520548702802, 0.4377041097405604, 0.5658002253844849,
      0.9430003756408082, 0.7156869651406362, 0.834968125997409, 1.0997179654365201,
      1.0997179654365201, 1.0997179654365201
    ]
  },
  {
    options: {operation: 'average-reciprocal'},
    expected: [9, 2.5, 6, 2.5, 2.5, 3, 3.5, 6, 2.5, 2.5]
  },
  {
    options: {operation: 'kernel', kernel: 'gaussian', bandwidth: 20},
    expected: [
      0.3989422804014327, 0.3520653267642995, 0.24197072451914337, 0.3520653267642995,
      0.33322460289179967, 0.31225393336676127, 0.28969155276148273, 0.24197072451914337,
      0.3520653267642995, 0.3520653267642995
    ]
  },
  {
    options: {operation: 'kernel', kernel: 'triangular', bandwidth: 20},
    expected: [1, 0.5, 0, 0.5, 0.4, 0.3, 0.2, 0, 0.5, 0.5]
  },
  {
    options: {operation: 'kernel', kernel: 'epanechnikov', bandwidth: 20},
    expected: [0.75, 0.5625, 0, 0.5625, 0.48, 0.3825, 0.27, 0, 0.5625, 0.5625]
  },
  {
    options: {operation: 'kernel', kernel: 'bisquare', bandwidth: 20},
    expected: [
      0.9375, 0.52734375, 0, 0.52734375, 0.384, 0.24384375, 0.1215, 0, 0.52734375, 0.52734375
    ]
  },
  {
    options: {operation: 'kernel', kernel: 'uniform', bandwidth: 20},
    expected: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
  },
  {
    options: {operation: 'kernel', kernel: 'gaussian', bandwidth: 'adaptive'},
    expected: [
      0.3989422804014327, 0.35206533556593145, 0.2419707487162134, 0.28191189498751384,
      0.2419707487162134, 0.272055019207752, 0.2419707487162134, 0.2419707487162134,
      0.2419707487162134, 0.2419707487162134
    ]
  },
  {
    options: {operation: 'kernel', kernel: 'triangular', bandwidth: 'adaptive'},
    expected: [
      1, 0.500000049999995, 9.99999900663795e-8, 0.16666674999999165, 9.99999899553572e-8,
      0.12500008749999125, 9.99999900663795e-8, 9.99999900663795e-8, 9.99999900663795e-8,
      9.99999900663795e-8
    ]
  },
  {
    options: {operation: 'kernel', kernel: 'epanechnikov', bandwidth: 'adaptive'},
    expected: [
      0.75, 0.5625000374999944, 1.4999997760556383e-7, 0.22916677083331766, 1.4999997743903037e-7,
      0.17578136484373277, 1.4999997760556383e-7, 1.4999997760556383e-7, 1.4999997760556383e-7,
      1.4999997760556383e-7
    ]
  },
  {
    options: {operation: 'kernel', kernel: 'bisquare', bandwidth: 'adaptive'},
    expected: [
      0.9375, 0.5273438203124919, 3.749998880278275e-14, 0.08752901475695056, 3.749998871951604e-14,
      0.05149848037720916, 3.749998880278275e-14, 3.749998880278275e-14, 3.749998880278275e-14,
      3.749998880278275e-14
    ]
  },
  {
    options: {operation: 'kernel', kernel: 'uniform', bandwidth: 'adaptive'},
    expected: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
  }
];
