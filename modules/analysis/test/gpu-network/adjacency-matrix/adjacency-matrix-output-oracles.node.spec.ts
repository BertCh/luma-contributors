// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeAdjacencyMatrixOrder} from '../../../src/gpu-network/adjacency-matrix/index';

it('adjacency order CPU helper inverts the hand-sorted list [1,2,0]', () => {
  expect(Array.from(computeAdjacencyMatrixOrder([2, 1, 1], [0, 1, 2]))).toEqual([2, 0, 1]);
});
