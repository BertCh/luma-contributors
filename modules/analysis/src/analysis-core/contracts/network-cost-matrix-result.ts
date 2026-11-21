// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';

/**
 * Non-owning cost matrix with row maps in the supplied CSR node domain.
 * Defined below both the network and spatial entries so that spatial consumers (location allocation)
 * can name it without importing the network layer; `gpu-network` re-exports it unchanged.
 */
export type GPUNetworkCostMatrixResult = {
  values: GraphDataView<'float32'>;
  rowCount: number;
  columnCount: number;
  startNodes: GraphDataView<'uint32'>;
  startRows?: GraphDataView<'uint32'>;
  startsPerRow: number;
  targetNodes?: GraphDataView<'uint32'>;
  direction: 'start-to-target' | 'target-to-start';
  converged?: GraphDataView<'uint32'>;
  sourceComplete?: GraphDataView<'uint32'>;
  costLimit?: GraphDataView<'float32'>;
};
