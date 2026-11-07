// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {OperatorDescriptor} from '../../computation-views/types';

/** Network point-process descriptors; density and second-order K remain distinct operators. */
export const NETWORK_POINT_PROCESS_OPERATORS: Record<string, OperatorDescriptor> = {
  GPUNetworkKernelDensity: {scope: 'pairwise', reduction: 'weighted-mean'},
  GPUNetworkInhomogeneousK: {scope: 'pairwise', scopeSecondary: 'global', reduction: 'sum'}
};
