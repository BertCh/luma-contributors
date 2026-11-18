// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {NetworkPathExtractionOperationProps} from '../../../src/gpu-network/network-analysis/gpu-network-path-extraction-operation';

const status = {count: 'count', requiredCount: 'required', overflow: 'overflow'};
const nodeOutput: NetworkPathExtractionOperationProps['output'] = {
  capacity: 2,
  ids: 'ids',
  status,
  // @ts-expect-error Edge rows belong only to the edge output.
  edgeRows: 'wrong'
};
const edgeOutput: NonNullable<NetworkPathExtractionOperationProps['edges']>['output'] = {
  capacity: 2,
  ids: 'edge-ids',
  status,
  // @ts-expect-error Node rows belong only to the node output.
  nodeRows: 'wrong'
};
void [nodeOutput, edgeOutput];
