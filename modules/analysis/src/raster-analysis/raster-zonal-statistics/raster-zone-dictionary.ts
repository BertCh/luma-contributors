// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getBinarySearchWGSL} from '../../analysis-core/kernels/binary-search-wgsl';

/** Sparse numeric IDs mapped through a caller-owned sorted dictionary, without float conversion. @internal */
export function getRasterZoneDictionaryNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    zones: GraphDataView<'uint32'>;
    dictionary: GraphDataView<'uint32'>;
    ignoredZone: number;
    sourceZoneIds: GraphDataView<'uint32'>;
    denseZoneRows: GraphDataView<'uint32'>;
    invalidCount: GraphDataView<'uint32'>;
  }
): {nodes: GPUCommandNode<Parameters>[]; zones: GraphDataView<'uint32'>} {
  const {id, dictionary, invalidCount} = props;
  const operation = 'GPURasterZonalStatistics';
  const zoneCapacity = dictionary.length;
  const invalidDictionary = createTransientView(graph, `${id}-dictionary-invalid`, 'uint32', 1);
  const zones = createTransientView(graph, `${id}-dense-zones`, 'uint32', props.zones.length);
  return {
    zones,
    nodes: [
      createFillNode(graph, {
        id: `${id}-dictionary-reset`,
        operation,
        view: invalidDictionary,
        type: 'u32',
        value: '0u'
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-dictionary-validate`,
        operation,
        invocationCount: zoneCapacity,
        bindings: [
          {name: 'dictionary', view: dictionary, type: 'u32', access: 'read'},
          {
            name: 'invalidDictionary',
            view: invalidDictionary,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        ],
        body: `if (index > 0u && dictionary[dictionaryOffset + index] <= dictionary[dictionaryOffset + index - 1u]) { atomicAdd(&invalidDictionary[invalidDictionaryOffset], 1u); }`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-dictionary-publish`,
        operation,
        invocationCount: zoneCapacity,
        bindings: [
          {name: 'dictionary', view: dictionary, type: 'u32', access: 'read'},
          {name: 'invalidDictionary', view: invalidDictionary, type: 'u32', access: 'read'},
          {name: 'sourceZoneIds', view: props.sourceZoneIds, type: 'u32', access: 'read_write'},
          {name: 'denseZoneRows', view: props.denseZoneRows, type: 'u32', access: 'read_write'},
          {name: 'invalidCount', view: invalidCount, type: 'u32', access: 'read_write'}
        ],
        body: `sourceZoneIds[sourceZoneIdsOffset + index] = dictionary[dictionaryOffset + index];
  denseZoneRows[denseZoneRowsOffset + index] = select(index, 0xffffffffu, invalidDictionary[invalidDictionaryOffset] != 0u);
  if (index == 0u) { invalidCount[invalidCountOffset] = invalidDictionary[invalidDictionaryOffset]; }`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-dictionary-map`,
        operation,
        invocationCount: props.zones.length,
        bindings: [
          {name: 'sourceZones', view: props.zones, type: 'u32', access: 'read'},
          {name: 'dictionary', view: dictionary, type: 'u32', access: 'read'},
          {name: 'invalidDictionary', view: invalidDictionary, type: 'u32', access: 'read'},
          {name: 'invalidCount', view: invalidCount, type: 'atomic<u32>', access: 'read_write'},
          {name: 'denseZones', view: zones, type: 'u32', access: 'read_write'}
        ],
        declarations: getBinarySearchWGSL('dictionary'),
        body: `let sourceZone = sourceZones[sourceZonesOffset + index];
  var denseZone = ${zoneCapacity}u;
  if (sourceZone != ${props.ignoredZone}u && invalidDictionary[invalidDictionaryOffset] == 0u) {
    let candidate = lowerBoundDictionary(0u, ${zoneCapacity}u, sourceZone);
    if (candidate < ${zoneCapacity}u && dictionary[dictionaryOffset + candidate] == sourceZone) { denseZone = candidate; }
    else { atomicAdd(&invalidCount[invalidCountOffset], 1u); }
  }
  denseZones[denseZonesOffset + index] = denseZone;`
      })
    ]
  };
}
