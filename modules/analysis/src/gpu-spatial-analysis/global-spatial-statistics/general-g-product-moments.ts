// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

const FIELDS = [
  'sum',
  'squares',
  'pairs',
  'squarePairs',
  'mixedPairs',
  'triples',
  'mixedTriples',
  'quadruples'
] as const;
type Field = (typeof FIELDS)[number];

/**
 * Positive distinct-index product moments for sparse General G, without subtracting raw moments.
 * pairs=sum(i!=j) xi*xj; squarePairs=sum(i!=j) xi²*xj²;
 * mixedTriples=sum(distinct i,j,k) xi²*xj*xk; quadruples=sum(distinct i,j,k,l) xi*xj*xk*xl.
 * The caller gates inputs to zero outside the sparse route, where the centered variance is stable.
 * Four merge passes keep every binding row-sized and every kernel at eight bindings or fewer.
 * @internal
 */
export function getGeneralGProductMomentNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    rows: number;
    bindings: WGSLKernelBinding[];
    value: string;
    destination: WGSLKernelBinding;
  }
): GPUCommandNode<Parameters>[] {
  const operation = 'GPUGlobalSpatialStatistics';
  const scratch = (suffix: string, count: number) =>
    Object.fromEntries(
      FIELDS.map(field => [
        field,
        createTransientView(graph, `${props.id}-${field}${suffix}`, 'float32', count)
      ])
    ) as Record<Field, GraphDataView<'float32'>>;
  let current = scratch('', props.rows);
  const bind = (name: Field, output = false, views = current): WGSLKernelBinding => ({
    name: output ? `next_${name}` : name,
    view: views[name],
    type: 'f32',
    access: output ? 'read_write' : 'read'
  });
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${props.id}-values`,
      operation,
      invocationCount: props.rows,
      bindings: [...props.bindings, bind('sum', true), bind('squares', true)],
      body: `let value = ${props.value}; next_sum[next_sumOffset + index] = value; next_squares[next_squaresOffset + index] = value * value;`
    })
  ];
  for (const field of FIELDS.slice(2))
    nodes.push(
      createFillNode(graph, {
        id: `${props.id}-clear-${field}`,
        operation,
        view: current[field],
        type: 'f32',
        value: '0.0'
      })
    );
  for (let count = props.rows, level = 0; count > 1; level++) {
    const nextCount = Math.ceil(count / 2);
    const next = scratch(`-${level}`, nextCount);
    const merge = (suffix: string, inputs: Field[], outputs: [Field, string][]) =>
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${props.id}-merge-${level}-${suffix}`,
          operation,
          invocationCount: nextCount,
          bindings: [
            ...inputs.map(field => bind(field)),
            ...outputs.map(([field]) => bind(field, true, next))
          ],
          body: `let left = index * 2u;
${inputs
  .map(
    field => `let left_${field} = ${field}[${field}Offset + left]; var right_${field} = 0.0;
if (left + 1u < ${count}u) { right_${field} = ${field}[${field}Offset + left + 1u]; }`
  )
  .join('\n')}
${outputs.map(([field, expression]) => `next_${field}[next_${field}Offset + index] = ${expression};`).join('\n')}`
        })
      );
    merge(
      'sums',
      ['sum', 'squares', 'pairs'],
      [
        ['sum', 'left_sum + right_sum'],
        ['squares', 'left_squares + right_squares'],
        ['pairs', 'left_pairs + right_pairs + 2.0 * left_sum * right_sum']
      ]
    );
    merge(
      'pairs',
      ['sum', 'squares', 'squarePairs', 'mixedPairs'],
      [
        [
          'squarePairs',
          'left_squarePairs + right_squarePairs + 2.0 * left_squares * right_squares'
        ],
        [
          'mixedPairs',
          'left_mixedPairs + right_mixedPairs + left_squares * right_sum + left_sum * right_squares'
        ]
      ]
    );
    merge(
      'triples',
      ['sum', 'squares', 'pairs', 'mixedPairs', 'triples', 'mixedTriples'],
      [
        [
          'triples',
          'left_triples + right_triples + 3.0 * (left_pairs * right_sum + left_sum * right_pairs)'
        ],
        [
          'mixedTriples',
          'left_mixedTriples + right_mixedTriples + left_squares * right_pairs + left_pairs * right_squares + 2.0 * (left_mixedPairs * right_sum + left_sum * right_mixedPairs)'
        ]
      ]
    );
    merge(
      'quadruples',
      ['sum', 'pairs', 'triples', 'quadruples'],
      [
        [
          'quadruples',
          'left_quadruples + right_quadruples + 4.0 * (left_triples * right_sum + left_sum * right_triples) + 6.0 * left_pairs * right_pairs'
        ]
      ]
    );
    current = next;
    count = nextCount;
  }
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${props.id}-store`,
      operation,
      invocationCount: 1,
      bindings: [
        bind('pairs'),
        bind('squarePairs'),
        bind('mixedTriples'),
        bind('quadruples'),
        props.destination
      ],
      body: `let mass = pairs[pairsOffset];
  counters[countersOffset + 29u] = bitcast<u32>(select(0.0, squarePairs[squarePairsOffset] / mass / mass, mass > 0.0));
  counters[countersOffset + 30u] = bitcast<u32>(select(0.0, mixedTriples[mixedTriplesOffset] / mass / mass, mass > 0.0));
  counters[countersOffset + 31u] = bitcast<u32>(select(0.0, quadruples[quadruplesOffset] / mass / mass, mass > 0.0));`
    })
  );
  return nodes;
}
