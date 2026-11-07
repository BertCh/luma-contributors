// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {addKeplerLinearKernel} from './analysis-scaling';

/**
 * Cause slots of {@link addKeplerRateCauseCounts}, in the documented priority order: each row is
 * attributed to the first cause that applies, so the counts sum to the row count.
 */
export const KEPLER_RATE_CAUSES = [
  'valid',
  'masked',
  'nullEvent',
  'nullPopulation',
  'nonFiniteEvent',
  'nonFinitePopulation',
  'negativeEvent',
  'negativePopulation',
  'zeroPopulation'
] as const;

export type KeplerRateCause = (typeof KEPLER_RATE_CAUSES)[number];

export type KeplerRateCauseCountsProps = {
  id?: string;
  /** Float32 event column (packed, one view; see `packAnalysisVector` for chunked columns). */
  events: GraphDataView<'float32'>;
  /** Float32 population column with the same row count. */
  populations: GraphDataView<'float32'>;
  /** Optional validity (nonzero = present) of the event column; absent means no nulls. */
  eventValidity?: GraphDataView<'uint32'>;
  /** Optional validity (nonzero = present) of the population column. */
  populationValidity?: GraphDataView<'uint32'>;
  /** Optional inclusion mask (nonzero = included); excluded rows count as `masked`. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Optional current selection (nonzero = selected). Unselected rows are not counted at all, so the
   * counts sum to the number of selected rows; `mask` still attributes its rows to `masked`.
   */
  selection?: GraphDataView<'uint32'>;
  /** Caller-owned `KEPLER_RATE_CAUSES.length` uint32 counters; this pass zeroes then fills them. */
  counts: GraphDataView<'uint32'>;
};

/**
 * Counts rate source rows per exclusion cause on the GPU (exact integer atomics, so the result is
 * independent of dispatch order). It replaces scanning a CPU copy of the source and also works for
 * GPU-resident columns with no CPU rows. Only `counts` (36 bytes) needs to be read back.
 */
export function addKeplerRateCauseCounts<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: KeplerRateCauseCountsProps
): void {
  const id = props.id ?? 'kepler-rate-causes';
  const rowCount = props.events.length;
  // Event, population, validity and mask columns must all describe the same rows.
  if (
    props.populations.length !== rowCount ||
    props.counts.length !== KEPLER_RATE_CAUSES.length ||
    [props.eventValidity, props.populationValidity, props.mask, props.selection].some(
      view => view && view.length !== rowCount
    )
  )
    throw new Error('Rate cause counts require matching columns and nine counters');
  addKeplerLinearKernel(
    graph,
    `${id}-clear`,
    KEPLER_RATE_CAUSES.length,
    [{name: 'counts', view: props.counts, write: true, atomic: true}],
    'atomicStore(&counts[countsOffset + index], 0u);'
  );
  addKeplerLinearKernel(
    graph,
    `${id}-count`,
    rowCount,
    [
      {name: 'events', view: props.events},
      {name: 'populations', view: props.populations},
      ...(props.eventValidity ? [{name: 'eventValidity', view: props.eventValidity}] : []),
      ...(props.populationValidity
        ? [{name: 'populationValidity', view: props.populationValidity}]
        : []),
      ...(props.mask ? [{name: 'mask', view: props.mask}] : []),
      ...(props.selection ? [{name: 'selection', view: props.selection}] : []),
      {name: 'counts', view: props.counts, write: true, atomic: true}
    ],
    `${props.selection ? 'if (selection[selectionOffset + index] == 0u) { return; }\n' : ''}let event = events[eventsOffset + index];
let population = populations[populationsOffset + index];
var cause = 0u;
if (${props.mask ? 'mask[maskOffset + index] == 0u' : 'false'}) { cause = 1u; }
else if (${props.eventValidity ? 'eventValidity[eventValidityOffset + index] == 0u' : 'false'}) { cause = 2u; }
else if (${props.populationValidity ? 'populationValidity[populationValidityOffset + index] == 0u' : 'false'}) { cause = 3u; }
else if (!isFiniteValue(event)) { cause = 4u; }
else if (!isFiniteValue(population)) { cause = 5u; }
else if (event < 0.0) { cause = 6u; }
else if (population < 0.0) { cause = 7u; }
else if (population == 0.0) { cause = 8u; }
atomicAdd(&counts[countsOffset + cause], 1u);`
  );
}
