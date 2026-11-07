// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  setFilterAnimationTime,
  setFilterPlot,
  toggleFilterAnimation,
  wrapTo
} from '@kepler.gl/actions';
import type {KeplerTimeFilterLike, ExactPlaybackHost} from './time-contract-native-bridge';
import type {TimeContractFilterRecord, TimeContractPanelProps} from './time-contract-panel';
import {makeResidentTimeHistogramPlan} from './resident-time-histogram';

type StoreLike = {
  getState(): {keplerGl: Record<string, {visState?: {filters?: unknown[]}} | undefined>};
  dispatch(action: unknown): unknown;
};

type FilterState = KeplerTimeFilterLike & {type?: string | null; name?: readonly string[]};

function getTimeFilters(store: StoreLike): {index: number; filter: FilterState}[] {
  const filters = (store.getState().keplerGl['compute']?.visState?.filters ?? []) as FilterState[];
  return filters.flatMap((filter, index) => (filter.type === 'timeRange' ? [{index, filter}] : []));
}

/** Binds the time contract panel to the native Kepler instance named "compute". */
export function makeTimeContractPanelProps(store: StoreLike): TimeContractPanelProps {
  const filters: TimeContractFilterRecord[] = getTimeFilters(store).map(({index, filter}) => {
    let steps: readonly number[] = [];
    try {
      const domain = filter.domain as readonly [number, number] | undefined;
      if (domain)
        steps = makeResidentTimeHistogramPlan(domain, filter.plotType?.interval).thresholds;
    } catch {}
    return {index, label: filter.name?.join(', ') || `Time filter ${index + 1}`, filter, steps};
  });
  return {
    filters,
    getHost(index): ExactPlaybackHost {
      return {
        getFilter: () => getTimeFilters(store).find(entry => entry.index === index)?.filter,
        setFilterValue: value =>
          void store.dispatch(wrapTo('compute', setFilterAnimationTime(index, 'value', value))),
        pauseNativeAnimation: () =>
          void store.dispatch(wrapTo('compute', toggleFilterAnimation(index))),
        requestFrame: callback => requestAnimationFrame(callback),
        cancelFrame: handle => cancelAnimationFrame(handle)
      };
    },
    onZoneChanged() {
      // Re-issue each filter's plot interval so the resident scope re-plans its histogram edges.
      for (const {index, filter} of getTimeFilters(store)) {
        const interval = filter.plotType?.interval;
        if (interval) store.dispatch(wrapTo('compute', setFilterPlot(index, {interval} as never)));
      }
    }
  };
}
