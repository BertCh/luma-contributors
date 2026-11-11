// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test, vi} from 'vitest';
import {
  createKeplerComputeTableClass,
  getKeplerComputeTableState,
  installKeplerComputeTableBridge,
  planKeplerComputeFilters,
  updateKeplerComputeFilterPlot,
  type KeplerComputeFilter,
  type KeplerComputeTableBinding
} from '../../../examples/deck/kepler-compute/src/kepler-table-bridge';
import {patchKeplerComputeFilterSource} from '../../../examples/deck/kepler-compute/kepler-compute-transform';

class SourceTable {
  id: string;
  fields = [
    {name: 'value', type: 'real'},
    {name: 'category', type: 'string'},
    {name: 'time', type: 'timestamp'}
  ];
  allIndexes = [0, 1, 2, 3];
  filteredIndex = this.allIndexes;
  filteredIndexForDomain = this.allIndexes;
  filteredIndexByLayer = {};
  filteredIdxCPU: number[] | undefined;
  dataContainer = {
    valueAt: vi.fn(() => {
      throw new Error('Unexpected source cell scan');
    }),
    map: vi.fn(() => {
      throw new Error('Unexpected source map');
    }),
    rows: vi.fn(() => {
      throw new Error('Unexpected source rows');
    }),
    getPlainIndex: vi.fn(() => {
      throw new Error('Unexpected source index allocation');
    }),
    flattenData: vi.fn(() => {
      throw new Error('Unexpected source flatten');
    })
  };
  constructor({id}: {id: string}) {
    this.id = id;
  }
  filterTable(_filters: readonly KeplerComputeFilter[]): SourceTable {
    this.dataContainer.rows();
    return this;
  }
  getColumnFilterProps(_name: string): Record<string, unknown> | null {
    this.dataContainer.rows();
    return null;
  }
  getColumnLayerDomain(_field: {name: string}, _scale: string): readonly unknown[] | null {
    this.dataContainer.rows();
    return null;
  }
  filterTableCPU(): void {
    this.dataContainer.rows();
  }
}

function makeBinding(): KeplerComputeTableBinding {
  return {
    fields: {
      value: {dimension: 'value', kind: 'range', domain: [0, 100], step: 1},
      category: {
        dimension: 'category',
        kind: 'category',
        domain: ['a', 'b', 'c'],
        categories: ['a', 'b', 'c']
      },
      time: {dimension: 'time', kind: 'time', domain: [-1000, 1000], step: 1}
    },
    explorer: {clear: vi.fn(), setRange: vi.fn(), setCategories: vi.fn(), setEmpty: vi.fn()},
    onSelectionChange: vi.fn(),
    onUnsupported: vi.fn()
  };
}

const valueFilter: KeplerComputeFilter = {
  id: 'value-filter',
  dataId: ['source'],
  name: ['value'],
  type: 'range',
  value: [20, 80],
  fixedDomain: true
};

describe('Kepler resident table boundary', () => {
  test('native filters and domains never call bulk or cell row APIs on repeated updates', () => {
    const binding = makeBinding();
    const Table = createKeplerComputeTableClass(SourceTable, id =>
      id === 'source' ? binding : undefined
    );
    const table = new Table({id: 'source'});
    for (let step = 0; step < 10; step++) {
      table.filterTable([{...valueFilter, value: [step, 80]}]);
      expect(table.getColumnFilterProps('value')).toMatchObject({
        fixedDomain: true,
        domain: [0, 100],
        gpu: true
      });
      expect(table.getColumnLayerDomain({name: 'value'}, 'linear')).toEqual([0, 100]);
    }
    expect(binding.explorer.setRange).toHaveBeenLastCalledWith('value', [9, 80]);
    expect(binding.onSelectionChange).toHaveBeenCalledTimes(10);
    expect(table.filteredIndex).toEqual([]);
    expect(table.filteredIndexForDomain).toEqual([]);
    expect(table.getColumnFilterProps('time')).toMatchObject({gpu: true});
    expect(table.getColumnFilterProps('category')).toMatchObject({gpu: false});
    for (const method of Object.values(table.dataContainer)) expect(method).not.toHaveBeenCalled();
    expect(() => new Table({id: 'cpu-source'}).filterTable([])).toThrow('Unexpected source rows');
  });

  test('Kepler copyTable keeps resident methods and shared state while updating the copied indices', () => {
    const binding = makeBinding();
    const original = installKeplerComputeTableBridge(new SourceTable({id: 'source'}), binding);
    const copied: typeof original = Object.assign(
      Object.create(Object.getPrototypeOf(original)),
      original
    );
    copied.filterTable([valueFilter]);
    expect(copied.filteredIndex).toEqual([]);
    expect(original.filteredIndex).toEqual([0, 1, 2, 3]);
    expect(copied.lumaCompute).toBe(original.lumaCompute);
    expect(binding.explorer.setRange).toHaveBeenCalledWith('value', [20, 80]);
  });

  test('plans intersections, category complement and dataset-specific names without modifying source filters', () => {
    const binding = makeBinding();
    const filters: KeplerComputeFilter[] = [
      valueFilter,
      {
        ...valueFilter,
        id: 'second',
        dataId: ['other', 'source'],
        name: ['unrelated', 'value'],
        value: [30, 90]
      },
      {
        id: 'category',
        dataId: ['source'],
        name: ['category'],
        type: 'multiSelect',
        value: ['b'],
        inverted: true,
        fixedDomain: true
      }
    ];
    expect(planKeplerComputeFilters('source', [], binding.fields, filters)).toEqual({
      display: [
        {kind: 'range', dimension: 'value', value: [30, 80]},
        {kind: 'category', dimension: 'category', value: [0, 2]}
      ],
      domain: [],
      filterIds: ['value-filter', 'second', 'category']
    });
    expect(valueFilter.value).toEqual([20, 80]);
  });

  test('empty native category selection bypasses while disjoint range intersection selects no rows', () => {
    const binding = makeBinding();
    const table = installKeplerComputeTableBridge(new SourceTable({id: 'source'}), binding);
    table.filterTable([
      valueFilter,
      {...valueFilter, id: 'other', value: [90, 100]},
      {
        id: 'category',
        dataId: ['source'],
        name: ['category'],
        type: 'multiSelect',
        value: [],
        fixedDomain: true
      }
    ]);
    expect(binding.explorer.setEmpty).toHaveBeenCalledWith('value');
    expect(binding.explorer.setCategories).not.toHaveBeenCalled();
    expect(binding.explorer.clear).toHaveBeenCalledWith('category');
  });

  test('unsupported requests fail atomically before changing valid GPU selections', () => {
    const binding = makeBinding();
    const table = installKeplerComputeTableBridge(new SourceTable({id: 'source'}), binding);
    table.filterTable([valueFilter, {...valueFilter, id: 'inverted', inverted: true}]);
    expect(table.lumaCompute.status).toBe('unsupported');
    expect(binding.explorer.clear).not.toHaveBeenCalled();
    expect(binding.explorer.setRange).not.toHaveBeenCalled();
    expect(binding.onUnsupported).toHaveBeenCalledWith(expect.stringContaining('Inverted'));
    table.filterTable([{...valueFilter, fixedDomain: false}]);
    expect(binding.onUnsupported).toHaveBeenLastCalledWith(
      expect.stringContaining('Dynamic layer domains')
    );
  });

  test('disabled native numeric, category, time and polygon filters bypass all selections and domains', () => {
    // Pinned filter-utils.getFilterFunction returns true before validating a disabled filter;
    // computePolygonFilteredIndexByLayer excludes disabled filters before parsing their geometry.
    const binding = makeBinding();
    binding.applyDomainSelection = vi.fn();
    binding.prepareSpatialFilter = vi.fn(() => ({layerIds: ['points'], apply: vi.fn()}));
    binding.clearSpatialFilters = vi.fn();
    const disabled: KeplerComputeFilter[] = [
      {...valueFilter, enabled: false, fixedDomain: false, value: 'invalid numeric value'},
      {
        ...valueFilter,
        id: 'category',
        name: ['category'],
        type: 'multiSelect',
        enabled: false,
        fixedDomain: false,
        value: 'invalid category value'
      },
      {
        ...valueFilter,
        id: 'time',
        name: ['time'],
        type: 'timeRange',
        enabled: false,
        fixedDomain: false,
        value: [0.5, Infinity]
      },
      {
        id: 'polygon',
        dataId: ['source'],
        type: 'polygon',
        enabled: false,
        layerId: ['points'],
        value: {geometry: 'invalid'}
      }
    ];
    expect(
      planKeplerComputeFilters(
        'source',
        new SourceTable({id: 'source'}).fields,
        binding.fields,
        disabled
      )
    ).toEqual({display: [], domain: [], filterIds: []});
    const table = installKeplerComputeTableBridge(new SourceTable({id: 'source'}), binding);
    table.filterTable([valueFilter]);
    vi.clearAllMocks();
    table.filterTable(disabled);
    expect(table.lumaCompute.plan).toEqual({display: [], domain: [], filterIds: []});
    expect(binding.explorer.clear).toHaveBeenCalledWith('value');
    expect(binding.explorer.clear).toHaveBeenCalledWith('category');
    expect(binding.explorer.clear).toHaveBeenCalledWith('time');
    expect(binding.explorer.setRange).not.toHaveBeenCalled();
    expect(binding.explorer.setCategories).not.toHaveBeenCalled();
    expect(binding.applyDomainSelection).toHaveBeenCalledWith([], 2);
    expect(binding.clearSpatialFilters).toHaveBeenCalledOnce();
    expect(binding.prepareSpatialFilter).not.toHaveBeenCalled();
    expect(binding.onUnsupported).not.toHaveBeenCalled();
    expect(binding.onSelectionChange).toHaveBeenCalledOnce();
    table.filterTable([{...valueFilter, enabled: true}]);
    expect(binding.explorer.setRange).toHaveBeenCalledWith('value', [20, 80]);
  });

  test('validates exact time endpoints before mutating any selected field', () => {
    const binding = makeBinding();
    binding.fields = {
      ...binding.fields,
      time: {
        ...binding.fields['time'],
        validateRange(minimum, maximum) {
          if (![minimum, maximum].every(Number.isSafeInteger))
            throw new Error('Exact time requires safe integer milliseconds');
        }
      }
    };
    const table = installKeplerComputeTableBridge(new SourceTable({id: 'source'}), binding);
    table.filterTable([
      valueFilter,
      {...valueFilter, id: 'time', name: ['time'], type: 'timeRange', value: [0.5, 1]}
    ]);
    expect(binding.explorer.clear).not.toHaveBeenCalled();
    expect(binding.explorer.setRange).not.toHaveBeenCalled();
    expect(binding.onUnsupported).toHaveBeenCalledWith(
      'Exact time requires safe integer milliseconds'
    );
  });

  test('domain selection excludes fixed-domain display filters and never reuses display domain by accident', () => {
    const binding = makeBinding();
    binding.applyDomainSelection = vi.fn();
    binding.getDynamicLayerDomain = vi.fn(() => [25, 70]);
    const table = installKeplerComputeTableBridge(new SourceTable({id: 'source'}), binding);
    table.filterTable([
      valueFilter,
      {...valueFilter, id: 'dynamic', value: [25, 70], fixedDomain: false}
    ]);
    expect(binding.applyDomainSelection).toHaveBeenCalledWith(
      [{kind: 'range', dimension: 'value', value: [25, 70]}],
      1
    );
    expect(table.getColumnLayerDomain({name: 'value'}, 'linear')).toEqual([25, 70]);
    expect(binding.getDynamicLayerDomain).toHaveBeenCalledWith('value', 'linear', 1);
  });

  test('execution setup failure suppresses result publication and exposes partial state as unsupported', () => {
    const binding = makeBinding();
    binding.explorer.setRange = () => {
      throw new Error('GPU resource limit');
    };
    const table = installKeplerComputeTableBridge(new SourceTable({id: 'source'}), binding);
    table.filterTable([valueFilter]);
    expect(table.lumaCompute.status).toBe('unsupported');
    expect(binding.onUnsupported).toHaveBeenCalledWith('GPU resource limit');
    expect(binding.onSelectionChange).not.toHaveBeenCalled();
    expect(table.lumaCompute.setHistogram('value', [], 1)).toBe(false);
  });

  test('native plots use only matching bounded summaries and never retain stale bin counts', () => {
    const table = installKeplerComputeTableBridge(new SourceTable({id: 'source'}), makeBinding());
    table.filterTable([valueFilter]);
    const bins = [{x0: 0, x1: 10, count: 3, indexes: []}];
    expect(table.lumaCompute.setHistogram('value', bins, 0)).toBe(false);
    expect(table.lumaCompute.setHistogram('value', bins, 1)).toBe(true);
    const result = updateKeplerComputeFilterPlot({source: table}, valueFilter);
    expect(result?.bins).toEqual({source: bins});
    table.filterTable([{...valueFilter, value: [30, 50]}]);
    expect(updateKeplerComputeFilterPlot({source: table}, valueFilter)?.bins).toEqual({
      source: undefined
    });
    expect(() => table.filterTableCPU()).toThrow('explicit asynchronous materialization');
  });

  test('time histogram interval changes cannot relabel cached bins and supported plots recover independently', () => {
    const binding = makeBinding();
    binding.fields = {
      ...binding.fields,
      time: {...binding.fields['time'], histogramInterval: '1-hour'}
    };
    binding.onPlotRecovery = vi.fn();
    binding.onPlotUnsupported = vi.fn();
    const table = installKeplerComputeTableBridge(new SourceTable({id: 'source'}), binding);
    const filter: KeplerComputeFilter = {
      ...valueFilter,
      id: 'timestamp',
      name: ['time'],
      type: 'timeRange',
      value: [-1000, 1000],
      plotType: {type: 'histogram', interval: '1-hour'}
    };
    table.filterTable([filter]);
    const revision = table.lumaCompute.revision;
    const bins = [{x0: 0, x1: 3600000, count: 3, indexes: []}];
    expect(table.getColumnFilterProps('time')).toMatchObject({plotType: {interval: '1-hour'}});
    expect(table.lumaCompute.setHistogram('time', bins, revision)).toBe(true);
    expect(table.lumaCompute.updateFilterPlot(filter)).toMatchObject({
      timeBins: {source: {'1-hour': bins}}
    });
    const unsupported = table.lumaCompute.updateFilterPlot({
      ...filter,
      plotType: {type: 'histogram', interval: '1-minute'}
    });
    expect(unsupported).toMatchObject({
      timeBins: undefined,
      bins: undefined,
      lumaComputePlot: {status: 'unsupported'}
    });
    expect(table.lumaCompute.status).toBe('unsupported');
    expect(binding.onUnsupported).not.toHaveBeenCalled();
    expect(binding.onPlotUnsupported).toHaveBeenCalledWith(
      'timestamp',
      expect.stringContaining('supports 1-hour')
    );
    const updatedBins = [{...bins[0], count: 7}];
    // Valid GPU results may continue arriving while only plot configuration is unsupported.
    expect(table.lumaCompute.setHistogram('time', updatedBins, revision)).toBe(true);
    expect(table.lumaCompute.status).toBe('unsupported');
    expect(table.lumaCompute.updateFilterPlot(filter)).toMatchObject({
      timeBins: {source: {'1-hour': updatedBins}},
      lumaComputePlot: {status: 'ready'}
    });
    expect(table.lumaCompute.status).toBe('ready');
    expect(table.lumaCompute.revision).toBe(revision);
    expect(binding.onPlotRecovery).toHaveBeenCalledOnce();
    expect(binding.onPlotUnsupported).toHaveBeenLastCalledWith('timestamp', null);
    // Correct plot metadata must never clear an actual invalid execution request.
    table.filterTable([{...filter, inverted: true}]);
    expect(table.lumaCompute.updateFilterPlot(filter)).toMatchObject({
      lumaComputePlot: {status: 'unsupported'}
    });
    expect(table.lumaCompute.setHistogram('time', bins, revision)).toBe(false);
    expect(table.lumaCompute.status).toBe('unsupported');
    expect(binding.onPlotRecovery).toHaveBeenCalledOnce();
  });

  test('same-field filters retain correct selection but refuse charts until only one active filter remains', () => {
    // Pinned plot.ts getSkipIndexes skips current filter channels; CPU plot bases omit only filter.id.
    const binding = makeBinding();
    binding.onPlotUnsupported = vi.fn();
    const table = installKeplerComputeTableBridge(new SourceTable({id: 'source'}), binding);
    const second = {...valueFilter, id: 'second-value', value: [40, 60]};
    const bins = [{x0: 0, x1: 10, count: 3, indexes: []}];
    table.filterTable([valueFilter, second]);
    expect(binding.explorer.setRange).toHaveBeenLastCalledWith('value', [40, 60]);
    expect(table.lumaCompute.setHistogram('value', bins, table.lumaCompute.revision)).toBe(true);
    for (const filter of [valueFilter, second]) {
      expect(table.lumaCompute.updateFilterPlot({...filter, bins: {source: bins}})).toMatchObject({
        bins: undefined,
        timeBins: undefined,
        lumaComputePlot: {status: 'unsupported'}
      });
    }
    expect(binding.onUnsupported).not.toHaveBeenCalled();
    expect(binding.onPlotUnsupported).toHaveBeenCalledWith(
      'value-filter',
      expect.stringContaining('per-filter GPU histograms')
    );
    expect(binding.onPlotUnsupported).toHaveBeenCalledWith(
      'second-value',
      expect.stringContaining('per-filter GPU histograms')
    );
    const disabled = {...second, enabled: false};
    table.filterTable([valueFilter, disabled]);
    expect(binding.explorer.setRange).toHaveBeenLastCalledWith('value', [20, 80]);
    expect(binding.onPlotUnsupported).toHaveBeenCalledWith('value-filter', null);
    expect(binding.onPlotUnsupported).toHaveBeenCalledWith('second-value', null);
    expect(table.lumaCompute.updateFilterPlot({...disabled, bins: {source: bins}})).toMatchObject({
      bins: undefined,
      lumaComputePlot: {status: 'disabled'}
    });
    expect(table.lumaCompute.setHistogram('value', bins, table.lumaCompute.revision)).toBe(true);
    expect(table.lumaCompute.updateFilterPlot(valueFilter)).toMatchObject({
      bins: {source: bins},
      lumaComputePlot: {status: 'ready'}
    });
    table.filterTable([valueFilter, second]);
    table.lumaCompute.updateFilterPlot(valueFilter);
    table.lumaCompute.updateFilterPlot(second);
    table.filterTable([valueFilter]);
    expect(binding.onPlotUnsupported).toHaveBeenLastCalledWith('second-value', null);
    expect(table.lumaCompute.setHistogram('value', bins, table.lumaCompute.revision)).toBe(true);
    expect(table.lumaCompute.updateFilterPlot(valueFilter)).toMatchObject({
      lumaComputePlot: {status: 'ready'}
    });
    expect(binding.onUnsupported).not.toHaveBeenCalled();
  });

  test('native filter owners publish exact independent histogram scopes for duplicate fields', () => {
    const binding = makeBinding();
    binding.applyNativeFilters = vi.fn();
    binding.filterScopedHistograms = true;
    binding.onPlotUnsupported = vi.fn();
    const table = installKeplerComputeTableBridge(new SourceTable({id: 'source'}), binding);
    const second = {...valueFilter, id: 'second-value', value: [40, 60], fixedDomain: false};
    table.filterTable([valueFilter, second]);
    expect(binding.explorer.clear).not.toHaveBeenCalled();
    expect(binding.explorer.setRange).not.toHaveBeenCalled();
    expect(binding.applyNativeFilters).toHaveBeenCalledWith(
      expect.objectContaining({
        filters: [valueFilter, second],
        revision: 1,
        plan: expect.objectContaining({
          display: [
            {kind: 'range', dimension: 'value', value: [20, 80]},
            {kind: 'range', dimension: 'value', value: [40, 60]}
          ]
        })
      })
    );
    const firstBins = [{x0: 0, x1: 10, count: 3, indexes: []}];
    const secondBins = [{x0: 0, x1: 10, count: 7, indexes: []}];
    // Field-level cache cannot masquerade as an exact own-filter scope.
    table.lumaCompute.setHistogram('value', firstBins, 1);
    expect(table.lumaCompute.updateFilterPlot(valueFilter)).toMatchObject({
      lumaComputePlot: {status: 'pending'}
    });
    expect(table.lumaCompute.setFilterHistogram(valueFilter.id, firstBins, 1)).toBe(true);
    expect(table.lumaCompute.setFilterHistogram(second.id, secondBins, 1)).toBe(true);
    expect(table.lumaCompute.updateFilterPlot(valueFilter)).toMatchObject({
      bins: {source: firstBins},
      lumaComputePlot: {status: 'ready'}
    });
    expect(table.lumaCompute.updateFilterPlot(second)).toMatchObject({
      bins: {source: secondBins},
      lumaComputePlot: {status: 'ready'}
    });
    expect(binding.onPlotUnsupported).not.toHaveBeenCalled();
    table.lumaCompute.invalidateScope();
    expect(table.lumaCompute.revision).toBe(1);
    expect(table.lumaCompute.updateFilterPlot(valueFilter)).toMatchObject({
      bins: {source: undefined},
      lumaComputePlot: {status: 'pending'}
    });
    expect(table.lumaCompute.setFilterHistogram(valueFilter.id, firstBins, 1)).toBe(true);
    const disabled = {...second, enabled: false};
    table.filterTable([valueFilter, disabled]);
    expect(table.lumaCompute.setFilterHistogram(second.id, secondBins, 1)).toBe(false);
    expect(table.lumaCompute.setFilterHistogram(second.id, secondBins, 2)).toBe(true);
    expect(table.lumaCompute.updateFilterPlot(disabled)).toMatchObject({
      bins: {source: secondBins},
      lumaComputePlot: {status: 'ready'}
    });
    expect(
      table.lumaCompute.updateFilterPlot({...disabled, plotType: {type: 'lineChart'}})
    ).toMatchObject({bins: undefined, lumaComputePlot: {status: 'disabled'}});
    expect(binding.onPlotUnsupported).not.toHaveBeenCalled();
  });

  test('scope completion without plots remains revision checked and cannot clear invalid execution', () => {
    const binding = makeBinding();
    const table = installKeplerComputeTableBridge(new SourceTable({id: 'source'}), binding);
    table.filterTable([]);
    expect(table.lumaCompute.status).toBe('pending');
    expect(table.lumaCompute.completeScope(0)).toBe(false);
    expect(table.lumaCompute.completeScope(1)).toBe(true);
    expect(table.lumaCompute.status).toBe('ready');
    table.filterTable([{...valueFilter, inverted: true}]);
    table.lumaCompute.invalidateScope();
    expect(table.lumaCompute.completeScope(1)).toBe(false);
    expect(table.lumaCompute.status).toBe('unsupported');
  });

  test('does not pretend min/max is a quantile or Jenks domain', () => {
    const binding = makeBinding();
    const table = installKeplerComputeTableBridge(new SourceTable({id: 'source'}), binding);
    expect(table.getColumnLayerDomain({name: 'value'}, 'quantile')).toBeNull();
    expect(binding.onUnsupported).toHaveBeenCalledWith(
      expect.stringContaining('prepared GPU domain')
    );
  });

  test('seeds native Jenks domains only from current derived thresholds', () => {
    const binding = makeBinding();
    binding.residentStyleDomains = true;
    let derived: readonly number[] | undefined;
    binding.getDerivedJenksDomain = () => derived;
    const table = installKeplerComputeTableBridge(new SourceTable({id: 'source'}), binding);
    expect(table.getColumnLayerDomain({name: 'value'}, 'jenks')).toEqual([]);
    derived = [4, 9];
    expect(table.getColumnLayerDomain({name: 'value'}, 'jenks')).toEqual([4, 9]);
  });

  test('resident style ownership keeps static distributions empty without changing valid filter state', () => {
    const binding = makeBinding();
    binding.residentStyleDomains = true;
    const table = installKeplerComputeTableBridge(new SourceTable({id: 'source'}), binding);
    table.filterTable([valueFilter]);
    table.lumaCompute.completeScope(table.lumaCompute.revision);
    expect(table.getColumnLayerDomain({name: 'value'}, 'quantile')).toEqual([]);
    expect(table.getColumnLayerDomain({name: 'value'}, 'jenks')).toEqual([]);
    expect(table.getColumnLayerDomain({name: 'value'}, 'linear')).toEqual([0, 100]);
    expect(table.getColumnLayerDomain({name: 'category'}, 'ordinal')).toEqual(['a', 'b', 'c']);
    expect(table.lumaCompute.status).toBe('ready');
    expect(table.lumaCompute.revision).toBe(1);
    expect(binding.onUnsupported).not.toHaveBeenCalled();
    for (const method of Object.values(table.dataContainer)) expect(method).not.toHaveBeenCalled();
  });

  test('resident style ownership bypasses dynamic distribution reads while retaining exact extents', () => {
    const binding = makeBinding();
    binding.residentStyleDomains = true;
    binding.applyDomainSelection = vi.fn();
    binding.getDynamicLayerDomain = vi.fn(() => [25, 70]);
    const table = installKeplerComputeTableBridge(new SourceTable({id: 'source'}), binding);
    table.filterTable([{...valueFilter, fixedDomain: false}]);
    table.lumaCompute.completeScope(table.lumaCompute.revision);
    expect(table.getColumnLayerDomain({name: 'value'}, 'quantile')).toEqual([]);
    expect(table.getColumnLayerDomain({name: 'value'}, 'log')).toEqual([]);
    expect(binding.getDynamicLayerDomain).not.toHaveBeenCalled();
    expect(table.getColumnLayerDomain({name: 'value'}, 'linear')).toEqual([25, 70]);
    expect(binding.getDynamicLayerDomain).toHaveBeenCalledExactlyOnceWith('value', 'linear', 1);
    expect(table.lumaCompute.status).toBe('ready');
    expect(binding.onUnsupported).not.toHaveBeenCalled();
    for (const method of Object.values(table.dataContainer)) expect(method).not.toHaveBeenCalled();
  });

  test('spatial masks are explicitly layer-specific, validated before mutations and cleared on removal', () => {
    const binding = makeBinding();
    const apply = vi.fn();
    binding.prepareSpatialFilter = vi.fn(() => ({layerIds: ['points'], apply}));
    binding.clearSpatialFilters = vi.fn();
    const table = installKeplerComputeTableBridge(new SourceTable({id: 'source'}), binding);
    const polygon = {
      id: 'polygon',
      dataId: ['source'],
      type: 'polygon',
      layerId: ['points'],
      value: {geometry: {type: 'Polygon'}}
    };
    table.filterTable([polygon, {...valueFilter, inverted: true}]);
    expect(apply).not.toHaveBeenCalled();
    expect(binding.clearSpatialFilters).not.toHaveBeenCalled();
    table.filterTable([polygon, valueFilter]);
    expect(apply).toHaveBeenCalledOnce();
    expect(table.lumaCompute.plan.spatial).toEqual([{filterId: 'polygon', layerIds: ['points']}]);
    table.filterTable([]);
    expect(binding.clearSpatialFilters).toHaveBeenCalledTimes(2);
    expect(table.lumaCompute.plan.spatial).toBeUndefined();
    expect(getKeplerComputeTableState(table)).toBe(table.lumaCompute);
    expect(getKeplerComputeTableState({})).toBeUndefined();
  });
});

describe('pinned Kepler filter plot hook', () => {
  const original = `export function updateFilterPlot(datasets, filter, dataId = undefined, layers = []) {
    datasets.source.dataContainer.rows();
    return filter;
  }
  export function restoreFilters(datasets, filter) { return updateFilterPlot(datasets, filter); }
  export function mergeFilterDomainStep(filter, filterProps) {
    const sortedDomain = !filter.domain ? filterProps.domain : [...filter.domain, ...filterProps.domain].sort((a,b) => a-b);
    const newFilter = {...filter, ...filterProps, domain: [sortedDomain[0], sortedDomain[sortedDomain.length-1]]};
    switch (filterProps.fieldType) { default: return newFilter; }
  }`;

  test('intercepts same-module restore calls before CPU scans and leaves other tables unchanged', async () => {
    const patched = patchKeplerComputeFilterSource(original);
    expect(patchKeplerComputeFilterSource(patched)).toBe(patched);
    const sourceModule = await import(`data:text/javascript,${encodeURIComponent(patched)}`);
    const table = installKeplerComputeTableBridge(new SourceTable({id: 'source'}), makeBinding());
    const result = sourceModule.restoreFilters({source: table}, valueFilter);
    expect(result.lumaComputePlot.status).toBe('pending');
    expect(table.dataContainer.rows).not.toHaveBeenCalled();
    expect(() =>
      sourceModule.restoreFilters({source: new SourceTable({id: 'source'})}, valueFilter)
    ).toThrow('Unexpected source rows');
  });

  test('preserves numeric resident category labels during native restore domain merging', async () => {
    const sourceModule = await import(
      `data:text/javascript,${encodeURIComponent(patchKeplerComputeFilterSource(original))}`
    );
    const binding = makeBinding();
    binding.fields = {
      ...binding.fields,
      category: {dimension: 'category', kind: 'category', domain: [0, 1, 2, 3]}
    };
    const source = new SourceTable({id: 'source'});
    source.fields[1].type = 'integer';
    const table = installKeplerComputeTableBridge(source, binding);
    const props = table.getColumnFilterProps('category');
    expect(sourceModule.mergeFilterDomainStep({}, props).domain).toEqual([0, 1, 2, 3]);
    expect(sourceModule.mergeFilterDomainStep({domain: [0, 1]}, props).domain).toEqual([
      0, 1, 2, 3
    ]);
    expect(
      sourceModule.mergeFilterDomainStep({}, {fieldType: 'integer', domain: [0, 1, 2, 3]}).domain
    ).toEqual([0, 3]);
  });

  test('linked filters keep per-dataset resident bins, compute CPU datasets natively and never sum', async () => {
    const nativeSource = `export function updateFilterPlot(datasets, filter, dataId = undefined, layers = []) {
      const bins = {...filter.bins};
      for (const id of filter.dataId) if (!bins[id]) bins[id] = ['native-' + id];
      return {...filter, bins, nativePass: true};
    }
    export function mergeFilterDomainStep(filter, filterProps) {
      const newFilter = {...filter, ...filterProps};
      switch (filterProps.fieldType) { default: return newFilter; }
    }`;
    const sourceModule = await import(
      `data:text/javascript,${encodeURIComponent(patchKeplerComputeFilterSource(nativeSource))}`
    );
    const linked: KeplerComputeFilter = {
      ...valueFilter,
      id: 'linked',
      dataId: ['source', 'other', 'cpu'],
      name: ['value', 'value', 'value'],
      domain: [0, 200]
    };
    const source = installKeplerComputeTableBridge(new SourceTable({id: 'source'}), makeBinding());
    const other = installKeplerComputeTableBridge(new SourceTable({id: 'other'}), makeBinding());
    const cpu = new SourceTable({id: 'cpu'});
    source.filterTable([linked]);
    other.filterTable([linked]);
    const bins = [{x0: 0, x1: 100, count: 5, indexes: []}];
    expect(source.lumaCompute.setHistogram('value', bins, source.lumaCompute.revision)).toBe(true);
    const result = sourceModule.updateFilterPlot({source, other, cpu}, linked);
    expect(result.nativePass).toBe(true);
    // The pending resident entry is cleared after native code was prevented from filling it.
    expect(result.bins).toEqual({source: bins, other: undefined, cpu: ['native-cpu']});
    expect(result.lumaComputePlot.status).toBe('pending');
    expect(source.dataContainer.rows).not.toHaveBeenCalled();
    expect(other.lumaCompute.setHistogram('value', bins, other.lumaCompute.revision)).toBe(true);
    expect(sourceModule.updateFilterPlot({source, other, cpu}, linked)).toMatchObject({
      bins: {source: bins, other: bins, cpu: ['native-cpu']},
      lumaComputePlot: {status: 'ready'}
    });
    // All-resident filters complete without a native pass.
    const resident = sourceModule.updateFilterPlot(
      {source, other},
      {...linked, dataId: ['source', 'other'], name: ['value', 'value']}
    );
    expect(resident.nativePass).toBeUndefined();
    expect(resident.bins).toEqual({source: bins, other: bins});
    // Callers outside the seam cannot obtain a partial linked plot.
    expect(source.lumaCompute.updateFilterPlot(linked)).toMatchObject({
      lumaComputePlot: {status: 'unsupported'}
    });
    expect(
      updateKeplerComputeFilterPlot({source, other, cpu}, linked, filter => ({
        ...filter,
        id: 'seam'
      }))
    ).toMatchObject({id: 'seam'});
    expect(updateKeplerComputeFilterPlot({source, other, cpu}, linked)).toBeUndefined();
  });

  test('linked time plots share the native default plot type, keep line charts and never retain a stale pending status', async () => {
    const nativeSource = `function getDefaultPlotType(filter) {
      return {interval: '1-day', type: 'histogram', aggregation: 'average'};
    }
    function getLineChart(datasets, filter) {
      return {series: filter.timeBins[filter.dataId[0]][filter.plotType.interval].map(bin => bin.count)};
    }
    export function updateFilterPlot(datasets, filter, dataId = undefined, layers = []) {
      const timeBins = {...filter.timeBins};
      for (const id of filter.dataId)
        if (!timeBins[id]?.[filter.plotType.interval]) timeBins[id] = {[filter.plotType.interval]: ['native-' + id]};
      return {...filter, timeBins, nativePass: true};
    }
    export function mergeFilterDomainStep(filter, filterProps) {
      const newFilter = {...filter, ...filterProps};
      switch (filterProps.fieldType) { default: return newFilter; }
    }`;
    const sourceModule = await import(
      `data:text/javascript,${encodeURIComponent(patchKeplerComputeFilterSource(nativeSource))}`
    );
    const makeTable = (id: string) => {
      const binding = makeBinding();
      binding.fields = {
        ...binding.fields,
        time: {...binding.fields['time'], histogramInterval: '1-hour'}
      };
      binding.filterScopedHistograms = true;
      binding.getTimeHistogramInterval = (_field, filter) => filter?.plotType?.interval ?? '1-hour';
      return installKeplerComputeTableBridge(new SourceTable({id}), binding);
    };
    const source = makeTable('source');
    const other = makeTable('other');
    const cpu = new SourceTable({id: 'cpu'});
    const linked: KeplerComputeFilter = {
      ...valueFilter,
      id: 'linked-time',
      dataId: ['source', 'other', 'cpu'],
      name: ['time', 'time', 'time'],
      type: 'timeRange',
      value: [0, 86400000 * 3],
      domain: [0, 86400000 * 3],
      // Resident-created time filters carry the resident default interval and no aggregation.
      plotType: {type: 'histogram', interval: '1-hour'}
    };
    source.filterTable([linked]);
    other.filterTable([linked]);
    const day = 86400000;
    const bins = [{x0: 0, x1: day, count: 4, indexes: []}];
    source.lumaCompute.setFilterHistogram(
      'linked-time',
      bins,
      source.lumaCompute.revision,
      '1-day'
    );
    other.lumaCompute.setFilterHistogram('linked-time', bins, other.lumaCompute.revision, '1-day');
    const datasets = {source, other, cpu};
    const result = sourceModule.updateFilterPlot(datasets, linked);
    // Native's default plot type replaces the resident-only interval for every dataset.
    expect(result.plotType).toMatchObject({interval: '1-day', aggregation: 'average'});
    expect(result.timeBins).toEqual({
      source: {'1-day': bins},
      other: {'1-day': bins},
      cpu: {'1-day': ['native-cpu']}
    });
    expect(result.lumaComputePlot.status).toBe('ready');
    // A previous pending status on the stored filter cannot leak into the next composition.
    const stale = sourceModule.updateFilterPlot(datasets, {
      ...result,
      lumaComputePlot: {status: 'pending', revision: 1}
    });
    expect(stale.lumaComputePlot.status).toBe('ready');
    // The user's line chart type is preserved and drawn from the first dataset's resident bins.
    const line = sourceModule.updateFilterPlot(datasets, {
      ...result,
      plotType: {...result.plotType, type: 'lineChart'}
    });
    expect(line.plotType.type).toBe('lineChart');
    expect(line.lineChart).toEqual({series: [4]});
    // Native draws only the first dataset; a CPU first dataset keeps native's own series.
    const cpuFirst = sourceModule.updateFilterPlot(datasets, {
      ...result,
      dataId: ['cpu', 'source', 'other'],
      name: ['time', 'time', 'time'],
      plotType: {...result.plotType, type: 'lineChart'}
    });
    expect(cpuFirst.lineChart).toBeUndefined();
  });

  test('resident line charts count rows only and reject unrepresentable native plot options', () => {
    const table = installKeplerComputeTableBridge(new SourceTable({id: 'source'}), makeBinding());
    const bins = [{x0: 0, x1: 100, count: 5, indexes: []}];
    table.filterTable([valueFilter]);
    table.lumaCompute.setHistogram('value', bins, table.lumaCompute.revision);
    const buildLineChart = vi.fn((datasets: Record<string, unknown>) => ({
      seriesDataset: Object.keys(datasets)
    }));
    const line = {...valueFilter, plotType: {type: 'lineChart', aggregation: 'average'}};
    expect(table.lumaCompute.updateFilterPlot(line, {buildLineChart})).toMatchObject({
      bins: {source: bins},
      lineChart: {seriesDataset: ['source']},
      lumaComputePlot: {status: 'ready'}
    });
    // A seam-less caller cannot obtain a chart from placeholder native code.
    expect(table.lumaCompute.updateFilterPlot(line)).toMatchObject({
      lineChart: undefined,
      lumaComputePlot: {status: 'unsupported', reason: expect.stringContaining('series builder')}
    });
    // Native y-axis measures aggregate each bin's source row array, which resident plots never expose.
    expect(
      table.lumaCompute.updateFilterPlot({...line, yAxis: {name: 'value'}}, {buildLineChart})
    ).toMatchObject({lineChart: undefined, lumaComputePlot: {status: 'unsupported'}});
    expect(
      table.lumaCompute.updateFilterPlot(
        {...valueFilter, plotType: {type: 'histogram', numericInterval: 5}},
        {buildLineChart}
      )
    ).toMatchObject({bins: undefined, lumaComputePlot: {status: 'unsupported'}});
    expect(
      table.lumaCompute.updateFilterPlot(
        {...valueFilter, plotType: {type: 'scatter'}},
        {buildLineChart}
      )
    ).toMatchObject({lumaComputePlot: {status: 'unsupported'}});
  });

  test('time line charts plot the GPU-reduced y axis of the first dataset and gap empty bins', () => {
    const binding = makeBinding();
    binding.fields = {
      ...binding.fields,
      time: {...binding.fields['time'], histogramInterval: '1-hour'}
    };
    binding.filterScopedHistograms = true;
    binding.getTimeHistogramInterval = (_field, filter) => filter?.plotType?.interval ?? '1-hour';
    const table = installKeplerComputeTableBridge(new SourceTable({id: 'source'}), binding);
    const filter: KeplerComputeFilter = {
      ...valueFilter,
      id: 'line',
      name: ['time'],
      type: 'timeRange',
      value: [0, 7200000],
      yAxis: {name: 'value'},
      plotType: {type: 'lineChart', interval: '1-hour', aggregation: 'average'}
    };
    table.filterTable([filter]);
    const bins = [
      {x0: 0, x1: 3600000, count: 3, indexes: [] as never[], measure: 2.5},
      {x0: 3600000, x1: 7200000, count: 2, indexes: [] as never[], measure: null}
    ];
    table.lumaCompute.setFilterHistogram('line', bins, table.lumaCompute.revision, '1-hour');
    const buildLineChart = vi.fn((datasets: Record<string, unknown>, built: unknown) => ({
      datasets,
      y: (built as {timeBins: Record<string, Record<string, {count?: number}[]>>}).timeBins[
        'source'
      ]['1-hour'].map(bin => bin.count)
    }));
    const result = table.lumaCompute.updateFilterPlot(filter, {buildLineChart});
    // The stub dataset has no y field, so the native builder reads each bin's measure as its value.
    expect(result.lineChart).toEqual({
      datasets: {source: {id: 'source', fields: []}},
      y: [2.5, undefined]
    });
    // The histogram bins themselves keep their row counts.
    expect(result.timeBins).toMatchObject({source: {'1-hour': [{count: 3}, {count: 2}]}});
    expect(
      table.lumaCompute.updateFilterPlot(
        {...filter, plotType: {...filter.plotType, groupBy: {fieldName: 'category'}}},
        {buildLineChart}
      )
    ).toMatchObject({lineChart: undefined, lumaComputePlot: {status: 'unsupported'}});
    // Linked: only the first dataset draws the chart; others plot histogram counts.
    const linked = {...filter, dataId: ['other', 'source'], name: ['time', 'time']};
    const second = table.lumaCompute.updateFilterPlot(linked, {buildLineChart, stage: 'finalize'});
    expect(second.lineChart).toBeUndefined();
  });

  test('numeric range line charts plot GPU-reduced y measures and certify native edges', () => {
    const binding = makeBinding();
    binding.filterScopedHistograms = true;
    const prepareNumericPlot = vi.fn();
    binding.prepareNumericPlot = prepareNumericPlot;
    const table = installKeplerComputeTableBridge(new SourceTable({id: 'source'}), binding);
    const filter: KeplerComputeFilter = {
      ...valueFilter,
      id: 'numeric-line',
      yAxis: {name: 'value'},
      plotType: {type: 'lineChart', aggregation: 'median', numericInterval: 25}
    };
    table.filterTable([filter]);
    const bins = [
      {x0: 0, x1: 25, count: 3, indexes: [] as never[], measure: 2.5},
      {x0: 25, x1: 50, count: 2, indexes: [] as never[], measure: null}
    ];
    table.lumaCompute.setFilterHistogram('numeric-line', bins, table.lumaCompute.revision);
    const buildLineChart = vi.fn((_datasets: Record<string, unknown>, built: unknown) => ({
      y: (built as {bins: Record<string, {count?: number}[]>}).bins['source'].map(bin => bin.count)
    }));
    const result = table.lumaCompute.updateFilterPlot(filter, {buildLineChart});
    // Native reads numeric bins from `bins`, so the measures replace the counts there.
    expect(result.lineChart).toEqual({y: [2.5, undefined]});
    expect(prepareNumericPlot).toHaveBeenCalledWith('value', filter);
    // updateFilterPlot adds lumaComputePlot, which the KeplerComputeFilter type does not declare.
    const {lumaComputePlot} = result as {lumaComputePlot?: {status: string}};
    expect(lumaComputePlot?.status).toBe('ready');
    prepareNumericPlot.mockImplementation(() => {
      throw new Error('Resident line charts do not support mode');
    });
    expect(table.lumaCompute.updateFilterPlot(filter, {buildLineChart})).toMatchObject({
      lineChart: undefined,
      lumaComputePlot: {status: 'unsupported', reason: 'Resident line charts do not support mode'}
    });
    // Without a binding that reduces measures over native edges, numeric intervals stay unsupported.
    const plain = installKeplerComputeTableBridge(new SourceTable({id: 'plain'}), makeBinding());
    plain.filterTable([{...valueFilter, dataId: ['plain']}]);
    expect(
      plain.lumaCompute.updateFilterPlot(
        {...valueFilter, dataId: ['plain'], plotType: {type: 'histogram', numericInterval: 25}},
        {buildLineChart}
      )
    ).toMatchObject({lumaComputePlot: {status: 'unsupported'}});
  });

  test('grouped line charts rebuild native series from per-series GPU values', () => {
    const binding = makeBinding();
    binding.fields = {
      ...binding.fields,
      time: {...binding.fields['time'], histogramInterval: '1-hour'}
    };
    binding.filterScopedHistograms = true;
    binding.groupedLineCharts = true;
    binding.getTimeHistogramInterval = (_field, filter) => filter?.plotType?.interval ?? '1-hour';
    const table = installKeplerComputeTableBridge(new SourceTable({id: 'source'}), binding);
    const filter: KeplerComputeFilter = {
      ...valueFilter,
      id: 'grouped',
      name: ['time'],
      type: 'timeRange',
      value: [0, 7200000],
      yAxis: {name: 'value'},
      plotType: {
        type: 'lineChart',
        interval: '1-hour',
        aggregation: 'stdev',
        groupBy: {fieldName: 'category', numGroups: 2, groupOthers: true}
      }
    };
    table.filterTable([filter]);
    const common = {groupLabels: ['b', 'a'], groupOthers: true, indexes: [] as never[]};
    const bins = [
      {x0: 0, x1: 3600000, count: 4, ...common, groups: [{slot: 1, count: 3, measure: 4}]},
      {
        x0: 3600000,
        x1: 7200000,
        count: 5,
        ...common,
        groups: [
          {slot: 0, count: 2, measure: null},
          {slot: 2, count: 3, measure: 7}
        ]
      }
    ];
    table.lumaCompute.setFilterHistogram('grouped', bins, table.lumaCompute.revision, '1-hour');
    const calls: {datasets: Record<string, unknown>; built: Record<string, unknown>}[] = [];
    const buildLineChart = (datasets: Record<string, unknown>, built: unknown) => {
      calls.push({datasets, built: built as Record<string, unknown>});
      return {
        title: calls.length === 1 ? 'pseudo' : 'reference',
        aggregation: 'average',
        yAxis: 'value'
      };
    };
    const result = table.lumaCompute.updateFilterPlot(filter, {buildLineChart});
    expect(calls).toHaveLength(2);
    const pseudo = calls[0].built as {
      yAxis: {name: string};
      plotType: {aggregation: string};
      timeBins: {source: Record<string, {indexes: number[]; count: number}[]>};
    };
    // Pseudo rows are always averaged; the title and aggregation come from the reference build.
    expect(pseudo.plotType.aggregation).toBe('average');
    expect(pseudo.yAxis.name).toBe('value');
    const pseudoBins = pseudo.timeBins.source['1-hour'];
    // The first bin lists every series in first-seen order (b, a, merged Others) to fix discovery order.
    expect(pseudoBins[0].indexes).toEqual([0, 1, 2]);
    expect(pseudoBins[1].indexes).toEqual([3, 4]);
    const dataset = calls[0].datasets['source'] as {
      fields: {name: string; valueAccessor: (row: {index: number}) => unknown}[];
    };
    const group = dataset.fields.find(field => field.name === 'category')!;
    const measure = dataset.fields.find(field => field.name === 'value')!;
    const read = (field: typeof group, indexes: number[]) =>
      indexes.map(index => field.valueAccessor({index}));
    expect(read(group, [0, 1, 2, 3, 4])).toEqual([
      'b',
      'a',
      '\u0000luma-others',
      'b',
      '\u0000luma-others'
    ]);
    expect(read(measure, [0, 1, 2, 3, 4])).toEqual([null, 4, null, null, 7]);
    expect(result.lineChart).toMatchObject({title: 'reference', aggregation: 'average', bins});
    // Without a y axis the binding supplies each series' row count as its value.
    calls.length = 0;
    const countBins = bins.map(bin => ({
      ...bin,
      groups: bin.groups.map(entry => ({...entry, measure: entry.count}))
    }));
    table.lumaCompute.setFilterHistogram(
      'grouped',
      countBins,
      table.lumaCompute.revision,
      '1-hour'
    );
    table.lumaCompute.updateFilterPlot({...filter, yAxis: undefined}, {buildLineChart});
    const countField = (calls[0].datasets['source'] as typeof dataset).fields[1];
    expect(countField.name).toBe('\u0000luma-count');
    expect(read(countField, [0, 1, 2, 3, 4])).toEqual([null, 3, null, 2, 3]);
  });

  test('refuses an unaudited category merge seam', () => {
    expect(() =>
      patchKeplerComputeFilterSource(
        original.replace('switch (filterProps.fieldType)', 'switch (filterProps.type)')
      )
    ).toThrow('category-domain signature changed');
  });

  test('refuses an unaudited upstream signature rather than silently dropping the hook', () => {
    expect(() =>
      patchKeplerComputeFilterSource('export const updateFilterPlot = () => {};')
    ).toThrow('signature changed');
  });

  test('supports the pinned package CommonJS signature as well as its ESM entry', () => {
    const compiled = original.replaceAll('export ', '');
    expect(patchKeplerComputeFilterSource(compiled)).toContain(
      '.lumaCompute.updateFilterPlot(filter, residentPlotOptions)'
    );
  });

  test('resolves native plot helpers from the ESM alias or the CommonJS namespace and detects drift', () => {
    const esm = `import { mergePolygonLayerIndexes, updateTimeFilterPlotType, updateRangeFilterPlotType } from './plot';
      export function updateFilterPlot(datasets, filter) { return filter; }
      export function mergeFilterDomainStep(filter, filterProps) { switch (filterProps.fieldType) { default: return filter; } }`;
    const patchedEsm = patchKeplerComputeFilterSource(esm);
    expect(patchedEsm).toContain('getLineChart as __lumaGetLineChart');
    expect(patchedEsm).toContain('buildLineChart: __lumaGetLineChart');
    const commonJs = `var _plot = require("./plot");
      function updateFilterPlot(datasets, filter) { return filter; }
      function mergeFilterDomainStep(filter, filterProps) { switch (filterProps.fieldType) { default: return filter; } }`;
    expect(patchKeplerComputeFilterSource(commonJs)).toContain(
      'buildLineChart: _plot.getLineChart'
    );
    expect(() =>
      patchKeplerComputeFilterSource(esm.replace('updateRangeFilterPlotType }', 'other }'))
    ).toThrow('plot import changed');
  });
});
