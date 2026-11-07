// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Plugin} from 'vite';

const FILTER_PLOT_HOOK = '/* luma-resident-filter-plot */';
const CATEGORY_DOMAIN_HOOK = '/* luma-resident-category-domain */';
const PLOT_REQUIRE = 'var _plot = require("./plot");';
const PLOT_IMPORT =
  "import { mergePolygonLayerIndexes, updateTimeFilterPlotType, updateRangeFilterPlotType } from './plot';";

/**
 * Patch the pinned consumer seam, including calls from inside filter-utils itself. An exported
 * wrapper cannot intercept those lexical calls. The original CPU path remains intact for other tables.
 */
export function patchKeplerComputeFilterSource(source: string): string {
  if (source.includes(FILTER_PLOT_HOOK) && source.includes(CATEGORY_DOMAIN_HOOK)) return source;
  const declaration = /\b(?:export\s+)?function updateFilterPlot\([^]*?\)\s*\{/g;
  const matches = [...source.matchAll(declaration)];
  if (matches.length !== 1) {
    throw new Error(
      'Pinned Kepler updateFilterPlot signature changed; audit the resident plot hook'
    );
  }
  const match = matches[0];
  const insertion = match.index! + match[0].length;
  const hook = `
  ${FILTER_PLOT_HOOK}
  if (!__lumaResidentPlotNative) {
    const residentDatasetIds = Array.isArray(filter.dataId) ? filter.dataId : [filter.dataId];
    const residentTableIds = residentDatasetIds.filter(
      residentDatasetId => datasets[residentDatasetId] && datasets[residentDatasetId].lumaCompute
    );
    const residentPlotOptions = {buildLineChart: LUMA_GET_LINE_CHART};
    if (residentTableIds.length && residentDatasetIds.length === 1) {
      return datasets[residentTableIds[0]].lumaCompute.updateFilterPlot(filter, residentPlotOptions);
    }
    if (residentTableIds.length) {
      // Native plots are keyed by dataset id: resident tables write their own entries, native code
      // computes any remaining CPU datasets around them, then pending placeholders are cleared.
      // Each composition starts from a clean status; a stale pending status must never persist.
      let linkedFilter = {...filter, lumaComputePlot: undefined};
      const linkedPlotType = filter.plotType;
      if (
        filter.type === 'timeRange' &&
        !(linkedPlotType && linkedPlotType.aggregation && linkedPlotType.interval)
      ) {
        // Native replaces an incomplete time plot type by one chosen from the merged domain (and any
        // CPU timestamps). Resolve it once so every resident table and CPU dataset share it.
        linkedFilter = {...linkedFilter, plotType: LUMA_GET_DEFAULT_PLOT_TYPE(filter, datasets)};
      }
      for (const residentTableId of residentTableIds) {
        linkedFilter = datasets[residentTableId].lumaCompute.updateFilterPlot(linkedFilter, {
          ...residentPlotOptions,
          stage: 'prepare'
        });
      }
      const linkedStatus = linkedFilter.lumaComputePlot && linkedFilter.lumaComputePlot.status;
      if (
        residentTableIds.length < residentDatasetIds.length &&
        linkedStatus !== 'unsupported' &&
        linkedStatus !== 'disabled'
      ) {
        __lumaResidentPlotNative = true;
        try {
          linkedFilter = updateFilterPlot(datasets, linkedFilter, arguments[2], arguments[3]);
        } finally {
          __lumaResidentPlotNative = false;
        }
      }
      for (const residentTableId of residentTableIds) {
        linkedFilter = datasets[residentTableId].lumaCompute.updateFilterPlot(linkedFilter, {
          ...residentPlotOptions,
          stage: 'finalize'
        });
      }
      return linkedFilter;
    }
  }
`;
  // Native plot helpers come from the module's own plot import (ESM alias or CommonJS namespace).
  let plotHelpers: {getLineChart: string; getDefaultPlotType: string};
  let importAlias = '';
  if (source.includes(PLOT_IMPORT)) {
    plotHelpers = {
      getLineChart: '__lumaGetLineChart',
      getDefaultPlotType: '__lumaGetDefaultPlotType'
    };
    importAlias = PLOT_IMPORT.replace(
      'updateRangeFilterPlotType }',
      'updateRangeFilterPlotType, getDefaultPlotType as __lumaGetDefaultPlotType, getLineChart as __lumaGetLineChart }'
    );
  } else if (source.includes(PLOT_REQUIRE)) {
    plotHelpers = {
      getLineChart: '_plot.getLineChart',
      getDefaultPlotType: '_plot.getDefaultPlotType'
    };
  } else if (/['"]\.\/plot['"]/.test(source)) {
    throw new Error('Pinned Kepler plot import changed; audit the resident plot hook');
  } else {
    // Self-contained modules (tests) declare their own helpers.
    plotHelpers = {
      getLineChart: "(typeof getLineChart === 'function' ? getLineChart : undefined)",
      getDefaultPlotType:
        "(typeof getDefaultPlotType === 'function' ? getDefaultPlotType : undefined)"
    };
  }
  const resolvedHook = hook
    .replaceAll('LUMA_GET_LINE_CHART', plotHelpers.getLineChart)
    .replaceAll('LUMA_GET_DEFAULT_PLOT_TYPE', plotHelpers.getDefaultPlotType);
  const patched =
    'var __lumaResidentPlotNative = false;\n' +
    source.slice(0, insertion) +
    resolvedHook +
    source.slice(insertion);
  // Kepler chooses categorical merging from fieldType, even when a numeric dictionary has an
  // explicitly categorical filter. Preserve its complete label domain only for resident metadata.
  const categorySeam = /\bswitch\s*\(\s*filterProps\.fieldType\s*\)\s*\{/g;
  const categoryMatches = [...patched.matchAll(categorySeam)];
  if (categoryMatches.length !== 1) {
    throw new Error(
      'Pinned Kepler category-domain signature changed; audit the resident domain hook'
    );
  }
  const categoryInsertion = categoryMatches[0].index!;
  const categoryHook = `
  ${CATEGORY_DOMAIN_HOOK}
  if (filterProps.lumaComputeDomain === 'category') {
    return {...newFilter, domain: Array.from(new Set(sortedDomain))};
  }
  `;
  const result =
    patched.slice(0, categoryInsertion) + categoryHook + patched.slice(categoryInsertion);
  return importAlias ? result.replace(PLOT_IMPORT, importAlias) : result;
}

export function createKeplerComputeFilterPlugin(): Plugin {
  return {
    name: 'kepler-resident-filter-plot',
    enforce: 'pre',
    transform(source, id) {
      const normalized = id.replaceAll('\\', '/').split('?')[0];
      if (!normalized.includes('/@kepler.gl/utils/') || !normalized.endsWith('/filter-utils.js'))
        return null;
      return {code: patchKeplerComputeFilterSource(source), map: null};
    }
  };
}
