// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright contributors to the kepler.gl project

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

const DECK_CORE_IMPORT = /import\s*\{([^}]*)\}\s*from\s*(['"])@deck\.gl\/core\2;?/g;
const COORDINATE_SYSTEM_VALUES =
  "Object.freeze({DEFAULT: 'default', LNGLAT: 'lnglat', METER_OFFSETS: 'meter-offsets', " +
  "LNGLAT_OFFSETS: 'lnglat-offsets', CARTESIAN: 'cartesian'})";

/**
 * Kepler 3.3 and its dependencies import `COORDINATE_SYSTEM` from `@deck.gl/core`; deck.gl 10 removed
 * that export and takes the coordinate system as a string. Drop the named import and declare the
 * constant locally with the same member names and the string values deck.gl 10 accepts. A bundler
 * alias would create a second `@deck.gl/core` instance next to the pre-bundled one, so this rewrites
 * the importing module instead (it runs both in the Vite dev optimizer and in `vite build`).
 */
export function patchDeckCoordinateSystemImport(source: string): string {
  if (!source.includes('COORDINATE_SYSTEM')) return source;
  return source.replace(DECK_CORE_IMPORT, (statement, specifierList: string, quote: string) => {
    const specifiers = specifierList
      .split(',')
      .map(specifier => specifier.trim())
      .filter(Boolean);
    const coordinateSystem = specifiers.find(
      specifier => specifier === 'COORDINATE_SYSTEM' || specifier.startsWith('COORDINATE_SYSTEM ')
    );
    if (!coordinateSystem) return statement;
    const localName = coordinateSystem.split(/\s+as\s+/)[1] ?? 'COORDINATE_SYSTEM';
    const remaining = specifiers.filter(specifier => specifier !== coordinateSystem);
    const declaration = `const ${localName} = ${COORDINATE_SYSTEM_VALUES};`;
    return remaining.length
      ? `import { ${remaining.join(', ')} } from ${quote}@deck.gl/core${quote}; ${declaration}`
      : declaration;
  });
}

const EDITABLE_PATH_ANCHOR = "'if (path.billboard != 0.0) {'";
const EDITABLE_PATH_ANCHOR_IN_VERTEX_MAIN = "'if (path.billboard != 0.0) {\\n#ifdef DASH_ENABLED'";

/**
 * `@deck.gl-community/editable-layers` 10.0.0-alpha.1 widens picking strokes by inserting WGSL
 * before the first `if (path.billboard != 0.0) {` of deck.gl 10.0.0-alpha.1's `path.wgsl`. That
 * text first occurs inside `getLineJoinOffset`, where `widthPixels` does not exist, so every
 * editor path shader fails to compile. The vertex entry point's occurrence is the only one
 * followed by `#ifdef DASH_ENABLED`; anchor on that one.
 */
export function patchEditablePathLayerSource(source: string): string {
  return source.includes(EDITABLE_PATH_ANCHOR)
    ? source.replace(EDITABLE_PATH_ANCHOR, EDITABLE_PATH_ANCHOR_IN_VERTEX_MAIN)
    : source;
}

const GLOBE_LABEL_CULL_WGSL = /* wgsl */ `
  if (project.projectionMode == PROJECTION_MODE_GLOBE) {
    let anchorCommon = geometry.position.xyz;
    let radius = length(anchorCommon);
    let cameraDistance = length(project.cameraPosition);
    if (radius > 0.0 && cameraDistance > radius) {
      let cosineSurface = dot(anchorCommon / radius, project.cameraPosition / cameraDistance);
      let cosineHorizon = radius / cameraDistance + 0.02;
      if (cosineSurface < cosineHorizon) {
        *position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
      }
    }
  }
`;

/** Kepler's glyph/background helper injects GLSL into deck 10's WGSL as well as its GLSL. */
export function patchKeplerGlobeBackfaceCullSource(source: string): string {
  if (source.includes('kepler-globe-label-horizon')) return source;
  const declaration = /export function injectGlobeBackfaceCull\(shaders\) \{[^]*?\n\}/;
  if (!declaration.test(source) || !source.includes('var BACKFACE_CULL_INJECT ='))
    throw new Error('Pinned Kepler globe backface cull helper changed; audit the shader hook');
  // Public luma ShaderPlugin injections select the language at assembly time. Keep all existing
  // base injections/plugins, the original GLSL and the exact original horizon margin/guard.
  // Move the inherited position hook into the appended plugin, retaining its order. Assembly
  // collects plugins before legacy inject maps, so order Infinity alone would still put culling
  // before an inherited Infinity hook. With both in the plugin, stable ties keep culling last.
  return source.replace(
    declaration,
    `export function injectGlobeBackfaceCull(shaders) {
  var positionHook = (shaders.inject || {})['vs:DECKGL_FILTER_GL_POSITION'];
  var inheritedPosition = typeof positionHook === 'string' ? positionHook : (positionHook && positionHook.injection) || '';
  var remainingInject = {...(shaders.inject || {})};
  delete remainingInject['vs:DECKGL_FILTER_GL_POSITION'];
  return {...shaders, inject: remainingInject, plugins: [...(shaders.plugins || []), {
    name: 'kepler-globe-label-horizon',
    injections: [...(inheritedPosition ? [{stage: 'vertex', target: 'DECKGL_FILTER_GL_POSITION',
      glsl: inheritedPosition, wgsl: inheritedPosition,
      order: typeof positionHook === 'object' ? (positionHook.order ?? 0) : 0}] : []),
      {stage: 'vertex', target: 'DECKGL_FILTER_GL_POSITION', order: Number.POSITIVE_INFINITY,
      glsl: BACKFACE_CULL_INJECT, wgsl: ${JSON.stringify(GLOBE_LABEL_CULL_WGSL)}}]
  }]};
}`
  );
}

/**
 * loaders.gl's 3D tiles source assumes raw glTF {json}; deck also accepts postprocessed glTF.
 * Read both metadata shapes without rejecting a successfully loaded flat GLB during onTileLoaded.
 */
export function patchTiles3DRemovedExtensionsSource(source: string): string {
  return source.replaceAll(
    'content?.gltf?.json.extensionsRemoved',
    '(content?.gltf?.json?.extensionsRemoved ?? content?.gltf?.extensionsRemoved)'
  );
}

export function createKeplerComputeDeckCompatPlugin(): Plugin {
  return {
    name: 'kepler-deck-10-coordinate-system',
    enforce: 'pre',
    transform(source, id) {
      const normalized = id.replaceAll('\\', '/').split('?')[0];
      if (!normalized.includes('/node_modules/') || !/\.(?:m?js)$/.test(normalized)) return null;
      const compatible = normalized.endsWith('/deckgl-layers/dist/esm/globe/globe-backface-cull.js')
        ? patchKeplerGlobeBackfaceCullSource(source)
        : source;
      const patchedConsumer = patchDeckCoordinateSystemImport(
        normalized.endsWith('/editable-layers/dist/editable-layers/editable-path-layer.js')
          ? patchEditablePathLayerSource(source)
          : compatible
      );
      const patched =
        normalized.includes('/@loaders.gl/tiles-3d/') && normalized.endsWith('/tiles-3d-source.js')
          ? patchTiles3DRemovedExtensionsSource(patchedConsumer)
          : patchedConsumer;
      return patched === source ? null : {code: patched, map: null};
    }
  };
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
