// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type AtlasRoute = {
  title: string;
  previewId: string;
  stops: readonly (readonly [sceneId: string, purpose: string])[];
};

export const ATLAS_ROUTES = [
  {
    title: 'Can the transit network absorb a disruption?',
    previewId: 'transit-frequency',
    stops: [
      ['transit-frequency', 'Map service frequency'],
      ['transit-reachability', 'Measure reachable places'],
      ['rail-resilience', 'Test a network disruption']
    ]
  },
  {
    title: 'What changes after a fire?',
    previewId: 'burn-severity',
    stops: [
      ['composite-evidence', 'Audit cloud-screened observations'],
      ['burn-severity', 'Classify burn severity'],
      ['vegetation-trends', 'Track vegetation response']
    ]
  },
  {
    title: 'How does a point pattern depend on its geography?',
    previewId: 'h3-aggregation',
    stops: [
      ['point-patterns', 'Inspect point clustering'],
      ['h3-aggregation', 'Aggregate to a global grid'],
      ['spherical-observation-support', 'Change the observation support']
    ]
  },
  {
    title: 'How does water move across terrain?',
    previewId: 'watersheds',
    stops: [
      ['watersheds', 'Trace contributing areas'],
      ['terrain-conditioning', 'Test how conditioning changes drainage'],
      ['flood-lab', 'Test a conditional flood scenario']
    ]
  },
  {
    title: 'Will the model travel to another place?',
    previewId: 'rainfall-interpolation',
    stops: [
      ['health-regression', 'Fit regional health associations'],
      ['rainfall-interpolation', 'Interpolate a measured surface'],
      ['meuse-validation', 'Test held-out soil predictions']
    ]
  },
  {
    title: 'When does a measured difference become evidence of change?',
    previewId: '',
    stops: [
      ['point-cloud-support', 'Audit neighborhood support and retained rows'],
      ['point-cloud-change', 'Compare real surveys with explicit uncertainty'],
      ['survey-frame-sensitivity', 'Test which control area defines the correction']
    ]
  },
  {
    title: 'What survives the query?',
    previewId: '',
    stops: [
      ['osm-history-crossfilter', 'Follow rows through linked brushes'],
      ['osm-history-import-waves', 'Inspect calendar summaries'],
      ['earthquake-query-lab', 'Audit filters and excluded records']
    ]
  }
] as const satisfies readonly AtlasRoute[];

/** Finds the next investigation in the first curated route containing this story. */
export function getNextAtlasStop(
  sceneId: string
): {title: string; sceneId: string; purpose: string} | null {
  for (const route of ATLAS_ROUTES) {
    const currentIndex = route.stops.findIndex(stop => stop[0] === sceneId);
    if (currentIndex < 0) continue;
    const next = route.stops[currentIndex + 1];
    return next ? {title: route.title, sceneId: next[0], purpose: next[1]} : null;
  }
  return null;
}
