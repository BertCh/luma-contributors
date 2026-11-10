// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import inventory from 'virtual:demo-data';
import {getDemo} from '../demos';

export type DemoDataSource = {
  id: string;
  label: string;
  url: string;
  credit: string;
  terms: string;
  termsUrl?: string;
};

export type DemoData = {
  sources: readonly DemoDataSource[];
  scenario?: string;
  caption?: string;
};

/** The source list follows the same flagship-before-run precedence as the scene loader. */
export function getDemoData(entryId: string): DemoData {
  const demo = getDemo(entryId);
  const entry = inventory.entries[demo ? `demos/${demo.id}` : `runs/${entryId}`];
  if (!entry) return {sources: [], scenario: 'Illustrative scene.'};
  return {
    sources: entry.sources.map(id => inventory.sources[id]),
    scenario: entry.scenario,
    caption: demo?.caption ?? entry.caption
  };
}

export function listDemoDataSources(): readonly DemoDataSource[] {
  return Object.values(inventory.sources).sort((first, second) =>
    first.label.localeCompare(second.label)
  );
}
