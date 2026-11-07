// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {FEATURED_LENS_SCENARIOS} from '../../lens/featured';

export type TopicSlug =
  | 'anatomy'
  | 'command-graph'
  | 'scheduling'
  | 'kernels'
  | 'data-on-the-gpu'
  | 'lenses'
  | 'this-page'
  | 'upstream-first';

export type Topic = {
  slug: TopicSlug;
  title: string;
  /** One sentence, shown under the title and on the index card. */
  lead: string;
  /** Lens scenarios the page draws; its route loader loads them. */
  scenarios: readonly string[];
};

export const TOPICS: readonly Topic[] = [
  {
    slug: 'anatomy',
    title: 'Anatomy of a run',
    lead: 'One call becomes a graph of GPU kernels; follow it from a table to a readback.',
    scenarios: [FEATURED_LENS_SCENARIOS.kernelDensity]
  },
  {
    slug: 'command-graph',
    title: 'The command graph',
    lead: 'Kernels declare what they read and write; the compiler orders them from that alone.',
    scenarios: [FEATURED_LENS_SCENARIOS.localMoran]
  },
  {
    slug: 'scheduling',
    title: 'Scheduling',
    lead: 'Step through a recorded run: kernels light up, buffers live and die, hazards become edges.',
    scenarios: [FEATURED_LENS_SCENARIOS.kernelDensity]
  },
  {
    slug: 'kernels',
    title: 'Kernels',
    lead: 'One shader, thousands of invocations: set the workgroup size and watch the data re-tile.',
    scenarios: []
  },
  {
    slug: 'data-on-the-gpu',
    title: 'Data on the GPU',
    lead: 'A column is a list of buffers, one per batch, and a format says what the bytes mean.',
    scenarios: [FEATURED_LENS_SCENARIOS.kernelDensity]
  },
  {
    slug: 'lenses',
    title: 'Lenses',
    lead: '409 of 438 function pages show a run recorded on a real GPU: graph, formula and the data it produced.',
    scenarios: [
      FEATURED_LENS_SCENARIOS.kernelDensity,
      FEATURED_LENS_SCENARIOS.localMoran,
      FEATURED_LENS_SCENARIOS.serviceAreas
    ]
  },
  {
    slug: 'this-page',
    title: 'This page',
    lead: 'The canvas is drawn by luma.gl right now; these are its real numbers.',
    scenarios: []
  },
  {
    slug: 'upstream-first',
    title: 'Upstream first',
    lead: 'Gridlight adds no GPU runtime; it stacks on published luma.gl.',
    scenarios: []
  }
];

export function getTopic(slug: string): Topic | undefined {
  return TOPICS.find(topic => topic.slug === slug);
}
