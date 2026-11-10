// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The how-it-works pages in reading order. A page's content lives in `pages/<slug>/`; its data.ts
// names the lens scenarios and code examples the route loader prepares.

import {pageData as anatomy} from './pages/anatomy/data';
import {pageData as buildingBlocks} from './pages/building-blocks/data';
import {pageData as commandGraph} from './pages/command-graph/data';
import {pageData as contract} from './pages/contract/data';
import {pageData as dataOnTheGpu} from './pages/data-on-the-gpu/data';
import {pageData as glossary} from './pages/glossary/data';
import {pageData as iteration} from './pages/iteration/data';
import {pageData as kernels} from './pages/kernels/data';
import {pageData as lenses} from './pages/lenses/data';
import {pageData as scheduling} from './pages/scheduling/data';
import {pageData as thisPage} from './pages/this-page/data';
import {pageData as upstreamFirst} from './pages/upstream-first/data';
import {pageData as workflows} from './pages/workflows/data';
import {TOPIC_SLUGS} from './topic-slugs';
import type {Topic, TopicPartId, TopicSlug} from './topic-types';

export type {Topic, TopicPartId, TopicSlug} from './topic-types';

export type TopicPart = {id: TopicPartId; title: string; lead: string};

export const TOPIC_PARTS: readonly TopicPart[] = [
  {id: 'run', title: 'One run', lead: 'Follow a single call from buffers to a picture.'},
  {
    id: 'parallel',
    title: 'Parallel analysis',
    lead: 'How spatial algorithms become kernels that never loop over every row.'
  },
  {id: 'practice', title: 'In practice', lead: 'Operations chained into a whole analysis.'},
  {id: 'system', title: 'The system', lead: 'Records, packages and vocabulary.'}
];

export const TOPICS: readonly Topic[] = [
  {
    slug: 'anatomy',
    part: 'run',
    title: 'Anatomy of a run',
    lead: 'One call becomes a graph of GPU kernels. Follow kernel density from the code to the raster.',
    ...anatomy
  },
  {
    slug: 'data-on-the-gpu',
    part: 'run',
    title: 'Data on the GPU',
    lead: 'Every input and output is a typed view of a buffer. Offsets give flat arrays their shape.',
    ...dataOnTheGpu
  },
  {
    slug: 'command-graph',
    part: 'run',
    title: 'The command graph',
    lead: 'Kernels declare what they read and write. The compiler orders them from that alone.',
    ...commandGraph
  },
  {
    slug: 'scheduling',
    part: 'run',
    title: 'Scheduling',
    lead: 'Compile fixes the order once. Each frame only encodes it, and one small buffer changes the answer.',
    ...scheduling
  },
  {
    slug: 'kernels',
    part: 'run',
    title: 'Kernels',
    lead: 'One shader, thousands of invocations: how a dispatch covers the data.',
    ...kernels
  },
  {
    slug: 'building-blocks',
    part: 'parallel',
    title: 'Parallel building blocks',
    lead: 'No kernel loops over every row. Scan, scatter, sort and a grid index do the work a loop would.',
    ...buildingBlocks
  },
  {
    slug: 'iteration',
    part: 'parallel',
    title: 'Loops without readback',
    lead: 'Rounds are laid out in advance and gated on the GPU. A finished search runs zero workgroups.',
    ...iteration
  },
  {
    slug: 'contract',
    part: 'parallel',
    title: 'The contributor contract',
    lead: 'Every operation follows the same rules, so every operation composes and fails the same way.',
    ...contract
  },
  {
    slug: 'workflows',
    part: 'practice',
    title: 'Workflows',
    lead: 'A real analysis chains operations in one graph. Five of them, step by step.',
    ...workflows
  },
  {
    slug: 'lenses',
    part: 'system',
    title: 'Lenses',
    lead: 'Function pages show a run recorded on a real GPU. This is what a record holds.',
    ...lenses
  },
  {
    slug: 'this-page',
    part: 'system',
    title: 'This page',
    lead: 'The canvas is drawn by luma.gl right now; these are its real numbers.',
    ...thisPage
  },
  {
    slug: 'upstream-first',
    part: 'system',
    title: 'Upstream first',
    lead: 'Gridlight adds no GPU runtime; it stacks on published luma.gl.',
    ...upstreamFirst
  },
  {
    slug: 'glossary',
    part: 'system',
    title: 'Glossary',
    lead: 'Every term on these pages, who defines it and where it is explained.',
    ...glossary
  }
];

// The registry and the slug list (which the build reads) must name the same pages in the same order.
if (
  TOPICS.length !== TOPIC_SLUGS.length ||
  TOPICS.some((topic, index) => topic.slug !== TOPIC_SLUGS[index])
) {
  throw new Error('topics.ts and topic-slugs.ts list different pages');
}

export function getTopic(slug: string): Topic | undefined {
  return TOPICS.find(topic => topic.slug === slug);
}

export function getTopicPart(id: TopicPartId): TopicPart {
  return TOPIC_PARTS.find(part => part.id === id) ?? TOPIC_PARTS[0];
}

/** The pages before and after a topic in reading order. */
export function getTopicNeighbors(slug: TopicSlug): {previous?: Topic; next?: Topic} {
  const index = TOPICS.findIndex(topic => topic.slug === slug);
  return {previous: TOPICS[index - 1], next: TOPICS[index + 1]};
}
