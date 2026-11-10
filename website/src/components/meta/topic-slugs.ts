// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The how-it-works slugs in reading order. No imports, so vite.config.ts can load this file to list
// the routes to prerender (topics.ts pulls in `?raw` examples, which a config cannot load).

export const TOPIC_SLUGS = [
  'anatomy',
  'data-on-the-gpu',
  'command-graph',
  'scheduling',
  'kernels',
  'building-blocks',
  'iteration',
  'contract',
  'workflows',
  'lenses',
  'this-page',
  'upstream-first',
  'glossary'
] as const;

export type TopicSlug = (typeof TOPIC_SLUGS)[number];
