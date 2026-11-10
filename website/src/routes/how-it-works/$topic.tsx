// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createFileRoute, notFound} from '@tanstack/react-router';
import {highlightCode} from '../../components/catalog/highlight-code';
import {extractExample} from '../../components/meta/examples';
import {TopicPage, type TopicCode, type TopicModels} from '../../components/meta/topic-pages';
import {getTopic, TOPICS} from '../../components/meta/topics';
import {getLensViewModel} from '../../lens';
import {SITE_NAME} from '../__root';

/** Prerender these paths: one per topic. */
export const HOW_IT_WORKS_PATHS = [
  '/how-it-works',
  ...TOPICS.map(topic => `/how-it-works/${topic.slug}`)
];

export const Route = createFileRoute('/how-it-works/$topic')({
  loader: async ({params}) => {
    const topic = getTopic(params.topic);
    if (!topic) throw notFound();
    // Each model is a small lazily loaded chunk, so the HTML ships with the finished SVG. Examples
    // are highlighted here, so the HTML ships with coloured code.
    const models: TopicModels = {};
    const code: TopicCode = {};
    await Promise.all([
      ...topic.scenarios.map(async id => {
        models[id] = await getLensViewModel(id);
      }),
      ...Object.entries(topic.examples).map(async ([name, source]) => {
        const text = extractExample(source);
        code[name] = {code: text, html: await highlightCode(text)};
      })
    ]);
    return {models, code};
  },
  head: ({params}) => {
    const topic = getTopic(params.topic);
    return {
      meta: [
        {title: topic ? `${topic.title} | ${SITE_NAME}` : SITE_NAME},
        {name: 'description', content: topic?.lead ?? ''}
      ]
    };
  },
  component: TopicRoute
});

function TopicRoute() {
  const {topic: slug} = Route.useParams();
  const {models, code} = Route.useLoaderData();
  const topic = getTopic(slug);
  return topic ? <TopicPage topic={topic} models={models} code={code} /> : null;
}
