// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createFileRoute, notFound} from '@tanstack/react-router';
import {loadEntry} from '../../catalog';
import {FunctionPage} from '../../components/catalog/function-page';
import {highlightCode} from '../../components/catalog/highlight-code';
import {SITE_NAME} from '../__root';

export const Route = createFileRoute('/fn/$id')({
  loader: async ({params}) => {
    const entry = await loadEntry(params.id);
    if (!entry) {
      throw notFound();
    }
    // Runs during prerendering, so the example ships highlighted.
    const exampleHtml = await highlightCode(entry.example);
    return {entry, title: entry.title, summary: entry.summary, exampleHtml};
  },
  head: ({loaderData}) => ({
    meta: [
      {title: loaderData ? `${loaderData.title} | ${SITE_NAME}` : SITE_NAME},
      {name: 'description', content: loaderData?.summary ?? ''}
    ]
  }),
  component: FunctionRoute
});

function FunctionRoute() {
  const {entry, exampleHtml} = Route.useLoaderData();
  return <FunctionPage entry={entry} exampleHtml={exampleHtml} />;
}
