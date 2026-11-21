// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createFileRoute, notFound} from '@tanstack/react-router';
import {LensGraphPanel} from '../../components/meta/lens-graph-panel';
import {getLensViewModel} from '../../lens';
import {SITE_NAME} from '../__root';

export const Route = createFileRoute('/lens/$scenario')({
  loader: async ({params}) => {
    const recording = await getLensViewModel(params.scenario);
    if (!recording) throw notFound();
    return recording;
  },
  head: ({loaderData}) => ({
    meta: [
      {
        title: loaderData ? loaderData.title.replaceAll('-', ' ') + ' | ' + SITE_NAME : SITE_NAME
      }
    ]
  }),
  component: LensRecordingPage
});

function LensRecordingPage() {
  const recording = Route.useLoaderData();
  return (
    <main className="page">
      <h1>{recording.title.replaceAll('-', ' ')}</h1>
      <LensGraphPanel scenario={recording} />
    </main>
  );
}
