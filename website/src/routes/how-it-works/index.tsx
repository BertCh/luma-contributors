// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createFileRoute, Link} from '@tanstack/react-router';
import {PipelineDiagram} from '../../components/meta/pipeline-diagram';
import {TopicCardVisual, type TopicCardModels} from '../../components/meta/topic-card-visual';
import {TOPICS} from '../../components/meta/topics';
import {Card} from '../../components/ui/card';
import {PageHeader} from '../../components/ui/page-header';
import {FEATURED_LENS_SCENARIOS, getLensViewModel} from '../../lens';
import {SITE_NAME} from '../__root';
import '../../components/meta/meta.css';

export const Route = createFileRoute('/how-it-works/')({
  loader: async (): Promise<{models: TopicCardModels}> => {
    const [density, moran] = await Promise.all([
      getLensViewModel(FEATURED_LENS_SCENARIOS.kernelDensity),
      getLensViewModel(FEATURED_LENS_SCENARIOS.localMoran)
    ]);
    return {models: {density, moran}};
  },
  head: () => ({
    meta: [
      {title: `How it works | ${SITE_NAME}`},
      {
        name: 'description',
        content:
          'The system explained with its own output: graphs, buffers and dispatches drawn by luma.gl.'
      }
    ]
  }),
  component: HowItWorksIndex
});

function HowItWorksIndex() {
  const {models} = Route.useLoaderData();
  return (
    <main className="page">
      <PageHeader
        eyebrow="How it works"
        title="The system, drawn by itself"
        lede="Real runs, drawn by luma.gl: graphs, buffers, dispatches."
      />
      <div className="topic-grid">
        {TOPICS.map(topic => (
          <Card
            key={topic.slug}
            interactive
            media={<TopicCardVisual slug={topic.slug} models={models} />}
            title={
              <Link to="/how-it-works/$topic" params={{topic: topic.slug}}>
                {topic.title}
              </Link>
            }
          >
            {topic.lead}
          </Card>
        ))}
      </div>
      <PipelineDiagram />
    </main>
  );
}
