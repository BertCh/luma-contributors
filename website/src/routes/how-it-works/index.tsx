// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createFileRoute, Link} from '@tanstack/react-router';
import {useLiveSlot} from '../../components/catalog/use-live-slot';
import {StatStrip} from '../../components/meta/page-kit';
import {TopicCardVisual, type TopicCardModels} from '../../components/meta/topic-card-visual';
import {TOPIC_PARTS, TOPICS, type Topic} from '../../components/meta/topics';
import {Card} from '../../components/ui/card';
import {PageHeader} from '../../components/ui/page-header';
import {Section} from '../../components/ui/section';
import {FEATURED_LENS_SCENARIOS, getLensViewModel} from '../../lens';
import {formatCount} from '../../lens/format';
import summary from '../../lens/summary.json';
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
          'The system explained with its own output: real code, recorded command graphs and the data they produced.'
      }
    ]
  }),
  component: HowItWorksIndex
});

function TopicCard({
  topic,
  position,
  models
}: {
  topic: Topic;
  position: number;
  models: TopicCardModels;
}) {
  const {live, ref, handlers} = useLiveSlot();
  return (
    <div ref={ref} {...handlers} className="topic-card-slot">
      <Card
        interactive
        media={<TopicCardVisual slug={topic.slug} models={models} live={live} />}
        title={
          <Link to="/how-it-works/$topic" params={{topic: topic.slug}}>
            <span className="topic-card__number">{position}</span>
            {topic.title}
          </Link>
        }
      >
        {topic.lead}
      </Card>
    </div>
  );
}

function HowItWorksIndex() {
  const {models} = Route.useLoaderData();
  const dependencies =
    summary.dependencies.raw + summary.dependencies.war + summary.dependencies.waw;
  return (
    <main className="page">
      <PageHeader
        eyebrow="How it works"
        title="The system, drawn by itself"
        lede={`${TOPICS.length} short pages in reading order. Real code, and command graphs recorded on a real GPU.`}
        actions={
          <Link to="/how-it-works/$topic" params={{topic: TOPICS[0].slug}} className="topic-start">
            Start with {TOPICS[0].title.toLowerCase()}
          </Link>
        }
      />
      <StatStrip
        stats={[
          {value: formatCount(summary.recordCount), label: 'runs recorded on a GPU'},
          {value: formatCount(summary.kernels), label: 'kernels in those runs'},
          {value: formatCount(dependencies), label: 'orderings the compiler inferred'},
          {value: formatCount(summary.gatedKernels), label: 'kernels gated on the GPU'},
          {value: formatCount(summary.snapshots), label: 'buffers captured to draw'}
        ]}
      />
      {TOPIC_PARTS.map(part => (
        <Section key={part.id} title={part.title} lede={part.lead}>
          <div className="topic-grid">
            {TOPICS.map((topic, index) =>
              topic.part === part.id ? (
                <TopicCard key={topic.slug} topic={topic} position={index + 1} models={models} />
              ) : null
            )}
          </div>
        </Section>
      ))}
    </main>
  );
}
