// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createFileRoute, Link} from '@tanstack/react-router';
import {PipelineDiagram} from '../../components/meta/pipeline-diagram';
import {TopicThumb} from '../../components/meta/topic-thumbs';
import {TOPICS} from '../../components/meta/topics';
import {Card} from '../../components/ui/card';
import {PageHeader} from '../../components/ui/page-header';
import {SITE_NAME} from '../__root';
import '../../components/meta/meta.css';

export const Route = createFileRoute('/how-it-works/')({
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
  return (
    <main className="page">
      <PageHeader
        eyebrow="How it works"
        title="The system, drawn by itself"
        lede="Every picture here is a real run: graphs, buffers and dispatches, drawn by luma.gl."
      />
      <PipelineDiagram />
      <div className="topic-grid">
        {TOPICS.map(topic => (
          <Card
            key={topic.slug}
            interactive
            media={
              <div className="topic-card__thumb">
                <TopicThumb slug={topic.slug} />
              </div>
            }
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
    </main>
  );
}
