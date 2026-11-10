// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createFileRoute} from '@tanstack/react-router';
import {PageHeader} from '../components/ui/page-header';
import {listDemoDataSources} from '../gpu/data/provenance';
import {SITE_NAME} from './__root';

export const Route = createFileRoute('/demo-data')({
  head: () => ({meta: [{title: `Demo data | ${SITE_NAME}`}]}),
  component: DemoDataPage
});

function DemoDataPage() {
  return (
    <main className="page">
      <PageHeader
        title="Demo data"
        lede="The catalog uses compact extracts of these datasets. Each operation links to its sources and describes how to explore the measurements or a modeled scenario."
      />
      <dl className="demo-source-list">
        {listDemoDataSources().map(source => (
          <div key={source.id} id={source.id}>
            <dt>
              <a href={source.url}>{source.label}</a>
            </dt>
            <dd>
              {source.credit} ·{' '}
              {source.termsUrl ? <a href={source.termsUrl}>{source.terms}</a> : source.terms}
            </dd>
          </div>
        ))}
      </dl>
    </main>
  );
}
