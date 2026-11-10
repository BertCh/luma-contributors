// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Every term the how-it-works pages use, grouped by the page that explains it. Other pages link
// here with <Term id>, which lands on `#term-<id>`.

import {Link} from '@tanstack/react-router';
import {useState} from 'react';
import {GLOSSARY, GLOSSARY_OWNER_LABELS, type GlossaryOwner} from '../../glossary';
import {StatStrip, TopicShell} from '../../page-kit';
import type {TopicPageProps} from '../../topic-types';
import {getTopicPart, TOPICS} from '../../topics';
import './glossary.css';

const OWNERS: readonly GlossaryOwner[] = ['webgpu', 'luma.gl', 'gridlight'];

export function Page({topic}: TopicPageProps) {
  const [query, setQuery] = useState('');
  const needle = query.trim().toLowerCase();
  const matches = GLOSSARY.filter(
    term =>
      needle === '' ||
      term.term.toLowerCase().includes(needle) ||
      term.definition.toLowerCase().includes(needle) ||
      (term.code ?? '').toLowerCase().includes(needle)
  );
  const groups = TOPICS.map(candidate => ({
    topic: candidate,
    terms: matches.filter(term => term.topic === candidate.slug)
  })).filter(group => group.terms.length > 0);

  return (
    <TopicShell topic={topic}>
      <StatStrip
        stats={[
          {value: GLOSSARY.length, label: 'terms on these pages'},
          ...OWNERS.map(owner => ({
            value: GLOSSARY.filter(term => term.owner === owner).length,
            label: `defined by ${GLOSSARY_OWNER_LABELS[owner]}`
          }))
        ]}
      />
      <div className="glossary-filter" style={{marginTop: 'var(--space-5)'}}>
        <label htmlFor="glossary-query">Find a term</label>
        <input
          id="glossary-query"
          type="search"
          value={query}
          onChange={event => setQuery(event.target.value)}
          placeholder="scan, overflow, GPUSort"
          autoComplete="off"
        />
        <span aria-live="polite">
          {needle === '' ? null : `${matches.length} of ${GLOSSARY.length}`}
        </span>
      </div>
      {groups.map(group => (
        <section key={group.topic.slug} className="glossary-group">
          <div className="glossary-group__head">
            <h2 className="glossary-group__title">
              <Link to="/how-it-works/$topic" params={{topic: group.topic.slug}}>
                {group.topic.title}
              </Link>
            </h2>
            <span className="glossary-group__part">{getTopicPart(group.topic.part).title}</span>
          </div>
          <dl className="glossary-list">
            {group.terms.map(term => (
              <div key={term.id} id={`term-${term.id}`} className="glossary-term">
                <dt>
                  <span className="glossary-term__name">{term.term}</span>
                  {term.code ? <code className="glossary-term__code">{term.code}</code> : null}
                </dt>
                <dd>{term.definition}</dd>
                <span className="glossary-term__owner" data-owner={term.owner}>
                  {GLOSSARY_OWNER_LABELS[term.owner]}
                </span>
              </div>
            ))}
          </dl>
        </section>
      ))}
      {groups.length === 0 ? <p className="glossary-empty">No term matches "{query}".</p> : null}
    </TopicShell>
  );
}
