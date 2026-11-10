// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Section} from '../../../ui/section';
import {Figure, InShort, Prose, Term, TopicShell} from '../../page-kit';
import type {TopicPageProps} from '../../topic-types';
import {AnatomyFigure} from './anatomy-figure';
import {BaselineTable} from './baseline-table';
import {RulesGrid} from './rules-grid';
import {StatusTimeline} from './status-timeline';
import {ValidatorDemo} from './validator-demo';

export function Page({topic, code}: TopicPageProps) {
  const example = code.serviceAreas;
  return (
    <TopicShell
      topic={topic}
      tryIt={[
        {kind: 'function', id: 'network-service-areas', note: 'The contributor on this page.'},
        {kind: 'function', id: 'kernel-density'},
        {kind: 'topic', slug: 'building-blocks', note: 'The primitives rule 2 points to.'},
        {kind: 'topic', slug: 'iteration', note: 'Rounds with a budget.'},
        {kind: 'topic', slug: 'data-on-the-gpu', note: 'Masks, compact output and overflow.'}
      ]}
    >
      <InShort
        points={[
          <>
            A <Term id="contributor">contributor</Term> is one class: props in, a validating
            constructor, kernels out.
          </>,
          <>Bad props fail at construction, by name. A kernel cannot throw, so it writes status.</>,
          <>A checker keeps every operation on the same rules.</>
        ]}
      />
      <Section title="One shape">
        <Prose>
          Every operation is a class with a props type and an output type. The constructor checks
          the props once; <code>getCommandNodes</code> returns the <Term id="kernel">kernels</Term>{' '}
          without touching the graph.
        </Prose>
        {example ? (
          <Figure
            notice="Props are checked before any kernel exists."
            caption="Select a step to mark its lines and its box."
          >
            <AnatomyFigure example={example} />
          </Figure>
        ) : null}
      </Section>
      <Section title="Fail early, by name">
        <Prose>
          These are the library's validators, running here. Each message starts with the contributor
          and names the prop and the limit. Nothing is checked per frame or per row.
        </Prose>
        <Figure
          notice="Every message names the contributor, the prop and the limit."
          caption="The real functions from analysis-core, called with your value. The first two defaults are out of range."
        >
          <ValidatorDemo />
        </Figure>
      </Section>
      <Section title="The rules">
        <Prose>Thirteen rules, in five families. Each links to the page that explains it.</Prose>
        <RulesGrid />
      </Section>
      <Section title="Status, not exceptions">
        <Prose>
          Construction can throw. The GPU run cannot, so a kernel records what happened in named
          ports and the caller chooses whether to look.
        </Prose>
        <Figure
          notice="Only construction throws; the run writes status."
          caption="Port names and definitions come from the contract's status types."
        >
          <StatusTimeline />
        </Figure>
        <Prose>
          <code>yarn check-contributor-contract</code> reports departures from the rules and fails
          on any not recorded in a baseline file. The baseline holds the known ones:
        </Prose>
        <BaselineTable />
      </Section>
    </TopicShell>
  );
}
