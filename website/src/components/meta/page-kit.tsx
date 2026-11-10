// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared parts of every how-it-works page: the shell (position, previous and next, further links),
// figures with a "what to notice" caption, a code walk that links steps to lines, glossary terms
// and stat strips. Pages in `pages/<slug>/` compose these; they do not restyle them.

import {Link} from '@tanstack/react-router';
import {type ReactNode, useId} from 'react';
import {getEntry} from '../../catalog';
import {PageHeader} from '../ui/page-header';
import {getGlossaryTerm} from './glossary';
import type {Topic, TopicExample, TopicSlug} from './topic-types';
import {getTopic, getTopicNeighbors, getTopicPart, TOPIC_PARTS, TOPICS} from './topics';
import './meta.css';
import './page-kit.css';

// Shell

/** A further link at the end of a page: a function page, a reference page or another topic. */
export type TryItLink =
  | {kind: 'function'; id: string; note?: string}
  | {kind: 'reference'; slug: string; label: string; note?: string}
  | {kind: 'topic'; slug: TopicSlug; note?: string};

export function TopicShell({
  topic,
  children,
  tryIt
}: {
  topic: Topic;
  children: ReactNode;
  /** Where to go to use what the page explained. */
  tryIt?: readonly TryItLink[];
}) {
  const part = getTopicPart(topic.part);
  const position = TOPICS.findIndex(candidate => candidate.slug === topic.slug) + 1;
  const {previous, next} = getTopicNeighbors(topic.slug);
  return (
    <main className="page topic-page">
      <PageHeader
        eyebrow={
          <>
            <Link to="/how-it-works" className="topic-page__home">
              How it works
            </Link>
            <span aria-hidden="true">/</span>
            <span>{part.title}</span>
            <span className="topic-page__position">
              {position} of {TOPICS.length}
            </span>
          </>
        }
        title={topic.title}
        lede={topic.lead}
      />
      {children}
      {tryIt?.length ? <TryIt links={tryIt} /> : null}
      <nav className="topic-pager" aria-label="Previous and next page">
        {previous ? <PagerLink topic={previous} direction="previous" /> : <span />}
        {next ? <PagerLink topic={next} direction="next" /> : <span />}
      </nav>
      <TopicNav current={topic.slug} />
    </main>
  );
}

function PagerLink({topic, direction}: {topic: Topic; direction: 'previous' | 'next'}) {
  return (
    <Link
      to="/how-it-works/$topic"
      params={{topic: topic.slug}}
      className="topic-pager__link"
      data-direction={direction}
    >
      <span className="topic-pager__label">{direction === 'next' ? 'Next' : 'Previous'}</span>
      <span className="topic-pager__title">{topic.title}</span>
      <span className="topic-pager__lead">{topic.lead}</span>
    </Link>
  );
}

function TopicNav({current}: {current: TopicSlug}) {
  return (
    <nav className="topic-map" aria-label="All how-it-works pages">
      {TOPIC_PARTS.map(part => (
        <div key={part.id} className="topic-map__part">
          <span className="topic-map__title">{part.title}</span>
          <ul>
            {TOPICS.filter(topic => topic.part === part.id).map(topic => (
              <li key={topic.slug}>
                <Link
                  to="/how-it-works/$topic"
                  params={{topic: topic.slug}}
                  aria-current={topic.slug === current ? 'page' : undefined}
                >
                  {topic.title}
                </Link>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </nav>
  );
}

function TryIt({links}: {links: readonly TryItLink[]}) {
  return (
    <section className="try-it" aria-label="Use it">
      <h2 className="try-it__title">Use it</h2>
      <ul className="try-it__list">
        {links.map(link => (
          <li key={`${link.kind}:${link.kind === 'function' ? link.id : link.slug}`}>
            <TryItAnchor link={link} />
            {link.note ? <span className="try-it__note">{link.note}</span> : null}
          </li>
        ))}
      </ul>
    </section>
  );
}

function TryItAnchor({link}: {link: TryItLink}) {
  if (link.kind === 'function') {
    const entry = getEntry(link.id);
    return (
      <Link to="/fn/$id" params={{id: link.id}} className="try-it__link">
        <span className="try-it__kind">function</span>
        {entry?.title ?? link.id}
      </Link>
    );
  }
  if (link.kind === 'reference') {
    return (
      <Link to="/docs/$" params={{_splat: link.slug}} className="try-it__link">
        <span className="try-it__kind">reference</span>
        {link.label}
      </Link>
    );
  }
  return (
    <Link to="/how-it-works/$topic" params={{topic: link.slug}} className="try-it__link">
      <span className="try-it__kind">explainer</span>
      {getTopic(link.slug)?.title ?? link.slug}
    </Link>
  );
}

// Text

/** A short paragraph at reading width. Keep it to three sentences. */
export function Prose({children}: {children: ReactNode}) {
  return <p className="topic-prose">{children}</p>;
}

/** The page in three or four lines, shown before the first section. */
export function InShort({points}: {points: readonly ReactNode[]}) {
  return (
    <aside className="in-short" aria-label="In short">
      <span className="in-short__title">In short</span>
      <ul>
        {points.map((point, index) => (
          <li key={index}>{point}</li>
        ))}
      </ul>
    </aside>
  );
}

/** A glossary term: dotted underline, the definition as a tooltip, a link to the glossary entry. */
export function Term({id, children}: {id: string; children?: ReactNode}) {
  const term = getGlossaryTerm(id);
  return (
    <Link
      to="/how-it-works/$topic"
      params={{topic: 'glossary'}}
      hash={`term-${id}`}
      className="term"
      title={term?.definition}
    >
      {children ?? term?.term.toLowerCase() ?? id}
    </Link>
  );
}

// Figures

/**
 * A figure with its caption. The caption says what to notice in at most two sentences; `notice`
 * renders it with a lead-in so the reader's eye lands on the point before any interaction.
 */
export function Figure({
  children,
  caption,
  notice
}: {
  children: ReactNode;
  caption?: ReactNode;
  /** The one thing to see in the figure, shown ahead of the caption. */
  notice?: ReactNode;
}) {
  return (
    <figure className="meta-figure">
      {children}
      {caption || notice ? (
        <figcaption className="meta-figure__caption">
          {notice ? <strong className="meta-figure__notice">{notice} </strong> : null}
          {caption}
        </figcaption>
      ) : null}
    </figure>
  );
}

export function MissingRun({scenarioId}: {scenarioId: string}) {
  return <p className="meta-figure__caption">No recorded run for {scenarioId}.</p>;
}

export type Stat = {value: ReactNode; label: ReactNode};

/** A row of large numbers with small labels. Numbers come from records or source, never typed in. */
export function StatStrip({stats}: {stats: readonly Stat[]}) {
  return (
    <dl className="stat-strip">
      {stats.map((stat, index) => (
        <div key={index}>
          <dt>{stat.label}</dt>
          <dd>{stat.value}</dd>
        </div>
      ))}
    </dl>
  );
}

// Code

export type CodeWalkStep = {
  id: string;
  title: string;
  /** One or two sentences. */
  text: ReactNode;
  /** 1-based inclusive line range in the example; see findExampleLines in examples.ts. */
  lines: readonly [number, number];
};

function escapeHtml(text: string): string {
  return text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
}

function getExampleHtml(example: TopicExample): string {
  if (example.html) return example.html;
  const lines = example.code
    .split('\n')
    .map(line => `<span class="line">${escapeHtml(line)}</span>`);
  return `<pre class="shiki"><code>${lines.join('\n')}</code></pre>`;
}

/** Highlighted example code. `lines` marks a 1-based inclusive range. */
export function ExampleCode({
  example,
  lines,
  label
}: {
  example: TopicExample;
  lines?: readonly [number, number];
  label?: string;
}) {
  const scope = useId();
  return (
    <div className="example-code" data-scope={scope} data-marked={lines ? 'true' : undefined}>
      {label ? <div className="example-code__label">{label}</div> : null}
      {lines ? (
        <style>{`[data-scope="${scope}"] .line:nth-child(n+${lines[0]}):nth-child(-n+${lines[1]}){background:var(--color-accent-soft) !important;box-shadow:inset 3px 0 0 var(--color-accent);opacity:1}`}</style>
      ) : null}
      <div dangerouslySetInnerHTML={{__html: getExampleHtml(example)}} />
    </div>
  );
}

/**
 * Code beside the steps that explain it. Selecting a step marks its lines; the page owns `active`
 * so the same selection can drive a diagram. Every step's text is visible without interaction.
 */
export function CodeWalk({
  example,
  steps,
  active,
  onActive,
  label
}: {
  example: TopicExample;
  steps: readonly CodeWalkStep[];
  active: string;
  onActive: (id: string) => void;
  label?: string;
}) {
  const current = steps.find(step => step.id === active) ?? steps[0];
  return (
    <div className="code-walk">
      <ExampleCode example={example} lines={current?.lines} label={label} />
      <ol className="code-walk__steps">
        {steps.map(step => (
          <li key={step.id}>
            <button
              type="button"
              className="code-walk__step"
              aria-pressed={step.id === current?.id}
              onClick={() => onActive(step.id)}
              onPointerEnter={event => {
                if (event.pointerType === 'mouse') onActive(step.id);
              }}
              onFocus={() => onActive(step.id)}
            >
              <strong>{step.title}</strong>
              <span>{step.text}</span>
            </button>
          </li>
        ))}
      </ol>
    </div>
  );
}

// Position in a run

export const RUN_STAGES = [
  {id: 'buffers', title: 'Buffers', topic: 'data-on-the-gpu'},
  {id: 'build', title: 'Build', topic: 'command-graph'},
  {id: 'compile', title: 'Compile', topic: 'scheduling'},
  {id: 'encode', title: 'Encode', topic: 'scheduling'},
  {id: 'dispatch', title: 'Dispatch', topic: 'kernels'},
  {id: 'readback', title: 'Read back', topic: 'anatomy'}
] as const satisfies readonly {id: string; title: string; topic: TopicSlug}[];

export type RunStageId = (typeof RUN_STAGES)[number]['id'];

/** "You are here": the six stages of a run with the ones this page covers lit. */
export function RunLocator({stages}: {stages: readonly RunStageId[]}) {
  return (
    <ol className="run-locator" aria-label="Stages of a run; this page covers the marked ones">
      {RUN_STAGES.map(stage => (
        <li key={stage.id} data-here={stages.includes(stage.id) ? 'true' : undefined}>
          <Link to="/how-it-works/$topic" params={{topic: stage.topic}}>
            {stage.title}
          </Link>
        </li>
      ))}
    </ol>
  );
}
