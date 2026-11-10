// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Section} from '../../../ui/section';
import {formatCount} from '../../../../lens/format';
import {PackageLayers} from '../../package-layers';
import {ExampleCode, Figure, InShort, Prose, StatStrip, Term, TopicShell} from '../../page-kit';
import type {TopicPageProps} from '../../topic-types';
import upstreamData from './upstream-data.json';
import './upstream-first.css';

type Rung = {rule: string; example: React.ReactNode};

const RUNGS: readonly Rung[] = [
  {
    rule: 'Use the public API, even if it is more verbose.',
    example: (
      <>
        Waiting for submitted work uses <code>device.createFence()</code> and{' '}
        <code>await fence.signaled</code>, not a private helper.
      </>
    )
  },
  {
    rule: 'Adapt to luma.gl’s design.',
    example: (
      <>
        Stepped capture needed a graph hook. A <code>GPUSnapshotTap</code> is an ordinary copy node,
        so the published graph orders it.
      </>
    )
  },
  {
    rule: 'Drop the feature.',
    example: (
      <>
        Strongly connected components on large graphs, and the Arrow batch stream: the register
        lists what callers lose.
      </>
    )
  },
  {
    rule: 'Last resort: a small helper named for the analysis domain.',
    example: (
      <>
        Sparse resistance connectivity keeps a Jacobi-PCG solver in{' '}
        <code>sparse-resistance-solver.ts</code>. Delete it when luma.gl exports one.
      </>
    )
  }
];

const maximumImports = Math.max(...upstreamData.imports.map(entry => entry.count));

export function Page({topic, code}: TopicPageProps) {
  const example = code.describeGraph;
  const gpuCore = upstreamData.imports.find(entry => entry.specifier === '@luma.gl/graph/gpu-core');
  const {workarounds} = upstreamData;
  return (
    <TopicShell
      topic={topic}
      tryIt={[
        {kind: 'reference', slug: 'api/computation-record', label: 'Computation record'},
        {kind: 'reference', slug: 'api/analysis-catalog', label: 'Analysis catalog'},
        {kind: 'topic', slug: 'lenses'},
        {kind: 'topic', slug: 'command-graph'}
      ]}
    >
      <InShort
        points={[
          <>This library imports published luma.gl through public entry points only.</>,
          <>
            When luma.gl lacks something, it uses the public API, adapts, drops the feature, or
            writes a small helper, in that order.
          </>,
          <>
            Recording a <Term id="command-graph" /> needs no luma.gl hook.
          </>
        ]}
      />
      <Section title="The stack">
        <Figure
          notice="Three packages here, three from luma.gl."
          caption="Each package depends on the next one to its right."
        >
          <PackageLayers version={upstreamData.lumaVersion} />
        </Figure>
      </Section>
      <Section title="Where imports go">
        <Figure
          notice={`${gpuCore ? Math.round((gpuCore.count / upstreamData.importStatementCount) * 100) : 0}% of imports are one entry point.`}
          caption={`${formatCount(upstreamData.importStatementCount)} import statements in ${formatCount(upstreamData.sourceFileCount)} source files, counted by entry point.`}
        >
          <ul className="upstream-imports" aria-label="Imports by luma.gl entry point">
            {upstreamData.imports.map(entry => (
              <li key={entry.specifier}>
                <span className="upstream-imports__name">{entry.specifier}</span>
                <span className="upstream-imports__track">
                  <span
                    className="upstream-imports__fill"
                    style={{width: `${(entry.count / maximumImports) * 100}%`}}
                  />
                </span>
                <span className="upstream-imports__count">{formatCount(entry.count)}</span>
              </li>
            ))}
          </ul>
        </Figure>
        <StatStrip
          stats={[
            {value: formatCount(upstreamData.sourceFileCount), label: 'source files'},
            {
              value: formatCount(upstreamData.contributorClassCount),
              label: 'contributor classes'
            },
            {value: upstreamData.lumaVersion, label: 'luma.gl release'}
          ]}
        />
      </Section>
      <Section title="When luma.gl lacks something">
        <Figure
          notice="Try each rung before the next."
          caption="The first three cost no local code. Improvements go upstream as ordinary luma.gl pull requests; they are not carried here."
        >
          <ol className="upstream-ladder" aria-label="Rules, in order">
            {RUNGS.map((rung, index) => (
              <li
                key={rung.rule}
                className="upstream-rung"
                data-rung={index === RUNGS.length - 1 ? 'last' : undefined}
              >
                <h3 className="upstream-rung__title">{rung.rule}</h3>
                <p className="upstream-rung__example">{rung.example}</p>
              </li>
            ))}
          </ol>
        </Figure>
        <StatStrip
          stats={[
            {value: workarounds.dropped, label: 'capabilities dropped'},
            {value: workarounds.workaround, label: 'local workarounds'},
            {value: workarounds['accepted regression'], label: 'accepted regressions'}
          ]}
        />
        <Prose>
          The workarounds register lists every one, with the file to delete and what luma.gl would
          need to ship.
        </Prose>
        <ul className="upstream-never" aria-label="Never done">
          <li>Fork or copy luma.gl source</li>
          <li>
            Import <code>_internal</code> or deep <code>dist</code> paths
          </li>
          <li>Re-export luma.gl APIs</li>
        </ul>
      </Section>
      <Section title="An example">
        <Prose>
          Recording needed no hook. The compiled graph already publishes its order, buffers and
          usage, and <code>describeGPUCommandGraph</code> reads only those.
        </Prose>
        {example ? (
          <Figure caption="Public luma.gl types in, a record graph out. The website typechecks this file.">
            <ExampleCode example={example} label="describe-graph.example.ts" />
          </Figure>
        ) : null}
        <Prose>
          <code>yarn check-upstream-first</code> runs in CI. It fails on any non-public{' '}
          <code>@luma.gl</code> import, any re-export of a luma.gl API and any source directory
          outside the analysis-owned list.
        </Prose>
      </Section>
    </TopicShell>
  );
}
