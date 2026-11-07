// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The five how-it-works pages. Each leads with a picture; captions stay at one or two sentences.

import {Link} from '@tanstack/react-router';
import type {ReactNode} from 'react';
import {formatBytes, formatCount} from '../../lens/format';
import {FEATURED_LENS_SCENARIOS} from '../../lens/featured';
import type {LensViewModel} from '../../lens/types';
import {CodeBlock} from '../ui/code-block';
import {PageHeader} from '../ui/page-header';
import {Section} from '../ui/section';
import {ColumnsDemo} from './columns-demo';
import {HazardDiagram} from './hazard-diagram';
import {LensGraphPanel} from './lens-graph-panel';
import {PackageLayers} from './package-layers';
import {PipelineDiagram} from './pipeline-diagram';
import {ResourceBars} from './resource-bars';
import {SnapshotMap} from './snapshot-map';
import {TOPICS, type Topic, type TopicSlug} from './topics';
import {WorkgroupDemo} from './workgroup-demo';
import './meta.css';

export type TopicModels = Record<string, LensViewModel | undefined>;

function Figure({children, caption}: {children: ReactNode; caption: ReactNode}) {
  return (
    <figure className="meta-figure" style={{margin: 0}}>
      {children}
      <figcaption className="meta-figure__caption">{caption}</figcaption>
    </figure>
  );
}

function TopicNav({current}: {current: TopicSlug}) {
  return (
    <nav className="topic-nav" aria-label="How it works topics">
      <Link to="/how-it-works">All topics</Link>
      {TOPICS.map(topic => (
        <Link
          key={topic.slug}
          to="/how-it-works/$topic"
          params={{topic: topic.slug}}
          aria-current={topic.slug === current ? 'page' : undefined}
        >
          {topic.title}
        </Link>
      ))}
    </nav>
  );
}

function TopicShell({topic, children}: {topic: Topic; children: ReactNode}) {
  return (
    <main className="page">
      <PageHeader
        eyebrow={
          <Link to="/how-it-works" style={{color: 'inherit', textDecoration: 'none'}}>
            How it works
          </Link>
        }
        title={topic.title}
        lede={topic.lead}
      />
      {children}
      <TopicNav current={topic.slug} />
    </main>
  );
}

function requireModel(models: TopicModels, scenarioId: string): LensViewModel | undefined {
  return models[scenarioId];
}

function Missing({scenarioId}: {scenarioId: string}) {
  return <p className="meta-figure__caption">No recorded run for {scenarioId}.</p>;
}

// Anatomy

function AnatomyPage({topic, models}: {topic: Topic; models: TopicModels}) {
  const density = requireModel(models, FEATURED_LENS_SCENARIOS.kernelDensity);
  return (
    <TopicShell topic={topic}>
      <Section title="Seven steps">
        <Figure caption="Hover a stage. Teal is built on the CPU, amber runs on the GPU.">
          <PipelineDiagram />
        </Figure>
      </Section>
      <Section title="A real run">
        {density ? (
          <Figure
            caption={
              <>
                Kernel density recorded on a real GPU: {density.recordedNodeCount} kernels,{' '}
                {density.recordedEdgeCount} dependencies. Particles are buffers handed from one
                kernel to the next.
              </>
            }
          >
            <LensGraphPanel scenario={density} />
          </Figure>
        ) : (
          <Missing scenarioId={FEATURED_LENS_SCENARIOS.kernelDensity} />
        )}
      </Section>
      <Section title="One dispatch">
        <Figure caption="A kernel launches as a grid of workgroups. The last lanes sit past the invocation count and do nothing.">
          <WorkgroupDemo />
        </Figure>
      </Section>
      <Section title="What comes back">
        {density?.map ? (
          <Figure
            caption={`The density raster (${formatCount(density.map.raster?.width ?? 0)} x ${formatCount(density.map.raster?.height ?? 0)} cells, downsampled) and its points, read back by the run's snapshot tap. The bright band sweeps the value range.`}
          >
            <div className="meta-canvas" style={{maxWidth: 560}}>
              <SnapshotMap map={density.map} label="Kernel density snapshot" />
            </div>
          </Figure>
        ) : null}
      </Section>
    </TopicShell>
  );
}

// Command graph

function CommandGraphPage({topic, models}: {topic: Topic; models: TopicModels}) {
  const moran = requireModel(models, FEATURED_LENS_SCENARIOS.localMoran);
  return (
    <TopicShell topic={topic}>
      <Section title="Kernels and buffers">
        {moran ? (
          <Figure
            caption={`Local Moran: ${formatCount(moran.recordedNodeCount)} kernels fold into ${moran.nodes.length} groups. xN marks unrolled loops and sort passes. Edge width is bytes handed over.`}
          >
            <LensGraphPanel scenario={moran} />
          </Figure>
        ) : (
          <Missing scenarioId={FEATURED_LENS_SCENARIOS.localMoran} />
        )}
      </Section>
      {moran ? (
        <>
          <Section title="Why that order">
            <Figure caption="The compiler infers every edge from read and write sets, so kernels never declare order by hand.">
              <div className="meta-panel">
                <HazardDiagram model={moran} />
              </div>
            </Figure>
          </Section>
          <Section title="Buffers">
            <Figure caption="Logical resources. The compiler may alias short-lived ones to share memory.">
              <ResourceBars model={moran} />
            </Figure>
          </Section>
        </>
      ) : null}
      <Section title="Compile, then encode">
        <div className="pipeline-steps" style={{marginTop: 0}}>
          <p
            style={{
              borderLeft: '2px solid var(--color-accent)',
              paddingLeft: 'var(--space-4)',
              fontSize: 'var(--text-small)',
              color: 'var(--color-muted)'
            }}
          >
            <strong style={{display: 'block', color: 'var(--color-text)'}}>Compile</strong>
            Order the nodes, add hazard edges, assign buffers.
          </p>
          <p
            style={{
              borderLeft: '2px solid var(--color-accent)',
              paddingLeft: 'var(--space-4)',
              fontSize: 'var(--text-small)',
              color: 'var(--color-muted)'
            }}
          >
            <strong style={{display: 'block', color: 'var(--color-text)'}}>Encode</strong>
            Record passes. Neighbouring compute nodes share a pass, except in timed runs.
          </p>
          <p
            style={{
              borderLeft: '2px solid var(--color-signal)',
              paddingLeft: 'var(--space-4)',
              fontSize: 'var(--text-small)',
              color: 'var(--color-muted)'
            }}
          >
            <strong style={{display: 'block', color: 'var(--color-text)'}}>Submit</strong>
            One command buffer. Results stay on the GPU until read.
          </p>
        </div>
      </Section>
    </TopicShell>
  );
}

// Data on the GPU

const FORMAT_ROWS = [
  {format: 'float32x3', bytes: [4, 4, 4], shader: 'vec3<f32>', note: 'three floats per row'},
  {format: 'unorm8x4', bytes: [1, 1, 1, 1], shader: 'vec4<f32>', note: 'four bytes, read as 0..1'},
  {
    format: 'vertex-list<float32x2>',
    bytes: [4, 4, 4, 4, 4, 4],
    shader: 'expanded by a model',
    note: 'variable length per row'
  }
] as const;

function FormatStrips() {
  return (
    <div className="meta-panel">
      <div className="meta-panel__bar">
        <span className="meta-panel__title">Stored bytes, then shader values</span>
      </div>
      <ul className="resource-bars" style={{gap: 12}}>
        {FORMAT_ROWS.map(row => (
          <li
            key={row.format}
            style={{gridTemplateColumns: 'minmax(0, 13rem) minmax(0, 1fr) minmax(0, 12rem)'}}
          >
            <span className="resource-bars__name">{row.format}</span>
            <span style={{display: 'flex', gap: 2}}>
              {row.bytes.map((width, index) => (
                <span
                  key={index}
                  style={{
                    flex: width,
                    height: 14,
                    maxWidth: 28 * width,
                    borderRadius: 3,
                    background: 'var(--color-accent)',
                    opacity: 0.35 + 0.15 * (index % 3)
                  }}
                />
              ))}
            </span>
            <span className="resource-bars__size" style={{textAlign: 'left'}}>
              {row.shader}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function DataOnTheGpuPage({topic, models}: {topic: Topic; models: TopicModels}) {
  const density = requireModel(models, FEATURED_LENS_SCENARIOS.kernelDensity);
  return (
    <TopicShell topic={topic}>
      <Section title="Columns of chunks">
        <Figure caption="Each dashed box is one GPUData: one buffer. Streaming keeps batch boundaries; nothing is repacked unless you ask.">
          <ColumnsDemo />
        </Figure>
      </Section>
      <Section title="Formats">
        <Figure caption="A format says how the stored bytes are laid out. The shader type is a separate question, answered where the vector meets a model.">
          <FormatStrips />
        </Figure>
      </Section>
      {density ? (
        <Section title="In a real run">
          <Figure
            caption={`Kernel density holds ${formatBytes(density.resources.totalBytes)} in ${density.resources.count} buffers. The output raster and the points dominate.`}
          >
            <ResourceBars model={density} />
          </Figure>
        </Section>
      ) : null}
    </TopicShell>
  );
}

// Lenses

const LENS_CAPTIONS: Record<string, string> = {
  [FEATURED_LENS_SCENARIOS.kernelDensity]: 'Points in, density raster out.',
  [FEATURED_LENS_SCENARIOS.localMoran]:
    'Tract values in; dots are tract centers colored by local I.',
  [FEATURED_LENS_SCENARIOS.serviceAreas]:
    'Road network in; dots are nodes colored by serving facility.'
};

function LensesPage({topic, models}: {topic: Topic; models: TopicModels}) {
  const density = requireModel(models, FEATURED_LENS_SCENARIOS.kernelDensity);
  const featured = topic.scenarios
    .map(id => models[id])
    .filter((model): model is LensViewModel => Boolean(model));
  return (
    <TopicShell topic={topic}>
      <Section title="Recorded, not mocked">
        <div className="snapshot-grid" style={{marginTop: 0}}>
          {featured.map(model => (
            <article className="snapshot-card" key={model.id}>
              {model.map ? (
                <SnapshotMap map={model.map} label={`Snapshot of ${model.title}`} />
              ) : null}
              <div className="snapshot-card__body">
                <h3 className="snapshot-card__title">{model.title}</h3>
                <p className="snapshot-card__text">{LENS_CAPTIONS[model.id]}</p>
                <ul className="snapshot-card__params">
                  {model.parameters.slice(0, 4).map(parameter => (
                    <li key={parameter.name}>
                      {parameter.label}{' '}
                      <b>
                        {parameter.value}
                        {parameter.unit ? ` ${parameter.unit}` : ''}
                      </b>
                    </li>
                  ))}
                </ul>
                <p className="snapshot-card__text">
                  {formatCount(model.recordedNodeCount)} kernels,{' '}
                  {formatBytes(model.resources.totalBytes)} of buffers
                </p>
              </div>
            </article>
          ))}
        </div>
      </Section>
      {density?.formulas[0] ? (
        <Section title="The math is part of the record">
          <Figure caption="Each lens carries the formula its kernels implement, next to the graph that computes it.">
            <pre className="formula">{density.formulas[0].ascii}</pre>
          </Figure>
        </Section>
      ) : null}
      <Section title="How a record is made">
        <ol className="pipeline-steps" style={{marginTop: 0}}>
          <li data-active="true">
            <strong>Run</strong>
            The operation executes on the GPU with a recorder attached.
          </li>
          <li data-active="true">
            <strong>Record</strong>
            Kernels, buffers, hazards and snapshot copies are written to one JSON file.
          </li>
          <li data-active="true">
            <strong>Compact</strong>
            At build time each record shrinks to a small view model; the full files never ship.
          </li>
          <li data-active="true">
            <strong>Draw</strong>
            Pages render the graph as SVG and the snapshots with luma.gl.
          </li>
        </ol>
      </Section>
    </TopicShell>
  );
}

// Upstream first

const BEFORE_AFTER = `// Not available on published luma.gl:
const graph = new GPUCommandGraph(device, {id: 'scan', recording: {sourceText: true}});

// Recording belongs to the analysis package:
const graph = new GPUCommandGraph(device, {id: 'scan'});
const recorder = props.record ? new GPUComputationRecorder({sourceText: true}) : undefined;`;

function UpstreamFirstPage({topic}: {topic: Topic; models: TopicModels}) {
  return (
    <TopicShell topic={topic}>
      <Section title="The stack">
        <Figure caption="A call descends left to right. Three packages here, three from published luma.gl 10.0.0-alpha.10.">
          <PackageLayers />
        </Figure>
      </Section>
      <Section title="What stays upstream">
        <ul className="pipeline-steps" style={{marginTop: 0}}>
          <li data-active="true">
            <strong>Graph and kernels</strong>
            Command graphs, compile and encode come from @luma.gl/graph as published.
          </li>
          <li data-active="true">
            <strong>No forks</strong>
            Public APIs only, even when that is more verbose.
          </li>
          <li data-active="true">
            <strong>Only analysis here</strong>
            Operations, lens records and adapters. Nothing that belongs in luma.gl.
          </li>
        </ul>
      </Section>
      <Section title="An example">
        <Figure caption="Recording needed no luma.gl hook. The compiled graph already publishes structure and hazards; the analysis recorder adds the domain facts.">
          <CodeBlock code={BEFORE_AFTER} language="ts" />
        </Figure>
      </Section>
    </TopicShell>
  );
}

export function TopicPage({topic, models}: {topic: Topic; models: TopicModels}) {
  switch (topic.slug) {
    case 'anatomy':
      return <AnatomyPage topic={topic} models={models} />;
    case 'command-graph':
      return <CommandGraphPage topic={topic} models={models} />;
    case 'data-on-the-gpu':
      return <DataOnTheGpuPage topic={topic} models={models} />;
    case 'lenses':
      return <LensesPage topic={topic} models={models} />;
    case 'upstream-first':
      return <UpstreamFirstPage topic={topic} models={models} />;
  }
}
