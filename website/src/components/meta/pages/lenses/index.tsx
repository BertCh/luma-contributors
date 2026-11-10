// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Link} from '@tanstack/react-router';
import {useState} from 'react';
import {catalog} from '../../../../catalog';
import {FEATURED_LENS_SCENARIOS} from '../../../../lens/featured';
import {formatBytes, formatCount} from '../../../../lens/format';
import summary from '../../../../lens/summary.json';
import type {LensSummary, LensViewModel} from '../../../../lens/types';
import {Section} from '../../../ui/section';
import {findExampleLines} from '../../examples';
import {
  CodeWalk,
  type CodeWalkStep,
  Figure,
  InShort,
  Prose,
  StatStrip,
  Term,
  TopicShell
} from '../../page-kit';
import {SnapshotMap} from '../../snapshot-map';
import type {TopicPageProps} from '../../topic-types';
import lensesData from './lenses-data.json';
import {KernelSpreadChart, RecordParts, SizeBars} from './record-parts';
import './lenses.css';

const totals: LensSummary = summary;

const CARD_CAPTIONS: Record<string, string> = {
  [FEATURED_LENS_SCENARIOS.kernelDensity]: 'Points in, density raster out.',
  [FEATURED_LENS_SCENARIOS.localMoran]:
    'Tract values in; dots are tract centers colored by local I.',
  [FEATURED_LENS_SCENARIOS.serviceAreas]:
    'Road network in; dots are nodes colored by serving facility.'
};

export function Page({topic, models, code}: TopicPageProps) {
  const [active, setActive] = useState('tap');
  const density = models[FEATURED_LENS_SCENARIOS.kernelDensity];
  const featured = topic.scenarios
    .map(id => models[id])
    .filter((model): model is LensViewModel => Boolean(model));
  const withRecord = catalog.filter(entry => entry.lensScenario).length;
  const example = code.recordRun;
  const steps: CodeWalkStep[] | null = example
    ? [
        {
          id: 'tap',
          title: 'Tap',
          text: 'A copy node, added last, reads the output into a buffer the CPU can map.',
          lines: findExampleLines(example.code, 'const arena', 'GPUSnapshotTap')
        },
        {
          id: 'compile',
          title: 'Compile',
          text: 'Published luma.gl orders the kernels and publishes the result.',
          lines: findExampleLines(example.code, 'graph.compile()')
        },
        {
          id: 'describe',
          title: 'Describe',
          text: 'Reads the compiled graph’s public id, stats and preflight report.',
          lines: findExampleLines(example.code, 'describeGPUCommandGraph(compiled)')
        },
        {
          id: 'record',
          title: 'Record',
          text: 'The graph, the device and the choices become one JSON record.',
          lines: findExampleLines(example.code, 'createGPUComputationRecord({', '});')
        },
        {
          id: 'submit',
          title: 'Run once',
          text: 'One encode and submit, then one read of the tap’s copy.',
          lines: findExampleLines(example.code, 'const commandEncoder', 'runId: record.runId')
        }
      ]
    : null;
  const kernelDensityStats = lensesData.featured[0];
  return (
    <TopicShell
      topic={topic}
      tryIt={[
        {kind: 'function', id: 'kernel-density'},
        {kind: 'function', id: 'local-morans-i'},
        {kind: 'function', id: 'network-service-areas'},
        {kind: 'topic', slug: 'command-graph'},
        {kind: 'topic', slug: 'anatomy'}
      ]}
    >
      <InShort
        points={[
          <>
            A <Term id="record" /> is the JSON of one run on a real GPU: the graph, its buffers, its
            values and its math.
          </>,
          <>
            The values come from one <Term id="snapshot-tap" /> copy at the end of the run.
          </>,
          <>
            A <Term id="lens" /> reads a record and draws it. Nothing on a function page is mocked.
          </>
        ]}
      />
      <Section title="Recorded, not mocked">
        <div className="snapshot-grid" style={{marginTop: 0}}>
          {featured.map(model => (
            <article className="snapshot-card" key={model.id}>
              {model.map ? (
                <SnapshotMap map={model.map} label={`Snapshot of ${model.title}`} />
              ) : null}
              <div className="snapshot-card__body">
                <h3 className="snapshot-card__title">{model.title}</h3>
                <p className="snapshot-card__text">{CARD_CAPTIONS[model.id]}</p>
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
                  {formatCount(model.recordedNodeCount)} graph nodes,{' '}
                  {formatBytes(model.resources.totalBytes)} of buffers
                </p>
                {model.formulas[0] ? (
                  <p className="lenses-card-formula">
                    <span className="lenses-card-formula__title">Formula in the record</span>
                    {model.formulas[0].ascii}
                  </p>
                ) : null}
              </div>
            </article>
          ))}
        </div>
      </Section>
      <Section title="What a record holds">
        {density ? (
          <Figure
            notice="A record is a table, not a recording."
            caption={`Counted from the kernel density record. Amber marks the one thing that leaves the graph.`}
          >
            <RecordParts model={density} />
          </Figure>
        ) : null}
        <Prose>
          Over all records: <Term id="kernel">kernels</Term> and dependencies by{' '}
          <Term id="hazard">hazard</Term> kind, buffers with sizes, and values.
        </Prose>
        <StatStrip
          stats={[
            {value: formatCount(totals.recordCount), label: 'records'},
            {value: formatCount(totals.kernels), label: 'kernels'},
            {value: formatCount(totals.gatedKernels), label: 'gated on the GPU'},
            {
              value: formatCount(
                totals.dependencies.raw + totals.dependencies.war + totals.dependencies.waw
              ),
              label: 'dependencies'
            },
            {
              value: formatCount(totals.buffers),
              label: `buffers, ${formatBytes(totals.bufferBytes)}`
            },
            {value: formatCount(totals.snapshots), label: 'captured values'},
            {
              value: formatCount(totals.formulas),
              label: `formulas in ${formatCount(totals.recordsWithFormulas)} records`
            }
          ]}
        />
        <Figure
          notice="Most records are small."
          caption="Records by kernel count. A few hundred kernels is rare."
        >
          <KernelSpreadChart
            spread={{
              bins: lensesData.bins,
              median: lensesData.median,
              smallest: totals.smallest,
              largest: totals.largest
            }}
          />
        </Figure>
      </Section>
      <Section title="How a record is made">
        {example && steps ? (
          <>
            <Prose>
              The browser test <code>capture-records.spec.ts</code> runs each scenario once and
              writes its record. The snapshot helper adds the tap; the same calls work on any graph.
            </Prose>
            <CodeWalk
              example={example}
              steps={steps}
              active={active}
              onActive={setActive}
              label="record-run.example.ts"
            />
          </>
        ) : null}
      </Section>
      <Section title="From records to a page">
        <Figure
          notice="A function page loads kilobytes, not megabytes."
          caption="Bytes on a log scale. The build keeps the compact extract of each record; the records never ship."
        >
          <SizeBars
            bars={[
              {
                label: 'All records',
                note: `${formatCount(totals.recordCount)} JSON files`,
                bytes: totals.recordBytes,
                shipped: false
              },
              {
                label: 'All view models',
                note: 'one chunk per page, loaded lazily',
                bytes: totals.viewModelBytes,
                shipped: true
              },
              {
                label: 'Kernel density record',
                note: 'one JSON file',
                bytes: kernelDensityStats.recordBytes,
                shipped: false
              },
              {
                label: 'Kernel density page',
                note: 'the chunk the page loads',
                bytes: kernelDensityStats.viewModelBytes,
                shipped: true
              }
            ]}
          />
        </Figure>
        <StatStrip
          stats={[
            {value: formatCount(withRecord), label: 'function pages with a record'},
            {value: formatCount(catalog.length), label: 'function pages'}
          ]}
        />
        <Prose>
          Open{' '}
          <Link to="/fn/$id" params={{id: 'kernel-density'}}>
            kernel density
          </Link>{' '}
          to see its recorded graph and the values the tap captured.
        </Prose>
      </Section>
    </TopicShell>
  );
}
