// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Link} from '@tanstack/react-router';
import {getFacet, getNextSteps, getPreviousSteps, resolveRelated} from '../../catalog';
import type {CatalogEntry} from '../../catalog/schema';
import {LensGraphPanel} from '../meta/lens-graph-panel';
import {Chip} from '../ui/chip';
import './catalog.css';
import {CodeBlock} from '../ui/code-block';
import {OutputTable, ParamTable} from '../ui/param-table';
import {Section} from '../ui/section';
import {DataFlow} from './data-flow';
import {EntryGrid} from './entry-grid';
import {getDemoData} from '../../gpu/data/provenance';
import {getRunLegend} from '../../gpu/runs/registry';
import {EntryPreview, hasOwnPicture} from './entry-preview';

/** Preview kinds drawn from the function's own parameters, not a generic stand-in. */
const BESPOKE_PREVIEW_KINDS: ReadonlySet<string> = new Set(['affine-transform', 'cell-cover']);

/** Inline `code` spans in entry prose. */
function renderInlineCode(text: string) {
  return text
    .split(/(`[^`]+`)/g)
    .map((part, index) =>
      part.startsWith('`') && part.endsWith('`') ? (
        <code key={index}>{part.slice(1, -1)}</code>
      ) : (
        part
      )
    );
}

function FacetChips({kind, ids}: {kind: 'use' | 'domain'; ids: string[]}) {
  return (
    <>
      {ids.map(id => (
        <Link
          key={id}
          to={kind === 'use' ? '/explore/use/$id' : '/explore/domain/$id'}
          params={{id}}
          className="chip-link"
        >
          <Chip>{getFacet(kind, id)?.label ?? id}</Chip>
        </Link>
      ))}
    </>
  );
}

export function FunctionPage({
  entry,
  exampleHtml
}: {
  entry: CatalogEntry;
  exampleHtml: string | null;
}) {
  // Each entry appears once: next steps win over before, and both win over related.
  const nextSteps = getNextSteps(entry, 3);
  const nextIds = new Set(nextSteps.map(step => step.id));
  const previousSteps = getPreviousSteps(entry, 3).filter(step => !nextIds.has(step.id));
  const shownIds = new Set([...nextIds, ...previousSteps.map(step => step.id)]);
  const related = resolveRelated(entry).filter(step => !shownIds.has(step.id));
  const primaryDataType =
    entry.dataTypes.input.find(id => id !== 'table') ?? entry.dataTypes.input[0] ?? 'table';
  const importLine = `import {${entry.symbol}} from '${entry.importPath}';`;
  const hasImport = entry.example.includes(entry.importPath);
  // Generic scenes per preview kind are stand-ins; demos, runs and bespoke previews run the function.
  const illustrativeNote =
    hasOwnPicture(entry) || BESPOKE_PREVIEW_KINDS.has(entry.preview.kind)
      ? null
      : "Illustrative scene, not this function's output.";

  const legend = getRunLegend(entry.id);
  const {sources, scenario, caption: sceneCaption} = getDemoData(entry.id);
  const caption = sceneCaption ?? entry.preview.caption;

  return (
    <main className="page function-page">
      <nav className="eyebrow" aria-label="Breadcrumb">
        <Link to="/explore">Explore</Link>
        <span className="crumb-separator">/</span>
        <Link to="/explore/data/$id" params={{id: primaryDataType}}>
          {getFacet('data', primaryDataType)?.label ?? primaryDataType}
        </Link>
        <span className="crumb-separator">/</span>
        <Link to="/explore/use/$id" params={{id: entry.useCases[0]}}>
          {getFacet('use', entry.useCases[0])?.label ?? entry.useCases[0]}
        </Link>
      </nav>

      <header className="function-title">
        <h1>{entry.title}</h1>
        <p className="lead">{entry.summary}</p>
      </header>

      <div className="function-visual">
        <EntryPreview entry={entry} aspectRatio={16 / 9} controls interactive />
        {legend ? (
          <ul className="function-legend" aria-label="Key">
            {legend.map(item => (
              <li key={item.label}>
                <span className="function-legend-swatch" style={{background: item.color}} />
                {item.label}
              </li>
            ))}
          </ul>
        ) : null}
        {caption || illustrativeNote ? (
          <p className="function-caption">
            {caption}
            {caption && illustrativeNote ? ' ' : null}
            {illustrativeNote ? <span className="function-hint">{illustrativeNote}</span> : null}
          </p>
        ) : null}
        {scenario || sources.length > 0 ? (
          <div className="function-caption demo-data">
            {scenario ? <p>{scenario}</p> : null}
            {sources.length > 0 ? (
              <details>
                <summary>Data · {sources.map(source => source.label).join(' · ')}</summary>
                <ul>
                  {sources.map(source => (
                    <li key={source.id}>
                      {source.credit} ·{' '}
                      {source.termsUrl ? (
                        <a href={source.termsUrl}>{source.terms}</a>
                      ) : (
                        source.terms
                      )}
                      {' · '}
                      <a href={source.url}>{source.label}</a>
                    </li>
                  ))}
                </ul>
              </details>
            ) : null}
          </div>
        ) : null}
      </div>

      <header className="function-header">
        <div className="function-meta">
          <DataFlow input={entry.dataTypes.input} output={entry.dataTypes.output} />
          <div className="function-facets">
            <span className="function-facets-group">
              <span className="function-facets-label">Use</span>
              <FacetChips kind="use" ids={entry.useCases} />
            </span>
            <span className="function-facets-group">
              <span className="function-facets-label">Domain</span>
              <FacetChips kind="domain" ids={entry.domains} />
            </span>
            {entry.maturity !== 'stable' ? <Chip>{entry.maturity}</Chip> : null}
          </div>
        </div>
      </header>

      <p className="function-explanation">{renderInlineCode(entry.explanation)}</p>

      {entry.params.length > 0 ? (
        <Section title="Parameters">
          <ParamTable params={entry.params} moreRequired={entry.moreRequired} />
        </Section>
      ) : null}

      {entry.outputs.length > 0 ? (
        <Section title="Returns">
          <OutputTable outputs={entry.outputs} />
        </Section>
      ) : null}

      <Section title="Example">
        {hasImport ? null : <CodeBlock code={importLine} language="typescript" />}
        <CodeBlock code={entry.example} language="typescript" html={exampleHtml ?? undefined} />
      </Section>

      {previousSteps.length > 0 ? (
        <Section title="Before">
          <EntryGrid entries={previousSteps} />
        </Section>
      ) : null}
      {nextSteps.length > 0 ? (
        <Section title="Next">
          <EntryGrid entries={nextSteps} />
        </Section>
      ) : null}
      {related.length > 0 ? (
        <Section title="Related">
          <EntryGrid entries={related} />
        </Section>
      ) : null}

      {entry.lensScenario ? (
        <Section title="Recorded run">
          <LensGraphPanel scenario={entry.lensScenario} />
          <p>
            <Link to="/how-it-works/$topic" params={{topic: 'lenses'}}>
              How lenses record a run
            </Link>
          </p>
        </Section>
      ) : null}

      {entry.reference ? (
        <p className="function-reference">
          <Link to="/docs/$" params={{_splat: entry.reference}}>
            Full reference
          </Link>
        </p>
      ) : null}
    </main>
  );
}
