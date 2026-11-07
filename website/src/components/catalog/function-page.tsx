// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Link} from '@tanstack/react-router';
import {getFacet, getNextSteps, getPreviousSteps, resolveRelated} from '../../catalog';
import type {CatalogEntry} from '../../catalog/schema';
import {LensGraphPanel} from '../meta/lens-graph-panel';
import {Chip} from '../ui/chip';
import './catalog.css';
import {CodeBlock} from '../ui/code-block';
import {ParamTable} from '../ui/param-table';
import {Section} from '../ui/section';
import {DataFlow} from './data-flow';
import {EntryGrid} from './entry-grid';
import {EntryPreview} from './entry-preview';

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
  const nextSteps = getNextSteps(entry, 3);
  const previousSteps = getPreviousSteps(entry, 3);
  const related = resolveRelated(entry);
  const importLine = `import {${entry.symbol}} from '${entry.importPath}';`;

  return (
    <main className="page function-page">
      <nav className="eyebrow" aria-label="Breadcrumb">
        <Link to="/explore">Explore</Link>
      </nav>

      <div className="function-visual">
        <EntryPreview entry={entry} aspectRatio={16 / 9} />
        {entry.preview.caption ? <p className="function-caption">{entry.preview.caption}</p> : null}
      </div>

      <header className="function-header">
        <h1>{entry.title}</h1>
        <p className="lead">{entry.summary}</p>
        <div className="function-meta">
          <DataFlow input={entry.dataTypes.input} output={entry.dataTypes.output} />
          <details className="function-more">
            <summary>+{entry.useCases.length + entry.domains.length + 1} tags</summary>
            <span className="function-chips">
              <FacetChips kind="use" ids={entry.useCases} />
              <FacetChips kind="domain" ids={entry.domains} />
              <Chip>{entry.maturity}</Chip>
            </span>
          </details>
        </div>
      </header>

      <p className="function-explanation">{renderInlineCode(entry.explanation)}</p>

      {entry.params.length > 0 ? (
        <Section title="Parameters">
          <ParamTable params={entry.params} />
        </Section>
      ) : null}

      <Section title="Use it">
        <CodeBlock code={importLine} language="typescript" />
        <CodeBlock code={entry.example} language="typescript" html={exampleHtml ?? undefined} />
      </Section>

      {entry.outputs.length > 0 ? (
        <Section title="Returns">
          <ParamTable params={entry.outputs} />
        </Section>
      ) : null}

      {entry.lensScenario ? (
        <Section title="How this runs">
          <LensGraphPanel scenario={entry.lensScenario} />
          <p>
            <Link to="/how-it-works/$topic" params={{topic: 'lenses'}}>
              How lenses record a run
            </Link>
          </p>
        </Section>
      ) : null}

      {previousSteps.length > 0 ? (
        <Section title="Before this">
          <EntryGrid entries={previousSteps} />
        </Section>
      ) : null}
      {nextSteps.length > 0 ? (
        <Section title="Next steps">
          <EntryGrid entries={nextSteps} />
        </Section>
      ) : null}
      {related.length > 0 ? (
        <Section title="Related">
          <EntryGrid entries={related} />
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
