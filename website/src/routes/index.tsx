// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createFileRoute, Link} from '@tanstack/react-router';
import {getFacetCounts} from '../catalog';
import type {PreviewKind} from '../catalog/schema';
import {DATA_TYPES, USE_CASES} from '../catalog/taxonomy';
import {FacetIcon} from '../components/brand/facet-icon';
import {Card} from '../components/ui/card';
import {getChipClassName} from '../components/ui/chip';
import {Section} from '../components/ui/section';
import {GpuCanvas} from '../gpu/gpu-canvas';
import {getPreviewScene, PreviewFallback} from '../gpu/previews';

export const Route = createFileRoute('/')({
  component: HomePage
});

/** Thumbnail kind for each data type facet. */
const DATA_TYPE_PREVIEW: Record<string, PreviewKind> = {
  points: 'points',
  lines: 'lines',
  polygons: 'polygons',
  raster: 'raster',
  terrain: 'terrain',
  network: 'network',
  table: 'histogram',
  'time-series': 'time-series',
  trajectories: 'trajectories',
  'point-cloud': 'point-cloud',
  cells: 'cells',
  'spatial-weights': 'spatial-weights'
};

const HERO_SEED = 7;

function HomePage() {
  const counts = getFacetCounts();
  return (
    <main>
      <section className="hero">
        <div className="hero__canvas">
          <GpuCanvas
            scene={getPreviewScene('density-surface')}
            fallback={<PreviewFallback kind="density-surface" seed={HERO_SEED} />}
            seed={HERO_SEED}
            label="Live luma.gl density surface drawn on the GPU"
          />
        </div>
        <div className="hero__content">
          <h1 className="hero__title">
            Spatial analysis, <em>on the GPU.</em>
          </h1>
          <p className="hero__lede">
            Every picture here is drawn by luma.gl, live. Pick a data type, find the operation.
          </p>
          <div className="hero__actions">
            <Link to="/explore" className="button">
              Explore operations
            </Link>
            <Link to="/how-it-works" className="button button--ghost">
              How it works
            </Link>
            <span className="hero__install">npm i @luma-contributors/analysis</span>
          </div>
        </div>
      </section>

      <div className="container">
        <Section
          title="Browse by data type"
          lede="What do you have?"
          aside={<Link to="/explore">All operations</Link>}
        >
          <div className="facet-grid">
            {DATA_TYPES.map(facet => (
              <Card
                key={facet.id}
                interactive
                className="facet-card"
                media={
                  <PreviewFallback
                    kind={DATA_TYPE_PREVIEW[facet.id] ?? 'points'}
                    seed={HERO_SEED}
                  />
                }
                title={
                  <>
                    <FacetIcon id={facet.id} size={18} />
                    <Link to="/explore/data/$id" params={{id: facet.id}}>
                      {facet.label}
                    </Link>
                    <span className="facet-card__count">{counts.data[facet.id] ?? 0}</span>
                  </>
                }
              >
                {facet.description}
              </Card>
            ))}
          </div>
        </Section>

        <Section title="Or by what you want to do" aside={<Link to="/explore">By domain too</Link>}>
          <div className="chip-row">
            {USE_CASES.map(useCase => (
              <Link
                key={useCase.id}
                to="/explore/use/$id"
                params={{id: useCase.id}}
                className={getChipClassName({})}
                title={useCase.description}
              >
                <FacetIcon id={useCase.id} size={16} />
                {useCase.label}
              </Link>
            ))}
          </div>
        </Section>

        <Section>
          <div className="teaser">
            <div>
              <h2 className="section__title">The system draws itself</h2>
              <p className="section__lede">
                Real command graphs, buffers and timings from the library, rendered the same way.
              </p>
              <p style={{marginTop: 'var(--space-4)'}}>
                <Link to="/how-it-works" className="button button--ghost">
                  See how it works
                </Link>
              </p>
            </div>
            <div className="teaser__steps" aria-label="Pipeline">
              <span className="teaser__step">data</span>
              <span className="teaser__arrow">→</span>
              <span className="teaser__step">GPU vectors</span>
              <span className="teaser__arrow">→</span>
              <span className="teaser__step">command graph</span>
              <span className="teaser__arrow">→</span>
              <span className="teaser__step">dispatch</span>
              <span className="teaser__arrow">→</span>
              <span className="teaser__step">readback</span>
            </div>
          </div>
        </Section>
      </div>
    </main>
  );
}
