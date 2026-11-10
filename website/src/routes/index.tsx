// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createFileRoute, Link} from '@tanstack/react-router';
import {getEntry, getFacetCounts} from '../catalog';
import {getEntrySeed} from '../catalog/entry-seed';
import {EntryPreview} from '../components/catalog/entry-preview';
import {getDemo} from '../gpu/demos';
import {useLiveSlot} from '../components/catalog/use-live-slot';
import {DATA_TYPES, USE_CASES, type DataTypeId} from '../catalog/taxonomy';
import {FacetIcon} from '../components/brand/facet-icon';
import {Card} from '../components/ui/card';
import {getChipClassName} from '../components/ui/chip';
import {Section} from '../components/ui/section';
import {GpuCanvas} from '../gpu/gpu-canvas';

export const Route = createFileRoute('/')({
  component: HomePage
});

/** Each facet shows one operation that accepts that data type. */
const DATA_TYPE_OPERATIONS = {
  points: 'dbscan-clustering',
  lines: 'line-density',
  polygons: 'map-coloring',
  raster: 'normalized-difference-water-index',
  terrain: 'terrain-flow',
  network: 'network-path-extraction',
  table: 'column-profile',
  'time-series': 'temporal-reduction',
  trajectories: 'trajectory-playhead',
  'point-cloud': 'pointcloud-ground',
  cells: 'cell-aggregation',
  'spatial-weights': 'spatial-weights-minimum-spanning-tree'
} satisfies Record<DataTypeId, string>;

function requireEntry(id: string) {
  const entry = getEntry(id);
  if (!entry) throw new Error(`Missing homepage operation: ${id}`);
  return entry;
}

const HERO_ENTRY = requireEntry('kernel-density');
const HERO_DEMO = getDemo(HERO_ENTRY.id);
if (!HERO_DEMO) throw new Error('Missing kernel density demo');
const HERO_SCENE = HERO_DEMO.loader;
const HERO_PARAMETERS = HERO_DEMO.parameters
  .filter(parameter => parameter.id === 'bandwidth')
  .map(parameter => ({...parameter, label: 'Default bandwidth'}));

function DataTypeCard({facet, count}: {facet: (typeof DATA_TYPES)[number]; count: number}) {
  const entry = requireEntry(DATA_TYPE_OPERATIONS[facet.id]);
  const {live, ref, handlers} = useLiveSlot();
  return (
    <div ref={ref} {...handlers} className="facet-card-slot">
      <Card
        interactive
        media={<EntryPreview entry={entry} live={live} interactive />}
        title={
          <>
            <FacetIcon id={facet.id} size={18} />
            <Link to="/explore/data/$id" params={{id: facet.id}}>
              {facet.label}
            </Link>
            <span className="facet-card__count">{count}</span>
          </>
        }
        meta={entry.title}
      >
        {facet.description}
      </Card>
    </div>
  );
}

function HomePage() {
  const counts = getFacetCounts();
  return (
    <main>
      <section className="hero">
        <div className="hero__canvas">
          <GpuCanvas
            scene={HERO_SCENE}
            fallback={<img className="hero__poster" src="/runs/kernel-density-dark.webp" alt="" />}
            seed={getEntrySeed(HERO_ENTRY)}
            theme="dark"
            label="Kernel density of magnitude 4 and larger earthquakes around Japan, 2020–2024"
            interactive
            controls
            parameters={HERO_PARAMETERS}
          />
        </div>
        <div className="hero__content">
          <h1 className="hero__title">
            Spatial analysis, <em>on the GPU.</em>
          </h1>
          <p className="hero__lede">
            Move across the map to see how kernel bandwidth changes the density of earthquakes. Your
            GPU recomputes the surface as you move.
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
          <p className="hero__hint">
            M4+ earthquakes around Japan · 2020–2024 ·{' '}
            <a href="https://earthquake.usgs.gov/data/comcat/">USGS ComCat</a>
          </p>
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
              <DataTypeCard key={facet.id} facet={facet} count={counts.data[facet.id] ?? 0} />
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
