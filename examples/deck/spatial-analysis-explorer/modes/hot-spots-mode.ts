// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Local spatial autocorrelation of San Francisco bike-parking capacity. Each point carries a value
 * column (parking spaces). A `GPUNeighborSearch` distance band writes a spatial-weights CSR, which
 * `GPUHotSpotAnalysis` (Getis-Ord Gi*) and `GPULocalMoran` (LISA) then consume. The search and both
 * statistics are compiled once, the statistics with and without false discovery rate correction, so
 * the five graphs exist before the first frame. The radius, the Moran significance level, the
 * statistic and the FDR choice are buffer writes or a choice between graphs that are already
 * compiled: the rebuild counter stays 0. The graphs run only when an input changed (the data is
 * static), and `GPUHistogram` counts the classes on the GPU so the only readback is a handful of
 * integers.
 *
 * Local Moran has a compile-time quadrant gating choice (`GPULocalMoran` `quadrantGating`):
 * analytic p-value, none (the ungated quadrant of every row), or gated by a conditional
 * permutation test (`GPULocalPermutationTest`, whose pseudo p-value tail `alternative` is also
 * compile-time). Every gating and every alternative is precompiled and selected between at run
 * time, so changing either is still a graph switch and a buffer write, never a rebuild.
 * `GPUGlobalPermutationTest` reports the pseudo p-value of the global Moran's I for the same tail.
 *
 * The compute graph, summary decoding, legends and palettes live in
 * `spatial-analysis-shared/hot-spots-analysis`; this file is the explorer adapter (controls,
 * readouts and Deck layers).
 */

import type {Layer} from '@deck.gl/core';
import {
  DEFAULT_HOT_SPOTS_PARAMETERS,
  HOT_SPOTS_ALTERNATIVES,
  HOT_SPOTS_HIDDEN,
  HOT_SPOTS_MAXIMUM_PERMUTATIONS,
  HOT_SPOTS_NO_DATA_VALUE,
  HOT_SPOTS_PERMUTATION_MAXIMUM_NEIGHBORS,
  createHotSpotsCompute,
  getHotSpotsInsignificantPalette,
  getHotSpotsSignificantPalette,
  hotSpotsStyle,
  type HotSpotsAlternative,
  type HotSpotsGating,
  type HotSpotsStatistic,
  type HotSpotsSummary
} from '../../spatial-analysis-shared/hot-spots-analysis';
import {SpatialAnalysisPointLayer} from '../spatial-analysis-layers';
import type {SpatialAnalysisModeDefinition} from '../spatial-analysis-mode';
import {formatCount} from '../spatial-analysis-resources';

export const hotSpotsMode: SpatialAnalysisModeDefinition = {
  id: 'hot-spots',
  title: 'Hot spots',
  contributors: [
    'GPUHotSpotAnalysis',
    'GPULocalMoran',
    'GPULocalPermutationTest',
    'GPUGlobalPermutationTest',
    'GPUHistogram'
  ],
  description:
    'Where is bike-parking capacity unusually high or low compared with its neighbors? Getis-Ord ' +
    'Gi* hot and cold spots or local Moran quadrants over San Francisco. Pick Local Moran, then ' +
    'switch the quadrant gating (analytic, none, permutation) and the pseudo p-value alternative ' +
    'and read the per-quadrant counts.',
  initialViewState: {longitude: -122.435, latitude: 37.765, zoom: 12.2},

  async create(context) {
    const parking = await context.data.getSanFranciscoBikeParking();
    context.signal.throwIfAborted();
    const {controls} = context;
    const pointCount = parking.spaces.length;
    let showNotSignificant = true;

    // Readouts are created below; the summary hook only runs after the first encoded frame.
    const readouts = {} as Record<string, {setValue: (value: string) => void}>;
    const compute = await createHotSpotsCompute(
      context.device,
      {
        positions: parking.positions,
        values: parking.spaces,
        spatialContext: parking.frame.spatialContext
      },
      {publishSummary: summary => publishSummary(summary)},
      DEFAULT_HOT_SPOTS_PARAMETERS
    );
    const initial = compute.getParameters();

    const publishSummary = (summary: HotSpotsSummary) => {
      const count = (index: number) => formatCount(summary.counts[index]);
      if (summary.statistic === 'gi-star') {
        // Bins -3..3 are histogram rows 0..6.
        readouts.hot.setValue(`${count(6)} / ${count(5)} / ${count(4)}`);
        readouts.cold.setValue(`${count(0)} / ${count(1)} / ${count(2)}`);
        readouts.outlier.setValue('n/a (Moran only)');
        readouts.insignificant.setValue(count(3));
        readouts.quadrant.setValue('n/a (Moran only)');
        readouts.quadrantTotal.setValue('n/a (Moran only)');
        readouts.global.setValue('n/a (Moran only)');
        readouts.permutation.setValue('n/a (Moran only)');
      } else {
        // Quadrant codes 0 (none), 1 HH, 2 LH, 3 LL, 4 HL are histogram rows 0..4.
        readouts.hot.setValue(count(1));
        readouts.cold.setValue(count(3));
        readouts.outlier.setValue(`${count(2)} / ${count(4)}`);
        readouts.insignificant.setValue(count(0));
        readouts.quadrant.setValue(`${count(1)} / ${count(2)} / ${count(3)} / ${count(4)}`);
        const withQuadrant =
          summary.counts[1] + summary.counts[2] + summary.counts[3] + summary.counts[4];
        readouts.quadrantTotal.setValue(
          `${formatCount(withQuadrant)} of ${formatCount(pointCount)} (${(
            (100 * withQuadrant) / pointCount
          ).toFixed(1)}%)`
        );
        readouts.permutation.setValue(
          summary.permutationCounts
            ? `${formatCount(summary.permutationCounts[0])} / ${formatCount(summary.permutationCounts[1])}`
            : 'n/a (permutation gating only)'
        );
        readouts.global.setValue(
          summary.globalMoran
            ? `${summary.globalMoran.observed.toFixed(3)} / ` +
                `${summary.globalMoran.pseudoPValue.toFixed(4)} / ` +
                `${summary.globalMoran.zSimulated.toFixed(2)}`
            : 'n/a (permutation gating only)'
        );
      }
      readouts.moments.setValue(
        `${summary.mean.toFixed(2)} / ${summary.standardDeviation.toFixed(2)} spaces`
      );
      readouts.capacity.setValue(
        summary.neighborOverflow
          ? 'overflow: neighbors truncated, reduce the radius'
          : summary.permutationOverflow
            ? `ok; rows over ${HOT_SPOTS_PERMUTATION_MAXIMUM_NEIGHBORS} neighbors skipped by the permutation test`
            : 'ok'
      );
    };

    controls.addSelect<HotSpotsStatistic>({
      label: 'Statistic (buffer-level switch between compiled graphs)',
      options: [
        {value: 'gi-star', label: 'Getis-Ord Gi* hot and cold spots'},
        {value: 'local-moran', label: 'Local Moran quadrants'}
      ],
      value: initial.statistic,
      onChange: value => {
        compute.setParameters({statistic: value});
        updateEnabled();
        updateLegendNote();
        context.updateLayers();
      }
    });
    const gatingSelect = controls.addSelect<HotSpotsGating>({
      label: 'Moran quadrant gating (compile-time option, all variants precompiled)',
      options: [
        {value: 'analytic', label: 'Analytic p-value (default)'},
        {value: 'none', label: 'None: ungated quadrant of every row'},
        {value: 'permutation', label: 'Conditional permutation test'}
      ],
      value: initial.gating,
      onChange: value => {
        compute.setParameters({gating: value});
        updateEnabled();
        updateLegendNote();
      }
    });
    const alternativeSelect = controls.addSelect<HotSpotsAlternative>({
      label: 'Permutation alternative (compile-time option, all variants precompiled)',
      options: [...HOT_SPOTS_ALTERNATIVES],
      value: initial.alternative,
      onChange: value => compute.setParameters({alternative: value})
    });
    controls.addNote(
      'Folded is symmetric about the simulated mean. On skewed data, low-low rows sit in the ' +
        'short tail and drop out, so folded keeps far fewer points than two-sided.'
    );
    const permutationsControl = controls.addSlider({
      label: 'Permutations (per-frame parameter)',
      min: 49,
      max: HOT_SPOTS_MAXIMUM_PERMUTATIONS,
      step: 50,
      value: initial.permutations,
      format: value => `${value}`,
      onChange: value => compute.setParameters({permutations: value})
    });
    const seedControl = controls.addSlider({
      label: 'Permutation seed (per-frame parameter)',
      min: 1,
      max: 20,
      step: 1,
      value: initial.seed,
      format: value => `${value}`,
      onChange: value => compute.setParameters({seed: value})
    });
    controls.addSlider({
      label: 'Neighborhood radius (per-frame parameter)',
      min: 100,
      max: 1500,
      step: 25,
      value: initial.radiusMeters,
      format: value => `${value} m`,
      onChange: value => compute.setParameters({radiusMeters: value})
    });
    const significanceControl = controls.addSlider({
      label: 'Moran significance level (per-frame parameter)',
      min: 0.001,
      max: 0.2,
      step: 0.001,
      value: initial.significanceLevel,
      format: value => `p ≤ ${value.toFixed(3)}`,
      onChange: value => compute.setParameters({significanceLevel: value})
    });
    const fdrToggle = controls.addToggle({
      label: 'Benjamini-Hochberg FDR (compile-time option, both variants precompiled)',
      value: initial.falseDiscoveryRate,
      onChange: value => compute.setParameters({falseDiscoveryRate: value})
    });
    const updateEnabled = () => {
      const {statistic, gating} = compute.getParameters();
      const moran = statistic === 'local-moran';
      const permutationGated = moran && gating === 'permutation';
      gatingSelect.setDisabled(!moran);
      alternativeSelect.setDisabled(!permutationGated);
      permutationsControl.setDisabled(!permutationGated);
      seedControl.setDisabled(!permutationGated);
      significanceControl.setDisabled(!moran || gating === 'none');
      fdrToggle.setDisabled(moran && gating !== 'analytic');
    };
    updateEnabled();
    controls.addToggle({
      label: 'Show not significant points',
      value: showNotSignificant,
      onChange: value => {
        showNotSignificant = value;
        context.updateLayers();
      }
    });
    for (const legend of hotSpotsStyle.getLegends(initial)) controls.addLegend(legend);
    const legendNote = controls.addNote('');
    const updateLegendNote = () =>
      legendNote.setValue(hotSpotsStyle.describe(compute.getParameters()));
    updateLegendNote();
    controls.addReadout('Points', formatCount(pointCount));
    controls.addReadout('Value', 'bike-parking spaces');
    readouts.hot = controls.addReadout('Hot 99 / 95 / 90% (HH)');
    readouts.cold = controls.addReadout('Cold 99 / 95 / 90% (LL)');
    readouts.outlier = controls.addReadout('Outliers LH / HL');
    readouts.insignificant = controls.addReadout('Not significant');
    readouts.quadrant = controls.addReadout('Quadrant points HH / LH / LL / HL');
    readouts.quadrantTotal = controls.addReadout('Points with a quadrant (of n)');
    readouts.permutation = controls.addReadout('Permutation significant / not (p ≤ level)');
    readouts.global = controls.addReadout('Global Moran I / pseudo p / z_sim');
    readouts.moments = controls.addReadout('Mean / std. deviation');
    readouts.capacity = controls.addReadout('Neighbor capacity');
    controls.addReadout('Data', parking.attribution);

    return {
      getCompiledGraphs: compute.getCompiledGraphs,
      encode: (commandEncoder, frame) => compute.encode(commandEncoder, frame.frameIndex),
      getLayers() {
        const {statistic} = compute.getParameters();
        const coordinateOrigin: [number, number, number] = [
          parking.origin[0],
          parking.origin[1],
          0
        ];
        const common = {
          coordinateOrigin,
          positions: compute.outputs.positions,
          instanceCount: compute.outputs.pointCount,
          values: compute.outputs.classes[statistic],
          valueFormat: 'uint32' as const,
          colormap: 'category' as const,
          noDataValue: HOT_SPOTS_NO_DATA_VALUE,
          noDataColor: HOT_SPOTS_HIDDEN
        };
        const layers: Layer[] = [];
        if (showNotSignificant) {
          layers.push(
            new SpatialAnalysisPointLayer({
              ...common,
              id: `hot-spots-insignificant-${statistic}`,
              radiusPixels: 3,
              palette: getHotSpotsInsignificantPalette()
            })
          );
        }
        layers.push(
          new SpatialAnalysisPointLayer({
            ...common,
            id: `hot-spots-significant-${statistic}`,
            radiusPixels: 5.5,
            palette: getHotSpotsSignificantPalette(statistic)
          })
        );
        return layers;
      },
      destroy: compute.destroy
    };
  }
};
