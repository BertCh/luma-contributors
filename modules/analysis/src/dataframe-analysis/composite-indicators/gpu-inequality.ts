// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  GraphDataView,
  GraphVectorView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGPUStorageBindingByteLength,
  validateMaximumCount,
  validateSinglePackedView,
  validateRequirement,
  validateCommandNodeBudget,
  getGraphViewChunks
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateGraphChunkBindingRanges,
  validateDisjointWritableViews
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH,
  getGPUInequalityMaximumLorenzKnotTotal,
  getGPUInequalityMaximumZoneCount,
  GPU_INEQUALITY_PARAMETER_LENGTH
} from './inequality-parameters';
import {
  getInequalityGiniTileAreaBody,
  getInequalityGiniTileTotalsBody,
  getInequalityKeysBody,
  getInequalityZoneStatsBody,
  INEQUALITY_GINI_FINISH_BODY,
  INEQUALITY_GINI_TILE_PREFIX_BODY,
  INEQUALITY_GINI_TILE_ROWS,
  INEQUALITY_GINI_FINISH_DECLARATIONS,
  INEQUALITY_GINI_TILE_PREFIX_DECLARATIONS,
  INEQUALITY_GLOBAL_SUMS_BODY,
  INEQUALITY_GLOBAL_SUMS_DECLARATIONS,
  INEQUALITY_REDUCTION_LANES,
  INEQUALITY_HELPERS_WGSL,
  INEQUALITY_STAT,
  INEQUALITY_STATS_STRIDE,
  INEQUALITY_ZONE_KEYS_BODY
} from './inequality-wgsl';

import {createInequalityParallelStatisticsNodes} from './inequality-parallel-statistics';
import {INEQUALITY_PREFIX_TILE_ROWS} from './inequality-lorenz-prefix';
import {validateGraphChunkGroupKernelSupport} from '../../analysis-core/kernels/graph-chunk-groups';
import {getBoundedDispatchLayout} from '@luma.gl/graph/gpu-core';

const OPERATION = 'GPUInequality';

/** Both published GPUSort passes accept at most 2^31 keys; reject larger metadata at construction. */
const MAXIMUM_ROW_COUNT = 0x80000000;
/** Two 32-bit published sorts, including their hierarchical histogram scans; conservatively bounded at 2^31 keys. */
const SORT_COMMAND_NODE_ALLOWANCE = 2048;
/** Four Lorenz summary levels plus carries, six term reductions and publication. */
const STATISTICS_COMMAND_NODE_ALLOWANCE = 64;

/** Smallest number of Lorenz knots. */
const MINIMUM_LORENZ_KNOT_COUNT = 2;
/** Largest number of Lorenz knots. */
const MAXIMUM_LORENZ_KNOT_COUNT = 1025;

/** Caller-owned outputs of {@link GPUInequality}. Every output is optional; at least one is needed. */
export type GPUInequalityOutput = {
  /** Gini coefficient per zone; NaN for an empty zone or zero total income. */
  gini?: GraphDataView<'float32'>;
  /** Theil T index per zone (0 at perfect equality, `ln n` at maximal inequality). */
  theilT?: GraphDataView<'float32'>;
  /** Theil L (mean log deviation) per zone; NaN when the zone has a zero value. */
  theilL?: GraphDataView<'float32'>;
  /**
   * Atkinson index per zone for the per-frame epsilon; NaN when epsilon is at least 1 and the zone
   * has a zero value.
   */
  atkinson?: GraphDataView<'float32'>;
  /** Hoover (Robin Hood) index per zone, the share of income to redistribute for equality. */
  hoover?: GraphDataView<'float32'>;
  /** Palma ratio per zone, income share of the top cut over the bottom cut; NaN if the bottom share is 0. */
  palma?: GraphDataView<'float32'>;
  /** Weighted mean value per zone; NaN for an empty zone. */
  mean?: GraphDataView<'float32'>;
  /** Number of included rows per zone. */
  counts?: GraphDataView<'uint32'>;
  /** Lorenz curve knots, row-major `zone * lorenzKnotCount + knot`; NaN for an empty zone. */
  lorenzKnots?: GraphDataView<'float32'>;
  /**
   * Pooled statistics and the Theil T decomposition, indexed by `GPU_INEQUALITY_GLOBAL_SUMMARY`.
   * At least `GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH` values.
   */
  globalSummary?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUInequality}.
 *
 * Per-frame (no rebuild or recompile): the contents of `values`, `zoneIds`, `weights`, `mask` and
 * `parameters` (Atkinson epsilon and the Palma cuts). Compile-time: the row count, `zoneCount`,
 * `lorenzKnotCount`, and which optional views are present.
 */
export type GPUInequalityProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'inequality'`. */
  id?: string;
  /** Parallel segmented Lorenz coordinates by default. Reference retains the original serial zone walk. */
  mode?: 'parallel' | 'reference';
  /** Packed float32 incomes. Negative and non-finite rows are excluded. */
  values: GraphDataView<'float32'>;
  /** Packed uint32 zone per row. `0xffffffff` or any value `>= zoneCount` excludes the row. */
  zoneIds: GraphDataView<'uint32'>;
  /**
   * Number of zones, at most {@link getGPUInequalityMaximumZoneCount} (3050402 on a default
   * device). Compile-time.
   */
  zoneCount: number;
  /** Optional packed float32 population weights; rows without a finite weight above 0 are excluded. */
  weights?: GraphDataView<'float32'>;
  /** Optional packed `uint32` row mask; zero excludes the row. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Per-frame parameters: float32 view of at least `GPU_INEQUALITY_PARAMETER_LENGTH` elements
   * written with `getGPUInequalityParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Lorenz knots per zone at evenly spaced population fractions, 2 to 1025. Defaults to 11.
   * `zoneCount * lorenzKnotCount` is at most {@link getGPUInequalityMaximumLorenzKnotTotal}
   * (33554432 on a default device). Compile-time.
   */
  lorenzKnotCount?: number;
  /** Caller-owned outputs. */
  output: GPUInequalityOutput;
};

/**
 * Per-zone income inequality indices from a column of non-negative values: Gini, Theil T and L,
 * Atkinson, Hoover, Palma, the Lorenz curve, plus the between/within-zone decomposition of Theil T.
 *
 * A row is included when its mask is non-zero, its zone is below `zoneCount`, its value is finite
 * and non-negative, and (with `weights`) its weight is finite and positive. Everything below is
 * weighted by `w` (1 without weights), with `W = sum w`, `S = sum w x` and `mu = S / W`. Zones with
 * `S = 0` report NaN for every index but the mean and count.
 *
 * - Gini: `1 - sum_i (P_i - P_(i-1)) (L_i + L_(i-1))`, the trapezoid area under the Lorenz curve
 *   of the rows sorted by value, `P` and `L` being the cumulative population and income shares.
 *   With unit weights this equals `2 sum_i i x_(i) / (n S) - (n + 1) / n`; one holder of all
 *   income among `n` rows gives `(n - 1) / n`.
 * - Theil T: `(1 / W) sum w r ln r` with `r = x / mu` and `0 ln 0 = 0`.
 * - Theil L: `-(1 / W) sum w ln r`. A zone containing a zero value reports NaN (zeros are not
 *   dropped, so a result never silently describes a different population).
 * - Atkinson with epsilon `e`: `1 - ((1 / W) sum w r^(1 - e))^(1 / (1 - e))`, and
 *   `1 - exp((1 / W) sum w ln r)` at `e = 1`. For `e < 1` zeros are fine; for `e >= 1` a zero
 *   gives NaN. A result that overflows f32 is NaN.
 * - Hoover: `sum w |x - mu| / (2 S)`.
 * - Palma: `(1 - L(1 - top)) / L(bottom)` for the per-frame cuts (0.1 and 0.4 by default), where
 *   `L(p)` is the Lorenz curve of the rows sorted by value. The curve is linear inside each row
 *   (a row's income is spread over its population share), so cuts that fall inside a row take a
 *   fractional share of it. NaN when `L(bottom)` is 0.
 * - Lorenz knots: `L(k / (K - 1))` for `k = 0 .. K - 1` with the same linear interpolation, so
 *   `L(0) = 0` and `L(1) = 1`.
 * - Global summary: pooled Theil T computed directly from all included rows, and the decomposition
 *   `between = sum_g s_g ln(s_g / p_g)`, `within = sum_g s_g T_g` with income share `s_g` and
 *   population share `p_g`; total equals between plus within up to f32 rounding. Pooled Gini
 *   comes from the globally value-sorted rows.
 *
 * Algorithm: stable value and zone sorts give each zone a value-ordered segment. The default
 * parallel route builds segmented population/income prefixes from bounded 256-row Lorenz walks,
 * hierarchical tail summaries and reverse carries. Row terms use shared fixed-order segment
 * reductions; Lorenz knots and Palma cuts binary-search the cumulative population. No float
 * atomics are used. Float64-oracle tolerances are scaled by `max(1, abs(expected))`: 1e-4 for
 * means, 2e-4 for Gini/Hoover/Lorenz, 5e-4 for Theil/Atkinson/Palma and 1e-3 for global summaries.
 * The parallel route changes the float32 summation order within these tolerances.
 * `mode: 'reference'` retains the original two serial walks and their rounding order, bounded by
 * the published sort maximum of 2^31 rows. Prefer the CPU twin for that reference route.
 * Every scalar row scratch binding and aligned caller binding must fit the device.
 */
export class GPUInequality implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUInequalityProps;
  /** Row count, `values.length`. */
  readonly rowCount: number;
  /** Lorenz knots per zone. */
  readonly lorenzKnotCount: number;
  readonly mode: 'parallel' | 'reference';
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPUInequalityProps) {
    this.id = props.id ?? 'inequality';
    this.props = props;
    const id = this.id;
    this.mode = props.mode ?? 'parallel';
    validateRequirement(
      id,
      this.mode === 'parallel' || this.mode === 'reference',
      'mode must be parallel or reference'
    );
    this.estimatedCommandNodeCount =
      SORT_COMMAND_NODE_ALLOWANCE + STATISTICS_COMMAND_NODE_ALLOWANCE;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    const {output, zoneCount} = props;
    // The device-dependent maximum is checked at declaration.
    validateCapacity(id, 'zoneCount', zoneCount, 1, 0xfffffffe);
    this.lorenzKnotCount = props.lorenzKnotCount ?? 11;
    validateCapacity(
      id,
      'lorenzKnotCount',
      this.lorenzKnotCount,
      MINIMUM_LORENZ_KNOT_COUNT,
      MAXIMUM_LORENZ_KNOT_COUNT
    );
    validateSinglePackedView(props.values, ['float32'], `${id} values`);
    validateSinglePackedView(props.zoneIds, ['uint32'], `${id} zoneIds`);
    this.rowCount = props.values.length;
    validateCapacity(id, 'values length', this.rowCount, 1, MAXIMUM_ROW_COUNT);
    if (props.zoneIds.length !== this.rowCount) {
      throw new Error(`${id} zoneIds length must equal the row count`);
    }
    if (props.weights) {
      validateSinglePackedView(props.weights, ['float32'], `${id} weights`);
      if (props.weights.length !== this.rowCount) {
        throw new Error(`${id} weights length must equal the row count`);
      }
    }
    if (props.mask) {
      validateSinglePackedView(props.mask, ['uint32'], `${id} mask`);
      if (props.mask.length !== this.rowCount) {
        throw new Error(`${id} mask length must equal the row count`);
      }
    }
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_INEQUALITY_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_INEQUALITY_PARAMETER_LENGTH} float32 values`
      );
    }
    const lengths: [keyof GPUInequalityOutput, number, 'float32' | 'uint32'][] = [
      ['gini', zoneCount, 'float32'],
      ['theilT', zoneCount, 'float32'],
      ['theilL', zoneCount, 'float32'],
      ['atkinson', zoneCount, 'float32'],
      ['hoover', zoneCount, 'float32'],
      ['palma', zoneCount, 'float32'],
      ['mean', zoneCount, 'float32'],
      ['counts', zoneCount, 'uint32'],
      ['lorenzKnots', zoneCount * this.lorenzKnotCount, 'float32'],
      ['globalSummary', GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH, 'float32']
    ];
    let outputCount = 0;
    for (const [name, length, format] of lengths) {
      const view = output[name];
      if (!view) {
        continue;
      }
      outputCount++;
      validateSinglePackedView(view, [format], `${id} output.${name}`);
      if (view.length < length) {
        throw new Error(`${id} output.${name} must hold at least ${length} values`);
      }
    }
    if (outputCount === 0) {
      throw new Error(`${id} needs at least one output`);
    }
    {
      const writableOutputs: Readonly<Record<string, GraphDataView | GraphVectorView | undefined>> =
        output;
      const publicationPasses: readonly (readonly (
        | GraphDataView
        | GraphVectorView
        | undefined
      )[])[] = [
        [output.gini, output.theilT, output.theilL, output.atkinson],
        [output.hoover, output.palma, output.mean],
        ...(this.mode === 'reference' ? [[output.counts, output.lorenzKnots]] : []),
        ...(output.globalSummary ? [[output.counts, output.globalSummary]] : [])
      ];
      // Model all logical writes as separate chunks; only the explicit publication groups coexist.
      const writableWordChunks = Object.values(writableOutputs)
        .flatMap(view => (view ? getGraphViewChunks(view) : []))
        .map(
          view =>
            new GraphDataView(view.buffer, {
              format: 'uint32',
              length: (view.length * view.rowByteLength) / 4,
              byteOffset: view.byteOffset,
              byteStride: 4,
              rowByteLength: 4
            })
        );
      const writableWordCount = writableWordChunks.reduce(
        (total, chunk) => total + chunk.length,
        0
      );
      validateDisjointWritableViews(id, {
        outputs: new GraphVectorView({
          id: `${id}-writable-ranges`,
          name: 'outputs',
          format: 'uint32',
          length: writableWordCount,
          valueLength: writableWordCount,
          stride: 1,
          byteStride: 4,
          rowByteLength: 4,
          data: writableWordChunks
        })
      });
      for (const fields of publicationPasses) {
        validateDisjointWritableViews(
          id,
          Object.fromEntries(fields.map((view, index) => [String(index), view]))
        );
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      lengths.map(([name]) => output[name]),
      [props.values, props.zoneIds, props.weights, props.mask, props.parameters]
    );
  }

  /**
   * Returns the key, value-sort, zone-key, zone-sort, zone-statistics, output-copy and optional
   * global-summary nodes in order. Node IDs are `${id}-keys`, `${id}-value-sort-*`,
   * `${id}-zone-keys`, `${id}-zone-sort-*`, `${id}-zone-stats`, `${id}-publish-indices`,
   * `${id}-publish-moments`, `${id}-global-sums` and `${id}-global-gini`.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, rowCount, lorenzKnotCount} = this;
    const {output, zoneCount, values, zoneIds, weights, mask, parameters} = props;
    validateCapacity(id, 'zoneCount', zoneCount, 1, getGPUInequalityMaximumZoneCount(graph.device));
    validateMaximumCount(
      id,
      'zoneCount x lorenzKnotCount',
      zoneCount * lorenzKnotCount,
      getGPUInequalityMaximumLorenzKnotTotal(graph.device)
    );
    validateGraphChunkBindingRanges(graph, id, {
      values,
      zoneIds,
      weights,
      mask,
      parameters,
      ...output
    });
    // Row sort planes, zone statistics and Lorenz knots are separate storage bindings.
    const tileCount = Math.ceil(rowCount / INEQUALITY_GINI_TILE_ROWS);
    for (const [name, length] of [
      ['row scratch', rowCount],
      ['stats scratch', zoneCount * INEQUALITY_STATS_STRIDE],
      ['counts scratch', zoneCount],
      ['lorenz scratch', zoneCount * lorenzKnotCount],
      ...(output.globalSummary
        ? ([['Gini tile scratch', Math.max(2, 2 * tileCount)]] as const)
        : [])
    ] as const) {
      validateMaximumCount(id, `${name} byteLength`, length * 4, graph.device.limits.maxBufferSize);
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        length * 4,
        'reduce values length, zoneCount or lorenzKnotCount'
      );
    }
    if (this.mode === 'parallel') {
      validateGraphChunkGroupKernelSupport(graph, id, 8);
      for (const length of [rowCount, zoneCount, zoneCount * lorenzKnotCount]) {
        getBoundedDispatchLayout(
          id,
          length,
          256,
          graph.device.limits.maxComputeWorkgroupsPerDimension
        );
      }
      for (const [name, length] of [
        ['zone offset scratch', zoneCount + 1],
        ['Lorenz summary scratch', 2 * Math.ceil(rowCount / INEQUALITY_PREFIX_TILE_ROWS)]
      ] as const) {
        validateGPUStorageBindingByteLength(
          graph.device,
          id,
          name,
          length * 4,
          'reduce row count or zoneCount'
        );
        validateMaximumCount(
          id,
          `${name} byteLength`,
          length * 4,
          graph.device.limits.maxBufferSize
        );
      }
    }
    const transient = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, Math.max(length, 1));
    const valueKeys = transient('value-keys', 'uint32', rowCount);
    const rowIndices = transient('row-indices', 'uint32', rowCount);
    const sortedValueKeys = transient('sorted-value-keys', 'uint32', rowCount);
    const sortedRows = transient('sorted-rows', 'uint32', rowCount);
    const zoneKeys = transient('zone-keys', 'uint32', rowCount);
    const sortedZoneKeys = transient('sorted-zone-keys', 'uint32', rowCount);
    const zoneSortedRows = transient('zone-sorted-rows', 'uint32', rowCount);
    const stats = transient('stats', 'float32', zoneCount * INEQUALITY_STATS_STRIDE);
    const counts = output.counts ?? transient('counts', 'uint32', zoneCount);
    const lorenz =
      output.lorenzKnots ?? transient('lorenz', 'float32', zoneCount * lorenzKnotCount);

    const read = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read'
    });
    const write = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read_write'
    });
    const declarations = `const ROW_COUNT: u32 = ${rowCount}u;
const ZONE_COUNT: u32 = ${zoneCount}u;
const KNOT_COUNT: u32 = ${lorenzKnotCount}u;
const STATS_STRIDE: u32 = ${INEQUALITY_STATS_STRIDE}u;
${INEQUALITY_HELPERS_WGSL}`;
    const weightExpression = weights ? 'weights[weightsOffset + row]' : '1.0';
    const nodes: GPUCommandNode<Parameters>[] = [];

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-keys`,
        operation: OPERATION,
        variant: 'keys',
        bindings: [
          read('values', values, 'f32'),
          read('zoneIds', zoneIds, 'u32'),
          ...(mask ? [read('rowMask', mask, 'u32')] : []),
          ...(weights ? [read('weights', weights, 'f32')] : []),
          write('valueKeys', valueKeys, 'u32'),
          write('rowIndices', rowIndices, 'u32')
        ],
        invocationCount: rowCount,
        declarations,
        body: getInequalityKeysBody({hasMask: Boolean(mask), hasWeights: Boolean(weights)})
      })
    );
    nodes.push(
      ...new GPUSort({
        id: `${id}-value-sort`,
        keys: valueKeys,
        values: rowIndices,
        outputKeys: sortedValueKeys,
        outputValues: sortedRows,
        keyBits: 32
      }).getCommandNodes(graph)
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-zone-keys`,
        operation: OPERATION,
        variant: 'zone-keys',
        bindings: [
          read('sortedValueKeys', sortedValueKeys, 'u32'),
          read('sortedRows', sortedRows, 'u32'),
          read('zoneIds', zoneIds, 'u32'),
          write('zoneKeys', zoneKeys, 'u32')
        ],
        invocationCount: rowCount,
        declarations,
        body: INEQUALITY_ZONE_KEYS_BODY
      })
    );
    // Stable radix keeps the value order inside each zone. Keys span [0, zoneCount].
    const zoneKeyBits = Math.max(1, Math.ceil(Math.log2(zoneCount + 1)));
    nodes.push(
      ...new GPUSort({
        id: `${id}-zone-sort`,
        keys: zoneKeys,
        values: sortedRows,
        outputKeys: sortedZoneKeys,
        outputValues: zoneSortedRows,
        algorithm: 'radix',
        keyBits: Math.min(32, zoneKeyBits)
      }).getCommandNodes(graph)
    );
    if (this.mode === 'parallel') {
      nodes.push(
        ...createInequalityParallelStatisticsNodes(graph, {
          id,
          sortedZones: sortedZoneKeys,
          sortedRows: zoneSortedRows,
          values,
          weights,
          zoneCount,
          parameters,
          stats,
          counts,
          lorenz,
          knotCount: lorenzKnotCount
        })
      );
    } else {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-zone-stats`,
          operation: OPERATION,
          variant: 'zone-stats',
          bindings: [
            read('sortedZones', sortedZoneKeys, 'u32'),
            read('sortedRows', zoneSortedRows, 'u32'),
            read('values', values, 'f32'),
            ...(weights ? [read('weights', weights, 'f32')] : []),
            read('params', parameters, 'f32'),
            write('stats', stats, 'f32'),
            write('lorenz', lorenz, 'f32'),
            write('counts', counts, 'u32')
          ],
          invocationCount: zoneCount,
          declarations,
          body: getInequalityZoneStatsBody(weightExpression)
        })
      );
    }

    const publish = (
      step: string,
      entries: [string, number, GraphDataView<'float32'> | undefined][]
    ) => {
      const present = entries.filter(([, , view]) => view);
      if (present.length === 0) {
        return;
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-${step}`,
          operation: OPERATION,
          variant: step,
          bindings: [
            read('stats', stats, 'f32'),
            ...present.map(([name, , view]) => write(name, view!, 'f32'))
          ],
          invocationCount: zoneCount,
          declarations,
          body: present
            .map(
              ([name, slot]) =>
                `${name}[${name}Offset + index] = stats[statsOffset + index * STATS_STRIDE + ${slot}u];`
            )
            .join('\n  ')
        })
      );
    };
    publish('publish-indices', [
      ['gini', INEQUALITY_STAT.GINI, output.gini],
      ['theilT', INEQUALITY_STAT.THEIL_T, output.theilT],
      ['theilL', INEQUALITY_STAT.THEIL_L, output.theilL],
      ['atkinson', INEQUALITY_STAT.ATKINSON, output.atkinson]
    ]);
    publish('publish-moments', [
      ['hoover', INEQUALITY_STAT.HOOVER, output.hoover],
      ['palma', INEQUALITY_STAT.PALMA, output.palma],
      ['mean', INEQUALITY_STAT.MEAN, output.mean]
    ]);

    if (output.globalSummary) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-global-sums`,
          operation: OPERATION,
          variant: 'global-sums',
          bindings: [
            read('stats', stats, 'f32'),
            write('counts', counts, 'u32'),
            write('summary', output.globalSummary, 'f32')
          ],
          invocationCount: INEQUALITY_REDUCTION_LANES,
          workgroupSize: INEQUALITY_REDUCTION_LANES,
          guardIndex: false,
          declarations: `${declarations}${INEQUALITY_GLOBAL_SUMS_DECLARATIONS}`,
          body: INEQUALITY_GLOBAL_SUMS_BODY
        })
      );
      // Pooled Gini by tiles: per-tile totals, a prefix over tiles only, per-tile Lorenz areas, and
      // a sum over tiles. Depth is O(rows / tile + tile) instead of two serial passes over all rows.
      const tileCount = Math.ceil(rowCount / INEQUALITY_GINI_TILE_ROWS);
      const tileTotals = transient('gini-tile-totals', 'float32', 2 * tileCount);
      const tileAreas = transient('gini-tile-areas', 'float32', tileCount);
      const pooledTotals = transient('gini-pooled-totals', 'float32', 2);
      const tileDeclarations = `${declarations}
const GINI_TILE_ROWS: u32 = ${INEQUALITY_GINI_TILE_ROWS}u;
const GINI_TILE_COUNT: u32 = ${tileCount}u;`;
      const sortedReads = [
        read('sortedValueKeys', sortedValueKeys, 'u32'),
        read('sortedRows', sortedRows, 'u32'),
        read('values', values, 'f32'),
        ...(weights ? [read('weights', weights, 'f32')] : [])
      ];
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-global-gini-tile-totals`,
          operation: OPERATION,
          variant: 'global-gini-tile-totals',
          bindings: [...sortedReads, write('tileTotals', tileTotals, 'f32')],
          invocationCount: tileCount,
          declarations: tileDeclarations,
          body: getInequalityGiniTileTotalsBody(weightExpression)
        })
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-global-gini-tile-prefix`,
          operation: OPERATION,
          variant: 'global-gini-tile-prefix',
          bindings: [
            write('tileTotals', tileTotals, 'f32'),
            write('pooledTotals', pooledTotals, 'f32')
          ],
          invocationCount: INEQUALITY_REDUCTION_LANES,
          workgroupSize: INEQUALITY_REDUCTION_LANES,
          guardIndex: false,
          declarations: `${tileDeclarations}${INEQUALITY_GINI_TILE_PREFIX_DECLARATIONS}`,
          body: INEQUALITY_GINI_TILE_PREFIX_BODY
        })
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-global-gini-tile-area`,
          operation: OPERATION,
          variant: 'global-gini-tile-area',
          bindings: [
            ...sortedReads,
            read('tileTotals', tileTotals, 'f32'),
            read('pooledTotals', pooledTotals, 'f32'),
            write('tileAreas', tileAreas, 'f32')
          ],
          invocationCount: tileCount,
          declarations: tileDeclarations,
          body: getInequalityGiniTileAreaBody(weightExpression)
        })
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-global-gini`,
          operation: OPERATION,
          variant: 'global-gini',
          bindings: [
            read('tileAreas', tileAreas, 'f32'),
            read('pooledTotals', pooledTotals, 'f32'),
            write('summary', output.globalSummary, 'f32')
          ],
          invocationCount: INEQUALITY_REDUCTION_LANES,
          workgroupSize: INEQUALITY_REDUCTION_LANES,
          guardIndex: false,
          declarations: `${tileDeclarations}${INEQUALITY_GINI_FINISH_DECLARATIONS}`,
          body: INEQUALITY_GINI_FINISH_BODY
        })
      );
    }
    return nodes;
  }
}
