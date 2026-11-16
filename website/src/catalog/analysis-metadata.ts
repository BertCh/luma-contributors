// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Adapt the public ontology source entry point, so development and extraction see unbuilt changes.
import {
  getGPUAnalysisOperatorCatalog,
  getGPUAnalysisOperatorQuestions,
  getGPUAnalysisWebsiteDataTypes,
  resolveGPUAnalysisOperatorKinds,
  type GPUAnalysisInterpretationSpec,
  type GPUAnalysisQuestion,
  type GPUSpatialAnalysisDataKind
} from '../../../modules/analysis/src/gpu-spatial-analysis/ontology/index';
import type {CatalogEntry} from './schema';

export type CatalogAnalysisMetadata = {
  questions: readonly GPUAnalysisQuestion[];
  dataKinds: {
    input: readonly GPUSpatialAnalysisDataKind[];
    output: readonly GPUSpatialAnalysisDataKind[];
  };
};

/** Website prose uses the shared caution vocabulary; numerical sentences need actual run facts. */
export type CatalogInterpretation = {
  summary: string;
  /** Shared cautions when applicable, or a method limit independent of any particular run. */
  cautions: readonly (GPUAnalysisInterpretationSpec['cautions'][number] | {text: string})[];
};

const operatorsBySymbol = new Map(
  getGPUAnalysisOperatorCatalog().map(operator => [operator.exportName, operator])
);

// These seven public constructors are inventory entries, but not ontology operators. Their kinds
// describe the real helper contract; keeping exceptions explicit makes new omissions fail closed.
export const CATALOG_HELPER_KINDS: Readonly<Record<string, CatalogAnalysisMetadata['dataKinds']>> =
  {
    createGPUComputationCaptures: {input: ['parameters'], output: ['summary']},
    createGPUComputationMetrics: {input: ['parameters'], output: ['summary']},
    createGPUComputationRecord: {input: ['parameters'], output: ['summary']},
    createHealpixDGGSJSON: {input: ['cells', 'attributes'], output: ['cells', 'attributes']},
    makeGPURasterBand: {input: ['attributes', 'parameters'], output: ['surface']},
    makeGPURasterMetadata: {input: ['parameters'], output: ['parameters']},
    makePointCloudLocalPositions: {input: ['points'], output: ['points']}
  };

/** No geometry or domain heuristic: every catalog symbol must resolve or be a named helper. */
export function getCatalogAnalysisMetadata(symbol: string): CatalogAnalysisMetadata {
  const operator = operatorsBySymbol.get(symbol);
  if (operator) {
    const kinds = resolveGPUAnalysisOperatorKinds(operator);
    return {
      questions: getGPUAnalysisOperatorQuestions(operator),
      dataKinds: {input: kinds.inputs, output: kinds.outputs}
    };
  }
  const helper = Object.hasOwn(CATALOG_HELPER_KINDS, symbol)
    ? CATALOG_HELPER_KINDS[symbol]
    : undefined;
  if (helper) return {questions: ['prepare-data'], dataKinds: helper};
  throw new Error(
    `website-catalog symbol "${symbol}" needs shared analysis metadata or an explicit helper contract`
  );
}

/** Project shared ports to website facets; dimensionality refines points into lidar clouds. */
export function getCatalogDataTypes(
  entry: Pick<CatalogEntry, 'symbol' | 'dataKinds'>
): CatalogEntry['dataTypes'] {
  const kinds = entry.dataKinds ?? getCatalogAnalysisMetadata(entry.symbol).dataKinds;
  const isPointCloud =
    operatorsBySymbol.get(entry.symbol)?.package ===
      '@luma-contributors/analysis/pointcloud-analysis' ||
    entry.symbol === 'makePointCloudLocalPositions';
  const projectKinds = (ports: readonly GPUSpatialAnalysisDataKind[]) => [
    ...new Set(
      ports.flatMap(kind =>
        kind === 'points' && isPointCloud
          ? ['point-cloud' as const]
          : getGPUAnalysisWebsiteDataTypes(kind)
      )
    )
  ];
  return {input: projectKinds(kinds.input), output: projectKinds(kinds.output)};
}

/** Method limits for the named operations. Dynamic cautions require actual run evidence. */
const NAMED_METHOD_CAUTIONS: Readonly<Record<string, readonly string[]>> = {
  GPUHotSpotAnalysis: [
    'Read the neighbourhood and minimum-neighbour settings before comparing scores; apply a multiple-testing correction when testing many locations.'
  ],
  GPULocalMoran: [
    'Neighbour weights set the meaning of local association. Analytic scores and permutation p-values use different reference distributions.'
  ],
  GPUQuantileLISA: [
    'The chosen quantile defines membership. Account for multiple locations when interpreting significance.'
  ],
  addHotSpotAnalysisRecipe: [
    'Cell size and grid rings define the scale. Cells below the minimum-neighbour count have no score.'
  ],
  addSpaceTimeHotSpotsRecipe: [
    'Time slices and neighbourhoods define the trend scale. Repeated local tests need a stated significance and correction policy.'
  ],
  addRateClusterMapRecipe: [
    'Inspect small populations and islands; smoothing changes the rates being compared. Permutation resolution limits attainable significance.'
  ],
  addSpatialRegressionRecipe: [
    'Inspect residual dependence and model diagnostics. Association does not establish causation.'
  ],
  GPUOrdinaryLeastSquares: [
    'Inspect residuals, collinearity and fit diagnostics. Coefficients describe association rather than causation.'
  ],
  GPUGeographicallyWeightedRegression: [
    'Bandwidth and local collinearity affect coefficients; local estimates are not independent tests.'
  ],
  GPUKriging: [
    'Prediction variance is conditional on the supplied variogram; it is not a guarantee of calibrated uncertainty.'
  ],
  GPUInverseDistanceWeighting: [
    'Distance units, search radius and power determine the surface; this method provides no prediction uncertainty.'
  ],
  GPUKernelDensity: [
    'Bandwidth changes the apparent pattern. Grid boundaries can truncate the kernel support.'
  ],
  GPUSpatialClustering: [
    'Radius and minimum points define density groups; noise points are not assigned to a cluster.'
  ],
  addClusterAndOutlineRecipe: [
    'Density thresholds define the groups; a hull is an outline and need not describe occupied space inside it.'
  ],
  addPointsInPolygonsChoroplethRecipe: [
    'Counts reflect population and area as well as intensity; use rates when that is the question.'
  ],
  addZonalDistributionRecipe: [
    'Zone boundaries determine membership; a summary can hide variation within each zone.'
  ],
  addDrainageNetworkRecipe: [
    'Pits and flat cells can interrupt drainage; depression filling and stream threshold change the network.'
  ],
  addDriveTimeCatchmentRecipe: [
    'Travel bands use the supplied edge costs; verify their units and convergence before calling them drive times.'
  ]
};

/** Keep authored explanations as the interpretation source; do not invent a result from a poster. */
export function attachCatalogAnalysisMetadata<
  Entry extends Pick<CatalogEntry, 'symbol'> & {
    explanation?: string;
    interpretation?: CatalogInterpretation;
  }
>(
  entry: Entry
): Omit<Entry, 'dataTypes' | keyof CatalogAnalysisMetadata> &
  CatalogAnalysisMetadata &
  Pick<CatalogEntry, 'dataTypes'> & {interpretation?: CatalogInterpretation} {
  const metadata = getCatalogAnalysisMetadata(entry.symbol);
  return {
    ...entry,
    ...metadata,
    dataTypes: getCatalogDataTypes({...entry, ...metadata}),
    ...(entry.explanation && !entry.interpretation
      ? {
          interpretation: {
            summary: entry.explanation,
            cautions: (NAMED_METHOD_CAUTIONS[entry.symbol] ?? []).map(text => ({text}))
          }
        }
      : {})
  };
}
