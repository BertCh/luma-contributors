// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUJacquezTest,
  GPULocalKnox,
  GPUNearRepeatKnox,
  GPUSpaceTimeHawkes,
  GPU_LOCAL_KNOX_SUMMARY,
  GPU_LOCAL_KNOX_SUMMARY_LENGTH,
  GPU_NEAR_REPEAT_CELL_SUMMARY,
  GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH,
  GPU_SPACE_TIME_SUMMARY,
  GPU_SPACE_TIME_SUMMARY_LENGTH,
  getGPUSpaceTimeParameterValues,
  getGPUSpaceTimeHawkesParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  GPUSpaceTimeDiggleTest,
  GPUSpaceTimeKFunction,
  GPU_SPACE_TIME_DIGGLE_SUMMARY,
  GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH,
  getGPUSpaceTimeDiggleRandomParameterValues,
  getGPUSpaceTimeKFunctionParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import type {CatalogGpuContext, CatalogRunOutput} from './catalog-binding-types';
import {appendCatalogRow, makeCatalogTable} from './catalog-geojson';
import {withCatalogGpuSession} from './catalog-gpu-session';
import type {Inputs, Values} from './catalog-platform3-b-common';
import {
  finiteSpacetimeValue,
  getSpacetimeSummary,
  makeSpacetimeGrid,
  makeSpacetimeNeighbors,
  makeSpacetimePointTable,
  makeSpacetimeSummaryTable,
  readSpacetimePoints,
  uploadSpacetimeNeighbors
} from './catalog-spacetime-common';

type EventKind = 'jacquez' | 'local-knox' | 'near-repeat' | 'hawkes' | 'diggle' | 'k-function';

export async function runSpacetimeEvents(
  kind: EventKind,
  context: CatalogGpuContext,
  inputs: Inputs,
  values: Values
): Promise<CatalogRunOutput> {
  const points = readSpacetimePoints(
    inputs['points'],
    kind === 'near-repeat'
      ? {
          ...values,
          temporalThreshold: Number(values['temporalThreshold']) / Number(values['bandCount'])
        }
      : values
  );
  if (points.count < 2) throw new Error('Spacetime event analysis needs at least two events');
  const permutations = Number(values['permutations'] ?? 0);
  const seed = Number(values['seed'] ?? 0);
  const summary: CatalogRunOutput['summary'] = {
    featureCount: points.count,
    permutations,
    seed,
    timeOrigin: points.timeOrigin,
    timeUnit: String(values['timeUnit']),
    maximumFeatures: 512
  };
  if (kind === 'hawkes') {
    const spatialThreshold = Number(values['spatialThreshold']);
    const temporalThreshold = Number(values['temporalThreshold']);
    const {bounds} = makeSpacetimeGrid(points, {
      ...values,
      spatialBandwidth: spatialThreshold,
      gridSize: 4
    });
    const duration = Math.max(1, ...points.times);
    const windowArea = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]);
    const maximumIterations = Number(values['maximumIterations']);
    const maximumParents = Number(values['maximumParents']);
    const result = await withCatalogGpuSession(context.device, 'catalog-hawkes', async session => {
      const estimates = session.output('estimates', 'float32', 4);
      const probability = session.output('probability', 'float32', points.count);
      const likelihood = session.output('fitted-likelihood', 'float32', 1);
      const status = {
        converged: session.output('converged', 'uint32', 1),
        iterationCount: session.output('iterations', 'uint32', 1),
        iterationLimitReached: session.output('limit', 'uint32', 1),
        candidateOverflow: session.output('candidate-overflow', 'uint32', 1),
        invalidCount: session.output('invalid', 'uint32', 1),
        approximation: session.output('approximation', 'uint32', 1)
      };
      session.graph.add(
        new GPUSpaceTimeHawkes({
          positions: session.input('positions', points.positions, 'float32x2'),
          times: session.input('times', points.times, 'float32'),
          parameters: session.input(
            'parameters',
            getGPUSpaceTimeHawkesParameterValues({
              windowArea,
              windowDuration: duration,
              maximumLag: temporalThreshold,
              maximumDistance: spatialThreshold,
              initialBackground: points.count / (2 * windowArea * duration),
              initialBranchingRatio: 0.5,
              initialDecayRate: 1 / temporalThreshold,
              initialSpatialStandardDeviation: spatialThreshold
            }),
            'float32'
          ),
          maximumIterations,
          maximumParents,
          output: {
            estimates: estimates.view,
            fittedLogLikelihood: likelihood.view,
            backgroundProbability: probability.view,
            status: Object.fromEntries(
              Object.entries(status).map(([name, output]) => [name, output.view])
            )
          }
        })
      );
      session.run();
      return {
        estimates: await session.readFloat32(estimates),
        probability: await session.readFloat32(probability),
        likelihood: await session.readFloat32(likelihood),
        status: Object.fromEntries(
          await Promise.all(
            Object.entries(status).map(async ([name, output]) => [
              name,
              await session.readScalar(output)
            ])
          )
        )
      };
    });
    Object.assign(
      summary,
      getSpacetimeSummary(result.estimates, {
        backgroundRate: 0,
        branchingRatio: 1,
        decayRate: 2,
        spatialStandardDeviation: 3
      }),
      result.status,
      {
        fittedLogLikelihood: Number.isFinite(result.likelihood[0])
          ? result.likelihood[0]
          : 'undefined',
        maximumIterations,
        maximumParents,
        spatialThreshold,
        temporalThreshold,
        windowArea,
        windowDurationSeconds: duration,
        tolerance: 1e-4
      }
    );
    return {
      tables: [
        makeSpacetimePointTable(points, {
          background_probability: Array.from(result.probability, finiteSpacetimeValue)
        }),
        makeSpacetimeSummaryTable(summary)
      ],
      diagnostics: [],
      summary,
      dispatchedContributors: ['GPUSpaceTimeHawkes']
    };
  }
  if (kind === 'diggle' || kind === 'k-function') {
    const {bounds} = makeSpacetimeGrid(points, {...values, gridSize: 4});
    const classCount = Number(values['classCount']);
    const cells = classCount * classCount;
    const result = await withCatalogGpuSession(
      context.device,
      'catalog-space-time-k',
      async session => {
        const parameters = session.input(
          'parameters',
          getGPUSpaceTimeKFunctionParameterValues({
            bounds,
            maximumDistance: Number(values['maximumDistance']),
            maximumTimeLag: Number(values['maximumTimeLag']),
            timeWindow: [-1, Math.max(1, ...points.times) + 1]
          }),
          'float32'
        );
        const common = {
          positions: session.input('positions', points.positions, 'float32x2'),
          times: session.input('times', points.times, 'float32'),
          parameters,
          gridSize: [4, 4] as const,
          distanceClassCount: classCount,
          timeClassCount: classCount,
          edgeCorrection: 'none' as const
        };
        const outputK = session.output('k', 'float32', cells);
        const outputD = session.output('d', 'float32', cells);
        const outputSummary =
          kind === 'diggle'
            ? session.output('summary', 'float32', GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH)
            : undefined;
        if (kind === 'diggle') {
          session.graph.add(
            new GPUSpaceTimeDiggleTest({
              ...common,
              randomParameters: session.input(
                'random',
                getGPUSpaceTimeDiggleRandomParameterValues(seed),
                'uint32'
              ),
              permutationCount: permutations,
              output: {
                d: outputD.view,
                k: outputK.view,
                statistics: session.output('statistics', 'float32', permutations + 1).view,
                summary: outputSummary!.view
              }
            })
          );
        } else {
          session.graph.add(
            new GPUSpaceTimeKFunction({...common, output: {k: outputK.view, d: outputD.view}})
          );
        }
        session.run();
        return {
          k: await session.readFloat32(outputK),
          d: await session.readFloat32(outputD),
          summary: outputSummary ? await session.readFloat32(outputSummary) : undefined
        };
      }
    );
    const table = makeCatalogTable('bands', 'Spacetime distance/time bands', {
      distance_class: 'int32',
      time_class: 'int32',
      distance_upper: 'float64',
      time_upper_seconds: 'float64',
      k: 'float64',
      d: 'float64'
    });
    for (let distance = 0; distance < classCount; distance++)
      for (let time = 0; time < classCount; time++) {
        const cell = distance * classCount + time;
        appendCatalogRow(table, {
          distance_class: distance,
          time_class: time,
          distance_upper: ((distance + 1) * Number(values['maximumDistance'])) / classCount,
          time_upper_seconds: ((time + 1) * Number(values['maximumTimeLag'])) / classCount,
          k: finiteSpacetimeValue(result.k[cell]),
          d: finiteSpacetimeValue(result.d[cell])
        });
      }
    Object.assign(
      summary,
      {
        maximumDistance: Number(values['maximumDistance']),
        maximumTimeLag: Number(values['maximumTimeLag']),
        studyWindowPadding: Number(values['spatialBandwidth']),
        classCount,
        edgeCorrection: 'none',
        maximumClassesPerAxis: 8
      },
      result.summary ? getSpacetimeSummary(result.summary, GPU_SPACE_TIME_DIGGLE_SUMMARY) : {}
    );
    return {
      tables: [table, ...(result.summary ? [makeSpacetimeSummaryTable(summary)] : [])],
      diagnostics: [],
      summary,
      dispatchedContributors: [
        kind === 'diggle' ? 'GPUSpaceTimeDiggleTest' : 'GPUSpaceTimeKFunction'
      ]
    };
  }
  const neighborCount = Number(values['neighborCount'] ?? 0);
  if (kind === 'jacquez' && neighborCount >= points.count)
    throw new Error('Jacquez neighborCount must be below event count');
  const spatialThreshold = Number(values['spatialThreshold'] ?? Infinity);
  const temporalThreshold = Number(values['temporalThreshold'] ?? 0);
  const neighbors = makeSpacetimeNeighbors(
    points.positions,
    spatialThreshold,
    kind === 'jacquez' ? neighborCount : undefined
  );
  const bandCount = Number(values['bandCount'] ?? 1);
  const cells = bandCount * bandCount;
  const distanceEdges = Array.from(
    {length: bandCount + 1},
    (_, index) => (index * spatialThreshold) / bandCount
  );
  const timeEdges = Array.from(
    {length: bandCount + 1},
    (_, index) => (index * temporalThreshold) / bandCount
  );
  const result = await withCatalogGpuSession(
    context.device,
    'catalog-space-time-test',
    async session => {
      const pairs = uploadSpacetimeNeighbors(session, neighbors);
      const times = session.input('times', points.times, 'float32');
      const parameters = session.input(
        'parameters',
        getGPUSpaceTimeParameterValues({
          seed,
          permutationCount: permutations,
          timeThreshold: temporalThreshold
        }),
        'uint32'
      );
      const output = session.output(
        'summary',
        'float32',
        kind === 'local-knox'
          ? points.count * GPU_LOCAL_KNOX_SUMMARY_LENGTH
          : kind === 'near-repeat'
            ? cells * GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH
            : GPU_SPACE_TIME_SUMMARY_LENGTH
      );
      if (kind === 'jacquez')
        session.graph.add(
          new GPUJacquezTest({
            neighbors: pairs,
            times,
            parameters,
            k: neighborCount,
            maximumPermutations: permutations,
            output: {
              statistics: session.output('statistics', 'uint32', permutations + 1).view,
              summary: output.view
            }
          })
        );
      else if (kind === 'local-knox')
        session.graph.add(
          new GPULocalKnox({
            pairs,
            times,
            parameters,
            maximumPermutations: permutations,
            output: {summary: output.view}
          })
        );
      else
        session.graph.add(
          new GPUNearRepeatKnox({
            pairs,
            times,
            parameters,
            distanceEdges,
            timeEdges,
            maximumPermutations: permutations,
            output: {
              counts: session.output('counts', 'uint32', (permutations + 1) * cells).view,
              summary: output.view
            }
          })
        );
      session.run();
      return session.readFloat32(output);
    }
  );
  Object.assign(summary, {
    maximumPermutations: 999,
    pairEntries: neighbors.neighbors.length,
    spatialThreshold: Number.isFinite(spatialThreshold) ? spatialThreshold : 'nearest neighbors',
    temporalThreshold,
    neighborCount
  });
  if (kind === 'jacquez') {
    Object.assign(summary, getSpacetimeSummary(result, GPU_SPACE_TIME_SUMMARY));
    return {
      tables: [makeSpacetimeSummaryTable(summary)],
      summary,
      diagnostics: [],
      dispatchedContributors: ['GPUJacquezTest']
    };
  }
  if (kind === 'local-knox') {
    const columns = Object.fromEntries(
      Object.entries(GPU_LOCAL_KNOX_SUMMARY).map(([name, offset]) => [
        name,
        Array.from({length: points.count}, (_, index) =>
          finiteSpacetimeValue(result[index * GPU_LOCAL_KNOX_SUMMARY_LENGTH + offset])
        )
      ])
    );
    Object.assign(summary, {
      observedTotal: columns['observed'].reduce<number>((total, value) => total + (value ?? 0), 0),
      pValueMinimum: Math.min(
        ...columns['pseudoPGreater'].filter((value): value is number => value !== null)
      ),
      inference: 'conditional permutation, greater tail, no multiple testing correction'
    });
    return {
      tables: [makeSpacetimePointTable(points, columns)],
      summary,
      diagnostics: [],
      dispatchedContributors: ['GPULocalKnox']
    };
  }
  const table = makeCatalogTable('bands', 'Spacetime distance/time bands', {
    distance_class: 'int32',
    time_class: 'int32',
    distance_lower: 'float64',
    distance_upper: 'float64',
    time_lower_seconds: 'float64',
    time_upper_seconds: 'float64',
    ...Object.fromEntries(
      Object.keys(GPU_NEAR_REPEAT_CELL_SUMMARY).map(name => [name, 'float64' as const])
    )
  });
  for (let distance = 0; distance < bandCount; distance++)
    for (let time = 0; time < bandCount; time++) {
      const offset = (distance * bandCount + time) * GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH;
      appendCatalogRow(table, {
        distance_class: distance,
        time_class: time,
        distance_lower: distanceEdges[distance],
        distance_upper: distanceEdges[distance + 1],
        time_lower_seconds: timeEdges[time],
        time_upper_seconds: timeEdges[time + 1],
        ...Object.fromEntries(
          Object.entries(GPU_NEAR_REPEAT_CELL_SUMMARY).map(([name, index]) => [
            name,
            finiteSpacetimeValue(result[offset + index])
          ])
        )
      });
    }
  Object.assign(summary, {
    bandCount,
    maximumBandsPerAxis: 8,
    observedTotal: (table.columns['observed'] as number[]).reduce(
      (total, value) => total + value,
      0
    ),
    pValueMinimum: Math.min(...(table.columns['pseudoPGreater'] as number[])),
    inference: 'time permutation, greater tail, no multiple testing correction'
  });
  return {tables: [table], summary, diagnostics: [], dispatchedContributors: ['GPUNearRepeatKnox']};
}
