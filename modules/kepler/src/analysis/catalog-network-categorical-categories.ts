// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUColocationQuotient,
  GPULocalCategoryDependence,
  GPUSpatialEntropy,
  GPUCategoryPairCounts,
  GPUGeographicalDetector,
  GPURedcapSingleLinkage,
  GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS,
  GPU_SPATIAL_ENTROPY_LAYOUT,
  GPU_GEOGRAPHICAL_DETECTOR_LAYOUT,
  getGeographicalDetectorPValue,
  getGPURedcapSingleLinkageParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {COORDINATE_SYSTEM_PARAMETER, type CatalogOperationBinding} from './catalog-binding-types';
import {withCatalogGpuSession} from './catalog-gpu-session';
import {buildDistanceBandWeights} from './catalog-platform4-weights';
import {makeLocalFrame} from './catalog-local-frame';
import {
  getPointPositions,
  type CatalogCoordinateSystem,
  type CatalogFeatureCollection
} from './catalog-geojson';
import {
  makeInputSpec,
  makeOutputSpec,
  makeNumberParameter,
  makeStringParameter
} from './catalog-platform3-b-common';
import {
  MAXIMUM_ROWS,
  makeBinding,
  makeResult,
  readPositions,
  readNumbers,
  readCategories,
  uploadWeights,
  getSourceRow,
  getNumber,
  getString
} from './catalog-network-categorical-common';

const OPERATIONS = [
  ['colocation-quotient', 'Colocation quotient', 'GPUColocationQuotient'],
  ['local-category-dependence', 'Local category dependence', 'GPULocalCategoryDependence'],
  ['spatial-entropy', 'Spatial entropy', 'GPUSpatialEntropy'],
  ['category-pair-counts', 'Category pair counts', 'GPUCategoryPairCounts'],
  ['geographical-detector', 'Geographical factor detector', 'GPUGeographicalDetector'],
  ['redcap-single-linkage', 'Constrained single linkage', 'GPURedcapSingleLinkage']
] as const;

/** Bounded CPU topology preparation reuses the catalog distance-band builder; no statistics run here. */
export function prepareCategoryWeights(
  collection: CatalogFeatureCollection,
  coordinateSystem: CatalogCoordinateSystem,
  radius: number
) {
  readPositions(collection, coordinateSystem === 'EPSG:4326');
  const coordinates = collection.features.map(
    feature => [...getPointPositions(feature.geometry)[0]] as [number, number]
  );
  const frame = makeLocalFrame(coordinateSystem, coordinates, {shift: 'minimum'});
  const projected = coordinates.map(position => frame.forward(position));
  if (projected.some(position => !position.every(Number.isFinite)))
    throw new Error('Points cannot be projected into the catalog local frame');
  return buildDistanceBandWeights(
    {
      x: Float64Array.from(projected.map(position => position[0])),
      y: Float64Array.from(projected.map(position => position[1])),
      featureIndex: Int32Array.from(coordinates.map((_, index) => index)),
      coordinates,
      count: coordinates.length,
      droppedCount: 0
    },
    radius
  );
}

export const CATEGORY_BINDINGS: readonly CatalogOperationBinding[] = OPERATIONS.map(
  ([id, title, contributor]) => {
    const detector = id === 'geographical-detector';
    const regionalization = id === 'redcap-single-linkage';
    const local = id === 'local-category-dependence' || regionalization;
    return makeBinding({
      id,
      title,
      domain: regionalization ? 'clustering' : 'analysis',
      summary: `${title} using the existing ${contributor} contributor.`,
      contributorNames: [contributor],
      inputs: [
        makeInputSpec('points', 'Categorical points', ['point'], MAXIMUM_ROWS, MAXIMUM_ROWS)
      ],
      parameters: [
        ...(!regionalization
          ? [makeStringParameter('categoryProperty', 'Category property', 'category')]
          : []),
        ...(detector || regionalization
          ? [makeStringParameter('valueProperty', 'Value property', 'value')]
          : []),
        ...(!detector
          ? [
              COORDINATE_SYSTEM_PARAMETER,
              makeNumberParameter('radius', 'Distance band radius', 1000, 1e-6, 1e9, 'crs')
            ]
          : []),
        ...(regionalization
          ? [
              makeNumberParameter(
                'regionCount',
                'Target region count',
                2,
                1,
                MAXIMUM_ROWS,
                undefined,
                'integer'
              ),
              makeNumberParameter(
                'componentIterations',
                'Component iterations',
                32,
                1,
                256,
                undefined,
                'integer'
              )
            ]
          : [])
      ],
      outputs: [
        makeOutputSpec(
          id,
          title,
          local ? 'point' : 'none',
          local
            ? ['source_index', 'source_id']
            : id === 'spatial-entropy' || detector
              ? ['statistic']
              : ['origin_category', 'destination_category'],
          Math.max(MAXIMUM_ROWS, 32 ** 2)
        ),
        ...(id === 'colocation-quotient'
          ? [
              makeOutputSpec(
                'colocation-local',
                'Local colocation quotients',
                'point',
                ['source_index', 'destination_category'],
                MAXIMUM_ROWS * 32
              )
            ]
          : [])
      ],
      declaredBehavior: [
        'Categories are encoded in first-seen order with type retained; null/missing categories use 0xffffffff. At most 32 categories and 1024 source rows.',
        'Each feature must contain one finite point. Invalid coordinates or required numeric values fail before dispatch. Non-finite output statistics publish null.',
        'Distance bands use the shared catalog local frame (meters for EPSG:4326, native planar units otherwise), with symmetric positive weights 1/(1+distance/radius). Self links are excluded.',
        'Single linkage cuts the heaviest spanning-tree edges; this is not the SSD partition of REDCAP. Disconnected components can exceed the requested region count. Convergence and exhausted iteration status are reported.'
      ],
      limitation: regionalization
        ? 'One finite attribute, standardized by the contributor. No minimum region size or SSD partition.'
        : detector
          ? 'One pre-stratified categorical factor and one response; no interaction detector.'
          : 'Point distance-band topology only, bounded CPU preparation. LICD uses binary links; CLQ uses distance-decay shares. No permutation inference.',
      runGpu: async (context, inputs, values) => {
        const points = inputs['points'];
        const rowCount = points.features.length;
        readPositions(points, !detector && values['coordinateSystem'] === 'EPSG:4326');
        const encoded = regionalization
          ? undefined
          : readCategories(points, getString(values, 'categoryProperty'));
        const categoryCount = encoded?.labels.length ?? 1;
        const numericValues =
          detector || regionalization
            ? readNumbers(points, getString(values, 'valueProperty'))
            : undefined;
        const preparedWeights = detector
          ? undefined
          : prepareCategoryWeights(
              points,
              values['coordinateSystem'] as CatalogCoordinateSystem,
              getNumber(values, 'radius')
            );
        return withCatalogGpuSession(context.device, `catalog-${id}`, async session => {
          const weights = preparedWeights ? uploadWeights(session, preparedWeights) : undefined;
          const categories = encoded
            ? session.input('categories', encoded.categories, 'uint32')
            : undefined;
          const pairCounts = session.output('pair-counts', 'uint32', categoryCount ** 2);
          const categoryRows = session.output('category-rows', 'uint32', categoryCount);
          const statistics = session.output(
            'statistics',
            'float32',
            rowCount * GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS.length
          );
          const globalQuotients = session.output('global', 'float32', categoryCount ** 2);
          const localQuotients = session.output('local', 'float32', rowCount * categoryCount);
          const summary = session.output('summary', 'float32', 8);
          const labels = session.output('labels', 'uint32', rowCount);
          const regionCount = session.output('region-count', 'uint32', 1);
          const converged = session.output('converged', 'uint32', 1);
          const exhausted = session.output('exhausted', 'uint32', 1);
          const invalid = session.output('invalid', 'uint32', 1);
          if (id === 'colocation-quotient')
            session.graph.add(
              new GPUColocationQuotient({
                weights: weights!,
                categories: categories!,
                categoryCount,
                output: {globalQuotients: globalQuotients.view, localQuotients: localQuotients.view}
              })
            );
          else if (id === 'local-category-dependence')
            session.graph.add(
              new GPULocalCategoryDependence({
                weights: weights!,
                categories: categories!,
                categoryCount,
                output: {statistics: statistics.view}
              })
            );
          else if (id === 'spatial-entropy')
            session.graph.add(
              new GPUSpatialEntropy({
                weights: weights!,
                categories: categories!,
                categoryCount,
                output: {summary: summary.view, pairCounts: pairCounts.view}
              })
            );
          else if (id === 'category-pair-counts')
            session.graph.add(
              new GPUCategoryPairCounts({
                weights: weights!,
                categories: categories!,
                categoryCount,
                output: {pairCounts: pairCounts.view, categoryRowCounts: categoryRows.view}
              })
            );
          else if (detector)
            session.graph.add(
              new GPUGeographicalDetector({
                values: session.input('values', numericValues!, 'float32'),
                labels: categories!,
                labelCapacity: categoryCount,
                output: {summary: summary.view}
              })
            );
          else
            session.graph.add(
              new GPURedcapSingleLinkage({
                weights: weights!,
                values: session.input('values', numericValues!, 'float32'),
                parameters: session.input(
                  'parameters',
                  getGPURedcapSingleLinkageParameterValues({
                    regionCount: getNumber(values, 'regionCount')
                  }),
                  'uint32'
                ),
                componentIterations: getNumber(values, 'componentIterations'),
                output: {
                  labels: labels.view,
                  regionCount: regionCount.view,
                  status: {
                    converged: converged.view,
                    iterationLimitReached: exhausted.view,
                    invalidCount: invalid.view
                  }
                }
              })
            );
          session.run();
          if (local) {
            const output = regionalization
              ? await session.readUint32(labels)
              : await session.readFloat32(statistics);
            const statisticNames = Object.entries(GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS).filter(
              ([name]) => name !== 'length'
            );
            const result = makeResult(
              id,
              title,
              {
                source_index: 'int32',
                source_id: 'string',
                geometry: 'string',
                ...(regionalization
                  ? {region: 'int32' as const}
                  : Object.fromEntries(statisticNames.map(([name]) => [name, 'float64' as const])))
              },
              points.features.map((_, index) => ({
                ...getSourceRow(points, index),
                ...(regionalization
                  ? {region: output[index]}
                  : Object.fromEntries(
                      statisticNames.map(([name, slot]) => [name, output[index * 6 + slot]])
                    ))
              })),
              regionalization
                ? {
                    regionCount: await session.readScalar(regionCount),
                    converged: await session.readScalar(converged),
                    iterationLimitReached: await session.readScalar(exhausted),
                    invalidCount: await session.readScalar(invalid)
                  }
                : {}
            );
            return result;
          }
          if (id === 'spatial-entropy' || detector) {
            const output = await session.readFloat32(summary);
            const layout = detector ? GPU_GEOGRAPHICAL_DETECTOR_LAYOUT : GPU_SPATIAL_ENTROPY_LAYOUT;
            const rows = Object.entries(layout)
              .filter(([name]) => name !== 'length')
              .map(([name, slot]) => ({statistic: name, value: output[slot]}));
            if (detector)
              rows.push({statistic: 'pValue', value: getGeographicalDetectorPValue(output)});
            return makeResult(id, title, {statistic: 'string', value: 'float64'}, rows);
          }
          const output =
            id === 'colocation-quotient'
              ? await session.readFloat32(globalQuotients)
              : await session.readUint32(pairCounts);
          const counts =
            id === 'category-pair-counts' ? await session.readUint32(categoryRows) : undefined;
          const result = makeResult(
            id,
            title,
            {
              origin_category: 'string',
              destination_category: 'string',
              value: 'float64',
              origin_rows: 'int32'
            },
            Array.from(output, (value, index) => ({
              origin_category: encoded!.labels[Math.floor(index / categoryCount)],
              destination_category: encoded!.labels[index % categoryCount],
              value,
              origin_rows: counts ? counts[Math.floor(index / categoryCount)] : null
            }))
          );
          if (id === 'colocation-quotient') {
            const quotients = await session.readFloat32(localQuotients);
            const localResult = makeResult(
              'colocation-local',
              'Local colocation quotients',
              {
                source_index: 'int32',
                source_id: 'string',
                geometry: 'string',
                destination_category: 'string',
                quotient: 'float64'
              },
              Array.from(quotients, (quotient, index) => ({
                ...getSourceRow(points, Math.floor(index / categoryCount)),
                destination_category: encoded!.labels[index % categoryCount],
                quotient
              }))
            );
            result.tables.push(...localResult.tables);
          }
          return result;
        });
      }
    });
  }
);
