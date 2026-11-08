// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {GPURasterFractionalCategoricalStatistics} from '../../../src/raster-analysis/gpu-raster-fractional-categorical-statistics';
import {GPURasterEVI} from '../../../src/raster-analysis/gpu-raster-spectral-indices';
import {
  GPURasterConservativeRemapping,
  GPURasterTexture,
  GPURasterTemporal,
  GPURasterRegistration,
  GPURasterRegionGraph,
  GPURasterSpectralUnmixing,
  GPURasterSeasonalBreak
} from '../../../src/raster-analysis/remote-sensing/index';
import type {BenchmarkCase, BenchmarkScope, Output} from './raster-performance-harness';

const floats = (count: number, value: (index: number) => number) =>
  Float32Array.from({length: count}, (_, index) => value(index));
const integers = (count: number, value: (index: number) => number) =>
  Uint32Array.from({length: count}, (_, index) => value(index));
const fill = (count: number, value: number) => Array<number>(count).fill(value);
const band = (
  scope: BenchmarkScope,
  name: string,
  values: Float32Array,
  updated?: Float32Array
) => ({
  id: name,
  format: 'float32' as const,
  storage: {kind: 'buffer' as const, values: scope.input(name, values, 'float32', updated)}
});

/** Independent double-precision normal-equation solve, only for these small well-conditioned fixtures. */
function leastSquares(rows: number[][], response: number[]) {
  const columns = rows[0].length;
  const system = Array.from({length: columns}, (_, column) =>
    Array.from({length: columns + 1}, (_, following) =>
      rows.reduce(
        (sum, row, index) =>
          sum + row[column] * (following === columns ? response[index] : row[following]),
        0
      )
    )
  );
  for (let column = 0; column < columns; column++) {
    let pivot = column;
    for (let row = column + 1; row < columns; row++)
      if (Math.abs(system[row][column]) > Math.abs(system[pivot][column])) pivot = row;
    [system[column], system[pivot]] = [system[pivot], system[column]];
    const divisor = system[column][column];
    for (let following = column; following <= columns; following++)
      system[column][following] /= divisor;
    for (let row = 0; row < columns; row++)
      if (row !== column) {
        const multiple = system[row][column];
        for (let following = column; following <= columns; following++)
          system[row][following] -= multiple * system[column][following];
      }
  }
  const coefficients = system.map(row => row[columns]);
  const residual = rows.reduce(
    (sum, row, index) =>
      sum +
      (row.reduce((prediction, value, column) => prediction + value * coefficients[column], 0) -
        response[index]) **
        2,
    0
  );
  return {coefficients, residual};
}

function textureReference(values: Float32Array, width: number): Output {
  const features: number[] = [],
    pairCount: number[] = [];
  for (let row = 0; row < width; row++)
    for (let column = 0; column < width; column++) {
      const histogram = [0, 0, 0, 0];
      for (
        let neighborRow = Math.max(0, row - 2);
        neighborRow <= Math.min(width - 1, row + 2);
        neighborRow++
      )
        for (
          let neighborColumn = Math.max(0, column - 2);
          neighborColumn < Math.min(width - 1, column + 2);
          neighborColumn++
        )
          histogram[
            values[neighborRow * width + neighborColumn] * 2 +
              values[neighborRow * width + neighborColumn + 1]
          ]++;
      const count = histogram.reduce((sum, value) => sum + value, 0);
      const probabilities = histogram.map(value => value / count);
      const angular = probabilities.reduce((sum, value) => sum + value * value, 0);
      const entropy = probabilities.reduce(
        (sum, value) => sum - (value ? value * Math.log(value) : 0),
        0
      );
      features.push(1, 1, 0.5, angular, Math.sqrt(angular), -1, entropy);
      pairCount.push(count);
    }
  return {features, pairCount};
}

function seasonalReference(values: Float32Array, cells: number, slices: number): Output {
  const result: Output = {
    coefficients: [],
    components: [],
    breakIndex: [],
    improvement: [],
    status: []
  };
  const baseRows = Array.from({length: slices}, (_, slice) => [
    1,
    slice / slices,
    Math.cos((2 * Math.PI * slice) / 8),
    Math.sin((2 * Math.PI * slice) / 8)
  ]);
  for (let cell = 0; cell < cells; cell++) {
    const response = Array.from(values.slice(cell * slices, (cell + 1) * slices));
    const energy = response.reduce((sum, value) => sum + value * value, 0);
    const score = (residual: number, parameters: number) =>
      slices * Math.log(Math.max(residual, Math.max(1e-20, energy * 1e-12)) / slices) +
      parameters * Math.log(slices);
    const baseline = leastSquares(baseRows, response);
    let chosen = [...baseline.coefficients, 0],
      boundary = 0xffffffff,
      gain = 0;
    for (let split = 8; split <= slices - 8; split++) {
      const fit = leastSquares(
        baseRows.map((row, slice) => [...row, Number(slice >= split)]),
        response
      );
      const candidateGain = score(baseline.residual, 4) - score(fit.residual, 6);
      if (candidateGain > gain) {
        gain = candidateGain;
        boundary = split;
        chosen = fit.coefficients;
      }
    }
    result['coefficients'].push(...chosen);
    result['breakIndex'].push(boundary);
    result['improvement'].push(gain);
    result['status'].push(0);
    for (let slice = 0; slice < slices; slice++) {
      const trend = chosen[0] + (chosen[1] * slice) / slices + (slice >= boundary ? chosen[4] : 0);
      const season = chosen[2] * baseRows[slice][2] + chosen[3] * baseRows[slice][3];
      result['components'].push(trend, season, response[slice] - trend - season);
    }
  }
  return result;
}

export function createRasterBenchmarkCases(): BenchmarkCase[] {
  const cases: BenchmarkCase[] = [];
  for (const large of [false, true]) {
    const side = large ? 128 : 16;
    const count = side * side;
    cases.push({
      name: `RS02-categorical-${side}`,
      shape: {side, cells: count, classes: 4},
      updateKind: 'reclassify all source cells; prepared fractional tuples retained',
      construct(scope) {
        const initial = integers(count, index => index % 4),
          updated = integers(count, index => index % 3);
        const coverage = {
          zoneIds: scope.input(
            'zoneIds',
            integers(count, () => 0),
            'uint32'
          ),
          cellIds: scope.input(
            'cellIds',
            integers(count, index => index),
            'uint32'
          ),
          fractions: scope.input(
            'fractions',
            floats(count, () => 0.5),
            'float32'
          ),
          zoneValidity: scope.input('zoneValidity', new Uint32Array([1]), 'uint32'),
          status: {
            count: scope.input('count', new Uint32Array([count]), 'uint32'),
            requiredCount: scope.input('requiredCount', new Uint32Array([count]), 'uint32'),
            overflow: scope.input('overflow', new Uint32Array([0]), 'uint32')
          }
        };
        scope.graph.add(
          new GPURasterFractionalCategoricalStatistics({
            grid: makeGPURasterMetadata({width: side, height: side}),
            input: {
              id: 'classes',
              format: 'uint32',
              storage: {kind: 'buffer', values: scope.input('classes', initial, 'uint32', updated)}
            },
            zoneCount: 1,
            categoryCapacity: 4,
            coverage,
            output: {
              coverageSums: scope.output('coverageSums', 4, 'float32'),
              classAreas: scope.output('classAreas', 4, 'float32'),
              proportions: scope.output('proportions', 4, 'float32'),
              coveredArea: scope.output('coveredArea', 1, 'float32'),
              validArea: scope.output('validArea', 1, 'float32'),
              missingArea: scope.output('missingArea', 1, 'float32'),
              majority: scope.output('majority', 1, 'uint32'),
              minority: scope.output('minority', 1, 'uint32'),
              validity: scope.output('validity', 1, 'uint32')
            }
          })
        );
        return {
          reference(changed) {
            const totals = [0, 0, 0, 0];
            for (const category of changed ? updated : initial) totals[category] += 0.5;
            const present = totals
              .map((value, category) => ({value, category}))
              .filter(entry => entry.value);
            const majority = [...present].sort(
              (left, right) => right.value - left.value || right.category - left.category
            )[0].category;
            const minority = [...present].sort(
              (left, right) => left.value - right.value || left.category - right.category
            )[0].category;
            return {
              coverageSums: totals,
              classAreas: totals,
              proportions: totals.map(value => value / (count * 0.5)),
              coveredArea: [count * 0.5],
              validArea: [count * 0.5],
              missingArea: [0],
              majority: [majority],
              minority: [minority],
              validity: [1]
            };
          }
        };
      }
    });
    cases.push({
      name: `RS03-remapping-${side}`,
      shape: {sourceSide: side, destinationSide: side / 2},
      updateKind: 'double source values on same compiled overlap and accumulation graph',
      construct(scope) {
        const initial = floats(
            count,
            index => 1 + (index % side) + Math.floor(index / side) * 0.25
          ),
          updated = floats(count, index => initial[index] * 2),
          outputCount = count / 4;
        scope.graph.add(
          new GPURasterConservativeRemapping({
            source: {width: side, height: side, origin: [0, 0], spacing: [1, 1]},
            destination: {width: side / 2, height: side / 2, origin: [0, 0], spacing: [2, 2]},
            values: scope.input('source', initial, 'float32', updated),
            normalization: 'covered-area',
            output: {
              values: scope.output('values', outputCount, 'float32'),
              coveredArea: scope.output('coveredArea', outputCount, 'float32'),
              status: scope.output('status', outputCount, 'uint32')
            }
          })
        );
        return {
          reference(changed) {
            const source = changed ? updated : initial;
            const values = Array.from({length: outputCount}, (_, index) => {
              const first = Math.floor(index / (side / 2)) * 2 * side + (index % (side / 2)) * 2;
              return (
                (source[first] +
                  source[first + 1] +
                  source[first + side] +
                  source[first + side + 1]) /
                4
              );
            });
            return {values, coveredArea: fill(outputCount, 4), status: fill(outputCount, 0)};
          }
        };
      }
    });
    cases.push({
      name: `RS04-texture-${side}`,
      shape: {side, cells: count, levels: 2, radius: 2},
      updateKind: 'invert checkerboard values, same directional histogram graph',
      construct(scope) {
        const initial = floats(count, index => ((index % side) + Math.floor(index / side)) % 2),
          updated = floats(count, index => 1 - initial[index]);
        scope.graph.add(
          new GPURasterTexture({
            grid: makeGPURasterMetadata({width: side, height: side}),
            values: scope.input('source', initial, 'float32', updated),
            levels: 2,
            radius: 2,
            range: [0, 1],
            offset: [1, 0],
            features: scope.output('features', count * 7, 'float32'),
            pairCount: scope.output('pairCount', count, 'uint32')
          })
        );
        return {reference: changed => textureReference(changed ? updated : initial, side)};
      }
    });
    cases.push({
      name: `RS09-evi-${side}`,
      shape: {side, cells: count, bands: 3, bandMathPasses: 6},
      updateKind: 'increase NIR reflectance by 0.1, keep red/blue resident',
      construct(scope) {
        const initial = floats(count, index => 0.6 + (index % 7) * 0.01),
          updated = floats(count, index => initial[index] + 0.1);
        new GPURasterEVI({
          width: side,
          height: side,
          nearInfrared: band(scope, 'nearInfrared', initial, updated),
          red: band(
            scope,
            'red',
            floats(count, () => 0.2)
          ),
          blue: band(
            scope,
            'blue',
            floats(count, () => 0.1)
          ),
          output: scope.output('values', count, 'float32'),
          outputValidity: scope.output('validity', count, 'uint32')
        }).addToGraph(scope.graph);
        return {
          reference: changed => ({
            values: Array.from(
              changed ? updated : initial,
              value =>
                (2.5 * (value - Math.fround(0.2))) /
                (value + 6 * Math.fround(0.2) - 7.5 * Math.fround(0.1) + 1)
            ),
            validity: fill(count, 1)
          })
        };
      }
    });
    for (const operation of ['mean', 'savitzky-golay'] as const) {
      const cells = large ? 512 : 16,
        slices = large ? 31 : 9,
        smoothing = operation === 'savitzky-golay';
      cases.push({
        name: `RS06-${operation}-${cells}`,
        shape: {cells, slices, window: 7, order: 2},
        updateKind: 'add 0.25 to every acquisition, reuse compiled graph',
        construct(scope) {
          const initial = floats(
              cells * slices,
              index =>
                1 +
                Math.floor(index / slices) * 0.01 +
                (index % slices) * 0.1 +
                (index % slices) ** 2 * 0.001
            ),
            updated = floats(initial.length, index => initial[index] + 0.25),
            outputCount = smoothing ? initial.length : cells;
          scope.graph.add(
            new GPURasterTemporal({
              cellCount: cells,
              sliceCount: slices,
              values: scope.input('source', initial, 'float32', updated),
              operation,
              windowLength: 7,
              polynomialOrder: 2,
              output: {
                values: scope.output('values', outputCount, 'float32'),
                counts: scope.output('counts', outputCount, 'uint32'),
                chosenSlice: scope.output('chosenSlice', outputCount, 'uint32'),
                status: scope.output('status', outputCount, 'uint32')
              }
            })
          );
          return {
            reference(changed) {
              const source = changed ? updated : initial;
              const values = Array.from({length: outputCount}, (_, index) => {
                const cell = smoothing ? Math.floor(index / slices) : index;
                if (!smoothing)
                  return (
                    Array.from(source.slice(cell * slices, (cell + 1) * slices)).reduce(
                      (sum, value) => sum + value,
                      0
                    ) / slices
                  );
                const center = index % slices,
                  first = Math.min(slices - 7, Math.max(0, center - 3));
                const rows = Array.from({length: 7}, (_, offset) => {
                  const coordinate = (first + offset - center) / 7;
                  return [1, coordinate, coordinate * coordinate];
                });
                return leastSquares(
                  rows,
                  Array.from(source.slice(cell * slices + first, cell * slices + first + 7))
                ).coefficients[0];
              });
              return {
                values,
                counts: fill(outputCount, smoothing ? 7 : slices),
                chosenSlice: fill(outputCount, 0xffffffff),
                status: fill(outputCount, 0)
              };
            }
          };
        }
      });
    }
    const registrationSide = large ? 64 : 16,
      registrationCount = registrationSide ** 2;
    cases.push({
      name: `RS07-registration-${registrationSide}`,
      shape: {side: registrationSide, cells: registrationCount, upsampleFactor: 4},
      updateKind: 'replace moving image with a different integer periodic translation',
      construct(scope) {
        let state = 0x5eed2026;
        const reference = floats(registrationCount, () => {
          state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
          return state / 4294967296 - 0.5;
        });
        const translated = (horizontal: number, vertical: number) =>
          floats(
            registrationCount,
            index =>
              reference[
                ((Math.floor(index / registrationSide) - vertical + registrationSide) %
                  registrationSide) *
                  registrationSide +
                  (((index % registrationSide) - horizontal + registrationSide) % registrationSide)
              ]
          );
        scope.graph.add(
          new GPURasterRegistration({
            grid: makeGPURasterMetadata({width: registrationSide, height: registrationSide}),
            reference: scope.input('reference', reference, 'float32'),
            moving: scope.input('moving', translated(3, -2), 'float32', translated(-2, 3)),
            upsampleFactor: 4,
            output: {
              shift: scope.output('shift', 1, 'float32x2'),
              peak: scope.output('peak', 1, 'float32'),
              status: scope.output('status', 1, 'uint32')
            }
          })
        );
        return {
          reference: changed => ({shift: changed ? [2, -3] : [-3, 2], status: [0]}),
          validate(actual) {
            expect(actual['peak'][0]).toBeGreaterThan(0.9);
            expect(actual['peak'][0]).toBeLessThanOrEqual(1.0001);
          }
        };
      }
    });
    const regionSide = large ? 128 : 16,
      regions = large ? 32 : 4,
      regionCells = regionSide ** 2;
    cases.push({
      name: `RS08-regions-${regionSide}`,
      shape: {side: regionSide, cells: regionCells, regions},
      updateKind: 'double region means: connected merge becomes separated regions',
      construct(scope) {
        const labels = integers(regionCells, index =>
            Math.floor((index % regionSide) / (regionSide / regions))
          ),
          initial = floats(regionCells, index => labels[index]),
          updated = floats(regionCells, index => initial[index] * 2);
        scope.graph.add(
          new GPURasterRegionGraph({
            grid: makeGPURasterMetadata({width: regionSide, height: regionSide}),
            labels: scope.input('labels', labels, 'uint32'),
            values: scope.input('values', initial, 'float32', updated),
            regionCount: regions,
            mergeThreshold: 1.5,
            maximumIterations: regions,
            output: {
              adjacency: scope.output('adjacency', regions * regions, 'uint32'),
              weights: scope.output('weights', regions * regions, 'float32'),
              mapping: scope.output('mapping', regions, 'uint32'),
              status: scope.output('status', 1, 'uint32')
            }
          })
        );
        return {
          reference(changed) {
            const adjacency = fill(regions * regions, 0);
            for (let row = 0; row < regionSide; row++)
              for (let column = 1; column < regionSide; column++) {
                const first = labels[row * regionSide + column - 1],
                  second = labels[row * regionSide + column];
                if (first !== second) {
                  adjacency[first * regions + second]++;
                  adjacency[second * regions + first]++;
                }
              }
            return {
              adjacency,
              weights: adjacency.map(value => (value ? (changed ? 2 : 1) : NaN)),
              mapping: Array.from({length: regions}, (_, index) => (changed ? index : 0)),
              status: [0]
            };
          }
        };
      }
    });
    const unmixingCells = large ? 256 : 16,
      bands = large ? 32 : 8,
      members = 3;
    cases.push({
      name: `RS10-simplex-${unmixingCells}`,
      shape: {cells: unmixingCells, bands, members, maximumIterations: 512},
      updateKind: 'change known simplex abundances in resident spectra',
      construct(scope) {
        const initialAbundances = [0.2, 0.3, 0.5],
          updatedAbundances = [0.6, 0.1, 0.3];
        const initial = floats(
            unmixingCells * bands,
            index => initialAbundances[(index % bands) % members]
          ),
          updated = floats(initial.length, index => updatedAbundances[(index % bands) % members]);
        scope.graph.add(
          new GPURasterSpectralUnmixing({
            cellCount: unmixingCells,
            bandCount: bands,
            endmemberCount: members,
            values: scope.input('spectra', initial, 'float32', updated),
            endmembers: scope.input(
              'endmembers',
              floats(bands * members, index =>
                Number((index % bands) % members === Math.floor(index / bands))
              ),
              'float32'
            ),
            constraint: 'simplex',
            maximumIterations: 512,
            tolerance: 1e-6,
            output: {
              abundances: scope.output('abundances', unmixingCells * members, 'float32'),
              residual: scope.output('residual', unmixingCells, 'float32'),
              status: scope.output('status', unmixingCells, 'uint32'),
              iterationCounts: scope.output('iterationCounts', unmixingCells, 'uint32')
            }
          })
        );
        return {
          reference: changed => ({
            abundances: Array.from(
              {length: unmixingCells * members},
              (_, index) => (changed ? updatedAbundances : initialAbundances)[index % members]
            ),
            residual: fill(unmixingCells, 0),
            status: fill(unmixingCells, 0)
          }),
          validate(actual) {
            for (const value of actual['iterationCounts']) {
              expect(value).toBeGreaterThan(0);
              expect(value).toBeLessThanOrEqual(512);
            }
          }
        };
      }
    });
    const seasonalCells = large ? 32 : 4,
      seasonalSlices = large ? 64 : 24;
    cases.push({
      name: `RS11-seasonal-${seasonalCells}`,
      shape: {cells: seasonalCells, slices: seasonalSlices, period: 8, minimumSegmentLength: 8},
      updateKind: 'change positive level break to negative break in same graph',
      construct(scope) {
        const source = (jump: number) =>
          floats(seasonalCells * seasonalSlices, index => {
            const slice = index % seasonalSlices;
            return (
              2 +
              Math.floor(index / seasonalSlices) * 0.1 +
              slice / seasonalSlices +
              0.5 * Math.cos((2 * Math.PI * slice) / 8) +
              0.25 * Math.sin((2 * Math.PI * slice) / 8) +
              (slice >= seasonalSlices / 2 ? jump : 0)
            );
          });
        const initial = source(3),
          updated = source(-2);
        scope.graph.add(
          new GPURasterSeasonalBreak({
            cellCount: seasonalCells,
            sliceCount: seasonalSlices,
            values: scope.input('source', initial, 'float32', updated),
            period: 8,
            minimumSegmentLength: 8,
            output: {
              coefficients: scope.output('coefficients', seasonalCells * 5, 'float32'),
              components: scope.output('components', seasonalCells * seasonalSlices * 3, 'float32'),
              breakIndex: scope.output('breakIndex', seasonalCells, 'uint32'),
              improvement: scope.output('improvement', seasonalCells, 'float32'),
              status: scope.output('status', seasonalCells, 'uint32')
            }
          })
        );
        return {
          reference: changed =>
            seasonalReference(changed ? updated : initial, seasonalCells, seasonalSlices)
        };
      }
    });
  }
  return cases;
}
