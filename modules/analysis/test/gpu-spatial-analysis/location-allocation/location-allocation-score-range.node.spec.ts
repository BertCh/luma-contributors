// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {GPULocationAllocation} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-location-allocation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {getObjective, runGreedyOnCPU} from './location-allocation-oracle';
import {computeLocationAllocationOnCPU} from '../../../src/gpu-spatial-analysis/location-allocation/cpu-location-allocation-operation';

function interpretSwapKernel(
  kernel: {source: string; constants?: Record<string, number>},
  memory: Record<string, Float32Array | Uint32Array>,
  reductionGroups?: number
): void {
  const bindings: Record<number, Record<number, Float32Array | Uint32Array>> = {0: {}};
  for (const binding of new WgslReflect(kernel.source).storage) {
    expect(memory[binding.name], binding.name).toBeDefined();
    bindings[binding.group][binding.binding] = memory[binding.name];
  }
  let source = kernel.source
    .replace(/^\s*_ = [^;]+;/gm, '')
    .replace(/\bfrexp\(/g, 'probeFrexp(')
    .replace(/\bldexp\(/g, 'probeLdexp(');
  if (reductionGroups !== undefined) {
    // WgslExec does not synchronize workgroup barriers. A one-lane reduction visits the same
    // four positive terms; raw totals overflow in either order and binary-scaled terms are exact.
    source = source
      .replace(/@workgroup_size\(\d+\)/, '@workgroup_size(1)')
      .replace(/const SWAP_WORKGROUP_SIZE: u32 = \d+u;/, 'const SWAP_WORKGROUP_SIZE: u32 = 1u;');
  }
  source += `
struct ProbeFraction {fract: f32, exp: i32}
fn probeFrexp(value: f32) -> ProbeFraction {
 if (value == 0.0) { return ProbeFraction(0.0, 0); }
 let bits = bitcast<u32>(value);
 return ProbeFraction(bitcast<f32>((bits & 0x807fffffu) | 0x3f000000u), i32((bits >> 23u) & 255u) - 126);
}
fn probeLdexp(value: f32, exponent: i32) -> f32 {return value * exp2(f32(exponent));}`;
  new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
    'main',
    [reductionGroups ?? 1, 1, 1],
    bindings,
    {constants: kernel.constants}
  );
}

for (const sparse of [false, true])
  for (const scale of [1, 2 ** 124])
    it(`swap tolerance retains a representable improvement, scale ${scale}, sparse ${sparse}`, () => {
      const costs = Float32Array.from(
        [5, 13, 14, 4, 4, 8, 4, 5, 13, 15, 5, 2],
        value => value * scale
      );
      const sparseCosts = {
        offsets: Uint32Array.of(0, 3, 6, 9, 12),
        facilities: Uint32Array.of(0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2),
        costs
      };
      const input = sparse ? {sparseCosts} : {costs};
      const options = {
        operation: 'p-median' as const,
        facilityCount: 3,
        demandCount: 4,
        maximumFacilityCount: 2
      };
      const greedy = computeLocationAllocationOnCPU(input, {...options, maximumSwapIterations: 0});
      const improved = computeLocationAllocationOnCPU(input, {
        ...options,
        maximumSwapIterations: 1
      });
      expect(Array.from(greedy.selectedFacilities)).toEqual([1, 1, 0]);
      expect(Array.from(improved.selectedFacilities)).toEqual([1, 0, 1]);
      expect(Number.isFinite(improved.objective[0])).toBe(true);
      expect(improved.objective[0]).toBe(15 * scale);
      expect(Array.from(improved.status.slice(4))).toEqual([1, 0, 1]);
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
        createTransientView(graph, `swap-range-${serial++}`, format, length);
      try {
        const contributor = new GPULocationAllocation({
          ...(sparse
            ? {
                sparseCosts: {
                  offsets: view('uint32', 5),
                  facilities: view('uint32', 12),
                  costs: view('float32', 12)
                }
              }
            : {costs: view('float32', 12)}),
          facilityCount: 3,
          operation: 'p-median',
          maximumFacilityCount: 2,
          maximumSwapIterations: 2,
          output: {
            selectedFacilities: view('uint32', 3),
            assignments: view('uint32', 4),
            objective: view('float32', 1),
            status: view('uint32', 7)
          }
        });
        const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
        const columnEntries = new Uint32Array(24);
        for (let facility = 0; facility < 3; facility++)
          for (let demand = 0; demand < 4; demand++) {
            columnEntries[facility * 8 + demand * 2] = demand;
            columnEntries[facility * 8 + demand * 2 + 1] = new Uint32Array(
              Float32Array.of(costs[demand * 3 + facility]).buffer
            )[0];
          }
        const memory: Record<string, Float32Array | Uint32Array> = {
          costs,
          rowCosts: costs,
          rowOffsets: sparseCosts.offsets,
          rowFacilities: sparseCosts.facilities,
          selected: Uint32Array.of(1, 1, 0),
          selectedList: new Uint32Array(2),
          slotOfFacility: new Uint32Array(3),
          chosen: Uint32Array.of(0, 0, 0x3f800000),
          status: new Uint32Array(7),
          swapState: new Uint32Array(3),
          demandData: new Uint32Array(12),
          swapTotals: new Float32Array(11),
          pairScores: new Float32Array(12),
          columnOffsets: Uint32Array.of(0, 4, 8, 12),
          columnEntries
        };
        memory['scoreScale'] = memory['chosen'];
        const run = (suffix: string, reductionGroups?: number) => {
          const kernel = kernels.find(kernel => kernel.id === `location-allocation-${suffix}`)!;
          expect(kernel, suffix).toBeDefined();
          interpretSwapKernel(kernel, memory, reductionGroups);
        };
        run('score-scale');
        run('swap-0-list');
        run('swap-0-prepare');
        run(sparse ? 'swap-0-removal' : 'swap-0-totals', sparse ? 3 : 1);
        run('swap-0-score');
        expect(memory['pairScores'][10], `tolerance scale ${memory['swapTotals'][6]}`).toBe(0);
        expect(Number.isFinite(memory['swapTotals'][6])).toBe(true);
        expect(Number.isFinite(memory['pairScores'][11])).toBe(true);
        expect(memory['pairScores'][11]).toBeGreaterThan(0);
        // The hand-worked gain is 3 and the old objective is 18, in the same scaled units.
        expect(memory['pairScores'][11] / memory['swapTotals'][6]).toBeCloseTo(3 / 18, 6);
      } finally {
        device.destroy();
      }
    });

for (const sparse of [false, true])
  for (const operation of ['p-median', 'maximum-coverage'] as const)
    for (const unusedCost of operation === 'p-median' ? [0, 2 ** 125] : [0])
      for (const candidateCost of operation === 'p-median' && unusedCost > 0 ? [0, 1] : [0])
        it(`swap scaling uses contributing products and finite coverage weights: ${operation}, sparse ${sparse}, unused cost ${unusedCost}, candidate cost ${candidateCost}`, () => {
          const scale = operation === 'p-median' ? 2 ** -40 : 2 ** 125;
          const costs =
            operation === 'p-median'
              ? Float32Array.from([
                  2 ** 125,
                  2 ** 125,
                  2 ** 125,
                  unusedCost,
                  0,
                  candidateCost,
                  ...[5, 13, 14, 4, 4, 8, 4, 5, 13, 15, 5, 2].map(value => value * scale)
                ])
              : Float32Array.of(0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1);
          const demandWeights =
            operation === 'p-median'
              ? Float32Array.of(0, 2 ** 125, 1, 1, 1, 1)
              : Float32Array.of(scale, scale, scale, scale, 2 ** 127);
          const demandCount = demandWeights.length;
          const offsets = Uint32Array.from({length: demandCount + 1}, (_, row) => row * 3);
          const facilities = Uint32Array.from({length: costs.length}, (_, entry) => entry % 3);
          const options = {
            operation,
            facilityCount: 3,
            demandCount,
            maximumFacilityCount: 2,
            maximumSwapIterations: 1
          };
          const cpu = computeLocationAllocationOnCPU(
            {
              ...(sparse ? {sparseCosts: {offsets, facilities, costs}} : {costs}),
              demandWeights,
              coverageDistance: Float32Array.of(0)
            },
            options
          );
          expect(Array.from(cpu.selectedFacilities)).toEqual(
            candidateCost ? [1, 1, 0] : operation === 'p-median' ? [1, 0, 1] : [0, 1, 1]
          );
          if (operation === 'p-median' && candidateCost === 0) {
            expect(Array.from(cpu.assignments)).toEqual([0, unusedCost === 0 ? 0 : 2, 0, 0, 0, 2]);
            expect(Array.from(cpu.assignedCosts)).toEqual([
              2 ** 125,
              0,
              5 * scale,
              4 * scale,
              4 * scale,
              2 * scale
            ]);
          }
          expect(cpu.objective[0]).toBe(
            (candidateCost ? 18 : operation === 'p-median' ? 15 : 4) * scale
          );
          expect(Array.from(cpu.status.slice(4))).toEqual(candidateCost ? [0, 1, 0] : [1, 0, 1]);
          const device = createNullWebGPUDevice();
          const graph = new GPUCommandGraph(device);
          let serial = 0;
          const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
            createTransientView(graph, `final2-${serial++}`, format, length);
          try {
            const contributor = new GPULocationAllocation({
              operation,
              facilityCount: 3,
              ...(sparse
                ? {
                    sparseCosts: {
                      offsets: view('uint32', demandCount + 1),
                      facilities: view('uint32', costs.length),
                      costs: view('float32', costs.length)
                    }
                  }
                : {costs: view('float32', costs.length)}),
              demandWeights: view('float32', demandCount),
              coverageDistance: operation === 'p-median' ? undefined : view('float32', 1),
              maximumFacilityCount: 2,
              maximumSwapIterations: 1,
              output: {
                selectedFacilities: view('uint32', 3),
                assignments: view('uint32', demandCount),
                objective: view('float32', 1),
                status: view('uint32', 7)
              }
            });
            const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
            const columnEntries = new Uint32Array(costs.length * 2);
            for (let facility = 0; facility < 3; facility++)
              for (let demand = 0; demand < demandCount; demand++) {
                columnEntries[(facility * demandCount + demand) * 2] = demand;
                columnEntries[(facility * demandCount + demand) * 2 + 1] = new Uint32Array(
                  Float32Array.of(costs[demand * 3 + facility]).buffer
                )[0];
              }
            const memory: Record<string, Float32Array | Uint32Array> = {
              costs,
              rowCosts: costs,
              rowOffsets: offsets,
              rowFacilities: facilities,
              demandWeights,
              coverageDistance: Float32Array.of(0),
              selected: Uint32Array.of(1, 1, 0),
              selectedList: new Uint32Array(2),
              slotOfFacility: new Uint32Array(3),
              chosen: new Uint32Array(3),
              status: new Uint32Array(7),
              swapState: new Uint32Array(3),
              demandData: new Uint32Array(demandCount * 3),
              swapTotals: new Float32Array(11),
              pairScores: new Float32Array(12),
              columnOffsets: Uint32Array.of(0, demandCount, 2 * demandCount, 3 * demandCount),
              columnEntries
            };
            memory['scoreScale'] = memory['chosen'];
            for (const suffix of [
              'score-scale',
              'swap-0-list',
              'swap-0-prepare',
              sparse ? 'swap-0-removal' : 'swap-0-totals',
              'swap-0-score'
            ]) {
              const kernel = kernels.find(kernel => kernel.id === `location-allocation-${suffix}`);
              expect(kernel, suffix).toBeDefined();
              interpretSwapKernel(
                kernel!,
                memory,
                suffix === 'swap-0-removal' ? 3 : suffix === 'swap-0-totals' ? 1 : undefined
              );
            }
            const pair = operation === 'p-median' ? 10 : 4;
            if (operation === 'p-median')
              expect(memory['pairScores'][pair], `objective scale ${memory['swapTotals'][6]}`).toBe(
                candidateCost ? -1 : 0
              );
            expect(Number.isFinite(memory['swapTotals'][6])).toBe(true);
            expect(memory['swapTotals'][6]).toBeGreaterThan(0);
            if (candidateCost) {
              // Clipping differences independently would cancel the costly candidate's penalty.
              expect(memory['pairScores'][4]).toBe(-1);
              return;
            }
            const gain = memory['pairScores'][pair + (operation === 'p-median' ? 1 : 0)];
            expect(gain).toBeGreaterThan(0);
            expect(Number.isFinite(gain)).toBe(true);
            expect(gain / memory['swapTotals'][6]).toBeCloseTo(
              operation === 'p-median' ? 3 / 18 : 1 / 8,
              6
            );
          } finally {
            device.destroy();
          }
        });

for (const sparse of [false, true])
  for (const unreachable of [false, true])
    it(`swap bounds retain the reachability scale and zero optimum: sparse ${sparse}, unreachable ${unreachable}`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
        createTransientView(graph, `swap-bound-${serial++}`, format, length);
      try {
        const contributor = new GPULocationAllocation({
          operation: 'p-median',
          facilityCount: 3,
          maximumFacilityCount: 2,
          maximumSwapIterations: 1,
          ...(sparse
            ? {
                sparseCosts: {
                  offsets: view('uint32', 3),
                  facilities: view('uint32', 6),
                  costs: view('float32', 6)
                }
              }
            : {costs: view('float32', 6)}),
          output: {
            selectedFacilities: view('uint32', 3),
            assignments: view('uint32', 2),
            objective: view('float32', 1),
            status: view('uint32', 7)
          }
        });
        const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
        const kernel = kernels.find(
          kernel => kernel.id === `location-allocation-swap-0-${sparse ? 'removal' : 'totals'}`
        )!;
        const totals = new Float32Array(11);
        const memory = {
          demandData: unreachable
            ? Uint32Array.of(0x7f800000, 0x7f800000, 0xffffffff, 0x3f800000, 0x40000000, 0)
            : Uint32Array.of(0, 0x7e000000, 0, 0, 0x7e000000, 0),
          selectedList: Uint32Array.of(0, 1),
          swapTotals: totals,
          scoreScale: Uint32Array.of(0, 20, 0x3f800000)
        };
        interpretSwapKernel(kernel, memory, sparse ? 3 : 1);
        expect(totals[6]).toBe(unreachable ? 2 ** -20 : 0);
        expect(totals[7]).toBe(2);
        expect(totals[8]).toBe(unreachable ? 20 : 0);
        expect(totals[10]).toBe(unreachable ? Infinity : 0);
      } finally {
        device.destroy();
      }
    });

it('shared oracle regression fixtures exclude the former finite cutoff and unreachable coverage', () => {
  expect(runGreedyOnCPU('p-median', Float32Array.of(2e30, 1e30), 2, 1, 0)).toEqual([1]);
  expect(runGreedyOnCPU('maximum-coverage', Float32Array.of(1, Infinity), 2, 1, Infinity)).toEqual([
    0
  ]);
  expect(getObjective('maximum-coverage', Float32Array.of(1, Infinity), 2, [1], Infinity)).toBe(0);
});

for (const sparse of [false, true])
  for (const scale of [1, 2 ** 125])
    it(`greedy scores stay finite and rank facility 1 first, scale ${scale}, sparse ${sparse}`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
        createTransientView(graph, `range-${serial++}`, format, length);
      try {
        const contributor = new GPULocationAllocation({
          ...(sparse
            ? {
                sparseCosts: {
                  offsets: view('uint32', 3),
                  facilities: view('uint32', 6),
                  costs: view('float32', 6)
                }
              }
            : {costs: view('float32', 6)}),
          facilityCount: 3,
          operation: 'p-median',
          maximumFacilityCount: 2,
          maximumSwapIterations: 0,
          output: {
            selectedFacilities: view('uint32', 3),
            assignments: view('uint32', 2),
            objective: view('float32', 1),
            status: view('uint32', 7)
          }
        });
        const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
        const kernel = kernels.find(kernel => kernel.id === 'location-allocation-score-0')!;
        const costs = Float32Array.from([7, 3, 6, 7, 5, 3], value => value * scale);
        const columnEntries = new Uint32Array(12);
        for (let facility = 0; facility < 3; facility++)
          for (let demand = 0; demand < 2; demand++) {
            columnEntries[facility * 4 + demand * 2] = demand;
            columnEntries[facility * 4 + demand * 2 + 1] = new Uint32Array(
              Float32Array.of(costs[demand * 3 + facility]).buffer
            )[0];
          }
        const memory: Record<string, Float32Array | Uint32Array> = {
          costs,
          selected: new Uint32Array(3),
          bestCosts: Float32Array.of(Infinity, Infinity),
          scores: new Float32Array(6),
          status: new Uint32Array(7),
          columnOffsets: Uint32Array.of(0, 2, 4, 6),
          columnEntries,
          chosen: Uint32Array.of(0, 0, 0x3f800000)
        };
        const bindings: Record<number, Record<number, Float32Array | Uint32Array>> = {0: {}};
        const scaleKernel = kernels.find(
          kernel => kernel.id === 'location-allocation-score-scale'
        )!;
        const scaleBindings: Record<number, Record<number, Float32Array | Uint32Array>> = {0: {}};
        for (const binding of new WgslReflect(scaleKernel.source).storage)
          scaleBindings[binding.group][binding.binding] = memory[binding.name];
        new WgslExec(
          new WgslParser().parse(scaleKernel.source.replace(/^\s*_ = [^;]+;/gm, ''))
        ).dispatchWorkgroups('main', [1, 1, 1], scaleBindings, {constants: scaleKernel.constants});
        expect(memory['chosen'][1]).toBe(Math.max(0, Math.floor(Math.log2(7 * scale)) + 1 - 120));
        for (const binding of new WgslReflect(kernel.source).storage)
          bindings[binding.group][binding.binding] = memory[binding.name];
        const source =
          kernel.source
            .replace(/^\s*_ = [^;]+;/gm, '')
            .replace(/\bfrexp\(/g, 'probeFrexp(')
            .replace(/\bldexp\(/g, 'probeLdexp(') +
          `
struct ProbeFraction {fract: f32, exp: i32}
fn probeFrexp(value: f32) -> ProbeFraction {
 if (value == 0.0) { return ProbeFraction(0.0, 0); }
 let bits = bitcast<u32>(value);
 return ProbeFraction(bitcast<f32>((bits & 0x807fffffu) | 0x3f000000u), i32((bits >> 23u) & 255u) - 126);
}
fn probeLdexp(value: f32, exponent: i32) -> f32 {return value * exp2(f32(exponent));}`;
        new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
          'main',
          [1, 1, 1],
          bindings,
          {constants: kernel.constants}
        );
        const scores = memory['scores'];
        for (const score of scores) expect(Number.isFinite(score)).toBe(true);
        expect(scores[3]).toBeLessThan(scores[1]);
        expect(scores[3]).toBeLessThan(scores[5]);
      } finally {
        device.destroy();
      }
    });
