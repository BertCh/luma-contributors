// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {addGPUCommandNodes} from '@luma.gl/graph/gpu-core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {GPUWideNumericScan} from '@luma-contributors/analysis/dataframe-analysis';
import type {KernelAuditCase} from './kernel-audit-context';

/** Build both word widths at the core binding limit, with empty and trailing chunks. */
export const WIDE_NUMERIC_SCAN_CASES: readonly KernelAuditCase[] = ([2, 4] as const).flatMap(
  wordsPerRow =>
    [false, true].map(
      (withValidity): KernelAuditCase => ({
        exportName: 'GPUWideNumericScan',
        name: `${wordsPerRow * 32}-bit maximum binding chunks, ${withValidity ? 'validity and selection' : 'unmasked'}`,
        build: ({device, graph, view, maximumRows}) => {
          const chunkLengths = [maximumRows(wordsPerRow * 4), 0, 1];
          const makeVector = <Format extends 'uint32' | 'uint32x2' | 'uint32x4'>(
            format: Format,
            components: number
          ) =>
            new GPUVector({
              name: `audit-${format}`,
              type: 'data',
              format,
              // The audit context owns all device buffers, including when a build or compile fails.
              ownsData: false,
              data: chunkLengths.map(
                (length, index) =>
                  new GPUData({
                    format,
                    length,
                    ownsBuffer: false,
                    buffer: device.createBuffer({
                      id: `audit-${format}-${index}`,
                      byteLength: Math.max(1, length) * components * 4,
                      usage: Buffer.STORAGE
                    })
                  })
              )
            });
          const words = wordsPerRow === 2 ? makeVector('uint32x2', 2) : makeVector('uint32x4', 4);
          const scan = new GPUWideNumericScan({
            id: 'wide-numeric-audit',
            words,
            wordsPerRow,
            validity: withValidity ? makeVector('uint32', 1) : undefined,
            selectionExpression: withValidity ? 'keepWideRow(globalRow)' : undefined,
            helperSource: withValidity
              ? 'fn keepWideRow(row: u32) -> bool {return (row & 1u) == 0u;}'
              : undefined,
            resultWordCount: 19,
            output: view('uint32', 19),
            phaseBodies: [
              `if (!selected) {return;}
if (!present) {atomicAdd(&result[2], 1u); return;}
let narrowing = classifyIntegerNarrowing(pair, true);
atomicMax(&result[0], narrowing.errorUnits.y);
${wordsPerRow === 4 ? 'atomicXor(&result[10], words[wordOffset + 3u]);' : ''}`,
              `if (!selected || !present) {return;}
let narrowing = classifyDoubleNarrowing(pair);
atomicMax(&result[1], narrowing.errorTop);`
            ]
          });
          const nodes = scan.getCommandNodes(graph);
          if (nodes.length !== scan.estimatedCommandNodeCount) {
            throw new Error(
              'GPUWideNumericScan audit node count must equal its construction estimate'
            );
          }
          addGPUCommandNodes(graph, nodes);
        }
      })
    )
);
