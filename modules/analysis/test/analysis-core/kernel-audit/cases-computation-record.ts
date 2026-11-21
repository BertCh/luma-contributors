// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUTransferMask} from '../../../src/computation-record/gpu-transfer-mask';
import type {KernelAuditCase} from './kernel-audit-context';

/**
 * Computation-record producers that compile shaders (CORE-115). `GPUTransferMask` is built for
 * both relation layouts and both modes with every view at the storage-binding limit; a CSR
 * relation's offsets hold one more entry than the source, so its source is one row shorter.
 * `GPUSnapshotTap` is a copy node with no shader; the audit records it as skipped with its reason.
 */
export const COMPUTATION_RECORD_AUDIT_CASES: KernelAuditCase[] = (['index', 'csr'] as const)
  .flatMap(layout => (['any', 'count'] as const).map(mode => ({layout, mode})))
  .map(
    ({layout, mode}): KernelAuditCase => ({
      exportName: 'GPUTransferMask',
      name: `${layout}-${mode} at the storage-binding limit`,
      build: ({graph, view, maximumRows}) => {
        const rows = maximumRows(4);
        const sourceLength = layout === 'index' ? rows : rows - 1;
        graph.add(
          new GPUTransferMask({
            id: `transfer-mask-${layout}-${mode}`,
            source: view('uint32', sourceLength, 'source'),
            relation:
              layout === 'index'
                ? {layout, targets: view('uint32', rows, 'targets')}
                : {
                    layout,
                    offsets: view('uint32', rows, 'offsets'),
                    neighbors: view('uint32', rows, 'neighbors')
                  },
            target: view('uint32', rows, 'target'),
            mode
          })
        );
      }
    })
  );
