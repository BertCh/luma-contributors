// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';

/** Records uploaded words and ownership without a GPU; failures identify the original allocation error. */
export function createWideNumericUploadDevice(failAllocation?: number) {
  const buffers: {
    data: Uint32Array;
    byteLength: number;
    usage: number;
    destroyCount: number;
    destroy(): void;
  }[] = [];
  const failure = new Error('injected buffer allocation failure');
  let allocationCount = 0;
  const device = {
    createBuffer(props: {data: Uint32Array; usage: number}) {
      allocationCount++;
      if (allocationCount === failAllocation) throw failure;
      const buffer = {
        data: new Uint32Array(props.data),
        byteLength: props.data.byteLength,
        usage: props.usage,
        destroyCount: 0,
        destroy() {
          this.destroyCount++;
        }
      };
      buffers.push(buffer);
      return buffer;
    }
  } as unknown as Device;
  return {device, buffers, failure};
}
