// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Pinned native `naturalBreaks` signature; typed arrays sort identically to number arrays. */
export type NativeNaturalBreaks = (values: ArrayLike<number>, classCount: number) => number[];

export type NativeJenksWorkerRequest = {
  /** Finite selected Float32 values, transferred to the worker. */
  values: Float32Array;
  classCount: number;
};

export type NativeJenksWorkerResponse = {breaks: number[]} | {error: string};

/** Computes the exact native breaks for a request; shared by the worker, the fallback and tests. */
export function computeNativeJenksBreaks(
  request: NativeJenksWorkerRequest,
  naturalBreaks: NativeNaturalBreaks
): NativeJenksWorkerResponse {
  try {
    return {breaks: naturalBreaks(request.values, request.classCount)};
  } catch (error) {
    return {error: error instanceof Error ? error.message : String(error)};
  }
}
