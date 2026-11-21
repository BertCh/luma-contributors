// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Compatibility entry: the implementation belongs to the public analysis package.
import {scanWideNumericColumnOnGpu as scanLibraryWideNumericColumnOnGpu} from '@luma-contributors/analysis/dataframe-analysis';

/** Preserve the example's numeric error field while delegating scanning to the library. */
export async function scanWideNumericColumnOnGpu(
  props: Parameters<typeof scanLibraryWideNumericColumnOnGpu>[0]
): Promise<
  Omit<Awaited<ReturnType<typeof scanLibraryWideNumericColumnOnGpu>>, 'maximumAbsoluteError'> & {
    maximumAbsoluteError: number;
  }
> {
  const report = await scanLibraryWideNumericColumnOnGpu(props);
  return {...report, maximumAbsoluteError: Number(report.maximumAbsoluteError)};
}
