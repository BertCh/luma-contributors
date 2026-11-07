// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LensTwinFile} from './lens-directive';

// Keep recorded figures in their own lazy reference chunk, out of story startup.
const TWIN_FILES = import.meta.glob<LensTwinFile>('../../../../../docs/lens-records/twins/*.json', {
  eager: true,
  import: 'default'
});

export const LENS_TWINS = new Map(
  Object.entries(TWIN_FILES).map(([path, twin]) => [path.split('/').pop()!.slice(0, -5), twin])
);
