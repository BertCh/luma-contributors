// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Kepler's nested dependencies import `vec2`, `vec3`, `vec4`, `mat3`, `mat4` and `quat` from
// `@math.gl/core`. The installed math.gl 5 beta exports those namespaces from its root entry but
// has no `./vec4`-style subpath exports, so re-export the root entry directly.
export * from '../../../../node_modules/@math.gl/core/dist/index.js';
