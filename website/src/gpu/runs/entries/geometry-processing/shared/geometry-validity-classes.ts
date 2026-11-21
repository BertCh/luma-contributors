// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Per-feature display classes; shared by the live picture and its device regression. */
export const GEOMETRY_VALIDITY_CLASSES_WGSL = `let bits = mask[maskOffset + index];
  var result = 4u;
  if (bits == 0u) { result = 0u; }
  else if ((bits & 16u) != 0u) { result = 1u; }
  else if ((bits & 96u) != 0u) { result = 2u; }
  else if (bits == 128u) { result = 3u; }
  else if ((bits & 256u) != 0u) { result = 6u; }
  // Repeated vertices and orientation are advisory; an unchecked row is incomplete.
  if (validity[validityOffset + index] == 0u && (bits & ~136u) == 0u) { result = 5u; }
  classes[classesOffset + index] = result;`;
