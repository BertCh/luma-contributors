// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Only this allowlisted opt-in flag crosses into the browser's minimal process shim.
globalThis.process.env['A5_AB'] = import.meta.env.LUMA_A5_AB;
