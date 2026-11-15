// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createExampleViteConfig} from '../../vite.config.shared';

export default createExampleViteConfig({
  config: {
    // examples/deck/arrow-deck.ts and deck-example-device.ts sit outside this example; resolve their
    // deck.gl imports to this example's deck.gl 9.4 rather than the root copy kepler-compute hoists.
    resolve: {dedupe: ['@deck.gl/core']}
  }
});
