// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {installScenegraphShadowPatch} from '../gpu-layers/index';

// Scenegraph layers (3D tiles, trip markers) also cast and receive WGSL shadow maps.
// deck.gl 10 needs no host patch for path picking or for the glTF `areaLighting` dependency:
// its path WGSL applies the picking filter and `scenegraphPbrMaterial` inherits luma's dependencies.
installScenegraphShadowPatch();
