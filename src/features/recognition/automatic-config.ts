/** Local UI profile derived from the supplied game UI, not a learned model. */
export const AUTOMATIC_BOARD_CONFIG = {
  maxPixels: 16_777_216,
  detectionEdge: 960,
  minCellPixels: 12,
  maxCandidates: 32,
  minMaskCoverage: 0.35,
  minComponentAspect: 0.45,
  maxComponentAspect: 1.4,
  minPitchFraction: 0.6,
  pitchStep: 0.25,
  projectionSamples: 48,
  minGridContrast: 3,
  emptyVariation: 9,
  tileVariation: 16,
  minTileSlope: 6,
  maxTileResidualRatio: 0.88,
} as const;
