export type Reservation<A> = {
  kind: "root" | "edge";
  rootIndex: number;
  depth: number;
  action: A | null;
};

/** Exhaustive conditional-root traversal; injectable scalar adapters keep unit tests transition-free. */
export function conditionalReference<S, A, R, E>(
  initial: S,
  initialRewards: R,
  api: {
    actions: (s: S) => A[];
    advance: (s: S, a: A, rewards: R) => { state: S; rewards: R };
    evaluate: (s: S, rewards: R) => E;
    compare: (a: E, b: E) => number;
    reserve: (entry: Reservation<A>) => void;
  },
  maxNodes: number,
  maxActions: number,
) {
  if (
    !Number.isSafeInteger(maxNodes) ||
    maxNodes < 1 ||
    !Number.isSafeInteger(maxActions) ||
    maxActions < 1
  )
    throw new Error("Reference bounds");
  const rootActions = api.actions(initial);
  const best: (E | null)[] = rootActions.map(() => null);
  const leaves: { rootIndex: number; path: A[]; evaluation: E }[] = [];
  let nodes = 1,
    maxDepth = 0;
  api.reserve({ kind: "root", rootIndex: -1, depth: 0, action: null });
  function incomplete(reason: string): never {
    const error = new Error(reason) as Error & { partial: unknown };
    error.partial = {
      complete: false,
      nodes,
      maxDepth,
      bestObserved: best,
      leaves,
    };
    throw error;
  }
  function edge(s: S, a: A, rewards: R, rootIndex: number, path: A[]) {
    if (nodes >= maxNodes) incomplete("Reference node cap; labels withheld");
    const depth = path.length + 1;
    if (depth > maxActions) incomplete("Reference depth cap; labels withheld");
    api.reserve({ kind: "edge", rootIndex, depth, action: a });
    nodes++;
    const next = api.advance(s, a, rewards);
    const nextPath = [...path, a],
      actions = api.actions(next.state);
    maxDepth = Math.max(maxDepth, depth);
    if (!actions.length) {
      const value = api.evaluate(next.state, next.rewards);
      leaves.push({ rootIndex, path: nextPath, evaluation: value });
      if (best[rootIndex] === null || api.compare(value, best[rootIndex]!) > 0)
        best[rootIndex] = value;
      return;
    }
    if (depth >= maxActions)
      incomplete("Reference depth cap; nonterminal leaf");
    for (const action of actions)
      edge(next.state, action, next.rewards, rootIndex, nextPath);
  }
  for (const [i, action] of rootActions.entries())
    edge(initial, action, initialRewards, i, []);
  if (!best.length || best.some((e) => e === null))
    incomplete("Missing conditional root evaluation");
  const completeBest = best as E[];
  const optimum = completeBest.reduce((a, b) =>
    api.compare(a, b) >= 0 ? a : b,
  );
  const qualityIndices = completeBest.flatMap((e, i) =>
    api.compare(e, optimum) === 0 ? [i] : [],
  );
  return {
    complete: true,
    nodes,
    maxDepth,
    best: completeBest,
    leaves,
    qualityIndices,
  };
}
