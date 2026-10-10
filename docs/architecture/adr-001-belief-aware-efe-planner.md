# ADR-001: Belief-aware EFE action selection for ALFWorld

- Status: accepted for the first experimental implementation
- Date: 2026-10-09

## Context

The research question is whether an LLM policy can make better choices when a
separate learned World Model estimates action outcomes and an explicit belief
state values information-seeking actions. The existing structured World Model
predicts factorized probabilities over parsed facts. Those marginals are useful
for predicted task progress, but they are not a calibrated joint posterior over
all possible hidden environments. Computing a principled posterior entropy or
Bayesian information gain directly from them would overstate what the current
model can support.

ALFWorld has a useful observable inspection protocol: moving to a surface
reveals its contents, while moving to a closed container and opening it takes
two primitive actions. The initial experiment can therefore represent one
task-relevant unknown explicitly: which currently admissible locations contain
the required object.

## Decision

Keep the policy model and World Model separate. The policy model proposes an
action; a bounded shortlist is evaluated with the learned World Model and an
explicit target-location belief. The policy proposal is preserved by default.
The EFE selector may override it only when the proposal is a fallback or
repeated action, or the previous observation made no progress, and only when a
candidate has a positive predicted-progress margin. The agent executes only
the first primitive action, receives the real observation, updates its belief,
and replans.

For `n` uninspected candidate locations and `m` remaining required objects, the
belief uses a uniform `m`-of-`n` approximation over distinct locations:

\[
H(B)=\log_2 {n \choose m}, \qquad
IG(\ell)=h_2(m/n),
\]

where `h₂` is binary entropy. For a surface, the inspection horizon is one; for
a closed container, `go to` is assigned a two-step horizon (one if the
container is already known open), and its expected IG is discounted until the
subsequent `open`. Empty inventories remove a location from the remaining
hypotheses; an observed target reduces the remaining target count. The same
location belief persists across the navigation/open sequence, so a delayed
observation can be linked back to the planned probe.

The candidate score minimized by the planner is:

\[
G(a)= -w_p\,\Delta S_{WM}(a)
      -\beta\,\gamma^{h(a)-1}IG(a)
      +w_c\,h(a)
      +c_{revisit}(a),
\]

where `ΔS_WM` is the change in a weighted, goal-specific score of predicted
fact marginals, `h(a)` is estimated primitive-step cost, and `c_revisit` is a
penalty for re-visiting or re-inspecting a location. Smaller `G` is preferred.
The policy proposal remains in the comparison shortlist. Only admissible
commands are eligible when the matched `admissible` action protocol is enabled.

## Consequences and limits

- The information-gain term is actionable and auditable without pretending
  that the factorized World Model provides a calibrated joint posterior.
- The first version models uncertainty about target location, not uncertainty
  about World Model parameters or arbitrary hidden state.
- The uniform location prior and distinct-location assumption are explicit
  approximations. Multi-object tasks, duplicate objects per location, and
  missing/incorrect admissible-location lists can violate them.
- The World Model contributes learned transition/task-progress predictions;
  this first planner is receding-horizon with a short inspection macro, not
  full tree search or a full Bayesian expected-free-energy derivation.
- Terminal fact scores are supplemented with small admissible-action subgoal
  bonuses for prerequisite steps such as `take`, `clean`, and `move`; repeated
  observation actions receive a stall penalty. This prevents an uncertain
  one-step model from rejecting necessary intermediate actions or selecting an
  unchanged-observation loop.
- The conservative gate is intentional: the current transition model is a
  ranking signal, not a trusted controller. Its validation metrics must be
  reported alongside any EFE result.
- Report the value as a task-location expected information gain proxy. Do not
  describe it as calibrated model epistemic uncertainty without additional
  calibration/ensemble experiments.
- Compare against policy-only under the *same* admissible-action constraint;
  use `--information-gain-weight 0` and `--progress-weight 0` as targeted
  ablations.
