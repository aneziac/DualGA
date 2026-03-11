#import "@local/superTemplate:0.4.0": *
#import "@local/ergo:0.2.0": *

#import math_mod:    *
#import physics_mod: *
#import cs_mod:      *
#import pstat_mod:   *

#import "@preview/touying:0.6.1": *
#import "@preview/numbly:0.1.0": numbly
#import "@preview/lovelace:0.3.1": *
#import themes.university: *

#show: university-theme.with(
  aspect-ratio: "16-9",
  config-info(
    title: [Towards Better Statistical Understanding of Watermarking LLMs],
    author: [Nate Annau],
    subtitle: [Based on the 2024 Paper by Cai et al.],
    date: datetime.today(),
    institution: [UC Santa Barbara],
  ),
)

#set heading(numbering: numbly("{1}.", default: "1.1"))

#title-slide()

== Outline <touying:hidden>

#components.adaptive-columns(outline(depth: 1, title: none, indent: 1em))

= Introduction

== Previous Approaches on Watermarking Natural Language

#v(2em)

- "Gumbel Softmax" Scheme --- _Aaronson (2023); Christ et al. (2023)_
// U
- Partition vocabulary and distort probabilities --- _Kirchenbauer et al. (2023)_
- Unique assignments of pseudorandom functions per token via secret keys --- _Kuditipudi et al. (2023)_
// - Curve Pareto optimum against log perplexity --- _Wouters (2023)_

#v(2em)

Key Question: _What is the minimal price for a watermarking algorithm to attain a certain level of detection ability?_

== Model Setup

A language model (LM) $cal(M)$ is a function that treats both a prompt $bu(x)$ and previous $t - 1$ tokens $y_1, dots, y_(t - 1)$ as _context_ $bu(y)_[t - 1]$ and maps this to a probability vector $bu(p)_t in Delta (cal(V))$ as a prediction, where $Delta (cal(V))$ denotes the set of all probability distributions over a vocabulary $cal(V)$.

The probability vector is obtained by applying softmax to a logit vector $bu(l)_t = (l_(t, 1), dots, l_(t, cal(V)))$:
$
  p_(t, k) := exp(l_(t, k)) / (sum_(k' in cal(V)) exp(l_(t, k'))).
$
New tokens $y_t$ are sampled until a terminate symbol $v_"term" in cal(V)$ is sampled or a maximum length $T_"max"$ is reached.

== Defining Watermarking

We wish to develop a model $cal(M)'$ based on a LM $cal(M)$ together with a _detection algorithm_, fitting the following ideal properties:
- The change in output is small enough so as to be unnoticeable to a human
- The change in output is large enough so as to be detectable by the algorithm with high certainty

#v(2em)

We want to formalize this tradeoff, clearly defining the classes of error at play.

== Classes of Error

Let $cal(K)$ be the space of all possible detection keys $K$, and $cal(T)$ be the space of token sequences $bu(y)_[t] = y_1, dots, y_t$.
Define a *watermarked model* $cal(L)'$ as consisting of an algorithm & key pair $(cal(A), cal(K))$ where $cal(A) : cal(K) times cal(T) -> Delta (cal(V))$, i.e., we get a PMF over $cal(V)$.

A *detection procedure* $cal(D) : cal(K) times cal(T) -> {0, 1}$ then determines whether the text came from $cal(L)'$.

Then the two types of errors are (for some prompt $bu(x) = bu(y)_[n < t]$):
$
  alpha_bu(p) (cal(D), bu(x)) &:= PP(cal(D)(K, bu(y)_[T_"max"]) = 1 | bu(y)_[T_"max"] sim bu(p)(x)) quad &&"(false positives)" \
  beta (cal(D), bu(x)) &:= PP(cal(D)(K, bu(y)_[T_"max"]) = 0 | bu(y)_[T_"max"] sim bu(q)^((cal(A), K))(x)) quad &&"(false negatives)"
$

= Distortion vs Detection Ability

== Kirchenbauer et al.

#grid(
  columns: (70%, 25%),
  column-gutter: 2em,
  text(20pt)[
    #pseudocode-list(title: smallcaps[Generalized Soft Watermarking])[
      + $t <- 1, y_0 <- k$
      + *while* $t <= T_"max"$ and $y_(t - 1) != v_"term"$ *do*
        + Use $f(k, y_(t - 1))$ to randomly partition $cal(V)$ into a _green list_ $G$ with $abs(G) = gamma abs(cal(V))$ and a _red list_ $abs(G^c) = (1 - gamma) abs(cal(V))$
        + $bu(q)_t <- "softmax"(bu(p)_t + bu(delta)_t dot bb(1)_G)$
        + Sample a token $y_t sim (q_(t, 1), dots, q_(t, abs(cal(V))))$
        + $t <- t + 1$
      + *end while*
      + *if* $t < T_"max"$ *then*
        + Set all remaining $y_(t + 1), dots, y_(T_"max")$ to be $v_"term"$
      + *end if*
    ]
  ],
  text(15pt)[
    *Input*
      + Language Model ${bu(p)}_(t in [T_"max"]) in Delta (cal(V))$
      + Prompt $bu(x) in cal(T)$
      + Green list ratio $gamma in (0, 1)$
      + Pseudorandom function $f$
      + Perturbations ${bu(delta)}_(t in [T_"max"]) in RR^abs(cal(V))_(> 0)$
      + Random seed $k$

    *Output*
      + Watermarked token sequence $bu(y)_[T_"max"]$
  ],
)

== Distance Measures

#v(1.5em)

With this manipulation, the idea of our detection algorithm $cal(D)$ is simple: how many more green words were there than expected?

#defn[Difference of Green Word Probability (DG)][
  $
    "DG"_t (bu(q)_t) := sum_(v in G) q_(t, v) - sum_(v in G) p_(t, v).
  $
]

This gives us a distinguishability condition.
Further, KL divergence gives us a natural distance measure between our PMFs:
$
  D_bu(p) (bu(q)_t) := D(bu(q)_t, bu(p)_t).
$

== Constrained Optimization Formulation

#v(2em)

Thus, we frame our goal as the following constrained optimization problem:
$
  "OPT"(Delta) &:= min_(bu(delta)_t) 1 / T sum_(t = 1)^T D_(bu(p)) (bu(q)_t (bu(delta)_t)) \
  &"such that" 1 / T sum_(t = 1)^T "DG"_t (bu(q)_t (bu(delta)_t)) >= Delta.
$
where the PMFs $bu(q)_t$ of the watermarked LM are treated as functions of $bu(delta)_t$ alone for brevity.

== Special Cases

We can make our problem
$
  "OPT"(Delta) &:= min_(bu(delta)_t) 1 / T sum_(t = 1)^T D_(bu(p)) (bu(q)_t (bu(delta)_t)) \
  &"such that" 1 / T sum_(t = 1)^T "DG"_t (bu(q)_t (bu(delta)_t)) >= Delta.
$
simpler by
- restricting $delta_(t, 1) = thin cdots thin = delta_(t, abs(cal(V))) = delta_t #h(5em) "OPT"_("static" v) (Delta)$
- restricting $delta_(t, k) = delta forall t in [T_"max"], v in cal(V) #h(3.3em) "OPT"_("static" t, v) (Delta)$

== Generalization Weakness

#text(size: 16pt)[
  We can deduce that the objective function can be viewed as an unbiased estimator for the KL divergence between the LMs.

  #prop[
    $
      EE[sum_(t = 1)^T_"max" D_bu(p) (bu(q)_t (bu(delta)_t))]
      &= sum_(t = 1)^T_"max" D(bu(q)_(t|[t - 1]) (bu(x)) concat bu(p)_(t|[t - 1])|bu(q)_[t - 1] (x)) \
      &= D(bu(q)(bu(x)) concat bu(p)(bu(x)))
    $
  ][]

  This leads to two important results.

  #prop[
    Suppose $"OPT"(Delta)$ is feasible.
    Then its optimal solution shares the same form of $"OPT"_("static" v)(Delta)$.
  ][]


  #prop[
    Suppose $"OPT"_("static" v)(Delta)$ is feasible.
    Then its optimal solution shares the same form of $"OPT"_("static" t, v)(Delta)$.
  ][]

  So, generalizing $delta$ to $bu(delta)_t$ brings no additional benefit.
  Further, it is forcibly online and, when viewing this as a multi-objective optimization problem, does not result in the Pareto optimum.
]

= Main Contribution

== An Online Algorithm with Adaptive $delta$

#text(size: 18pt)[
  We desire an algorithm that
  - Ensures Pareto optimality
  - Achieves the detection threshold $Delta$
  - Converges to optimal $delta^*$ quickly

  Our idea is to perform an online gradient ascent on the Lagrangian dual function for the dual variable $lambda_t$, and use this to approximate the optimal dual variable $lambda^*$.
  To see this, first introduce the Lagrangian of $"OPT"_("static" t, v) (Delta)$:
  $
    L(delta, lambda)
    &:= 1 / T sum_(t = 1)^T D_bu(p) (bu(q)_t (delta)) - lambda dot (1 / T sum_(t = 1)^T "DG"_t (delta) - Delta) \
    &= 1 / T sum_(t = 1)^T D_bu(p) (bu(q)_t (delta)) - 1 / T sum_(t = 1)^T lambda ("DG"_t (delta) - Delta).
  $

  Denote the corresponding primal function and dual functions by
  $
    f(delta) := sup_(lambda >= 0) L(delta, lambda) #h(5em) g(lambda) := inf_delta L(delta, lambda).
  $
]

== Key Properties of the Optimization Problem

#v(2em)

#text(18pt)[
  #lemma[
    1. The infimum defining $g(lambda)$ can always be achieved by setting $delta = lambda$.
    2. The Lagrangian dual $g(lambda)$ can be decomposed token-wise by defining $L_t (delta) := D_bu(p) (delta) - lambda ("DG"_t (delta) - Delta)$.
      Then with $f_t$ and $g_t$ defined accordingly, part 1 holds for each $g_t (lambda)$, and $ g(lambda) = 1 / T sum_(t = 1)^T g_t (lambda) $ where each $g_t$ is concave with $dv(g_t, lambda) = Delta - "DG"_t (lambda).$
    3. If the primal problem is feasible, then the strong duality holds:
      $
        inf_delta f(delta) = sup_(lambda >= 0) g(lambda).
      $
      with $delta^* = lambda^*$, where $lambda^*$ is exactly the optimal $lambda$ for maximizing the Lagrangian dual $g(lambda)$.
  ][]
]

== Dual Gradient Ascent Algorithm

#grid(
  columns: (70%, 25%),
  column-gutter: 2em,
  text(13pt)[
    #pseudocode-list(title: smallcaps[Dual Gradient Ascent for Soft Watermarking (henceforth _DualGA_)])[
      + $t <- 1, y_0 <- k$
      + *while* $t <= T_"max"$ and $y_(t - 1) != v_"term"$ *do*
        + Use $f(k, y_(t - 1))$ to randomly partition $cal(V)$ into a _green list_ $G$ with $abs(G) = gamma abs(cal(V))$ and a _red list_ $abs(G^c) = (1 - gamma) abs(cal(V))$
        + Set $delta_t$ according to the Lagrangian dual $ delta_t <- lambda_t $
        + $bu(q)_t <- "softmax"(bu(p)_t + delta_t dot bb(1)_G)$
        + Sample a token $y_t sim (q_(t, 1), dots, q_(t, abs(cal(V))))$
        + Compute the online gradient of the dual function $ "gd"_t <- Delta - "DG"_t (delta_t) $
        + Update the dual variable via online gradient ascent $ lambda_(t + 1) <- lambda_t + eta dot "gd"_t $
        + $t <- t + 1$
      + *end while*
      + *if* $t < T_"max"$ *then*
        + Set all remaining $y_(t + 1), dots, y_(T_"max")$ to be $v_"term"$
      + *end if*
    ]
  ],
  text(15pt)[
    *Input*
      + Language Model ${bu(p)}_(t in [T_"max"]) in Delta (cal(V))$
      + Prompt $bu(x) in cal(T)$
      + Green list ratio $gamma in (0, 1)$
      + Pseudorandom function $f_gamma$
      + DG constraint $Delta$
      + Step size $eta$
      + Initial dual variable $delta_1$
      + Random seed $k$

    *Output*
      + Watermarked token sequence $bu(y)_[T_"max"]$
  ],
)

= Experiments

== Setup

#v(4.5em)

The authors use a LLaMa-7B model, while I used a 350M model due to limited compute resources.

In terms of datasets, both of our implementations use the C4 dataset of English language texts for prompt samples.

== Kirchenbauer Implementation

#text(11pt)[
  #grid(
    columns: 2,
  ```py
  for _ in range(max_new_tokens):
      outputs = model(next_input, past_key_values=past_key_values, use_cache=True)
      past_key_values = outputs.past_key_values
      logits = outputs.logits[:, -1, :]  # (1, vocab_size)

      # 3. Get green list based on previous token
      prev_token = int(generated[0, -1:].item())
      green_set = get_green_list(prev_token, vocab_size, gamma)
      green_mask = torch.zeros(vocab_size, device=device)
      green_indices = torch.tensor(list(green_set), device=device, dtype=torch.long)
      green_mask[green_indices] = 1.0

      # p_t: original probabilities
      original_probs = torch.softmax(logits, dim=-1).squeeze(0)

      # Add delta to green list logits
      watermarked_logits = logits.clone()
      watermarked_logits[0, green_indices] += delta

      # 4. Watermarked probabilities
      wm_probs = torch.softmax(watermarked_logits, dim=-1).squeeze(0)

      # Compute DG_t = sum_{v in G} q_{t,v} - sum_{v in G} p_{t,v}
      green_prob_original = (original_probs * green_mask).sum().item()
      green_prob_wm = (wm_probs * green_mask).sum().item()
      dg_t = green_prob_wm - green_prob_original
      dg_values.append(dg_t)
  ```,

  ```py

      # Compute KL(q_t || p_t)
      kl_t = (wm_probs * (torch.log(wm_probs + 1e-30) - torch.log(original_probs + 1e-30))).sum().item()
      kl_values.append(kl_t)

      # 5. Sample from watermarked distribution
      next_token = torch.multinomial(wm_probs.unsqueeze(0), num_samples=1)
      generated = torch.cat([generated, next_token], dim=-1)
      next_input = next_token

      # Track if sampled token is green
      is_green = next_token.item() in green_set
      green_hits.append(1.0 if is_green else 0.0)

      # Stop on EOS
      if next_token.item() == tokenizer.eos_token_id:
          break
  ```
)]

== Dual GA Implementation

#text(9pt)[
  #grid(
    columns: 2,
    ```py

    for t in range(max_new_tokens):
        outputs = model(next_input, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]  # (1, vocab_size)

        # 3. Get green list based on previous token
        prev_token = int(generated[0, -1:].item())
        green_set = get_green_list(prev_token, vocab_size, gamma)
        green_mask = torch.zeros(vocab_size, device=device)
        green_indices = torch.tensor(list(green_set), device=device, dtype=torch.long)
        green_mask[green_indices] = 1.0

        # p_t: original probabilities
        original_probs = torch.softmax(logits, dim=-1).squeeze(0)

        # g = sum of original probs on green list
        g = (original_probs * green_mask).sum()

        # 4. delta = lambda
        delta_t = max(0.0, min(lambda_t, 15.0))

        # Add delta to green list logits
        watermarked_logits = logits.clone()
        watermarked_logits[0, green_indices] += delta_t

        # 5. Watermarked probabilities
        wm_probs = torch.softmax(watermarked_logits, dim=-1).squeeze(0)

        # Compute DG_t
        green_prob_original = (original_probs * green_mask).sum().item()
        green_prob_wm = (wm_probs * green_mask).sum().item()
        dg_t_val = green_prob_wm - green_prob_original
        dg_values.append(dg_t_val)
        ```,

    ```py
    # Compute KL(q_t || p_t)
    kl_t = (wm_probs * (torch.log(wm_probs + 1e-30) - torch.log(original_probs + 1e-30))).sum().item()
    kl_values.append(kl_t)

    # 6. Sample from watermarked distribution
    next_token = torch.multinomial(wm_probs.unsqueeze(0), num_samples=1)
    generated = torch.cat([generated, next_token], dim=-1)
    next_input = next_token

    # Track if sampled token is green
    is_green = next_token.item() in green_set
    green_hits.append(1.0 if is_green else 0.0)

    # DG_t as a function of delta and g.
    # Not sure if this is explicitly given in the paper but it's in their code
    g_val = g.item()
    delta_tensor = torch.tensor(delta_t)
    g_tensor = torch.tensor(g_val)
    denom = g_tensor * torch.exp(delta_tensor) - g_tensor + 1
    dg_closed = (1 - g_tensor) * (1 - 1 / denom)

    # Gradient: d g_t / d lambda = Delta - DG_t(lambda)
    gradient = D_target - dg_closed.item()

    # Step size: eta = autoeta / sqrt(t+1)
    eta = autoeta / np.sqrt(t + 1)

    # Update lambda (projected gradient ascent)
    lambda_t = max(0.0, min(lambda_t + eta * gradient, 15.0))

    # Stop on EOS
    if next_token.item() == tokenizer.eos_token_id:
        break
    ```
)]

== Comparison of Results

#text(15pt)[
  #figure(
    grid(
      columns: 2,
      column-gutter: 40pt,
      image(
        "paperfigure.png",
        height: 70%,
      ),
      image(
        "figure1_replication.png",
        height: 60%,
      )
    ),
    caption: [
      The scatter plot of $z$-score vs realized DG for different algorithms.
      SRL stands for the Kirchenbauer algorithm, and DualGA is the Dual Gradient Ascent algorithm.
      Each point represents one generated sequence, and for each algorithm 200 sequences are generated.
    ]
  )
]

== Limitations

- We assumed here that green list partitions are independent across positions, but since we only have dependence on our fixed seed $k$ and the previous token, this is not necessarily true for a model that repeatedly generates the same token

- Robustness: what if someone modifies the watermarked text after generation?
  - The authors show that good enough $p$-values are still obtained in the algorithm for up to half the text being deleted

- Distinguishability: as mentioned at the start, making a perturbation to the underlying distribution is inherently limiting, vs. e.g. the Scott Aaronson approach

== Further Research

- Replacement of KL divergence?
  - While mathematically convenient, its generality possibly implies an information theoretic measure more specific to text might be better

#v(3em)

- What if have multiple LLMs with specified parameters?
  - Can we modify the algorithm to track which individual modified a particular piece of text?

= Questions?
