# Systematic Review Protocol: Information and Thermodynamic Cost in Neural Systems

Status: **v1.3.** v1.2 was approved by the author on 2026-10-01 and frozen before the Phase 2 search. Later changes go only through §B.13; v1.3 revised the search strings after the recall check failed. Registration: not registered. The git commit of this version is the timestamped protocol (see §B.11).

This review supports the Map-STDP hypothesis (`docs/SPEC.md` §1) but is designed to test it, not to confirm it. Its findings feed the open items in SPEC §6 ("Energy model") and §1 ("information-dynamic entropy is a surrogate for thermodynamic entropy").

---

## Part A. Research Question Brief

### Topic

How neural signalling dissipates energy (its thermodynamics and metabolic cost), and what evidence links information-theoretic descriptions of neural activity to that cost.

### Primary research question

In neural systems, what quantitative relationships between information-theoretic measures of neural activity and its thermodynamic or metabolic cost have been measured, modelled or derived?

### Sub-questions

1. **SQ1 (mechanism).** Where along a neural signalling pathway is energy dissipated (resting potentials, action potentials, synaptic transmission, plasticity, axonal wiring), how large is each term, and which formal frameworks describe it (ATP/ion-flux accounting, Landauer-type bounds, stochastic thermodynamics and entropy production)?
2. **SQ2 (information–cost linkage).** What empirical, model-based or theoretical evidence relates information measures (mutual information, information rate, entropy rate, coding efficiency in bits per ATP or per joule, transfer entropy) to those costs, with what direction, magnitude and conditions? This explicitly includes evidence that the two are decoupled or trade off.
3. **SQ3 (network level).** Does evidence relate network-level communication structure (modularity, long-range versus local traffic, wiring cost, flow- or description-length-based measures such as the map equation) to metabolic or thermodynamic cost?

### Sub-question bindings

All three inherit: population = neural systems (biological, any species, or biophysically grounded models of them); timeframe = no lower bound, searches run to the Phase 2 search date; language = English; domain = neuroscience, biophysics and the statistical physics of neural systems. Deviations: none.

### FINER assessment

| Criterion | Score | Justification |
| --- | --- | --- |
| Feasible | 4/5 | Pilot searches return roughly 300–650 records per block in OpenAlex (§B.5). Only open APIs are available (no Scopus/WoS), and screening is done by one AI screener with an author audit. |
| Interesting | 5/5 | It is unresolved whether information measures and energetic cost are proportional, bounded or decoupled at different scales, and this decides whether a description length can serve as a cost surrogate. |
| Novel | 3/5 | Narrative reviews exist on brain energetics and on energy-efficient coding. A protocol-driven synthesis that keeps ATP cost apart from thermodynamic entropy production and adds the network level (SQ3) does not appear to exist yet. Phase 2 must verify this. |
| Ethical | 5/5 | Literature only, with no human-subjects activity. |
| Relevant | 5/5 | It directly informs the Map-STDP energy proxy (SPEC §6) and the choice among candidate description lengths. |
| **Average** | **4.4/5** | |

### Scope boundaries

**In scope:**

- Biological neurons, synapses, axons, glia when they support neuronal signalling energetics, circuits and whole brains, in any species.
- Biophysical or spiking models whose energy terms come from physiology (ion fluxes, ATP, heat).
- Analytic theory that applies thermodynamic or information-theoretic bounds to neural signalling.
- Network-level studies that relate connectome or communication structure to metabolic or wiring cost.

**Out of scope:**

- Energy use of artificial hardware: GPUs, neuromorphic chips and memristive crossbars. The device physics is different, and hardware is handled in SPEC §7.
- The variational free-energy principle and active inference, where "free energy" is a variational bound and not physical dissipation. Such studies are included only if they also report a physical energy quantity.
- Artificial neural networks with no biophysical energy model, such as ANN "energy" regularizers.
- Clinical metabolism studies with no information-theoretic or signalling component.

**Key assumptions:**

- Metabolic cost (ATP, O₂, glucose) and thermodynamic dissipation (heat, entropy production) are related but **not interchangeable**. The review keeps them as separate outcome classes (§B.3).
- A null or negative relationship is a valid finding.

### Candidate questions considered

| # | Candidate | FINER avg | Why not selected |
| --- | --- | --- | --- |
| 1 | Primary RQ above | 4.4 | Selected |
| 2 | Is the information efficiency of neurons (bits per ATP) conserved across species and cell types? | 3.8 | Too narrow. It answers part of SQ2 and none of SQ1 or SQ3. |
| 3 | Does the map-equation codelength of neural activity predict metabolic cost? | 2.6 | Pilot search: about 0 directly relevant records. It would be an empty review. It is kept as an explicit gap check within SQ3. |
| 4 | How do neurons dissipate energy? | 3.2 | Descriptive only and unbounded. It is folded into the review as SQ1. |

---

## Part B. Protocol (PRISMA-P 2015)

### B.1 Rationale

The Map-STDP project assumes that a description length of neural activity flow tracks neurons' thermodynamic cost. This assumption needs three kinds of support that the project does not yet have:

- a physical account of where neural signalling dissipates energy;
- evidence on whether information measures track that dissipation;
- evidence on whether the relationship holds for network-level communication structure.

The relevant literature is spread across physiology, computational neuroscience and statistical physics, and it uses incompatible notions of "cost" and "entropy". A protocol-driven review with explicit outcome classes makes those distinctions visible.

### B.2 Eligibility (adapted PICOS / PECOS)

| Element | Definition |
| --- | --- |
| **P**opulation | Neural systems at any scale: biological (any species, in vivo, in vitro, ex vivo) or biophysically grounded models of them. |
| **E**xposure | An information-theoretic characterisation of neural activity or signalling (SQ2), a network-level communication or flow structure (SQ3), or, for SQ1, the signalling process itself. |
| **C**omparator | Within-study variation: across firing rates, cell types, species, network configurations or model parameters. Analytic studies use a bound or an optimum as the reference. |
| **O**utcome | A thermodynamic or metabolic cost quantity (§B.3). For SQ2/SQ3 it must be quantitatively related to the exposure. |
| **S**tudy design | Experimental measurement, computational biophysical modelling, analytic theory, and quantitative analysis of existing datasets. |

| Criterion | Include | Exclude |
| --- | --- | --- |
| Study type | Primary empirical, modelling or theoretical studies | Reviews, editorials, commentaries and textbooks. Reviews are used for citation chasing only. |
| Publication date | Any, up to the search date | — |
| Language | English | Other languages, because no translation capacity is available |
| Publication status | Peer-reviewed articles, peer-reviewed full conference papers, and preprints (arXiv, bioRxiv), flagged as Tier 2 | Abstract-only conference records, theses, and records without retrievable full text or abstract |
| Energy notion | A physical energy quantity (§B.3, classes M and T) | Variational free energy only. Abstract "cost" with no physical grounding, such as spike count used as a cost with no stated energy model; these are recorded as class P (proxy) and excluded from SQ2/SQ3 pooling. |

### B.3 Outcomes

Each cost outcome is assigned exactly one **cost class**, and outcomes from different classes are never pooled:

- **M (metabolic):** ATP consumption, O₂ or glucose utilisation, CMRO₂, or ion-flux-derived ATP estimates.
- **T (thermodynamic):** heat dissipation, entropy production rate, free-energy dissipation, or comparison with the Landauer bound.
- **W (wiring/material):** axonal volume or length and conduction cost, reported in physical units or as a model with a stated energy mapping.
- **P (proxy):** spike counts or firing rates used as a cost with no energy model. This class is recorded for completeness and excluded from the primary synthesis.

**Primary outcomes:**

1. The quantitative information–cost relationship (SQ2): information efficiency (bits per ATP, bits per joule), the correlation or scaling exponent between an information measure and cost, a trade-off curve or an optimal operating point, and an analytic bound.
2. The energy budget breakdown by signalling process (SQ1), reported as fractions and absolute values with units and species.

**Secondary outcomes:**

1. Relations between network structure and cost (SQ3), such as the cost of cross-module versus within-module traffic, or modularity versus wiring or metabolic cost.
2. The gap between measured or modelled dissipation and the thermodynamic lower bound (the efficiency ratio relative to Landauer or to entropy production).
3. Any direct test of a description-length measure, including the map equation, against cost. This is a recorded gap check: its absence is reported.

### B.4 Information sources

| Source | Access | Rationale |
| --- | --- | --- |
| OpenAlex | Public API, title and abstract search | Multidisciplinary, covering much of the WoS and Scopus content |
| PubMed | E-utilities | Physiology and neuroscience |
| arXiv | Public API (q-bio.NC, physics.bio-ph, cond-mat.stat-mech) | Statistical physics and theory, which is often preprint-first |
| bioRxiv | Via OpenAlex, flagged | Preprints in neuroscience |
| Semantic Scholar / OpenAlex citation graph | API | Backward and forward citation chasing from included studies and seed papers |

Scopus, Web of Science and PsycINFO are **not accessible** in this environment. This is a recorded limitation.

### B.5 Search strategy (OpenAlex syntax; adapted per database in Phase 2)

Counts are OpenAlex hits from the executed search on 2026-10-01 (`search.py`, `data/search_log.json`). Each string is translated to PubMed and arXiv syntax with the same terms.

**S-A (SQ2 linkage), 370 hits (amended in v1.3):**

```text
("mutual information" OR "information rate" OR "bits per" OR "description length"
 OR "map equation" OR "coding efficiency" OR "information transmission"
 OR "transfer entropy" OR "entropy rate")
AND ("metabolic cost" OR "energy cost" OR "energetic cost" OR "energy efficiency"
 OR "energy consumption" OR ATP OR "entropy production")
AND (neuron OR neurons OR neural OR synapse OR synaptic OR axon OR brain OR cortex OR spiking)
NOT ("deep learning" OR "neural network accelerator")
```

**S-B (SQ3 network), 171 hits:**

```text
(neuron OR neurons OR neural OR synapse OR synaptic OR axon OR brain OR cortex OR cortical)
AND ("metabolic cost" OR "energy cost" OR "energetic cost" OR "wiring cost" OR metabolic)
AND ("wiring cost" OR "wiring economy" OR "communication cost" OR "network economy"
 OR "modular organization" OR "brain modularity" OR "network modularity")
```

With the bare term `modularity`, the pilot returned 638 hits, mostly unrelated uses of "modular". It was narrowed to the three phrases above.

**S-C1 (SQ1 budgets), 322 hits:**

```text
("energy budget" OR "ATP consumption" OR "ATP cost" OR "energy use")
AND ("action potential" OR "action potentials" OR "synaptic transmission"
 OR "resting potential" OR "grey matter" OR "gray matter" OR signalling OR signaling)
AND (neuron OR neurons OR brain OR cortex)
```

**S-C2 (SQ1 thermodynamics), 582 hits (amended in v1.3):**

```text
("entropy production" OR Landauer OR "stochastic thermodynamics"
 OR "nonequilibrium thermodynamics" OR "non-equilibrium thermodynamics")
AND (neuron OR neurons OR synapse OR synaptic OR "neural activity" OR "brain dynamics"
 OR "human brain" OR "brain activity" OR "whole-brain")
```

**S-D (foundational titles; title field only), 20 hits (added in v1.3):**

```text
("neural information" OR "neural code" OR "neural codes" OR "neural coding"
 OR "cortical computation" OR "neural computation" OR "neural signalling" OR "neural signaling")
AND ("metabolic cost" OR "energy cost" OR "energetic cost" OR "energy efficiency"
 OR "energy consumption" OR metabolic OR "cost of")
```

The executed search returned 1,870 records across the three databases, which came to 1,276 after deduplication. 147 of them have no abstract in any source. Adding the bare term `neural` to S-C2 raised it to more than 2,000 hits, mostly artificial-network papers, so S-C2 uses brain-specific phrases instead.

**Seed set for recall checking and citation chasing.** Ten seeds are eligible under §B.2. Still et al. (2012) and Rosvall & Bergstrom (2008) are not about neural systems; they are used only for citation chasing and the gap check, and are not counted in the recall check. The seeds are:

- Attwell & Laughlin (2001), the grey-matter energy budget;
- Laughlin, de Ruyter van Steveninck & Anderson (1998), the metabolic cost of neural information;
- Levy & Baxter (1996), energy-efficient neural codes;
- Lennie (2003), the cost of cortical computation;
- Alle, Roth & Geiger (2009), energy-efficient action potentials;
- Sengupta et al. (2010), action-potential energy efficiency across neuron types;
- Harris, Jolivet & Attwell (2012), synaptic energy use;
- Niven & Laughlin (2008), energy limitation in sensory systems;
- Still, Sivak, Bell & Crooks (2012), the thermodynamics of prediction;
- Lynn et al. (2021), broken detailed balance in the human brain;
- Bullmore & Sporns (2012), the economy of brain network organisation;
- Rosvall & Bergstrom (2008), the map equation (the gap check).

**Recall check.** If the database searches miss more than 2 eligible seeds, the strings are revised and re-run before screening proceeds. Every revision is logged in §B.13. **Result:** the v1.2 strings missed 3 of 10 eligible seeds (Laughlin et al. 1998, Lennie 2003, Lynn et al. 2021), so they were revised (amendment 3). The v1.3 strings retrieve all 10.

### B.6 Study records and selection

1. **Deduplication:** by DOI, then by normalised title and year.
2. **Title/abstract screening:** one AI screener applies §B.2. Records it marks "unclear" go to full-text review.
3. **Author audit:** the author independently screens a random 15% sample plus every "unclear" record. Agreement is reported as Cohen's κ. If κ < 0.6, the criteria are clarified and the whole set is re-screened.
4. **Full-text screening:** the AI screener records an exclusion reason for each excluded record, and the author checks all exclusions in SQ2 and SQ3.
5. **Pilot:** the first 50 records are screened by both reviewers before the full screen, to calibrate.

This is a **deviation from PRISMA dual independent screening**, and it is recorded as a limitation.

### B.7 Data items

| Category | Items |
| --- | --- |
| Metadata | Authors, year, venue, peer-review status (Tier 1/2), design (empirical / model / theory / data analysis) |
| System | Species, preparation, cell type or circuit, scale (synapse / neuron / circuit / brain), temperature |
| Exposure | Information measure and estimator (bias correction, sample size, stimulus ensemble) and network measure |
| Outcome | Cost class (M/T/W/P), quantity, units, measurement or estimation method, values with uncertainty |
| Relationship | Form (efficiency ratio, correlation, scaling exponent, trade-off, bound, optimum), direction, magnitude, stated conditions, and whether firing rate is controlled for |
| Thermodynamic framing | Equilibrium or non-equilibrium assumptions, coarse-graining level, whether the cost is a bound or an actual dissipation |
| Relevance to Map-STDP | Whether the result bears on per-spike, per-synaptic-event or wiring terms of the SPEC §6 energy proxy (recorded, not used for inclusion) |

### B.8 Appraisal (replacing RoB 2 and ROBINS-I)

RoB 2 and ROBINS-I assess intervention effects in clinical designs and do not fit physiological measurement, modelling or theory. A design-specific appraisal is used instead. Each domain is rated Low, Some concerns or High:

- **Empirical:** validity of the energy measurement (direct measurement or inferred from ion fluxes with assumed stoichiometry); bias of the information estimator (finite-sample correction); representativeness of the preparation (temperature, in vitro versus in vivo); selective reporting.
- **Model:** provenance of energy parameters (measured, fitted or assumed); validation against independent data; sensitivity analysis reported.
- **Theory:** assumptions stated (steady state, detailed balance, coarse-graining); whether the result is a bound or an equality; whether it connects to measurable quantities.

Use in synthesis: a sensitivity analysis excludes studies rated High on the energy-validity domain.

### B.9 Synthesis

- **SQ1:** a structured narrative and tabular synthesis of energy budgets, with species, temperature and method as stratifiers, and a separate section for thermodynamic frameworks (class T).
- **SQ2 meta-analysis (conditional):** information efficiency (bits per ATP) is pooled only if at least 3 studies report it for comparable signalling stages (for example, graded-potential photoreceptor or synaptic transmission) with extractable uncertainty. Pooling uses a random-effects model on the log scale, with I², τ² and a prediction interval. Leave-one-out and "exclude High-appraisal-concern" are the sensitivity analyses. Otherwise SWiM is used, with an effect-direction plot per cost class.
- **SQ3:** narrative synthesis with an evidence map (network measure × cost class).
- **Gap check:** a table of description-length measures (map equation, MDL, SBM codelength, entropy rate) against cost classes, marking each cell as tested, indirectly addressed or untested.
- **Certainty:** a GRADE-style rating adapted to non-intervention evidence. Start at "moderate" for direct empirical measurement and "low" for model-only or theory-only evidence. Downgrade for appraisal concerns, inconsistency, indirectness (wrong cost class or scale) and imprecision. Upgrade for convergence across independent methods. The adaptation is reported as a deviation.

### B.10 Meta-bias

- **Publication bias:** if a pooled outcome has at least 10 studies, use a funnel plot and Egger's test. Otherwise it is noted qualitatively.
- **Selective reporting:** compare the cost classes defined in each study's methods with those it reports.

### B.11 Registration

PROSPERO accepts health-related outcomes only, so this review is out of its scope. OSF Registries would be the applicable platform. **Author decision (2026-10-01): not registered.** The git commit that freezes v1.2 before any search is the timestamped protocol, and the report will say so. Later amendments are committed separately, so the git log shows what changed after the search began.

### B.12 Reporting

PRISMA 2020 (27 items), with the deviations in §B.6, §B.8 and §B.9 listed in the report's limitations.

### B.13 Amendments

| Date | Section | Change | Rationale |
| --- | --- | --- | --- |
| 2026-10-01 | §A, §B.2, §B.3 | v1.0 → v1.1: added cost classes M/T/W/P with no cross-class pooling; excluded the variational free-energy principle; made decoupling evidence an explicit SQ2 target; added S-B and the seed recall check | Devil's Advocate Checkpoint 1 (Appendix C) |
| 2026-10-01 | §B.5, §B.11, Appendix C | v1.1 → v1.2: added transfer entropy and entropy rate to S-A (308 hits); narrowed S-B (171 hits); recorded author decisions (SQ1 kept, author audit committed, git as registration) | Author approval at the end of Phase 1; made before the search, so no data had been seen |
| 2026-10-01 | §B.5 | v1.2 → v1.3: added `neural` and a deep-learning exclusion to S-A; added brain-scale phrases to S-C2; added the title-only string S-D. Tried and rejected: `bit OR bits` in S-A (2,589 hits, because "a bit" is common English) and `spikes` plus "energy consumption" in S-C1 (2,333 hits). | The recall check failed (3 of 10 eligible seeds missed). Two of the misses have no abstract in OpenAlex and their titles lack the S-A terms; Lynn et al. 2021 says "human brain", which S-C2 did not cover. Titles and abstracts were inspected only for the missed seeds, and none were screened. |

---

## Appendix C. Devil's Advocate Report, Checkpoint 1

### Verdict: PASS (after v1.1 revisions)

### Critical issues

No critical issues identified.

### Major issues

1. **Conflation of "thermodynamic entropy" with "metabolic cost"** (resolved in v1.1).
   - Type: method / scope.
   - Location: SPEC §1 hypothesis; protocol v1.0 outcomes.
   - Problem: SPEC equates thermodynamic entropy with metabolic cost. Measured ATP turnover in neurons is many orders of magnitude above Landauer-type bounds, so a result about one does not transfer to the other.
   - Recommendation: separate the cost classes and never pool across them. Applied in §B.3.
2. **Confirmation bias from project stake** (resolved in v1.1).
   - Type: bias.
   - Problem: the review was commissioned by a project that needs a positive linkage. Search strings built from "efficiency" vocabulary would favour studies reporting information–energy alignment.
   - Recommendation: make decoupling and trade-off evidence an explicit SQ2 target, and report the description-length gap check whatever its result. Applied in §A SQ2 and §B.3 (secondary outcome 3).
3. **"Free energy" homonym** (resolved in v1.1).
   - Type: scope.
   - Problem: the variational free-energy principle literature matches thermodynamic keywords but concerns a statistical bound, not dissipation.
   - Recommendation: exclude it unless the study reports a physical energy quantity. Applied in §A and §B.2.
4. **SQ1 is a poor fit for systematic-review machinery** (resolved by author decision: SQ1 stays inside the review, answered by the framework approach below).
   - Type: method.
   - Problem: "How does neural thermodynamics work" is a mechanistic and conceptual question. Its best answer may come from a few foundational theory papers that keyword search under-retrieves, while PICOS screening and GRADE are built for effect estimates.
   - Recommendation: keep SQ1 in the review but treat its answer as a structured framework built from included studies plus seed and citation chasing. Label SQ1 certainty ratings as descriptive. The alternative is to move SQ1 to a separate narrative background section outside the PRISMA flow.
5. **Single screener** (resolved: the author committed to the §B.6 audit).
   - Type: method.
   - Problem: AI-only screening is not independent dual screening, and its errors are correlated with the same model's synthesis.
   - Recommendation: the author audit in §B.6. The κ threshold makes the audit consequential rather than ceremonial. The author must commit the time, roughly 15% of the deduplicated records, recalculated after deduplication.

### Minor issues

- The adapted appraisal and GRADE reduce comparability with other reviews. Report them as deviations, as planned in §B.8 and §B.9.
- English-only, with no Scopus or WoS: the Russian- and Chinese-language biophysics literature is missed. Record this as a limitation.

### Observations

- The SPEC hypothesis ("a neuron pays more to talk to cortices it does not normally talk to") is closest to the wiring-economy literature (class W), not to information-efficiency studies. SQ3 is where the hypothesis lives or dies.
- The pilot found about 0 direct tests of the map equation against energy. The expected answer to the gap check is "untested", which is itself a publishable motivation for Map-STDP.

### Strongest counter-argument

"Information measures and metabolic cost co-vary only because both scale with spike count. Once firing rate is controlled for, no information-specific cost remains. A description length would then be a costly proxy for a firing-rate penalty." The extraction must record whether each SQ2 study controls for rate. This item is added to §B.7 under Relationship → conditions.

### Stress test

| Test | Result |
| --- | --- |
| Remove the strongest expected source (Laughlin et al. 1998). Does the review still stand? | Yes. SQ1 and SQ3 do not depend on it. |
| Flip the question ("information and cost are decoupled"). Is the opposing view credible? | Yes. It is now an explicit SQ2 target. |
| Apply to a different context (hardware). Does it generalise? | No. Hardware is excluded by design (SPEC §7). |
| "So what?" Is the significance justified? | Yes. It decides the SPEC §6 energy proxy and candidate ranking. |
