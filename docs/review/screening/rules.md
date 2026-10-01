# Screening Rules (title/abstract)

These rules apply `protocol.md` §B.2 to individual records. Both reviewers use them as written. Clarifications not stated in the protocol are marked **(clarification)** and are logged in protocol §B.13.

## Decisions

- **I**: include. Proceeds to full-text screening.
- **U**: unclear. Proceeds to full-text screening. Use U when the abstract does not show whether a criterion is met.
- **E**: exclude, with exactly one reason code. Use the **first** code in the list below that applies.

## Exclusion codes (in priority order)

| Code | Reason | Examples |
| --- | --- | --- |
| E4 | Not a primary study: a review, comment, editorial, peer-review document, meeting report, encyclopedia entry, book or book contents. | "We review…"; eLife decision letters |
| E5 | Publication status: a thesis or dissertation; a Zenodo or Figshare deposit or dataset; an abstract-only conference record. **(clarification)** Preprints on preprint servers (arXiv, bioRxiv, Research Square, Preprints.org, SSRN) are eligible and flagged as Tier 2. | Nuklearmedizin congress abstracts; Zenodo "theorem" deposits |
| E1 | Not a neural system. | Nanofluids, reactors, smart grids, optical computing |
| E7 | Explicitly out of scope: artificial hardware energy (neuromorphic, memristive, spintronic); ANNs or SNNs used as engineering models with no biophysical energy model; the variational free-energy principle with no physical energy; clinical or pathological metabolism with no signalling-cost quantity. | Spiking transformers; TBI pharmacology |
| E2 | No physical energy or thermodynamic cost quantity (class M, T or W), or cost is a proxy only (class P). | Network-efficiency graph metrics; firing-rate "cost" with no energy model |
| E3 | A cost quantity is present but not related to an information measure (SQ2), a network structure (SQ3), or a signalling process (SQ1). | Whole-brain glucose use versus age, with no signalling decomposition |

## Inclusion test, by sub-question

- **SQ1:** the study measures, models or bounds the energy or entropy cost of a neural signalling process (resting potential, action potential, synaptic transmission, plasticity, axonal conduction), or applies a thermodynamic framework (entropy production, Landauer, stochastic thermodynamics) to neural activity.
- **SQ2:** the study quantitatively relates an information measure to a class M or T cost. Evidence of decoupling counts.
- **SQ3:** the study relates network structure or communication (modularity, long-range traffic, wiring) to a class M, T or W cost.

## Records with no abstract **(clarification)**

Judge from the title. If the title clearly fails a criterion, exclude with that code. Otherwise mark U. The record's abstract is then retrieved (PubMed, Semantic Scholar, publisher) before full-text screening.
