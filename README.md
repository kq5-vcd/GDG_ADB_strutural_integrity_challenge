# Case Study 1: Structural Integrity Monitoring

## Case Introduction
ADB Safegate's airfield lighting fixtures operate in some of the harshest environments imaginable. They are routinely exposed to aircraft jet blast, ground vehicle impacts, thermal cycling, moisture ingress, and continuous vibration. Over time, these stresses produce structural fatigue, corrosion, mounting wear, and physical damage — any of which can render a fixture a safety hazard before it fails outright.

Today, structural condition is largely confirmed reactively — through physical inspection or after a failure occurs. The company wants to move toward **proactive structural monitoring**: catching mechanical compromise early so it can be addressed before becoming an operational risk.

To make this possible, ADB Safegate is developing a **dedicated sensor board** that mounts inside each fixture and continuously monitors structural state via 3-axis vibration sensing. The hackathon dataset is from the lab characterization of this sensor board — a controlled acoustic excitation rig where a subwoofer plays known frequency sweeps into a test fixture while 15 sensor boards record the resulting vibration response at 27 kHz. Different excitation patterns were tested (different frequency bands, with some tests playing two excitation sweeps simultaneously) under three controlled bolt conditions:

* **Run A — 30 NM** (clean baseline, all bolts torqued to spec)
* **Run B — Loose** (bolts in a loosened state)
* **Run C — Mix-45 Deg** (bolts loosened by 45° from torqued spec)

Each test produces a brief snapshot of the fixture's vibration response — 8,192 samples (~303 ms) per sensor board per test. Test conditions, excitation parameters, and bolt status for every recording are captured in a master `Test.csv` metadata file accompanying the dataset. ➤ Simply put: the data tells you how the fixture *responds* to known vibration, under three known mounting conditions. Your job is to figure out what that response actually says about structural state.

## The Challenge
Your mission: Using the labeled vibration data, build a structural fault classifier that distinguishes between a healthy fixture and the two fault conditions — and rigorously characterize where it works, where it breaks down, and what would be needed to take it from a lab benchmark toward a deployable monitoring capability.

You will work with the controlled-test vibration data from the sensor board characterization rig and must build a solution that can:

✅ Distinguish between the three labeled bolt conditions (30 NM, Loose, Mix-45 Deg) from the 3-axis vibration response, with a clearly described methodology
✅ Justify your feature engineering and classifier logic from physics-based reasoning about how mechanical resonance, damping, and modal response change when fixture mounting is compromised — not just statistical correlation
✅ Make meaningful use of the experimental structure of the dataset — the multiple excitation patterns, the parallel sensor boards on each test, and the replicate captures — both when training and when evaluating your model
✅ Honestly characterize how well your classifier generalizes across excitation patterns and replicates, and identify which excitation patterns and sensor channels carry the strongest discriminative signal
✅ Provide a clear gap analysis — what about this lab dataset is and isn't representative of what an in-fixture sensor board would see in deployed conditions, and what additional data would be needed to evolve this into a deployable monitoring capability

**Out of scope:** Predicting time-to-failure, detecting fault types not represented in the dataset, or claiming deployment-grade accuracy from controlled lab data alone. This is a benchmark — your goal is to characterize what is and isn't learnable from it, not to ship a production model.

## Bonus Track: From Algorithm to Product
*(Optional — additional points)*

Once you have built your algorithm, step back and ask: does it fit the kind of product CORTEX Service is, and does it deliver what an airport customer would actually pay for? Sketch a commercial roadmap for it — anchored in how you think customers would want to use this capability.

This case study (Structural Integrity) and Case Study 3 (EFD Localization) are intended to ship as part of the same algorithm package within CORTEX Service. You may consider this algorithm **together with** the other algorithm in the package, or treat it as a **standalone offering** — whichever framing produces a more credible commercial story.

Address the following dimensions:

* **Commercialization strategy** — Where does this product live? Inside CORTEX Service as a built-in capability, alongside CORTEX Service as a separate entity airports buy on its own terms, or as some hybrid? And within whichever choice you make, is it sold as one bundle with the other algorithm or as separate modules? Anchor both decisions in how you think airport customers would actually want to use this.
* **Pricing model & roadmap** — How do you charge for this: flat fee, per-fixture, per-airport, tiered, value-based per incident? And how does pricing evolve as the package matures from v1 (current data, current capability) through v2 and v3 (additional data streams, additional algorithms in the suite)?
* **Value articulation** — In concrete terms, what is the customer actually buying? Failures avoided, inspections deferred, safety incidents prevented, runway closures averted, fixtures replaced before they become hazards?
* **Algorithm-package lifecycle** — What ships at launch with the data and capability you have today? What improves as data quality and coverage grow? What additional algorithms (e.g. LED degradation, once production-ready) might join the package over time?

## Why It Matters
A reactive approach to structural integrity at airport scale carries real cost:

* **Safety risk** — a structurally compromised runway or taxiway light can fail without warning under jet blast or vehicle impact, creating an immediate operational hazard
* **Emergency response cost** — unscheduled structural repairs are dramatically more expensive than planned maintenance and frequently disrupt operations
* **Inspection burden** — without prioritisation, structural inspections are blanket and time-consuming; targeted inspection is far more efficient
* **Asset planning** — knowing which fixtures are degrading helps the company forecast replacement budgets across an airport's lifecycle

An effective in-fixture vibration-based monitoring capability gives ADB Safegate something fundamentally new: continuous structural awareness without dispatching anyone to inspect anything. This dataset is the validation foundation for that capability — evidence that the signal the sensor board picks up actually carries the structural information the product strategy is built on.

## Judging Focus
* **Quality of feature engineering and classifier logic** — does the team's approach reflect how mechanical resonance and damping change when a fixture's mounting is compromised? Are FFT, modal, or other physics-grounded transformations applied thoughtfully, or is the time series being treated as a black box?
* **Use of experimental structure** — does the team meaningfully exploit the multiple excitation patterns, the parallel sensor boards, and the replicate captures? Or has the data been flattened in a way that throws away the experimental design?
* **Generalization rigor** — are train/test splits set up to genuinely test whether the classifier generalizes (e.g. to unseen excitation patterns, unseen sensor boards, unseen replicates)? Or is the reported accuracy an artifact of leakage?
* **Excitation and channel insight** — has the team identified which excitation patterns and sensor channels carry the strongest discriminative signal, and articulated *why* — what about those frequency bands or axes makes them informative for this fault type?
* **Specificity of the gap analysis** — does the team clearly distinguish what is learnable from this lab dataset and what isn't, and concretely articulate what additional data would be needed for real-world deployment?
* **Bonus — Commercial coherence** *(stretch)* — does the team's commercialization strategy, pricing, and value articulation hang together as a credible CORTEX Service offering, driven by how customers would actually want to use the product?
