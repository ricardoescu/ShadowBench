# ShadowBench

## Personal inspiration for shadowbench
My grandmother dealt with schizophrenia most of her life. Several studies prove that when patients deal with a mental health condition (or other unrelated conditions), the treatment they need tends to get downgraded. I wanted to use my ML background to ensure that AI used in a medical context don't follow this pattern, and stress test them using counterfactuals.

Shortly after starting the project I realized that this same idea could be extended to other areas. I work in insurance, where AI usage is growing, and claimants should not be treated differently because bias in AI systems affects its decisions.
Governments move in the same direction. In none of these settings is plain accuracy enough. Tighter control is needed.

ShadowBench grew from a healthcare experiment into a broader AI assurance framework for finding hidden inconsistencies in consequential AI systems. This could be turned into a start-up reaching far beyond these areas, ensuring fairness in any AI system for government units, companies in any area, healthcare, etc.

## Abstract
**Adversarial assurance for high-stakes AI.**

ShadowBench automatically stress-tests AI systems with controlled counterfactuals to find cases where information that should not matter changes the model's decision.

The goal: if governments, insurers, healthcare providers, and other high-stakes organisations increasingly rely on AI, they need more than average benchmark accuracy. They need to know which people and cases their systems are likely to fail.

ShadowBench is an early-stage framework for finding those failures systematically.

---

## Why ShadowBench?

An AI system can perform well on a benchmark while still behaving inconsistently on particular groups or situations.

Consider two otherwise identical cases:

> **Case A:** A patient presents with the same physical symptoms.  
> **Case B:** The same patient presents with the same symptoms, but their history of schizophrenia is mentioned.

If the model changes its recommendation, that difference deserves investigation.

The same principle extends far beyond healthcare:

- **Government:** Does the same constituent case receive different priority depending on communication style, socioeconomic cues, or other irrelevant context?
- **Insurance:** Does the same claim receive different fraud suspicion or escalation depending on claimant characteristics or writing style?
- **Healthcare:** Do irrelevant demographic or medical-history attributes alter triage recommendations?

ShadowBench turns these questions into repeatable automated tests.

---

## How it works

ShadowBench currently follows a simple behavioural red-teaming pipeline:

```text
Benchmark case
      ↓
Controlled counterfactual generation
      ↓
Original + modified case
      ↓
Target AI model
      ↓
Structured decisions
      ↓
Counterfactual comparison
      ↓
Potential failure / instability
