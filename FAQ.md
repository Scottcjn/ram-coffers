# RAM Coffers FAQ

This FAQ answers common onboarding questions for contributors and operators who
are reading RAM Coffers as part of the broader Proof of Physical AI and
RustChain ecosystem.

## Is RAM Coffers a token treasury?

No. In this repository, "coffers" refers to NUMA-aware memory banks for LLM
inference. RAM Coffers is an inference optimization and research codebase, not a
custodial wallet, exchange, bank, or treasury contract.

The project is connected to RustChain because the same physical hardware that
runs RAM Coffers can also participate in Proof of Antiquity. That means the
hardware can do useful AI work and provide physical-device proof, but this repo
does not hold user funds or define payout policy by itself.

## How does RAM Coffers differ from traditional token treasuries?

Traditional token treasuries usually focus on managing balances, grants, votes,
or spending from an on-chain or organizational pool.

RAM Coffers focuses on hardware-local memory routing:

- Coffer placement maps model knowledge to NUMA nodes or cache tiers.
- Resonance routing selects the relevant memory bank for a query.
- POWER8-specific primitives such as DCBT, VSX, and vcipher reduce memory and
  attention cost during inference.

The "value" being organized here is physical compute and memory locality, not a
pooled token balance.

## What is the minimum RTC amount for treasury participation?

This repository does not define a minimum RTC amount for participation. RAM
Coffers documents the inference side of the stack; RTC participation rules live
in the RustChain ecosystem and may change over time.

For current RTC onboarding, mining, wallet, and bounty details, check the
RustChain documentation and active bounty board rather than relying on this FAQ
as a payment policy source.

## How are allocation decisions made?

For RAM Coffers, allocation decisions are technical rather than financial:

- Model shards are assigned to coffers based on memory capacity and domain role.
- Inference requests route to coffers by resonance matching.
- Threads and prefetch hints are chosen to keep hot paths close to the relevant
  POWER8 NUMA node.

For contribution rewards or bounty allocation, maintainers evaluate public
issues, pull requests, reviews, and validation evidence in the relevant GitHub
repositories.

## Can I withdraw my treasury contribution?

RAM Coffers does not accept or custody treasury deposits, so there is nothing to
withdraw from this repository.

If you are asking about RTC mining, bounty payouts, wallet transfers, or any
other token movement, use the official RustChain wallet and bounty instructions.
Do not share private keys, seed phrases, keystores, passwords, or verification
codes in issues or pull requests.

## How does antiquity affect treasury rewards?

Proof of Antiquity is the RustChain mechanism that rewards real physical
hardware, with vintage or distinctive hardware contributing to the proof model.
RAM Coffers is relevant because it shows that the same physical machine can run
useful AI inference, not only attest to its existence.

In practical terms:

- RAM Coffers improves inference performance on verified physical hardware.
- Proof of Antiquity belongs to RustChain's reward and attestation layer.
- Reward amounts, eligibility, and payout status should be checked against the
  current RustChain rules and public bounty records.

## Where should new readers start?

Start with:

1. `README.md` for the research context and performance summary.
2. `QUICK_START.md` for the shortest build and run path.
3. `CONTRIBUTING.md` for contribution expectations.
4. `BCOS.md` for certification context.
5. `docs/PPA_INTEGRATION.md` for how RAM Coffers fits into Proof of Physical AI.
