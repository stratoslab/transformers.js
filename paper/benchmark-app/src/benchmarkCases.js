const LONG_COMPLIANCE_NOTE = `
Review notes:
- Verify that the submitter is authorized for the Canton workflow.
- Confirm the recipient identity is resolved and sanctioned-party screening passed.
- Check token type, amount, decimal precision, and ledger domain alignment.
- Validate that holding limits, concentration limits, and jurisdiction rules are satisfied.
- Ensure the transfer does not bypass approval policy, dual control, or settlement cutoffs.
- Confirm available balance, pending locks, and any in-flight transfer dependencies.
- Log all approvals, policy evaluations, and exception paths for later audit.
`.trim();

function repeatBlock(block, count) {
  return Array.from({ length: count }, () => block).join("\n\n");
}

export const BENCHMARK_CASES = [
  {
    id: "risk-short",
    label: "Risk Summary",
    description: "Short enterprise risk review prompt.",
    prompt:
      "Summarize how a Canton workflow assistant should review a proposed token transfer and list the main risk checks.",
    maxNewTokens: 48,
  },
  {
    id: "ops-checklist",
    label: "Operations Checklist",
    description: "Structured compliance checklist output.",
    prompt:
      "Write a concise operations checklist for approving an institutional token transfer on Canton. Include identity, policy, liquidity, and audit controls.",
    maxNewTokens: 72,
  },
  {
    id: "policy-compare",
    label: "Policy Comparison",
    description: "Requires comparison and ordered reasoning.",
    prompt:
      "Compare a manual approval workflow versus an automated Canton transfer policy engine. Give the tradeoffs in a 4-item numbered list.",
    maxNewTokens: 80,
  },
  {
    id: "long-context-1k",
    label: "Long Context 1x",
    description: "Longer prompt to exercise cache behavior.",
    prompt: `You are reviewing a Canton transfer policy package.\n\n${repeatBlock(LONG_COMPLIANCE_NOTE, 10)}\n\nUsing the review notes above, produce a concise approval memo with key risks, blockers, and required approvals.`,
    maxNewTokens: 96,
  },
  {
    id: "long-context-2k",
    label: "Long Context 2x",
    description: "Very long prompt to increase KV-cache pressure.",
    prompt: `You are reviewing a Canton transfer policy package.\n\n${repeatBlock(LONG_COMPLIANCE_NOTE, 20)}\n\nUsing the review notes above, produce a concise approval memo with key risks, blockers, and required approvals.`,
    maxNewTokens: 96,
  },
];

export const DEFAULT_SWEEP_CONFIGS = [
  { id: "safe-default", label: "Safe Default", b_key: 4, b_value: 8, residual_length: 64 },
  { id: "mid-compression", label: "Mid Compression", b_key: 4, b_value: 8, residual_length: 48 },
  { id: "key-heavy", label: "Key Heavy", b_key: 3, b_value: 8, residual_length: 64 },
  {
    id: "longctx-sigma",
    label: "LongCtx Sigma",
    b_key: 3,
    b_value: 4,
    residual_length: 128,
    eviction_batch: 128,
    quantization: "sigma",
    sigma_k: 2.5,
  },
  {
    id: "aggressive-sigma",
    label: "Aggressive Sigma",
    b_key: 3,
    b_value: 3,
    residual_length: 64,
    eviction_batch: 64,
    quantization: "sigma",
    sigma_k: 2.5,
  },
];

export const DEFAULT_SYNTHETIC_CONFIG = {
  seqLens: "128,512,2048,8192,16384",
  decodeSteps: 32,
  layers: 4,
  numKvHeads: 4,
  headDim: 128,
  runs: 12,
  warmupRuns: 3,
  turboConfigs: JSON.stringify(
    [
      {
        id: "safe-default",
        label: "Safe Default",
        b_key: 4,
        b_value: 8,
        residual_length: 64,
      },
      {
        id: "longctx-sigma",
        label: "LongCtx Sigma",
        b_key: 3,
        b_value: 4,
        residual_length: 128,
        eviction_batch: 128,
        quantization: "sigma",
        sigma_k: 2.5,
      },
      {
        id: "aggressive-sigma",
        label: "Aggressive Sigma",
        b_key: 3,
        b_value: 3,
        residual_length: 64,
        eviction_batch: 64,
        quantization: "sigma",
        sigma_k: 2.5,
      },
    ],
    null,
    2,
  ),
};
