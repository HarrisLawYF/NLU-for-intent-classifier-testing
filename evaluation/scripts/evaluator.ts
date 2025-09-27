import { CORRECTNESS_PROMPT, createLLMAsJudge } from "openevals";
import type { EvaluationResult } from "langsmith/evaluation";

export const correctnessEvaluator = createLLMAsJudge({
  prompt: CORRECTNESS_PROMPT,
  feedbackKey: "correctness",
  model: "openai:gpt-5-mini",
});

// strict equal evaluator
type StrictEqualEvaluatorParams = {
  outputs: Record<string, any>;
  referenceOutputs?: Record<string, any>;
};

export function strictEqualEvaluator(
  params: StrictEqualEvaluatorParams
): EvaluationResult {
  const { outputs, referenceOutputs } = params;

  // outputs:
  // refer to the return type from the evaluate function's callback - { intent, confidence }
  // outputs: { intent: '<intent>', confidence: '<confidence>' }

  // referenceOutputs:
  // refer to the data formatter outputs: { intent: '<intent>' }
  // referenceOutputs: { intent: '<intent>' }

  const score = outputs.intent === referenceOutputs?.intent;

  return {
    key: "strict_equal",
    score,
  };
}
