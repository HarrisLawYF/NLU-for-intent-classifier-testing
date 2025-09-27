import "dotenv/config";
import { evaluate } from "langsmith/evaluation";
import { classifyIntent } from "../../src/intent_classifier/index.js";
import { strictEqualEvaluator } from "./evaluator.js";
import type { ExperimentResultRow, EvaluatorT } from "langsmith/evaluation";

type IntentClassifierEvaluationParams = {
  datasetName: string;
  chatHistory?: string;
  mostRecentIntent?: string;
  config?: {
    experimentPrefix?: string;
    maxConcurrency?: number;
  };
  extra?: {
    evaluators?: EvaluatorT[];
  };
};

type IntentClassifierEvaluationResult = {
  results: ExperimentResultRow[];
  count: number;
  experiment: string;
};

export async function intentClassifierEvaluation(
  params: IntentClassifierEvaluationParams
): Promise<IntentClassifierEvaluationResult> {
  const { datasetName, chatHistory, mostRecentIntent, config, extra } = params;

  const evaluators: EvaluatorT[] = [
    strictEqualEvaluator,
    ...(extra?.evaluators ?? []),
  ];

  const result = await evaluate(
    async (inputs: { input: string }) => {
      const { intent, confidence } = await classifyIntent({
        user_query: inputs.input,
        chat_history: chatHistory ?? "N/A",
        most_recent_intent: mostRecentIntent ?? "N/A",
      });

      // always return everything in the schema
      // change evaluator's params to match the schema

      // 1. return intent (string) ONLY - in the params of the evaluator's function: outputs: { outputs: '<intent>' }
      // It adds the field "outputs" automatically
      // 2. return object { intent, confidence } - in the params of the evaluator's function: outputs: { intent: '<intent>', confidence: '<confidence>' }
      return { intent, confidence };
    },
    {
      data: datasetName,
      evaluators,
      experimentPrefix:
        config?.experimentPrefix ?? "intent-classifier, gpt-4o-mini, baseline", // optional, experiment name prefix
      maxConcurrency: config?.maxConcurrency ?? 5, // optional, add concurrency
    }
  );

  return {
    results: result.results,
    count: result.processedCount,
    experiment: result.experimentName,
  };
}
