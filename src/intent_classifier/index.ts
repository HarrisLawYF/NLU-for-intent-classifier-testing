import { ChatPromptTemplate } from "@langchain/core/prompts";
import { constructSystemPrompt } from "../../prompts/system_prompt.js";
import { ChatOpenAI } from "@langchain/openai";
import { intentClassificationOutputSchema } from "./schema.js";

type PromptParams = {
  user_query: string;
  chat_history: string;
  most_recent_intent: string;
};

const model = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0.2 });

const modelWithOutputSchema = model.withStructuredOutput(
  intentClassificationOutputSchema
);

const systemPrompt = await constructSystemPrompt();

const prompt = ChatPromptTemplate.fromMessages<PromptParams>([
  ["system", systemPrompt],
  ["user", "User query: {user_query}"],
]);

const intentClassifier = prompt.pipe(modelWithOutputSchema);

export async function classifyIntent(params: PromptParams) {
  const { intent, confidence } = await intentClassifier.invoke(params);

  return { intent, confidence };
}
