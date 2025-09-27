import { mkdir, readdir, readFile, writeFile } from "fs/promises";
import { intentClassificationOutputSchema } from "../../src/intent_classifier/schema.js";
import { intentClassifierEvaluation } from "./eval.js";
import { FILE_PATHS } from "../../src/constants/index.js";
import { z } from "zod";

const schema = intentClassificationOutputSchema.shape.intent;
type Intent = z.infer<typeof schema>;

type RunResult = {
  description: string;
  input: string;
  expectedOutput: Intent;
  generatedOutput: Intent;
  confidence: number;
};

type generateReportParams = {
  datasetName: string;
  config?: {
    experimentPrefix?: string;
    concurrency?: "none" | "low" | "high";
  };
};

const concurrencyMapping = {
  none: 1,
  low: 5,
  high: 20,
} as const;

export async function generateReport(params: generateReportParams) {
  const { datasetName, config } = params;
  const { experimentPrefix, concurrency } = config ?? {};
  const maxConcurrency = concurrencyMapping[concurrency ?? "low"];

  const { results, count } = await intentClassifierEvaluation({
    datasetName,
    config: {
      experimentPrefix,
      maxConcurrency,
    },
  });

  // results: ExperimentResultRow[]
  // result.evaluationResults: EvaluationResults
  // result.evaluationResults.results: EvaluationResult[]
  // result.evaluationResults.results[0].score: boolean (run score)
  const failedResults = results.filter(
    (result) => !result.evaluationResults.results[0].score
  );

  const benchmark: number = (count - failedResults.length) / count;
  const benchmarkPercentage: string = `${benchmark * 100}%`;

  const cleanedResults: RunResult[] = [];
  for (let index = 0; index < failedResults.length; index++) {
    const result = failedResults[index];
    // example
    // {
    //   "inputs": { "input": "What is the weather in Tokyo?" },
    //   "outputs": { "intent": "weather" }
    // }
    // run:
    // { "outputs": { "intent": "weather", "confidence": 0.9 } }
    const input = result.example.inputs.input;
    const expectedOutput = result.example.outputs?.intent;
    const generatedOutput = result.run.outputs?.intent;

    // validate the output with the schema
    const validatedExpectedOutput = schema.parse(expectedOutput);
    const validatedGeneratedOutput = schema.parse(generatedOutput);

    cleanedResults.push({
      description: `### Run ${index + 1}
Input: ${input}
Expected Output: ${expectedOutput}
Generated Output: ${generatedOutput}`,
      expectedOutput: validatedExpectedOutput,
      generatedOutput: validatedGeneratedOutput,
      input,
      confidence: result.run.outputs?.confidence,
    });
  }

  // report per intent
  const resultsPerIntent: Map<Intent, RunResult[]> = new Map();
  const intentReports: { intent: Intent; report: string }[] = [];
  let finalReport = "";
  // group failed (cleaned) results by intent
  if (cleanedResults.length > 0) {
    // for each intent (expected output), generate a report

    for (const result of cleanedResults) {
      const intent = result.expectedOutput;
      if (!resultsPerIntent.has(intent)) {
        resultsPerIntent.set(intent, []);
      }
      resultsPerIntent.get(intent)?.push(result);
    }

    for (const [intent, runResults] of resultsPerIntent) {
      const report = await generateIntentReport({
        intent,
        results: runResults,
      });
      intentReports.push({ intent, report });
    }

    finalReport = `# 任务：优化意图的识别能力
  
  **背景：**
  你是一个专注于大语言模型提示词优化的AI助手。当前，模型在识别用户意图时出现了一些错误。
  
  **基准：**
  | 总案例 | 正确案例 | 错误案例 | 基准 |
  | :--- | :--- | :--- | :--- |
  | ${count} | ${count - failedResults.length} | ${
      failedResults.length
    } | ${benchmarkPercentage} |

  **失败案例数据：**
  ${intentReports.map((report) => report.report).join("\n")}
  
  **你的任务：**
  基于以上失败案例，请完成以下一项或多项任务：
  1.  **模式分析：** 总结导致模型错误分类的可能原因或共同模式（例如，是否因为输入中包含某些特定词汇？）。
  2.  **数据增强建议：** 提供3-5个新的、高质量的训练样本，这些样本应该能帮助模型更好地区分 \`<intent>\` 和其他意图。
  3.  **意图定义修正：** 如果你认为 \`<intent>\` 的定义本身存在模糊地带，请提出修正建议。
  `;

    // create directory if not exists
    await mkdir(`./evaluation/reports`, { recursive: true });

    // write to file
    await writeFile(
      `./evaluation/reports/${datasetName}.md`,
      finalReport,
      "utf-8"
    );
  } else {
    finalReport = `所有意图都正确分类，没有错误案例。`;
  }

  return {
    finalReport,
    intentReports,
    resultsPerIntent,
    benchmark,
  };
}

type GenerateIntentReportParams = {
  intent: Intent;
  results: RunResult[];
};

async function generateIntentReport(
  params: GenerateIntentReportParams
): Promise<string> {
  const { intent, results } = params;
  const intentDefinition = await loadIntentDefinition(intent);

  const tableRows = results.map((result, index) => {
    return `| ${index + 1} | ${result.input} | ${result.expectedOutput} | ${
      result.generatedOutput
    } | ${result.confidence} |`;
  });

  return `下表是模型将 **"${intent}"** 意图错误分类的详细记录。请仔细分析这些案例。
  
| 运行 | 输入 (Input) | 预期输出 (Expected) | 实际输出 (Generated) | 置信度 (Confidence) |
| :--- | :------------- | :------------------ | :------------------- | :------------------ |
${tableRows.join("\n")}


**"${intent}"意图定义:**
${intentDefinition}
  `;
}

async function loadIntentDefinition(intent: Intent): Promise<string> {
  try {
    // lowercase intent and replace space with underscore
    const intentLowercase = intent.toLowerCase().replace(/ /g, "_").trim();

    const intentFiles = await readdir(FILE_PATHS.INTENTS_DIR);

    const selectedFiles = intentFiles.filter(
      (file) =>
        file.endsWith(".md") &&
        !file.endsWith(FILE_PATHS.LOOSE_INTENTS_IDENTIFIER) &&
        file.includes(intentLowercase)
    );

    if (selectedFiles.length !== 1) {
      throw new Error(
        "Only one intent definition file should be found for the intent"
      );
    }

    const content = await readFile(
      `${FILE_PATHS.INTENTS_DIR}/${selectedFiles[0]}`,
      "utf-8"
    );

    return content;
  } catch (error) {
    console.error(error);
    throw error;
  }
}
