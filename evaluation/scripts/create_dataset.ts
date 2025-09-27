import "dotenv/config";
import type { Client, Dataset } from "langsmith";
import { ExampleCreate } from "langsmith/schemas";
import { formatTestData } from "./data_formatter.js";

type CreateDatasetParams = {
  client: Client;
  datasetName: string;
  fileName?: string;
  config?: {
    type?: "test" | "loose";
  };
};

export async function createDataset(
  args: CreateDatasetParams
): Promise<Dataset> {
  const { client, datasetName, config, fileName } = args;

  try {
    if (datasetName.trim().length === 0) {
      throw new Error("Dataset name cannot be empty");
    }

    console.log(`🔍 Checking if dataset '${datasetName}' exists...`);
    const isDatasetExists = await client.hasDataset({ datasetName });
    const dataset = isDatasetExists
      ? await client.readDataset({ datasetName })
      : await client.createDataset(datasetName);

    console.log(
      `📊 Dataset ${isDatasetExists ? "found" : "created"}: ${dataset.name}`
    );

    const examples: ExampleCreate[] = await formatTestData({
      dataset_id: dataset.id,
      type: config?.type ?? "test",
      fileName,
    });

    console.log(`📝 Uploading ${examples.length} examples to dataset...`);
    await client.createExamples(examples);

    console.log(`✅ Successfully uploaded ${examples.length} examples`);
    return dataset;
  } catch (error) {
    console.error(
      "❌ Error creating dataset:",
      error instanceof Error ? error.message : error
    );
    throw error;
  }
}
