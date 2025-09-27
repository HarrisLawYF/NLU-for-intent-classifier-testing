import "dotenv/config";
import { parseArgs as parseNodeArgs } from "node:util";
import { Dataset } from "langsmith";
import { langsmith } from "../client.js";

interface CLIOptions {
  operation: string;
}

const HELP_MESSAGE = `
📖 Usage:    langsmith:basic --operation <operation> [-o <operation>]
📖 Usage:    langsmith:basic <operation> [--help]

📝 Examples: langsmith:basic list-datasets
             langsmith:basic --operation list-datasets
             langsmith:basic --help

🔍 Options:  --operation, -o  Operation to perform (required)
             --help, -h      Show this help message

📋 Available Operations:
  list-datasets    List all available datasets

📋 Description: Perform basic LangSmith operations like listing datasets.
`;

// Available operations
const OPERATIONS = {
  "list-datasets": listDatasets,
} as const;

type OperationName = keyof typeof OPERATIONS;

async function listDatasets() {
  const datasets = langsmith.listDatasets(); // returns AsyncIterable<Dataset>
  const datasetList: Dataset[] = [];
  console.log("📊 Available Datasets:\n");

  for await (const dataset of datasets) {
    console.log(`📀 Dataset: ${dataset.name}`);
    console.log(`📊 ID: ${dataset.id}`);
    console.log(`📄 Description: ${dataset.description}`);
    console.log(`📅 Created At: ${dataset.created_at}`);
    console.log(`📅 Updated At: ${dataset.modified_at}`);
    console.log(`📈 Size: ${dataset.example_count}`);
    console.log(`--------------------------------`);
    datasetList.push(dataset);
  }

  if (datasetList.length === 0) {
    console.log("❌ No datasets found.");
  } else {
    console.log(`✅ Found ${datasetList.length} dataset(s)`);
  }

  return datasetList;
}

// Parse command line arguments
function parseArgs(): CLIOptions {
  try {
    const args = process.argv.slice(2);

    // Handle help flag - special case
    if (args.includes("--help") || args.includes("-h")) {
      console.log(HELP_MESSAGE);
      process.exit(0);
    }

    const { values, positionals } = parseNodeArgs({
      args: process.argv.slice(2),
      options: {
        operation: {
          type: "string",
          short: "o",
          description: "Operation to perform",
        },
      },
      allowPositionals: true,
      strict: true,
    });

    // Get operation from either positional args or --operation flag
    const operation = values.operation || positionals[0];

    // Validate required operation
    if (!operation) {
      console.error("❌ Error: operation is required");
      console.error("💡 Use --help to see available operations");
      process.exit(1);
    }

    // Validate operation exists
    if (!isValidOperation(operation)) {
      console.error(`❌ Error: Unknown operation '${operation}'`);
      console.error("💡 Use --help to see available operations");
      process.exit(1);
    }

    return {
      operation,
    };
  } catch (error) {
    if (error instanceof Error && error.message.includes("Unknown option")) {
      console.error("❌ Error: Unknown option provided");
      console.error(HELP_MESSAGE);
      process.exit(1);
    }
    throw error;
  }
}

async function main() {
  try {
    const { operation } = parseArgs();

    console.log(`🚀 Executing LangSmith operation: ${operation}`);
    console.log(`📋 Operation: ${operation}`);
    console.log("");

    // Execute the operation
    await OPERATIONS[operation as OperationName]();

    console.log(`\n✅ Operation '${operation}' completed successfully!`);
  } catch (error) {
    console.error("❌ Error:", error instanceof Error ? error.message : error);
    console.error("💡 Please check your input parameters and try again.");
    process.exit(1);
  }
}

function isValidOperation(operation: string): operation is OperationName {
  return operation in OPERATIONS;
}

main();
