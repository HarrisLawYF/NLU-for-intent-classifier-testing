import { parseArgs as parseNodeArgs } from "node:util";
import { generateReport } from "./eval_report.js";

interface CLIOptions {
  datasetName: string;
  experimentPrefix?: string;
  concurrency: "none" | "low" | "high";
}

const HELP_MESSAGE = `
📖 Usage:    eval-report <datasetName> [--prefix <experimentPrefix>] [--concurrency <concurrency>]
             eval-report <datasetName> [-p <experimentPrefix>] [-c <concurrency>]
📝 Examples: eval-report my-dataset
             eval-report my-dataset --prefix "experiment_1"
             eval-report my-dataset -p "experiment_1" -c high
🔍 Options:  --prefix, -p        Experiment prefix for the evaluation run
             --concurrency, -c   Concurrency level: 'none', 'low', or 'high' (defaults to 'low')
             --help, -h          Show this help message
📋 Description: Generate an evaluation report for a dataset with intent classification results.
`;

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
        prefix: {
          type: "string",
          short: "p",
          description: "Experiment prefix for the evaluation run",
        },
        concurrency: {
          type: "string",
          short: "c",
          description:
            "Concurrency level: 'none', 'low', or 'high' (defaults to 'low')",
        },
      },
      allowPositionals: true,
      strict: true,
    });

    // Validate required datasetName
    if (positionals.length === 0) {
      console.error("❌ Error: datasetName is required");
      process.exit(1);
    }

    const datasetName = positionals[0];

    // Validate and set concurrency (default to "low")
    let concurrency: "none" | "low" | "high" = "low";
    if (values.concurrency) {
      const concurrencyValue = values.concurrency.toLowerCase().trim();
      concurrency = validateConcurrency(concurrencyValue);
    }

    return {
      datasetName,
      experimentPrefix: values.prefix,
      concurrency,
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
    const { datasetName, experimentPrefix, concurrency } = parseArgs();

    console.log(`🚀 Starting evaluation report generation...`);
    console.log(`📊 Dataset Name: ${datasetName}`);
    console.log(`🔧 Concurrency: ${concurrency}`);

    if (experimentPrefix) {
      console.log(`🏷️  Experiment Prefix: ${experimentPrefix}`);
    }

    await generateReport({
      datasetName,
      config: {
        experimentPrefix,
        concurrency,
      },
    });

    console.log(`✅ Report generated successfully!`);
    console.log(`📄 Report saved to: ./evaluation/reports/${datasetName}.md`);
  } catch (error) {
    console.error("❌ Error:", error instanceof Error ? error.message : error);
    console.error("💡 Please check your input parameters and try again.");
    process.exit(1);
  }
}

main();

export function validateConcurrency(
  concurrency: string
): "none" | "low" | "high" {
  if (
    concurrency === "none" ||
    concurrency === "low" ||
    concurrency === "high"
  ) {
    return concurrency;
  }
  console.error(
    `❌ Error: Invalid concurrency '${concurrency}'. Must be 'none', 'low', or 'high'. Defaulting to 'low'`
  );
  return "low";
}
