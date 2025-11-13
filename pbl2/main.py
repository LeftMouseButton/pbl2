#!/usr/bin/env python3
"""
Main pipeline launcher for Knowledge Graph build process.

Runs sequentially:
1. Module 1 – Crawler
2. Module 2 – Clean
3. Module 3 – Entity & Relationship Extraction
4. Module 4 – JSON Validation
5. Module 5 – Combine JSON Files for Analysis
6. Utilities: Predict Token Count – Ensure token count is within LLM limits
7. Module 6 – Read combined JSON file, analyse, export results.
"""

import subprocess
import sys
import datetime


def run_step(description: str, command: list[str]):
    """Run one pipeline step, print status, and stop on error."""
    print("\n" + "=" * 80)
    print(f"[{datetime.datetime.now().isoformat()}] 🚀 Starting: {description}")
    print("=" * 80)
    print(" ".join(command))
    print("-" * 80)

    try:
        result = subprocess.run(command, check=True)
        print(f"\n✅ Completed: {description}\n")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR in {description}")
        print(f"Command failed with return code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\n❌ Unexpected error in {description}: {e}")
        sys.exit(1)


def main():
    steps = [
        (
            "Module 1: Crawler",
            ["python", "-m", "src.kg.module1_crawler.crawler"],
        ),
        (
            "Module 2: Clean",
            ["python", "-m", "src.kg.module2_clean.clean"],
        ),
        (
            "Module 3: Entity & Relationship Extraction",
            [
                "python",
                "-m",
                "src.kg.module3_extraction_entity_relationship.extraction_entity_relationship",
                "--all",
            ],
        ),
        (
            "Module 4: Validate Extracted JSON",
            ["python", "-m", "src.kg.module4_validate_json.validate_json", "data/json/"],
        ),
        (
            "Module 5: Combine JSON Files for Analysis",
            [
                "python",
                "-m",
                "src.kg.module5_prepare_for_analysis.combine_json_files",
            ],
        ),
        (
            "Utilities: Predict Token Count",
            ["python", "-m", "src.kg.utils.tokencount_predictor", "--input_path", "data/combined/all_diseases.json"],
        ),
        (
            "Module 6: Graph Construction & Analysis",
            [
                "python",
                "-m",
                "src.kg.module6_analysis.analyse",
                "--input",
                "data/combined/all_diseases_matched.json",
                "--outdir",
                "data/analysis",
                "--viz-html",
                "graph.html",
                "--graphml",
                "graph.graphml",
                "--topk",
                "30",
                "--seed",
                "Breast cancer",
                "--seed",
                "Lung cancer",
                "--betweenness-sample",
                "200",
                "--random-state",
                "42",
            ],
        ),

    ]

    print("\n" + "#" * 80)
    print("#               KNOWLEDGE GRAPH PIPELINE LAUNCHER")
    print("#" * 80)

    for desc, cmd in steps:
        run_step(desc, cmd)

    print("\n🎉 All modules completed successfully! The data is ready for analysis.\n")


if __name__ == "__main__":
    main()

