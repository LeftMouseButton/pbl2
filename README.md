# Knowledge Graph Pipeline

Builds and analyses a knowledge graph.

Modular design; each step is independently executable and may be reused for other tasks.

## Dependencies
```
conda install beautifulsoup4 lxml networkx pandas -c conda-forge python-slugify pydantic pyvis
pip install google-generativeai
```
Set your API key (required for Module 3):
```
export GOOGLE_API_KEY="YOUR_API_KEY"
```

## Limitations
```
Module 1 -- More sources are required beyond Wikipedia/MedlinePlus. Code is structured such that additional sources can be added in the future.
Module 3 -- Major issues
                1) Reproducibility: LLMs generate different information with each run.
                2) Hallucinations/etc: LLMs may fabricate facts or utilize external information.
                3) Naming issues:
                        1) Some entries do not share the same naming scheme (eg: "Tobacco smoking", "Smoking tobacco", "Smoking (active and passive)", "Smoking", and "Smoking cigarettes").
                        2) Excessive verbosity: "Being overweight (possibly due to smoking-related lower body weight)"
...
```

## Usage

Edit disease_names.txt as desired, then:
```
1) python main.py

OR

1) python -m src.kg.module1_crawler.crawler
2) python -m src.kg.module2_clean.clean
3) python -m src.kg.module3_extraction_entity_relationship.extraction_entity_relationship.py --all
4) python -m src.kg.module4_validate_json.validate_json data/json/
5) python -m src.kg.module5_prepare_for_analysis.combine_json_files.py
6) python -m src.kg.module6_analysis.analyse     --input data/combined/all_diseases.json     --outdir data/analysis     --viz-html graph.html      --graphml graph.graphml     --topk 30     --seed "Breast cancer"     --seed "Lung cancer"      --betweenness-sample 200      --random-state 42

```
## Modules/Steps:
### 1) Module 1 – Web Crawler
-----------------------------------
Collects raw natural-language content (HTML or plain text) for diseases.

Current Targets: 

- Wikipedia (REST API)
- MedlinePlus (HTML via XML search API).

Saves results under data/raw/ with provenance metadata (for Module 2 cleaning).

```
Output:
    data/raw/{slug}_{source}.{ext}
    data/raw/metadata.jsonl    # one JSON record per fetched resource
```

### 2) Module 2 – Cleaning / Preprocessing
-----------------------------------
Converts raw HTML or plain-text files (from Module 1) into normalized,
clean text suitable for LLM-based entity extraction (Module 3).
```
Input:
    data/raw/*.html or .txt   (from Module 1 - Web Crawler)
Output:
    data/processed/{disease}_-_{source}.txt
    data/processed/metadata.jsonl    # metadata record includes source filename, processed filename, checksums, and timestamp.
```
### 3) Module 3 – LLM-based Entity and Relationship Extraction
-----------------------------------
Uses the Google AI Studio API (Gemini 2.5 Flash Lite) to perform structured entity and relationship extraction for knowledge-graph population.

This step also combines all input text from multiple sources into one file per disease.
If we have a lot of sources for each disease in the future, this is a weak point and will need to be changed (will run into a token limit otherwise).

Note: this is also the weakest point in the entire project for scientific reproducibility. Would be ideal to avoid use of an LLM here.

```
Parameters:
  • Single-disease mode:
      python extraction_entity_relationship.py --disease breast-cancer
  • Batch mode (process all disease prefixes):
      python extraction_entity_relationship.py --all
  • Force (existing .json files are skipped unless --force):
      python extraction_entity_relationship.py --all --force
- Retries failed extractions up to a configurable limit.
```

```
Input:
    .txt file(s) from data/processed/
        options:
            --disease {name}
                OR
            --all
Output:
    data/json/{disease-name}.json
```

### 4) Module 4 - Validator with Auto-Repair for LLM Step-1 JSON Outputs
-------------------------------------------------
Checks and fixes structural issues in extracted JSONs.
No backup files are produced — JSONs are overwritten in place.
```
Input:
    data/json/{disease-name}.json
Output:
    data/json/{disease-name}.json    # validated, schema-consistent
```

### 5) Module 5 - Analysis Preparation
-----------------------------------
Dealing with token limits constitutes a major challenge for LLM-based graph analysis.
We need to employ RAG, Summarization, Chunking, Compression/Encoding, etc.
Running the "TokenCount Predictor" (Utilities) suggests we should be able to store information for approximately 300 diseases before this step becomes essential, assuming the LLM provider = ChatGPT Plus.

Also, we should try to remove duplicates/etc (caused by slightly different names/descriptions produced by the LLM in Module 3) from the .json files.

For now, we'll just combine all the .json's into a single file, for either manually uploading to chatgpt or just using with NetworkX (non-LLM) in Module 6.
```
Input:
    data/json/*.json
Output:
    data/combined/all_diseases.json
```

### 6) Module 6 - NetworkX Analysis
-----------------------------------
Using NetworkX, produces the graph, performs analysis, exports results.

```
Input:
    data/combined/all_diseases.json
        OR
    data/json/*.json
Output:
    data/analysis/
        report_module6.md                 # Human-readable analysis report (paste-able into paper)
        graph.graphml                     # Full graph (labels & types as node/edge attributes)
        graph.html                        # Interactive PyVis visualization
        centrality.csv                    # Degree, betweenness, eigenvector (top-k and full)
        communities.csv                   # Node → community mapping
        link_predictions.csv              # Top predicted links (ensemble-ranked)
```


### Utilities
### TokenCount Predictor
-----------------------------------
Estimates token counts for JSON files (useful for planning LLM-based analysis or RAG setups).
```
Usage: 
        # For a directory
            python src/kg/utils/tokencount_predictor.py --input_path data/json
        # For a single file
            python src/kg/utils/tokencount_predictor.py --input_path data/example.json
Input:
    .json file(s) from --input_path
Output:
    Token count printed to the command line.
```

