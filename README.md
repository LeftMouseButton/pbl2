# Knowledge Graph Pipeline

Builds and analyses a knowledge graph.

Modular design.

## Dependencies
conda install beautifulsoup4

conda install -c conda-forge python-slugify

conda install lxml

conda install pydantic -c conda-forge

pip install google-generativeai
    (package not available in conda)

conda install networkx

conda install -c conda-forge pyvis

conda install pandas

Set your API key, in a given terminal window:
export GOOGLE_API_KEY="YOUR_API_KEY"


## Usage
```
1) python main.py

OR

1) python -m src.kg.module1_crawler.crawler
2) python -m src.kg.module2_clean.clean
3) python -m src.kg.module3_extraction_entity_relationship.extraction_entity_relationship.py --all
4) python -m src.kg.module4_validate_json.validate_json data/json/
5) python src/kg/module5_prepare_for_analysis/combine_json_files.py
6) python -m src.kg.module6_analysis.analyse     --input data/combined/all_diseases.json     --outdir data/analysis     --viz-html graph.html      --graphml graph.graphml     --topk 30     --seed "Breast cancer"     --seed "Lung cancer"      --betweenness-sample 200      --random-state 42

```
## Modules/Steps:
### 1) Module 1 – Web Crawler
-----------------------------------
Collects raw natural-language content (HTML or plain text) for diseases.

Current Targets: Wikipedia (API) and MedlinePlus (HTML).

Saves results under data/raw/ with provenance metadata (for Module 2 cleaning).

```
Output:
    data/raw/{slug}_{source}.{ext}
    data/raw/metadata.jsonl  (records url, timestamp, checksum)
```

### 2) Module 2 – Cleaning / Preprocessing
-----------------------------------
Converts raw HTML or plain-text files (from Module 1) into normalized,
clean text suitable for LLM-based entity extraction (Module 3).
```
Input:
    data/raw/*.html or .txt   (from module1_crawler)
Output:
    data/processed/*.txt      (normalized text)
    data/processed/metadata.jsonl (checksum + provenance)
```
### 3) Module 3 – LLM-based Entity and Relationship Extraction
-----------------------------------
Uses the Google AI Studio API (Gemini 2.5 Flash Live) to perform structured entity and relationship extraction for knowledge-graph population.

This step also combines all input text from multiple sources into one file.
If we have a lot of sources in the future, this is a weak point and will need to be changed (will run into a token limit otherwise).

```
Parameters:
  • Single-disease mode:
      python extraction_entity_relationship.py --disease breast-cancer
  • Batch mode (process all disease prefixes):
      python extraction_entity_relationship.py --all
  • Force (existing .json files are skipped unless --force):
      python extraction_entity_relationship.py --all --force
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
Checks and fixes structural issues in extracted JSONs before DB ingestion.
No backup files are produced — JSONs are overwritten in place.
```
Input:
    data/json/{disease-name}.json
Output:
    data/json/{disease-name}.json
```

### 5) Module 5 - Analysis Preparation
-----------------------------------
Dealing with token limits constitutes a major challenge for graph analysis.
We need to employ RAG, Summarization, Chunking, Compression/Encoding, etc.
Running the "TokenCount Predictor" (Utilities) suggests we should be able to store information for approximately 180 diseases before this step becomes essential, assuming the LLM provider = ChatGPT Plus.

Also, we should try to remove duplicates/etc (caused by slightly different names/descriptions produced by the LLM in Module 3) from the .json files.

For now, we'll just combine all the .json's into a single file for manually uploading to chatgpt.
```
Input:
    data/json/*.json
Output:
    data/combined/all_diseases.json
```

### 6) Module 6 - Analysis
-----------------------------------
Produces the graph, performs analysis, exports results.

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
'''

...

### Utilities
### TokenCount Predictor
```
Usage: 
        # For a directory
            python src/kg/utils/tokencount_predictor.py --input_path data/json
        # For a single file
            python src/kg/utils/tokencount_predictor.py --input_path data/example.json
Input:
    .json file(s) from --input_path parameter
Output:
    Command Line <- integer: Predicted Token Count
```

