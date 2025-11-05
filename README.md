# Knowledge Graph Pipeline

Builds and analyses a knowledge graph.

Modular design.

## Dependencies
conda install beautifulsoup4

conda install -c conda-forge python-slugify

conda install lxml

conda install pydantic -c conda-forge

## Usage
```
1) python -m src.kg.module1_crawler.crawler
2) python -m src.kg.module2_clean.clean
3) manual -- see src/kg/module3_extraction_entity_relationship/readme.txt
4) 
    Validate a single file:
      python -m src.kg.module4_validate_json.validate_json data/json/leukemia.json
    
    Validate all extracted JSONs:
      python -m src.kg.module4_validate_json.validate_json data/json/
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
Read "src/kg/module3_extraction_entity_relationship/readme.txt" for more info.

ChatGPT will be tasked with performing entity/relationship extraction, saving the result to a json file.
This step also combines all input text from multiple sources into one file.
If we have a lot of sources in the future, this is a weak point and will need to be changed (will run into a token limit otherwise).

For now, this step will be performed manually. We could use the chatgpt api, but that costs money.
```
Input:
    data/processed/{disease-name}_*.txt 
    example_entity_extraction.json
    {prompt}
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

### 5) Module 5 - 
-----------------------------------
...

## Results and Interpretation
(this is just copypasted for now)


<img width="666" height="1840" alt="Unsaved Image 1" src="https://github.com/user-attachments/assets/a87a9522-e718-4d6f-a06b-f1e88ed1c58d" />
