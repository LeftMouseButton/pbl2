
ChatGPT will be tasked with performing entity/relationship extraction, saving the result to a json file.
This step also combines all input text from multiple sources into one file.
If we have a lot of sources in the future, this is a weak point and will need to be changed (will run into a token limit otherwise).

For now, this step will be performed manually. We could use the chatgpt api, but that costs money.

Troubleshooting:
* If download doesn't work, retry.
	If still doesn't work, change model to Thinking and retry.
* If file is not produced, prompt with the last line again (output as a .json file, no commentary.)

1) All processed text files corresponding to one disease must be uploaded manually (in a browser window) to chatgpt.
	eg: breast-cancer-wikipedia.txt, breast-cancer-medlineplus.txt

2) Also upload example_entity_extraction.json.

3) Paste the following prompt:
Task: Perform structured entity and relationship extraction for knowledge-graph population.

Input: The uploaded file(s) contain cleaned natural-language text describing a single disease.

Goal: Identify and organize all relevant biomedical entities and their relationships, and output as valid JSON following the schema below.

Schema:
{
  "disease_name": "",
  "synonyms": [],
  "summary": "",
  "causes": [],
  "risk_factors": [],
  "symptoms": [],
  "diagnosis": [],
  "treatments": [],
  "related_genes": [],
  "subtypes": [
    {
      "name": "",
      "typical_patients": "",
      "five_year_survival": "",
      "common_treatments": []
    }
  ],
  "epidemiology": {
      "global_cases_2015": "",
      "global_deaths_2015": "",
      "most_common_in": "",
      "five_year_survival_rate": ""
  },
  "relationships": [
    {"source": "", "relation": "", "target": ""}
  ]
}

Instructions:

Use concise entity names (no full sentences in lists).

Include both diseases and subtypes (e.g., ALL, AML, etc.) as nodes in relationships.

Extract gene associations, treatments, and symptoms even if implicit.

When a quantitative fact (e.g., survival rate, global deaths) is mentioned, include it numerically in epidemiology.

Include at least 5 relationships connecting main entities (has_symptom, has_cause, treated_with, associated_gene, etc.).

If multiple files for the same disease are uploaded, merge their information.

Output only JSON, no commentary.

output as a .json file, no commentary.

4) Copy the resulting JSON and save it as: data/json/{disease-name}.json