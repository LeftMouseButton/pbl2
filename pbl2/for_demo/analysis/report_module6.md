# Enhanced Cancer Knowledge Graph Analysis Report
## Executive Summary
- **1421** nodes and **2383** edges
- **5** connected components (giant component: 1404 nodes, 98.8% of all nodes)
- **23** communities detected
- Average degree: **3.35**; density: **0.0024**
- Community sizes – mean: **61.8**, largest: **152**, smallest: **1**

## Graph Summary
- Nodes: **1421**
- Edges: **2383**
- Types: subtype: 406 | treatment: 246 | gene: 237 | risk_factor: 149 | cause: 147 | diagnosis: 134 | disease: 102
- Connected components: **5**; giant component size: **1404** (98.80% of nodes)
- Average degree: **3.35**
- Graph density: **0.0024**
- Isolates (preview): carcinoid syndrome

## Community Detection
- Detected **23** communities.

| community | leader | size |
|---|---|---|
| 0 | tonsillitis | 8 |
| 1 | pancreatic cancer | 73 |
| 2 | radiation therapy | 125 |
| 3 | neuroendocrine tumors | 62 |
| 4 | colorectal cancer | 84 |
| 5 | brain tumor | 72 |
| 6 | carcinoid syndrome | 1 |
| 7 | multiple myeloma | 24 |
| 8 | skin cancer | 107 |
| 9 | ovarian cancer | 127 |
| 10 | lymphoma | 74 |
| 11 | salivary gland cancer | 53 |

## Centrality (Top Hubs)
### Degree
| label | type | score |
|---|---|---|
| lung cancer | disease | 0.07746478873239436 |
| colorectal cancer | disease | 0.06830985915492958 |
| ovarian cancer | disease | 0.056338028169014086 |
| radiation therapy | treatment | 0.0528169014084507 |
| head and neck cancer | risk_factor | 0.047183098591549295 |
| endometrial cancer | disease | 0.045774647887323945 |
| chemotherapy | treatment | 0.04295774647887324 |
| pancreatic cancer | disease | 0.04295774647887324 |
| salivary gland cancer | subtype | 0.03732394366197183 |
| brain tumor | disease | 0.03591549295774648 |
| liver cancer | disease | 0.03591549295774648 |
| bladder cancer | disease | 0.03380281690140845 |
| male breast cancer | disease | 0.03380281690140845 |
| skin cancer | disease | 0.030985915492957747 |
| cholangiocarcinoma | disease | 0.029577464788732393 |
| melanoma | subtype | 0.028169014084507043 |
| rhabdomyosarcoma | subtype | 0.027464788732394368 |
| hepatocellular carcinoma | disease | 0.026056338028169014 |
| surgery | treatment | 0.02535211267605634 |
| leukemia | disease | 0.02535211267605634 |

### Betweenness
| label | type | score |
|---|---|---|
| radiation therapy | treatment | 0.2557494739977772 |
| chemotherapy | treatment | 0.1554693860550341 |
| lung cancer | disease | 0.11441453440924891 |
| colorectal cancer | disease | 0.09484743666697994 |
| tp53 | gene | 0.0889522969606736 |
| ovarian cancer | disease | 0.07245579597869894 |
| surgery | treatment | 0.07180118769131842 |
| head and neck cancer | risk_factor | 0.0584310269588853 |
| pancreatic cancer | disease | 0.05813817855160966 |
| salivary gland cancer | subtype | 0.05413417359481374 |
| targeted therapy | treatment | 0.05384685918179204 |
| brain tumor | disease | 0.051969851121977696 |
| endometrial cancer | disease | 0.048110550896524926 |
| liver cancer | disease | 0.04340420145906766 |
| bladder cancer | disease | 0.0383559356498625 |
| skin cancer | disease | 0.036542429153343506 |
| melanoma | subtype | 0.03488458306922649 |
| medulloblastoma | subtype | 0.0348756063287566 |
| lymphoma | subtype | 0.03407284901790436 |
| male breast cancer | disease | 0.033986871346105854 |

### Eigenvector
| label | type | score |
|---|---|---|
| radiation therapy | treatment | 0.32650585746386146 |
| chemotherapy | treatment | 0.27868383372797206 |
| colorectal cancer | disease | 0.2021709918436731 |
| lung cancer | disease | 0.20059007327055342 |
| ovarian cancer | disease | 0.1790973933074122 |
| tp53 | gene | 0.169374271216982 |
| targeted therapy | treatment | 0.1632375401178195 |
| head and neck cancer | risk_factor | 0.1621932579792695 |
| surgery | treatment | 0.14867719896031262 |
| endometrial cancer | disease | 0.1449960725363471 |
| biopsy | diagnosis | 0.12269217138325747 |
| bladder cancer | disease | 0.11705798375499697 |
| male breast cancer | disease | 0.11656723593302408 |
| salivary gland cancer | subtype | 0.11535412314032954 |
| obesity | risk_factor | 0.10770761607229362 |
| lymphoma | subtype | 0.1071804269575406 |
| rhabdomyosarcoma | subtype | 0.10617195204811108 |
| cholangiocarcinoma | disease | 0.10154677346459164 |
| pancreatic cancer | disease | 0.09911670524877436 |
| skin cancer | disease | 0.09653289642821176 |

## Link Prediction (Top Suggestions)
| u | type_u | v | type_v | ensemble_score |
|---|---|---|---|---|
| radiation therapy | treatment | chemotherapy | treatment | 0.867 |
| radiation therapy | treatment | surgery | treatment | 0.477 |
| radiation therapy | treatment | tp53 | gene | 0.431 |
| radiation therapy | treatment | targeted therapy | treatment | 0.424 |
| radiation therapy | treatment | brain tumor | disease | 0.344 |
| myb | gene | nfib | gene | 0.343 |
| myb | gene | mybl1 | gene | 0.343 |
| nfib | gene | mybl1 | gene | 0.343 |
| crest | treatment | fluid | treatment | 0.343 |
| crest | treatment | humidifier | treatment | 0.343 |
| crest | treatment | tonsillectomy | treatment | 0.343 |
| fluid | treatment | humidifier | treatment | 0.343 |
| fluid | treatment | tonsillectomy | treatment | 0.343 |
| nf2 | gene | bap1 | gene | 0.234 |
| nf2 | gene | axl | gene | 0.234 |
| radiation therapy | treatment | prostate cancer | disease | 0.191 |
| radiation therapy | treatment | immunotherapy | treatment | 0.189 |
| nf2 | gene | pdgfrb | gene | 0.179 |
| radiation therapy | treatment | myelodysplastic syndromes | disease | 0.164 |
| radiation therapy | treatment | neuroendocrine tumors | disease | 0.160 |
| radiation therapy | treatment | chronic myeloid leukemia | disease | 0.153 |
| radiation therapy | treatment | breast cancer | disease | 0.148 |
| nf2 | gene | hyperthermic intraperitoneal chemotherapy | treatment | 0.146 |
| radiation therapy | treatment | biological therapy | treatment | 0.139 |
| radiation therapy | treatment | hormone therapy | treatment | 0.132 |

## Statistical Validation
- Degree distribution does **not strongly support a power-law** over an exponential model (AIC comparison).
- Spearman correlation (degree vs betweenness): **r = 0.975**, p = 0
- Spearman correlation (degree vs eigenvector): **r = 0.653**, p = 9.87e-174

## Node Property Prediction
- Neighbor-majority accuracy: **8.45%** on 142 hidden nodes.

## Traversal & Shortest Paths
### BFS (depth ≤ 3) from seeds
```text
Seed: lung cancer
  lung cancer → surgery → chemotherapy → radiation therapy → immunotherapy → laser therapy → targeted therapy → photodynamic therapy → cryosurgery → cisplatin → carboplatin → etoposide → irinotecan → topotecan → lurbinectedin → paclitaxel → docetaxel → vinorelbine → gemcitabine → osimertinib

Seed: liver cancer
  liver cancer → liver transplantation → radiation therapy → chemotherapy → immunotherapy → transarterial chemoembolization → transarterial radioembolization → photodynamic therapy → chimeric antigen receptor t-cell therapy → chest computed tomography → magnetic resonance imaging → liver biopsy → endoscopic retrograde cholangiopancreatography → magnetic resonance cholangiopancreatography → cirrhosis → nonalcoholic fatty liver disease → nonalcoholic steatohepatitis → fascioliasis → hereditary hemochromatosis → primary biliary cirrhosis
```

### DFS (preorder) from seeds
```text
Seed: lung cancer
  lung cancer → surgery → adrenal gland cancer → chemotherapy → anal cancer → radiation therapy → acoustic neuroma → nf2 → malignant mesothelioma → targeted therapy → anaplastic thyroid cancer → adjuvant radiotherapy → adenoid cystic carcinoma → palliative radiation therapy → adrenocortical carcinoma → genetic counseling → tp53 → bladder cancer → transurethral resection of bladder tumor → radical cystectomy

Seed: liver cancer
  liver cancer → liver transplantation → cholangiocarcinoma → radiation therapy → acoustic neuroma → nf2 → malignant mesothelioma → targeted therapy → anaplastic thyroid cancer → chemotherapy → adrenal gland cancer → surgery → anal cancer → cisplatin → astroblastoma → craniotomy → etoposide → lung cancer → immunotherapy → chronic lymphocytic leukemia
```

### Shortest paths among seeds
```text
lung cancer → radiation therapy → liver cancer
```

## Biological Interpretation (High-Level)
- The graph is highly connected, suggesting that the underlying disease–gene–treatment landscape is strongly interlinked. Many entities participate in shared pathways or therapeutic contexts.
- The rich community structure indicates multiple functional modules or thematic clusters, potentially corresponding to disease subtypes, shared genetic pathways, or co-occurring symptom groups.
- The highest-scoring predicted relation is between **radiation therapy** and **chemotherapy** (treatment–treatment), with ensemble score ≈ **0.867**. This edge is a strong candidate for follow-up curation.

---
Generated by Module 6 (enhanced analysis).
