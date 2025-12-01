# Scripts Overview

This repository contains two main groups of scripts under the `scripts/` folder: **Assets Extraction** and **Threat Elicitation**. Each group implements a series of processes that support the extraction, processing, and analysis of legal texts and threat information using AI (OpenAI API) and data processing techniques.

**Methodology:**
The implementation of these scripts is based on a structured methodological framework that combines natural language processing (NLP) techniques, machine learning for clustering and vector computations, and batch processing strategies. This methodology is implemented by these scripts to ensure comprehensive analysis and consolidation of asset and threat data extracted from legal texts.

---

## Assets Extraction

- <a id="extract-asset"></a> **extract_asset.py**
  This script loads legal text documents and extracts nouns using spaCy. It then computes noun vectors, runs clustering experiments (varying thresholds for cluster selection and merging), and prepares a batch of requests. These requests are sent to the AI API to extract structured asset clusters from the legal text.

- <a id="read-extract-asset-result"></a> **read_extract_asset_result_by_batch.py**
  This script processes the raw batch results from the AI API. For each result, it parses the response JSON, applies duplicate removal (using NLP lemmatization) within each cluster, and then updates an intermediate JSON file with the final clusters. The updated results are saved for later review or further analysis.

- <a id="plot-result"></a> **plot_result.py**
  After the experiments have been executed, this script loads the results and plots key metrics. It displays the number of extracted assets and the cohesion of clusters (average distance to the centroid) per experiment. The visual output helps to identify the top experiments based on asset extraction quality.

---

## Threat Elicitation

- <a id="extract-threat-with-batch"></a> **extract_threat_with_batch.py**
  This script is responsible for processing legal text files to identify potential security threats. It prepares messages for the AI API using given instructions, creates batched requests (multiple executions per file), and sends these requests to the API. The responses are then used to extract threat-related information from the legal articles.

- <a id="read-threat-from-batch"></a> **read_threat_from_batch.py**
  This script retrieves batch results specific to threat elicitation. It extracts the necessary threat information from the responses, aggregates the data, and saves the final threat summary into an output file.

- <a id="group-threat-by-articles-runs"></a> **group_threat_by_articles_runs.py**
  This script aggregates threat information by grouping the responses according to legal articles. It reads aggregated batch responses and produces an output JSON that summarizes the threats associated with each article.

- <a id="merge-similar-threat-by-articles"></a> **merge_similar_threat_by_articles.py**
  This script further processes the threat data by merging similar threat entries based on their associated articles. It produces a consolidated output where redundant or similar threat items are merged to simplify analysis.

- <a id="merge-similar-threat"></a> **merge_similar_threat.py**
  This script performs a similar consolidation process for threat data. It transforms the merged results and creates one or more batch entries to update the threat analysis by merging similar threat entries from the raw batch results.

- <a id="get-final-threat"></a> **get_final_threat.py**
  This script downloads the batch results using a provided batch ID. It extracts the embedded JSON from the content field of each response, consolidates the threat information, and saves the final, aggregated threat data to a specified file.

---

## Compositionality Phase

- Consolidated threats are automatically converted into Subject–Attribute–Verb–Object (SAVO) and Subject–Verb–Object (SVO) structures so they can be composed into heterogeneous cause–effect scenarios. The current experiments use **three** regulatory factors (AI Act, NIS 2, ISO 9241-210), yet the methodology scales to $N$ independent factors.
- Full outputs for the three oracle configurations live in `results/Compositionality/oracle_0/results.json`, `results/Compositionality/oracle_1/results.json`, and `results/Compositionality/oracle_2/results.json`.
- Each oracle is orchestrated through dedicated n8n workflows stored under `scripts/compositionality/oracle_*/Compositionality 3 context, Oracolo *.json`, which encode the pipeline logic leveraged during experiments.
- The runtime logic for the auxiliary Python services (required by Oracles 1 and 2) is exposed via HTTP through `scripts/compositionality/oracles_server/main.py`; the n8n workflows invoke these endpoints (for example `/filter_oracol_1` and `/filter_oracol_2`) whenever analyst-driven filtering is needed.

### Low-Human-Intervention Oracle

1. **Fully automated SAVO/SVO conversion** – an `o1-mini` assistant processes the AI Act, NIS 2, and ISO 9241-210 threat sets independently and emits both structured formats.
2. **Fully automated heterogeneous composition** – a second `o1-mini`, guided by the compositionality prompt, evaluates every cross-context pairing across the three factors and generates multi-factor causal chains (`results/Compositionality/oracle_0/results.json`).

### Medium-Human-Intervention Oracle

1. **SVO conversion** – same automated step as above, limited to the SVO format.
2. **Analyst-driven topical filter** – for each context (AI Act, NIS 2, ISO 9241-210) an analyst defines a hypernym sentence; BERTopic compares its topics to every threat and retains only those with similarity > 0.55 (endpoint `/filter_oracol_0`).
3. **LLM composition** – the filtered threats are recombined with the same assistant used in the low-human oracle, yielding tighter scenarios (`results/Compositionality/oracle_1/results.json`).

### Medium-High-Human-Intervention Oracle

1. **SAVO/SVO conversion** – identical to step 1 of the low-human oracle.
2. **Explicit per-role hypernyms** – analysts define WordNet-derived hypernym lists for Subject, Verb, and Object. The `/filter_oracol_1` endpoint computes cosine similarities against those lists and keeps the top 3 occurrences per role and context.
3. **Combinatorial composition** – the filtered sets from the three contexts are multiplied via Cartesian product to enumerate all plausible combinations (`results/Compositionality/oracle_2/results.json`).

---

## Cross-Validation of Composed Threats

- The CLI script `scripts/compositionality/cross_check/cross_check.py` implements the cross-validation step described in the paper (Section "Cross Validation"). It can either ingest a pre-shaped JSON payload (`--input`) or read the three oracle outputs directly via `--oracle-files` (defaulting to the JSON artifacts in `results/Compositionality/oracle_*`).
- Each oracle run is parsed into normalized SVO triples; unique subjects, verbs, and objects are embedded with `SentenceTransformer` (`stsb-roberta-base-v2`). The script then evaluates all cross-run combinations in parallel (Joblib + multicore) and scores them with per-role cosine similarity plus aggregate mean/std statistics.
- Thresholds replicate the paper’s semantic-overlap criteria (defaults: subject ≥ 0.30, verb ≥ 0.30, object ≥ 0.30, mean ≥ 0.50, std ≤ 0.15). Additional legacy filters can be applied, and results can be capped via `--max-results`.
- When provided with a composition file (typically the Low-Human oracle output), the script attaches the original Threat A/B pairs and the composed path for traceability, then collapses matches to the reference run to highlight which low-intervention threats are corroborated by the other oracles.
- Outputs (matches + metadata) are saved as JSON—e.g., `results/Compositionality/cross_check/result_055_tok_k_3.json`—and have shown that roughly 30% of low-intervention compositions are independently rediscovered by the higher-intervention pipelines, confirming semantic consistency across workflows.

---

## Workflow for Applying the Methodology

1. **Preparation & Data Collection**
   - Gather the legal text documents to be analyzed.
   - Ensure the texts are accessible as expected.

2. **Assets Extraction Workflow**
   - **Extraction:**
     - Run `extract_asset.py` to load legal texts, extract nouns using spaCy, compute noun vectors, and carry out clustering experiments.
     - The script prepares batched requests that are submitted to the AI API to extract asset clusters.
   - **Post-Processing:**
     - Use `read_extract_asset_result_by_batch.py` to process the raw API responses. The script parses the JSON content, removes duplicates with NLP lemmatization, and updates an intermediate JSON file with the final clusters.
   - **Visualization:**
     - Execute `plot_result.py` to visualize key metrics like the number of extracted assets and cluster cohesion, aiding in the identification of top-performing experiments.

3. **Threat Elicitation Workflow**
   - **Threat Extraction:**
     - Run `extract_threat_with_batch.py` to process legal texts for potential security threats. This script prepares and sends batched API requests for threat identification.
   - **Aggregation & Grouping:**
     - Use `read_threat_from_batch.py` to extract and aggregate threat details from API responses.
     - Run `group_threat_by_articles_runs.py` to group threats by legal articles, creating an initial organized summary.
   - **Consolidation:**
     - Execute `merge_similar_threat_by_articles.py` to merge similar threat entries grouped by articles.
     - Run `merge_similar_threat.py` for further consolidation of batch results.
   - **Finalization:**
     - Run `get_final_threat.py` to download and compile the comprehensive threat data using a specified batch ID, yielding the final aggregated threat summary.

4. **Compositionality Workflows (n8n + HTTP services)**
   - Start the FastAPI service in `scripts/compositionality/oracles_server/main.py` so that Oracles 1 and 2 can call the `/filter_oracol_0` and `/filter_oracol_1` endpoints during execution.
   - Import and run the n8n workflow files located under `scripts/compositionality/oracle_*` (one JSON per oracle). Provide the SAVO/SVO threat exports for AI Act, NIS 2, and ISO 9241-210 as inputs; each workflow orchestrates the LLM calls and filtering logic described above.
   - Collect the resulting composed threats from `results/Compositionality/oracle_0/results.json`, `results/Compositionality/oracle_1/results.json`, and `results/Compositionality/oracle_2/results.json`.

5. **Cross-Validation Workflow**
   - Run `python scripts/compositionality/cross_check/cross_check.py --oracle-files <path_oracle0> <path_oracle1> <path_oracle2> --output results/Compositionality/cross_check/<file>.json` (adjust options as needed).
   - Optional flags such as `--mean-sim`, `--sim-subject`, and `--sim-verb` let you tune the semantic-overlap thresholds cited in the paper; `--composition-file` attaches the detailed trace for the selected reference run.
   - Inspect the generated JSON (e.g., `results/Compositionality/cross_check/result_055_tok_k_3.json`) to review overlaps across oracles and quantify the ~30% consistency rate.

---

This README provides a high-level overview of each script’s function and the workflow to apply the underlying methodology. Use the appropriate scripts based on whether you wish to extract assets from legal texts or analyze potential threats described within those texts.