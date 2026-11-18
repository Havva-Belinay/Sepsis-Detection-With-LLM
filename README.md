# Sepsis-Detection-With-LLM
Analysis and early detection of critical clinical conditions in emergency department patients—such as sepsis risk, antibiotic treatment management, and the need for intensive care—using decision support systems powered by Large Language Models (LLMs).

## Methodology & Repository Structure

* **Dataset & Clinical Guidelines:** To accurately extract the required clinical outcomes (sepsis risk, antibiotic therapy, and ICU admission need), the model was provided with clinical guideline documents alongside a patient dataset. *Due to patient privacy and data protection regulations, the raw dataset is not included in this repository.*
* **Iterative Model & Prompt Optimization:** To continuously improve prediction accuracy, various Large Language Models (LLMs), reference guidelines, and prompt engineering strategies were systematically tested and evaluated.
* **Sampling & Testing Strategy:** To optimize processing time during the experimentation phase, evaluations were conducted on a subset consisting of the first 10 patient records rather than the entire dataset.
* **Outputs & Best Performing Configuration:** All experimental outputs are stored in structured `.json` format within their respective directories. The configuration yielding the most consistent and accurate results is implemented in `en_iyi_calısan.py`, with its final outputs saved to `son_deneme.json`.
