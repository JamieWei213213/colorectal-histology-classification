# Error analysis: observations first

No histology confusions were observed locally: official data and pretrained weights were not downloaded. Synthetic class errors have no tissue interpretation.

After Colab, read each model's `final_test/confusion_pairs.csv`, row-normalized confusion matrix, `predictions.csv`, and first indexed incorrect/correct examples. The generated `error_analysis.md` describes measured counts only. Rows are true labels; columns are predictions. Divide a pair's count by the number of true-class images to compare classes of different sizes. The displayed example grid is deliberately selected by split order, not by dramatic appearance; it may not cover the most frequent pair.

Complete this worksheet for the top two or three measured pairs:

| Field | What to record |
|---|---|
| Observation | True A predicted B: actual count and fraction; model and checkpoint |
| Visible evidence | Specific indexed patches; describe only visible colors, texture, layout or mixed regions |
| Hypothesis | Possible loss of detail, similar appearance, ambiguous patch content or distribution shift; label it unverified |
| Check needed | Original-resolution images, provenance, comparison across models, pathologist review |
| Limit | Cannot infer a cellular mechanism, prognosis, immune activity or diagnostic validity |

Conditional questions, not observed findings: **if** smooth muscle and stroma are confused, inspect whether coarse texture distinguishes them at 28 pixels. **If** normal mucosa and tumor epithelium are confused, inspect whether gland structure was lost by downsampling. **If** background and mucus are confused, inspect light regions and tissue coverage. Do not say any of these occurred unless the saved table shows it; even then, these proposed explanations remain unverified.

Test-error review is descriptive. Do not use it to tune this same experiment and present the same test split as untouched. A future protocol needs a separately protected evaluation design.
