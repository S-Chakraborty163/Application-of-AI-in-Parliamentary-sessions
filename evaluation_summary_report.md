# Parliamentary AI Arena: Double-Blind Evaluation Report

## 1. Executive Summary
| Model            |   Quality Score |   Faithfulness |   Relevance |   FollowUp Quality |   Citation Accuracy |   Entity Grounding |   Domain Expertise |
|:-----------------|----------------:|---------------:|------------:|-------------------:|--------------------:|-------------------:|-------------------:|
| llama3.1:latest  |           88.41 |          90.64 |       87.45 |              94.69 |               89.98 |              89.83 |              62.71 |
| mistral:latest   |           86.74 |          91.59 |       86.63 |              92.78 |               89.59 |              89.91 |              51.43 |
| qwen2.5:latest   |           82.97 |          86.24 |       82.74 |              89.27 |               85.19 |              85.18 |              53.97 |
| sansad-v2:latest |           91.03 |          92.89 |       90.49 |              95.04 |               91.18 |              91.59 |              60.03 |

## 2. Visualizations
### Head-to-Head Win Rate
Percentage of topics where a model scored the absolute highest Quality Score.

![Win Rate](viz_win_rate.png)

### Domain Mastery Radar
Visualizes which model covers the most ground across the 5 hardest parliamentary metrics.

![Radar Chart](viz_radar_chart.png)

### Advanced Domain & Fine-Tuning Performance
Comparing the models across 8 rigorous parliamentary metrics to highlight domain expertise.

![Advanced Metrics](viz_advanced_metrics.png)

### Quality Distribution (Consistency)
Shows the consistency of each model across all topics. A wider 'fat' top means the model consistently gives perfect answers, while a long tail means it occasionally hallucinates or fails completely.

![Quality Distribution](viz_quality_distribution.png)

