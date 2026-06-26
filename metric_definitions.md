# Evaluation Metrics Definitions

To ensure a highly rigorous, academic standard for our Double-Blind Benchmark, we utilized an 8-metric grading system evaluated by a massive Cloudflare-hosted Judge Model. Each metric specifically tests a different dimension of LLM performance within the highly specialized domain of the Indian Parliamentary Knowledge Graph (2019-2024).

Below is a detailed breakdown of each metric.

### 1\. Overall Quality Score (0-100)

**Definition:** A holistic, weighted aggregate of all other metrics.
**Purpose:** Provides a single, digestible number to rank the models. This is not just a simple average, but a weighted score that heavily penalizes hallucinations (Faithfulness/Citation Accuracy) while rewarding strong Domain Expertise.

### 2\. Faithfulness (0-100)

**Definition:** The degree to which the model's answer is strictly derived from the provided Knowledge Graph context.
**Purpose:** In legal and parliamentary domains, an LLM cannot invent facts. A score of 100 means the model made zero claims outside of the provided context. If a model uses its own pre-trained knowledge to answer a question instead of the provided context, it is severely penalized.

### 3\. Citation Accuracy (0-100)

**Definition:** The precision with which the model cites specific Constitutional Articles, Bills, and Dates.
**Purpose:** A model might be "Faithful" (not lying), but still fail to properly cite its sources. This metric ensures that when the model mentions the *Digital Personal Data Protection Act*, it accurately associates it with the correct 2023 timeline and parliamentary session found in the graph.

### 4\. Entity Grounding (0-100)

**Definition:** How accurately the model identifies and links key entities (Politicians, Ministries, Committees, Acts).
**Purpose:** The Indian Parliament has complex overlapping entities. This metric grades whether the model correctly maps a speaker to their respective Ministry or Committee without mixing up roles (e.g., falsely attributing a quote by the IT Minister to the Finance Minister).

### 5\. Relevance (0-100)

**Definition:** How directly the model answers the user's specific prompt without hallucinating unnecessary filler.
**Purpose:** Models often "yap" or provide long, winding answers to sound smart. This metric forces the model to stay concise, direct, and entirely focused on the user's core question regarding the policy.

### 6\. Follow-Up Quality (0-100)

**Definition:** The model's ability to generate highly intelligent, critical counter-questions or follow-up prompts.
**Purpose:** We want our AI to be an active debate partner, not just a search engine. High scores indicate the model is capable of analyzing the policy it just explained and asking deep, structural questions about its potential loopholes or implementation challenges.

### 7\. Domain Expertise (0-100)

**Definition:** The tone, formality, and structural phrasing of the response.
**Purpose:** A parliamentary AI should not talk like a generic chatbot. It must adopt the formal, authoritative, and precise lexicon of a legal expert or parliamentarian. Models that use casual language or overly enthusiastic emojis fail this metric.

