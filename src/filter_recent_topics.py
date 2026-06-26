import re
import random

def main():
    # Keywords strongly associated with 2019-2024 parliamentary events
    recent_keywords = [
        "2020", "2021", "2022", "2023", "2024", "2025", "2026",
        "farm laws", "pegasus", "data protection", "dpdp", "criminal law", "bharatiya nyaya", 
        "new tax regime", "monsoon session", "article 370", "caa", "citizenship amendment",
        "women's reservation", "nari shakti", "covid", "pandemic", "vaccin", "nep 2020",
        "electoral bonds", "ayodhya", "infrastructure", "smart cities", "gst compensation",
        "telecom bill", "broadcasting", "digital india", "semiconductor", "ai", "deepfake"
    ]
    
    with open("indian_policy_discussions_250.md", "r", encoding="utf-8") as f:
        content = f.read()
        
    # Extract all questions (lines starting with a number and dot)
    matches = re.findall(r'^\d+\.\s+(.*?)$', content, re.MULTILINE)
    
    if not matches:
        print("No questions found!")
        return
        
    scored_topics = []
    for q in matches:
        # Give higher weight to questions containing recent keywords
        score = sum(1 for kw in recent_keywords if kw.lower() in q.lower())
        scored_topics.append((score, q))
        
    # Sort by score (descending), then shuffle within same scores to keep it diverse
    scored_topics.sort(key=lambda x: (x[0], random.random()), reverse=True)
    
    # Take top 150
    final_150 = [q for score, q in scored_topics[:150]]
    
    with open("final_150_eval_topics.md", "w", encoding="utf-8") as f:
        f.write("# Final 150 Curated Topics (Filtered for 2019-2024 Relevance)\n\n")
        for i, q in enumerate(final_150, 1):
            f.write(f"{i}. {q}\n")
            
    print(f"Successfully filtered {len(matches)} questions down to {len(final_150)} recent topics.")
    print("Saved to final_150_eval_topics.md")

if __name__ == "__main__":
    main()
