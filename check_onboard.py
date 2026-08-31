from indexer.repo_onboarder import onboard_repo

result = onboard_repo("big_repo")
print(f"\nFiles: {result['files_total']} | unchanged: {result['files_unchanged']}")

hits = result["retriever"].retrieve("apply a discount to a product's price", top_k=3)
for h in hits:
    print(h["id"], h.get("rrf_score"))