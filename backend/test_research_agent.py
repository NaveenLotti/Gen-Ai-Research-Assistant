from app.agents.research_agent import research_agent


query = "What deep learning architecture is proposed for predicting the remaining useful life of Li-ion batteries?"

result = research_agent(query)


print("\n" + "=" * 60)
print("RESEARCH AGENT")
print("=" * 60)

print("\nQUESTION:")
print(query)

print("\nANSWER:")
print(result["answer"])

print("\nAGENT:")
print(result["agent"])

print("\n" + "=" * 60)
print("SOURCES")
print("=" * 60)

for i, source in enumerate(result["sources"], start=1):
    print(f"\n[{i}] Page {source['page']}")
    print("-" * 40)
    print(source["content"][:300])