from app.agents.supervisor import research_graph


questions = [
    "What is the main objective of these research papers?",
    "Give me a summary of the research paper.",
    "Compare the methodologies used in these papers."
]


for question in questions:

    print("\n" + "=" * 80)
    print("QUESTION:", question)
    print("=" * 80)

    result = research_graph.invoke({
        "query": question,
        "answer": "",
        "sources": [],
        "agent": ""
    })

    print("\nSELECTED AGENT:")
    print(result["agent"])

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")

    for source in result["sources"][:5]:

        print(
            f"- {source.get('paper_name', 'Unknown')} "
            f"(Page {source.get('page', '?')})"
        )