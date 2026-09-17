from app.agents.supervisor import supervisor_node


questions = [
    "What deep learning architecture is proposed?",
    "Give me a summary of this research paper.",
    "Compare the methodology with another approach."
]


for question in questions:

    state = {
        "query": question,
        "answer": "",
        "sources": [],
        "agent": ""
    }

    result = supervisor_node(state)

    print("\nQuestion:", question)
    print("Selected Agent:", result["agent"])