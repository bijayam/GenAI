from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_embeddings import retriever, vector_store

model = OllamaLLM(model="llama3.2:3b")

template = """
You are a helpful AI assistant.

Answer the question ONLY using the provided context.

If the answer is not found in the context, say:
"I could not find the answer in the provided documents."

Context:
{rag_result}

Question:
{question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:

    print("\n-------------------------------")

    question = input("Ask your question (q to quit): ")

    if question.lower() == "q":
        break

    #docs = retriever.invoke(question)
    results = vector_store.similarity_search_with_score(
                        question,
                        k=5
                    )

    
    # Print similarity scores and retrieved documents
    # How to interpret similarity scores: 
    # For ChromaDB: with similarity_search_with_score(), the score is 
    # typically a distance, not a similarity percentage.
    # That means:
    # LOWER score = more similar
    # HIGHER score = less similar
    # Example: 
    # Similarity Score: 0.21   ← excellent
    # Similarity Score: 0.35   ← good
    # Similarity Score: 0.78   ← weak
    # Similarity Score: 1.20   ← probably irrelevant
    docs = []
    for i, (doc, score) in enumerate(results):

        print(f"\nSOURCE {i+1}")
        print(f"Similarity Score: {score}")

        print(doc.page_content[:300])

        print("\n-----------------------------")

        docs.append(doc)

    # Join the retrieved documents into a single string to pass to the LLM
    rag_result = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    # Invoke the chain with the retrieved context and the user's question
    answer = chain.invoke({
    "rag_result": rag_result,
    "question": question
    })

    # Print the answer from the LLM
    print("\nAnswer:\n")
    print(answer)