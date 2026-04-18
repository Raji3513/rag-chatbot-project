"""
RAG Chain Module
Combines retrieval and LLM generation to answer questions
using context from the FAISS vector database.
"""

from utils.retriever import retrieve_documents


# Prompt template for answer generation
PROMPT_TEMPLATE = (
    "Answer the question based on the context below. "
    "If the context does not contain enough information, "
    "say 'I don't have enough information to answer that question.'\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def generate_answer(llm, question, context):
    """
    Generate an answer using the LLM with the given context.

    Args:
        llm: A LangChain-compatible LLM instance.
        question (str): The user's question.
        context (str): Retrieved context from documents.

    Returns:
        str: The generated answer.
    """
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    answer = llm.invoke(prompt)

    # Ensure we return a clean string
    if isinstance(answer, str):
        return answer.strip()
    return str(answer).strip()


def ask_question(vector_store, llm, question, k=3):
    """
    Full RAG pipeline: retrieve context from vector store and generate answer.

    Args:
        vector_store: A loaded FAISS vector store instance.
        llm: A LangChain-compatible LLM instance.
        question (str): The user's question.
        k (int): Number of documents to retrieve.

    Returns:
        dict: Contains 'answer', 'sources', 'context', and 'documents'.
    """
    # Handle empty question
    if not question or not question.strip():
        return {
            "answer": "Please enter a valid question.",
            "sources": [],
            "context": "",
            "documents": []
        }

    try:
        # Step 1: Retrieve relevant documents
        print("\n  [RAG] Step 1: Retrieving relevant documents...")
        documents = retrieve_documents(vector_store, question, k=k)

        # Step 2: Build context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in documents])
        sources = [doc.metadata.get("source", "Unknown") for doc in documents]

        # Step 3: Generate answer using LLM
        print("  [RAG] Step 2: Generating answer with LLM...")
        answer = generate_answer(llm, question, context)
        print("  [RAG] Step 3: Answer generated successfully")

        return {
            "answer": answer,
            "sources": sources,
            "context": context,
            "documents": documents
        }

    except ValueError as e:
        return {
            "answer": "Error: {}".format(str(e)),
            "sources": [],
            "context": "",
            "documents": []
        }
    except Exception as e:
        return {
            "answer": "Error generating answer: {}".format(str(e)),
            "sources": [],
            "context": "",
            "documents": []
        }
