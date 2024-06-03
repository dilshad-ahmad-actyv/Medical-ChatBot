prompt_template = """
    Use the following pieces of the information to answer the User's question.
    If you don't know the answer, just say that you don't know, don't try to make up the random answer.

    context: {context}
    question: {question}

    Only return the helpful answer bellow nothing else
    Helpful answer:
"""