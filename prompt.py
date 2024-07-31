from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from process_files import get_db

llm = ChatOpenAI(model_name="gpt-4", temperature=0)


def get_results(question):
    """
        Retrieve answers from a language model based on the input question.

        Args:
            question (str): The question to be answered.

        Returns:
            tuple: A tuple containing the response to the question, the source document where the information was retrieved from, and the page number of the source document.
    """
    # question = "Give me details about the work of providing sluice valves to Gajapathinagaram?"
    # Build prompt
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""

    # Initilaize chain
    # Set return_source_documents to True to get the source document
    # Set chain_type to prompt template defines
    QA_CHAIN_PROMPT = PromptTemplate.from_template(
        template)  # Run chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=get_db().as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    # question = "Is probability a class topic?"
    result = qa_chain({"query": question})
    # print(result)
    # Check the result of the query
    # print(result["result"])
    # Check the source document from where we
    # print(result["source_documents"][0])
    # print(result["source_documents"][0].metadata["source"])
    # print(result["source_documents"][0].metadata["page"])
    return result["result"], result["source_documents"][0].metadata[
        "source"], result["source_documents"][0].metadata["page"]
