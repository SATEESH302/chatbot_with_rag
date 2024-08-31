import dotenv
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough
import pandas as pd

REVIEWS_CSV_PATH = "reviews.csv"
REVIEWS_CHROMA_PATH = "chroma_data"


def load_environment():
    dotenv.load_dotenv()


def load_reviews(file_path):
    loader = CSVLoader(file_path=file_path, source_column="review")
    return loader.load()


def initialize_vector_db(reviews, persist_directory):
    return Chroma.from_documents(
        reviews, OpenAIEmbeddings(), persist_directory=persist_directory
    )


def create_prompt_templates():
    review_template_str = """Your job is to use patient
    reviews to answer questions about their experience at
    a hospital. Use the following context to answer questions.
    Be as detailed as possible, but don't make up any information
    that's not from the context. If you don't know an answer, say
    you don't know.

    {context}
    """

    review_system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["context"],
            template=review_template_str,
        )
    )

    review_human_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["question"],
            template="{question}",
        )
    )
    messages = [review_system_prompt, review_human_prompt]
    review_prompt_template = ChatPromptTemplate(
        input_variables=["context", "question"],
        messages=messages,
    )
    return review_prompt_template


def create_review_chain(reviews_vector_db, review_prompt_template):
    reviews_retriever = reviews_vector_db.as_retriever(k=3)
    chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    review_chain = (
        {"context": reviews_retriever, "question": RunnablePassthrough()}
        | review_prompt_template
        | chat_model
        | StrOutputParser()
    )
    return review_chain
