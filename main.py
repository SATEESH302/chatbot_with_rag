from utils import (
    create_review_chain,
    create_prompt_templates,
    initialize_vector_db,
    load_reviews,
    load_environment,
)
from utils import REVIEWS_CSV_PATH, REVIEWS_CHROMA_PATH


def main(query):
    load_environment()
    reviews = load_reviews(REVIEWS_CSV_PATH)
    reviews_vector_db = initialize_vector_db(reviews, REVIEWS_CHROMA_PATH)
    review_prompt_template = create_prompt_templates()
    review_chain = create_review_chain(reviews_vector_db, review_prompt_template)
    # question = """ How was the service at City Hospital?"""
    res = review_chain.invoke(query)
    print("res#####", res)
    return res


query = """ How was the service at City Hospital?"""
result = main(query)
print("result :", result)
