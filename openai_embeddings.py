import os
import openai
from neo4j import GraphDatabase, Result
import pandas as pd
from openai.error import APIError
from time import sleep
from bs4 import BeautifulSoup


def remove_html_markup(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    plain_text = soup.get_text()
    return plain_text


def generate_embeddings(file_name, limit=None):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )

    driver.verify_connectivity()

    query = """MATCH (a:Article) WHERE a.body IS NOT NULL
    RETURN a.id AS id, a.title AS title, a.body AS body"""

    if limit is not None:
        query += f" LIMIT {limit}"

    articles = driver.execute_query(
        query,
        result_transformer_=Result.to_df
    )

    print(len(articles))

    embeddings = []

    for _, n in articles.iterrows():
        body = remove_html_markup(n['body'])

        successful_call = False
        while not successful_call:
            try:
                res = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=f"{n['title']}: {body}",
                    encoding_format="float"
                )
                successful_call = True
            except APIError as e:
                print(e)
                print("Retrying in 5 seconds...")
                sleep(5)

        print(n['title'])

        embeddings.append({"id": n['id'], "embedding": res['data'][0]['embedding']})

    embedding_df = pd.DataFrame(embeddings)
    embedding_df.head()
    embedding_df.to_csv(file_name, index=False)

#generate_embeddings('openai-embeddings.csv',limit=1)
generate_embeddings('openai-embeddings-full.csv',limit=None)
