import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.config import get_settings

s = get_settings()

# 1. Neo4j
driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_username, s.neo4j_password))
driver.verify_connectivity()
with driver.session(database=s.neo4j_database) as session:
    count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
print(f"✅ Neo4j connected — {count} nodes")
driver.close()

# 2. OpenAI embeddings
emb = OpenAIEmbeddings(model=s.openai_embed_model, api_key=s.openai_api_key)
print(f"✅ Embeddings OK — vector dim {len(emb.embed_query('hello'))}")

# 3. OpenAI chat
llm = ChatOpenAI(model=s.openai_chat_model, api_key=s.openai_api_key, temperature=0)
print(f"✅ Chat OK — {llm.invoke('Reply with just: pong').content}")