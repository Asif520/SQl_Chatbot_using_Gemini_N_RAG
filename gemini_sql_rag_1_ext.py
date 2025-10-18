import google.generativeai as genai
import re
import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sentence_transformers import SentenceTransformer
import faiss
import re
import json
import psycopg2
import warnings
warnings.filterwarnings("ignore")

def load_schema():
    with open("data/tables.json", "r") as f:
        tables = json.load(f)

    # Create schema dictionary
    schema_dict = {}
    for db in tables:
        db_id = db["db_id"]
        schema_parts = []
        for table_idx, table_name in enumerate(db["table_names_original"]):
            cols = []
            for col in db["column_names_original"]:
                if col[0] == table_idx:
                    col_name = col[1]
                    col_type = db["column_types"][db["column_names_original"].index(col)]
                    cols.append(f"{col_name} {col_type.upper()}")
            if cols:
                schema_parts.append(f"{table_name}({', '.join(cols)})")

        # Add foreign keys if available
        fk_str = ""
        if "foreign_keys" in db:
            for fk in db["foreign_keys"]:
                from_col = db["column_names_original"][fk[0]][1]
                to_col = db["column_names_original"][fk[1]][1]
                from_table = db["table_names_original"][db["column_names_original"][fk[0]][0]]
                to_table = db["table_names_original"][db["column_names_original"][fk[1]][0]]
                fk_str += f"FOREIGN KEY {from_table}({from_col}) REFERENCES {to_table}({to_col}); "

        schema_str = f"{db_id}({', '.join(schema_parts)})"
        if fk_str:
            schema_str += f" {fk_str.strip()}"

        schema_dict[db_id] = schema_str
    return schema_dict

def data_examples(schema_dict):
    # Load train_spider.json for examples
    with open("data/train_spider.json", "r") as f:
        train_data = json.load(f)

    sql_examples = []
    for ex in train_data:
        db_id = ex["db_id"]
        schema_str = schema_dict.get(db_id, f"{db_id} (Schema details not found)")
        request = ex["question"]
        query = ex["query"]
        example_str = f"Schema: {schema_str}\nRequest: {request}\nQuery: {query};"
        sql_examples.append(example_str)
    return sql_examples

@st.cache_resource
def prepare_sql_examples():
    schema_dict = load_schema()
    sql_examples = data_examples(schema_dict)

    return sql_examples

@st.cache_resource
def qus_data_examples():

    with open("data/business_qus_data.json", 'r') as f:
        qus_answer = json.load(f)

    unique = []
    for record in qus_answer:
        unique.append(record)

    #create formatted qus_ans dataset
    qus_examples = []
    for ex in unique:
        data = ex["Data"]
        qus = ex["Question"]
        ans = ex["Answer"]
        qus_example = f"Data: {data}\nQuestion: {qus}\nAnswer: {ans};"
        qus_examples.append(qus_example)

    return qus_examples

@st.cache_resource
def load_embedder():
    return SentenceTransformer("BAAI/bge-base-en-v1.5")

@st.cache_resource
def load_indexes():
    index = faiss.read_index("data/sql_examples.index")
    index_qus = faiss.read_index("data/qus_example.index")
    return index,index_qus

# def embed_store_tokens(sql_examples,qus_examples):
#
#     example_embeddings = np.load("data/example_embeddings.npy")
#
#     dimension = example_embeddings.shape[1]  # Embedding size (e.g., 768 for bge-base)
#     index = faiss.IndexFlatL2(dimension)  # Simple L2 distance index; for large datasets, consider IndexIVFFlat for faster search
#     index.add(example_embeddings)  # Add vectors to index
#
#     qus_example_embedding = np.load("data/qus_example_embedding.npy")
#
#     dimension_qus = qus_example_embedding.shape[1]  # Embedding size (e.g., 768 for bge-base)
#     index_qus = faiss.IndexFlatL2(dimension_qus)  # Simple L2 distance index; for large datasets, consider IndexIVFFlat for faster search
#     index_qus.add(qus_example_embedding)  # Add vectors to index
#
#     return index,index_qus


def generate_sql_with_custom_rag(gemini_model,schema, embedder,request,conversation_history,sql_examples,index, max_length=1024, temperature=0.4, top_p=0.9, k=3):
    try:
        # Step 1: Create a query string for retrieval
        query_text = f"Schema: {schema}\nRequest: {request}\nContext: {conversation_history}"
        query_embedding = embedder.encode([query_text], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')

        # Step 2: Retrieve top-k similar examples using FAISS
        distances, indices = index.search(query_embedding, k)
        retrieved_examples = [sql_examples[idx] for idx in indices[0] if idx != -1]

        # Step 3: Format retrieved examples for prompt
        examples_str = "\n\n".join(retrieved_examples) if retrieved_examples else "No similar examples found."

        # Step 4: Build prompt
        prompt = f"""
        You are a SQL expert. 
        Use the following examples, schema, and conversation context to generate a single, correct SQL query. 
        Assume a standard SQL database (PostgreSQL/MySQL).
        Return only the SQL query — no explanations.

        Examples:
        {examples_str}

        Database Schema:
        {schema}

        Conversation Context:
        {conversation_history}

        Request:
        {request}
        """

        # Step 5: Generate SQL using Gemini
        response = gemini_model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": 300,
            }
        )

        text = response.text.strip()

        sql_match = re.search(r"(SELECT.*?\n)", text, re.DOTALL | re.IGNORECASE)
        if sql_match:
            text = sql_match.group(1).strip()

        return text
    except Exception as e:
        return f"Error: {str(e)}"


def fetch_data_from_database(sql_query: str):
    conn = psycopg2.connect(
        host="ep-long-tooth-a1zzotwg-pooler.ap-southeast-1.aws.neon.tech",  # e.g., ep-silent-sunset-123456.neon.tech
        dbname="neondb",
        user="neondb_owner",
        password="npg_Bd06StQryYlV",
        sslmode="require")

    conn.cursor()
    df = pd.read_sql(sql_query, conn)
    conn.close()
    records = df.to_dict(orient="records")
    json_data = json.dumps(records, indent=2)

    return json_data


def generate_answer_from_json_data(gemini_model,json_data,embedder, request,conversation_history,qus_examples,index_qus, max_length=1024, temperature=0.5, top_p=0.9, k=3):
    try:
        # Step 1: Create query for retrieval
        query_text = f"Data: {json_data}\nQuestion: {request}\nContext: {conversation_history}"
        query_embedding = embedder.encode([query_text], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')

        # Step 2: Retrieve similar examples using FAISS
        distances, indices = index_qus.search(query_embedding, k)
        retrieved_examples = [qus_examples[idx] for idx in indices[0] if idx != -1]
        examples_str = "\n\n".join(retrieved_examples) if retrieved_examples else "No similar examples found."

        # Step 3: Build prompt for Gemini
        prompt = f"""
        You are a helpful AI assistant. 
        Use the provided data and conversation context to answer the question.
        Be concise and human-readable. 
        Do not include extra commentary or repeat data.

        Examples:
        {examples_str}

        Data:
        {json_data}

        Conversation Context:
        {conversation_history}

        Question:
        {request}
        """

        # Step 4: Generate answer using Gemini
        response = gemini_model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": 300,
            }
        )

        text = response.text.strip()

        # Optional cleanup for safety
        answer_match = re.search(r'(?i)(answer:)?\s*(.*)', text, re.DOTALL)
        if answer_match:
            text = answer_match.group(2).strip()

        return text
    except Exception as e:
        return f"Error: {str(e)}"

@st.cache_resource
def load_llm_model():
    # Configure Gemini
    genai.configure(api_key="AIzaSyCiGgeMMHrELnvKg-1ydHCVWlFm9LFLYpU")
    # Choose model
    return genai.GenerativeModel("gemini-2.0-flash")

def generate_text(gemini_model,schema,embedder, request, conversation_history,sql_examples, index,qus_examples, index_qus):
    # Step 1: Generate SQL
    sql_query = generate_sql_with_custom_rag(gemini_model,schema, embedder,request,conversation_history,sql_examples,index)

    # Step 2: Fetch data from DB using your existing function
    result_data = fetch_data_from_database(sql_query)

    # Step 3: Generate final natural-language answer
    answer = generate_answer_from_json_data(gemini_model,result_data,embedder, request,conversation_history,qus_examples,index_qus)

    return answer

def format_conversation_history(conversation_history):
    """Format the dictionary into readable text for passing to the model."""
    formatted = ""
    for msg in conversation_history["messages"]:
        formatted += f"{msg['role'].capitalize()}: {msg['content']}\n"
    return formatted.strip()

if __name__=="__main__":
    st.set_page_config(page_title="SQL Chatbot", page_icon="🤖", layout="centered")
    st.title("🤖 SQL Chatbot (Gemini + RAG Ready)")
    st.caption("Ask me anything about your database. Type below to start chatting!")

    schema = """ecommerce(customers(customer_id INT, first_name TEXT, last_name TEXT, email TEXT, phone TEXT, address TEXT, city TEXT, country TEXT, created_at TIMESTAMP)
    ,orders(order_id INT, customer_id INT, order_date TIMESTAMP, status TEXT, amount DECIMAL))"""

    sql_examples = prepare_sql_examples()

    qus_examples = qus_data_examples()

    embedder = load_embedder()

    index, index_qus = load_indexes()

    # Load model
    gemini_model = load_llm_model()

    # Ensure proper structure in session_state
    if "conversation_history" not in st.session_state or not isinstance(st.session_state.conversation_history, dict):
        st.session_state.conversation_history = {"messages": []}
    elif "messages" not in st.session_state.conversation_history:
        st.session_state.conversation_history["messages"] = []

    # Display previous messages
    for msg in st.session_state.conversation_history["messages"]:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        elif msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["content"])

    # --- Chat input box ---
    if user_input := st.chat_input("Type your question or SQL request..."):
        # Add user message
        st.session_state.conversation_history["messages"].append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Format history for prompt (if your generate_text uses it)
        history_text = format_conversation_history(st.session_state.conversation_history)

        # --- Generate model answer (your function here) ---
        response = generate_text(gemini_model, schema, embedder, user_input, history_text, sql_examples, index,
                                 qus_examples, index_qus)

        # Add assistant response
        st.session_state.conversation_history["messages"].append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

    # Sidebar options
    st.sidebar.header("⚙️ Settings")
    if st.sidebar.button("🧹 Clear Conversation"):
        st.session_state.conversation_history = []
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.info("Built with ❤️ using Streamlit + Python\n\nModel backend: Gemini + Custom RAG")




