import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import subprocess
import streamlit as st


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower()  
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


def create_textual_representation(df):
    return df.apply(lambda row: ' | '.join(row.astype(str)), axis=1)


def vectorize_data(text_data, model):
    embeddings = model.encode(text_data.tolist())
    return embeddings


def retrieve_relevant_rows(query_vector, embeddings, df, top_n=5):
    similarities = cosine_similarity([query_vector], embeddings)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]  # Get top N indices
    return df.iloc[top_indices]


def run_model(query, context):
    full_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    result = subprocess.run(
        ["ollama", "run", "llama3.1", full_prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )
    return result.stdout


def main():
    st.title("Intelligent Dataset Query with RAG")

    file_upload = st.file_uploader("Upload your CSV dataset", type=["csv"])

    if file_upload is not None:
        df = load_data(file_upload)

        if df is not None:
           
            text_data = create_textual_representation(df)

            
            model = SentenceTransformer('all-MiniLM-L6-v2')  
            embeddings = vectorize_data(text_data, model)

            user_query = st.text_input("Ask a question about the dataset:")

            if user_query:
                query_vector = model.encode(user_query)  
                relevant_rows = retrieve_relevant_rows(query_vector, embeddings, df)
                context = relevant_rows.to_string(index=False)  

                result = run_model(user_query, context)
                st.write("Model Output:")
                st.text(result)


if __name__ == "__main__":
    main()
