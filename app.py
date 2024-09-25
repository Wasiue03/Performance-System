import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import subprocess
from flask import Flask, request, jsonify

app = Flask(__name__)


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower()  
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}") 
        return pd.DataFrame()  


def create_textual_representation(df):
    if df.empty:
        return []  
    return df.apply(lambda row: ' | '.join(row.astype(str)), axis=1)


def vectorize_data(text_data, model):
    embeddings = model.encode(text_data.tolist())
    return embeddings


def retrieve_relevant_rows(query_vector, embeddings, df, top_n=5):
    similarities = cosine_similarity([query_vector], embeddings)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]  
    return df.iloc[top_indices]


def run_model(query, context):
    full_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    result = subprocess.run(
        ["ollama", "run", "llama3.1", full_prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
        encoding='utf-8'
    )
    return result.stdout


df = load_data('sales_performance_data.csv')  
text_data = create_textual_representation(df)
model = SentenceTransformer('all-MiniLM-L6-v2')  

if not df.empty:  
    embeddings = vectorize_data(text_data, model)  
else:
    embeddings = None 


@app.route('/query', methods=['POST'])
def query_dataset():
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    if embeddings is None:  
        return jsonify({'error': 'Dataset not loaded correctly.'}), 500

    query_vector = model.encode(query)  
    relevant_rows = retrieve_relevant_rows(query_vector, embeddings, df)
    context = relevant_rows.to_string(index=False)  

    result = run_model(query, context)
    return jsonify({'model_output': result})

@app.route('/team_performance', methods=['POST'])
def team_performance():
    query = "What is the performance of the team?"  
    if embeddings is None: 
        return jsonify({'error': 'Dataset not loaded correctly.'}), 500

    query_vector = model.encode(query) 
    relevant_rows = retrieve_relevant_rows(query_vector, embeddings, df)
    context = relevant_rows.to_string(index=False)  

    result = run_model(query, context)
    return jsonify({'model_output': result})

@app.route('/total_tours', methods=['POST'])
def total_tours():
    query = "What is the total number of tours?"  
    if embeddings is None: 
        return jsonify({'error': 'Dataset not loaded correctly.'}), 500

    query_vector = model.encode(query) 
    relevant_rows = retrieve_relevant_rows(query_vector, embeddings, df)
    context = relevant_rows.to_string(index=False)  

    result = run_model(query, context)
    return jsonify({'model_output': result})

@app.route('/camila_ali_performance', methods=['POST'])
def camila_ali_performance():
    query = "How has Camila Ali performed?"  
    if embeddings is None:  
        return jsonify({'error': 'Dataset not loaded correctly.'}), 500

    query_vector = model.encode(query)  
    relevant_rows = retrieve_relevant_rows(query_vector, embeddings, df)
    context = relevant_rows.to_string(index=False)  

    result = run_model(query, context)
    return jsonify({'model_output': result})


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
