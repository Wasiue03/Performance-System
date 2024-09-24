import pandas as pd
import subprocess
import streamlit as st

# Load and preprocess dataset
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower()  # Standardize column names
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Generate a summary or text-based context from the dataset
def generate_dataset_context(df):
    # For simplicity, convert a part of the dataset to string format
    context = df.head(10).to_string(index=False)
    return f"Here is a preview of the dataset:\n{context}"

# Function to run the model with text input, including the dataset context
def run_model(query, dataset_context):
    try:
        # Combine query with dataset context
        full_prompt = f"Dataset context:\n{dataset_context}\n\nBased on this context, please answer the following question: {query}"
        
        # Running the model with the query and context as a string
        result = subprocess.run(
            ["ollama", "run", "llama3.1", full_prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        st.error(f"Error running the model: {e.stderr}")
        return None

# Main function for Streamlit app
def main():
    st.title("Intelligent Dataset Query with LLM")

    # Step 1: Upload the dataset
    file_upload = st.file_uploader("Upload your CSV dataset", type=["csv"])

    if file_upload is not None:
        df = load_data(file_upload)

        if df is not None:
            # Display a preview of the dataset
            st.write("Dataset Preview:")
            st.dataframe(df.head())

            # Step 2: Generate the dataset context (summary)
            dataset_context = generate_dataset_context(df)

            # Step 3: Take the user's question as input
            user_query = st.text_input("Ask a question about the dataset (e.g., 'How many calls were made on Monday?')")

            if user_query:
                # Step 4: Run the model with the query and dataset context
                result = run_model(user_query, dataset_context)

                if result:
                    # Display the model's output
                    st.write("Model Output:")
                    st.text(result)

# Run the Streamlit app
if __name__ == "__main__":
    main()
