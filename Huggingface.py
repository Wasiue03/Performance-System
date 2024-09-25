import pandas as pd
import json
from transformers import pipeline

def ingest_df(file_path, file_type='csv', delimiter=','):
    if file_type == 'csv':
        df = pd.read_csv(file_path, delimiter=delimiter)
    elif file_type == 'json':
        with open(file_path, 'r') as file:
            df = json.load(file)
        df = pd.DataFrame(df)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    return df

# Load the dataset
df = ingest_df('sales_performance_data.csv', file_type='csv')

# Data processing and calculations
df['dated'] = pd.to_datetime(df['dated'], errors='coerce')
df['created'] = pd.to_timedelta(df['created'], errors='coerce')
numeric_columns = ['lead_taken', 'tours_booked', 'applications', 'revenue_confirmed', 'revenue_pending', 'revenue_runrate']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
df['tour_to_lead_conversion'] = df['tours_booked'] / df['lead_taken']
df['application_to_tour_conversion'] = df['applications'] / df['tours_booked']
df['revenue_per_tour'] = df['revenue_confirmed'] / df['tours_booked']

# Summarize data for the entire team
team_summary = df.agg({
    'lead_taken': 'sum',
    'tours_booked': 'sum',
    'applications': 'sum',
    'revenue_confirmed': 'sum',
    'revenue_pending': 'sum',
    'revenue_runrate': 'mean',  
})

# Generate a text summary for each representative
def generate_rep_summary(row):
    return f"""Employee: {row['employee_name']}
    Leads Taken: {row['lead_taken']}
    Tours Booked: {row['tours_booked']}
    Applications: {row['applications']}
    Revenue Confirmed: {row['revenue_confirmed']}
    Revenue Pending: {row['revenue_pending']}
    Conversion Rate (Tours to Leads): {row['tour_to_lead_conversion']:.2f}
    """

df['summary'] = df.apply(generate_rep_summary, axis=1)

# Prepare context for the transformer model
team_summary_text = f"""
The team took a total of {team_summary['lead_taken']} leads, booked {team_summary['tours_booked']} tours, and confirmed ${team_summary['revenue_confirmed']} in revenue. The average revenue run rate is ${team_summary['revenue_runrate']}.
"""
context = team_summary_text + "\n" + "\n".join(df['summary'].tolist())

# Load the QA model
qa_pipeline = pipeline("question-answering")

def answer_query(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

def query_insights(question):
    answer = answer_query(question, context)
    return answer

# Example usage
questions = [
    "What is the total revenue confirmed by the team?",
    "How many tours were booked?",
    "What is the conversion rate of tours to leads for each employee?"
]

for question in questions:
    answer = query_insights(question)
    print(f"Question: {question}\nAnswer: {answer}\n")
