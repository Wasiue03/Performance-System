from transformers import pipeline
import pandas as pd

# Load the model using Hugging Face Transformers
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')

# Sample sales data
data = {
    "employee_id": [183],
    "employee_name": ["Camilla Ali"],
    "revenue_confirmed": [1365],
    "revenue_pending": [700],
    "applications": [2],
    "tours_booked": [2]
}

sales_data = pd.DataFrame(data)

# Format data for prompt
formatted_data = sales_data.to_string(index=False)
prompt = f"Analyze the following sales data and provide feedback:\n{formatted_data}\nProvide qualitative feedback and actionable insights."

# Generate insights from the model
insights = generator(prompt, max_length=100)
print(insights[0]['generated_text'])
