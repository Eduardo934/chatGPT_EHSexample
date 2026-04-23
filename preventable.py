import pandas as pd
from openai import OpenAI

# -----------------------------
# Configuration
# -----------------------------
API_KEY = "[CHATGPT-API-KEY]"  # Replace with your actual API key
MODEL_NAME = "gpt-4o"
FILE_PATH = "EHSample/DOT ACCIDENTS HISTORY REPORT.csv"
SAMPLE_SIZE = 5

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

# -----------------------------
# Prompt Templates
# -----------------------------
SYSTEM_PROMPT = (
    "You're a helpful Safety assistant who classifies "
    "preventable and non-preventable accidents."
)

USER_PROMPT_TEMPLATE = """
<instructions>
Consider the below criteria to classify preventable accidents:

1. Driver behavior
    • Speeding 
    • Aggressive driving 
    • Failure to obey traffic laws 
    • Distracted driving (phone, GPS, etc.) 

2. Driver condition
    • Fatigue or drowsiness 
    • Stress or emotional state 
    • Illness or impairment (including medication effects) 

3. Training and competency
    • Defensive driving training gaps 
    • Lack of familiarity with vehicle type 
    • Inexperience in certain road conditions (rain, night, etc.) 

4. Vehicle condition / maintenance
    • Brake or tire issues 
    • Lights not working 
    • Poor preventive maintenance practices 

5. Journey planning / route management
    • Poor route selection 
    • High-risk areas (traffic congestion, unsafe zones) 
    • Unrealistic schedules leading to rushing 

6. Supervision and company policies
    • Lack of enforcement of driving rules 
    • Weak or unclear fleet safety policies 
    • Pressure to meet deadlines over safety 

7. Fatigue management systems
    • Long driving hours 
    • Insufficient rest breaks 
    • No monitoring of driving time 

8. Communication issues
    • Unclear instructions (routes, priorities) 
    • Miscommunication with dispatch 
    • Last-minute changes while driving 

9. Environmental / road conditions
    • Weather (rain, fog, heat) 
    • Road quality (potholes, construction) 
    • Traffic density 

10. Incident response preparedness
    • Driver does not know what to do after a near miss or emergency 
    • No protocols for breakdowns or accidents
    • Lack of emergency contact systems  

Classify the following accident by using the criteria described above:
"{accident_description}"

Rules:
- Use the listed criteria
- Do NOT include markdown
- Respond with ONE word only:
  Preventable or Non-Preventable
</instructions>
"""

# -----------------------------
# Functions
# -----------------------------
def generate_accident_response(accident_description: str) -> str:
    """Call OpenAI API to classify accident."""
    
    user_prompt = USER_PROMPT_TEMPLATE.format(
        accident_description=accident_description
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content.strip()


def load_data(file_path: str, sample_size: int) -> pd.DataFrame:
    """Load and prepare dataset."""
    
    df = pd.read_csv(
        file_path,
        encoding="latin1",
        skiprows=4
    )

    df = df.sample(n=sample_size)
    df = df.loc[:, ["EE Name", "Accident Description"]]

    return df


def classify_accidents(df: pd.DataFrame) -> pd.DataFrame:
    """Run classification loop over dataset."""
    
    predictions = []

    for accident in df["Accident Description"]:
        print(f"Processing: {accident}")

        prediction = generate_accident_response(accident)
        predictions.append(prediction)

    df["Preventable_prediction"] = predictions

    return df


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    
    pbna_incidents = load_data(FILE_PATH, SAMPLE_SIZE)
    pbna_incidents = classify_accidents(pbna_incidents)

    print(pbna_incidents)

