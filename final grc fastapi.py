import cohere
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise RuntimeError("COHERE_API_KEY not set in environment")

# Initialize Cohere client
co = cohere.Client(COHERE_API_KEY)

# FastAPI app initialization
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Enhanced Control Validator API with Cohere is working."}

@app.post("/validate-control")
async def validate_control(
    process: str = Query(...),
    subprocess: str = Query(...),
    risk: str = Query(...),
    frequency: str = Query(...),
    risk_description: str = Query(...),
    control: str = Query(...),
    control_description: str = Query(...)
):
    try:
        prompt = f"""
You are an expert internal auditor.

Your task is to critically validate the effectiveness of the control in mitigating the risk by performing a comprehensive analysis of all inputs.

### Input Data:
- Process: {process}
- Subprocess: {subprocess}
- Risk: {risk}
- Risk Frequency: {frequency}
- Risk Description: {risk_description}
- Control Name: {control}
- Control Description: {control_description}

### Instructions:
1. Verify if the subprocess logically belongs under the given process.
2. Assess whether the specified risk is relevant and genuinely associated with the process/subprocess.
3. Consider the risk frequency — assess how often the risk may occur and whether that makes the risk high-priority.
4. Evaluate if the risk description accurately reflects the risk name. Flag it if it's too vague or mismatched.
5. Review both the control name and control description:
   - Does the control clearly mitigate the identified risk?
   - Is the control relevant to the frequency and severity of the risk?
   - Is the control concrete, specific, and implementable?

### Output Requirements:
- Begin your response with one of these three categories:
    - VALID – if the control is clearly aligned with the risk and all metadata is consistent (80–100% alignment).
    - PARTIALLY VALID – if the control is somewhat aligned but lacks depth, clarity, or partial mismatch in inputs (40–79% alignment).
    - INVALID – if the control is vague, irrelevant, or the data inputs are poorly aligned (<40% alignment).
- Follow this with a brief, precise justification.
- If the control is INVALID or PARTIALLY VALID, suggest an improved control description that better mitigates the risk.
"""


        response = co.generate(
            model='command-xlarge',  # or command-r-plus if you're using the latest
            prompt=prompt,
            max_tokens=300,
            temperature=0.5
        )

        result = response.generations[0].text.strip()
        return {"result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
