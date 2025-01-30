import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables from .env file (for local development)
load_dotenv()

# Initialize OpenAI client with API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define Pydantic models
class ResumeData(BaseModel):
    full_name: str
    professional_title: str
    email: str
    phone_number: str
    location: str
    website_url: Optional[str] = None
    linkedin_url: Optional[str] = None
    professional_summary: str

class ResumeRequest(BaseModel):
    resume_text: str

class ResumeResponse(BaseModel):
    resume_data: ResumeData

@app.post("/extract_resume", response_model=ResumeResponse)
async def extract_resume(request: ResumeRequest):
    try:
        # Call OpenAI API with Structured Outputs
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant that extracts resume information into structured JSON."},
                {
                    "role": "user",
                    "content": (
                        "Extract the following information from the resume text and provide it in JSON format:\n\n"
                        "Full Name, Professional Title, Email, Phone Number, Location, Website URL, LinkedIn URL, Professional Summary.\n\n"
                        f"Resume Text:\n{request.resume_text}"
                    )
                }
            ],
            response_format=ResumeData,
        )

        # Extract parsed data
        resume_data = completion.choices[0].message.parsed

        return ResumeResponse(resume_data=resume_data)

    except OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
    except KeyError:
        raise HTTPException(status_code=500, detail="Unexpected response format from OpenAI.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
