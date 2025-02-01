import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import httpx

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

# Load X_API_KEY from environment variables
X_API_KEY = os.getenv("LINKEDIN_API_KEY")

# Define Pydantic models
class Experience(BaseModel):
    company: Optional[str] = None
    position: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None
    key_achievements: Optional[List[str]] = None

class GitHubProject(BaseModel):
    title: Optional[str] = None
    subtitle: Optional[str] = None
    date: Optional[str] = None
    description: Optional[str] = None

class Education(BaseModel):
    institution_name: Optional[str] = None
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    grade_gpa: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None

class ResumeData(BaseModel):
    full_name: str
    professional_title: str
    email: str
    phone_number: str
    location: str
    website_url: Optional[str] = None
    linkedin_url: Optional[str] = None
    professional_summary: str
    skills: Optional[List[str]] = None
    experience: Optional[List[Experience]] = None
    github_projects: Optional[List[GitHubProject]] = None
    education: Optional[List[Education]] = None

class ResumeRequest(BaseModel):
    resume_text: str
    job_description: Optional[str] = None

class ResumeResponse(BaseModel):
    resume_data: ResumeData

# Define Pydantic models for LinkedIn responses
class JobResponse(BaseModel):
    job_id: str
    status: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None

# Define Pydantic models for GitHub profiles
class GitHubProfile(BaseModel):
    login: Optional[str]
    name: Optional[str]
    bio: Optional[str]
    company: Optional[str]
    blog: str
    location: Optional[str]
    email: Optional[str]
    twitter_username: Optional[str]
    public_repos: int
    followers: int
    following: int
    created_at: str
    updated_at: str

class Repository(BaseModel):
    name: str
    description: Optional[str]
    html_url: str
    language: Optional[str]
    stargazers_count: int
    forks_count: int
    created_at: str
    updated_at: str

class GitHubResponse(BaseModel):
    profile: GitHubProfile
    repositories: List[Repository]

@app.post("/extract_resume", response_model=ResumeResponse)
async def extract_resume(request: ResumeRequest):
    try:
        # Create dynamic user message based on job_description
        if request.job_description:
            user_intro = f"For a resume of a Job Seeker of the JD - {request.job_description}, I need to extract the following information:\n\n"
        else:
            user_intro = "For a resume of a Job Seeker, I need to extract the following information:\n\n"
        
        user_content = (
            user_intro +
            "Extract the following information from the resume text and provide it in JSON format:\n\n"
            "Full Name, Professional Title, Email, Phone Number, Location, Website URL, LinkedIn URL, Professional Summary, skills ( Best Skills based on the projects any best 10, only if none is present with linkedIn), experience, GitHub projects (Best 5 based on Stars or Relatable to Job needs, and add Subtitle as one liner), education, try to fill  related data but dont give false data.\n\n"
            f"User Data:\n{request.resume_text}"
        )
        
        # Call OpenAI API with Structured Outputs
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an data expert assistant that extracts or digest and give resume related information into structured JSON from the data provided"},
                {"role": "user", "content": user_content}
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

@app.get("/api/v1/linkedin/profile/{username}", response_model=JobResponse)
async def crawl_linkedin_profile(username: str):
    url = f"https://profile-fetch.hyrenet-staging.in/api/v1/linkedin/profile/{username}"
    headers = {"X-API-Key": X_API_KEY}
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers)
        response.raise_for_status()
        return response.json()

@app.get("/api/v1/linkedin/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    url = f"https://profile-fetch.hyrenet-staging.in/api/v1/linkedin/job/{job_id}"
    headers = {"X-API-Key": X_API_KEY}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

@app.get("/api/v1/github/profile/{username}", response_model=GitHubResponse)
async def fetch_github_profile(username: str):
    url = f"https://profile-fetch.hyrenet-staging.in/api/v1/github/profile/{username}"
    headers = {
        "accept": "application/json",
        "X-API-Key": X_API_KEY
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

@app.post("/api/v1/resume/analyze")
async def analyze_resume(file: UploadFile = File(...)):
    headers = {"X-API-Key": X_API_KEY}
    file_bytes = await file.read()
    files = {"file": (file.filename, file_bytes, file.content_type)}
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://profile-fetch.hyrenet-staging.in/api/v1/resume/analyze",
            headers=headers,
            files=files
        )
        response.raise_for_status()
        return response.json()

