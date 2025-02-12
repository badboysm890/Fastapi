import os
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, validator
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import httpx
import re
import base64

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


# -------------------------------------------------------------------
# MODELS FOR RESUME EXTRACTION (Remain the same, if not parsed with "response_format")
# -------------------------------------------------------------------
class Experience(BaseModel):
    company: Optional[str] = None
    position: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None
    key_achievements: Optional[List[str]] = None

    class Config:
        extra = "allow"


class GitHubProject(BaseModel):
    title: Optional[str] = None
    subtitle: Optional[str] = None
    date: Optional[str] = None
    description: Optional[str] = None

    class Config:
        extra = "allow"


class Education(BaseModel):
    institution_name: Optional[str] = None
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    grade_gpa: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None

    class Config:
        extra = "allow"


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

    class Config:
        extra = "allow"


class ResumeRequest(BaseModel):
    resume_text: str
    job_description: Optional[str] = None

    class Config:
        extra = "allow"


class ResumeResponse(BaseModel):
    resume_data: ResumeData

    class Config:
        extra = "allow"


# -------------------------------------------------------------------
# MODELS FOR LINKEDIN RESPONSES (Not used in structured parse)
# -------------------------------------------------------------------
class JobResponse(BaseModel):
    job_id: str
    status: str

    class Config:
        extra = "allow"


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None

    class Config:
        extra = "allow"


# -------------------------------------------------------------------
# MODELS FOR GITHUB PROFILES (Not used in structured parse)
# -------------------------------------------------------------------
class GitHubProfile(BaseModel):
    login: Optional[str] = None
    name: Optional[str] = None
    bio: Optional[str] = None
    company: Optional[str] = None
    blog: str
    location: Optional[str] = None
    email: Optional[str] = None
    twitter_username: Optional[str] = None
    public_repos: int
    followers: int
    following: int
    created_at: str
    updated_at: str

    class Config:
        extra = "allow"


class Repository(BaseModel):
    name: str
    description: Optional[str] = None
    html_url: str
    language: Optional[str] = None
    stargazers_count: int
    forks_count: int
    created_at: str
    updated_at: str

    class Config:
        extra = "allow"


class GitHubResponse(BaseModel):
    profile: GitHubProfile
    repositories: List[Repository]

    class Config:
        extra = "allow"


# -------------------------------------------------------------------
# MODELS FOR /analyze_image (Used in "response_format")
# => Must have extra = "forbid" to comply with OpenAI's structured parse
# -------------------------------------------------------------------
class EducationSection(BaseModel):
    content: Optional[List[str]] = None
    match: Optional[bool] = None

    class Config:
        extra = "forbid"


class ExperienceSection(BaseModel):
    total_years: Optional[int] = None
    relevant_experience_years: Optional[int] = None
    job_hopping_risk: Optional[str] = None
    content: Optional[List[str]] = None
    match: Optional[bool] = None

    class Config:
        extra = "forbid"


class SkillsSection(BaseModel):
    requirement: Optional[List[str]] = None
    content: Optional[List[str]] = None
    missing: Optional[List[str]] = None
    matched_skills_percentage: Optional[str] = None
    transferable_skills: Optional[List[str]] = None
    Matches: Optional[List[str]] = None

    class Config:
        extra = "forbid"


class CertificationsSection(BaseModel):
    certifications: Optional[Dict[str, str]] = None

    class Config:
        extra = "forbid"


class WorkPreferencesSection(BaseModel):
    notice_period: Optional[str] = None
    relocation: Optional[str] = None
    remote_work_preferred: Optional[str] = None

    class Config:
        extra = "forbid"


class CompensationSection(BaseModel):
    current_salary: Optional[str] = None
    expected_salary: Optional[str] = None

    class Config:
        extra = "forbid"


class SoftSkillsSection(BaseModel):
    communication: Optional[str] = None
    teamwork: Optional[str] = None
    problem_solving: Optional[str] = None
    leadership: Optional[str] = None

    class Config:
        extra = "forbid"


class ProjectsSection(BaseModel):
    content: Optional[List[str]] = None
    match: Optional[bool] = None

    class Config:
        extra = "forbid"


class SummarySection(BaseModel):
    content: Optional[List[str]] = None

    class Config:
        extra = "forbid"


class AIInsightsSection(BaseModel):
    recommendation: Optional[str] = None
    upskilling_suggestions: Optional[List[str]] = None

    class Config:
        extra = "forbid"


class ImageAnalysisResponse(BaseModel):
    name: Optional[str] = None
    overall_match_score: Optional[str] = None
    education_match_score: Optional[str] = None
    experience_match_score: Optional[str] = None
    skill_match_score: Optional[str] = None
    education: Optional[EducationSection] = None
    experience: Optional[ExperienceSection] = None
    skills: Optional[SkillsSection] = None
    certifications: Optional[CertificationsSection] = None
    work_preferences: Optional[WorkPreferencesSection] = None
    compensation: Optional[CompensationSection] = None
    soft_skills: Optional[SoftSkillsSection] = None
    projects: Optional[ProjectsSection] = None
    summary: Optional[SummarySection] = None
    ai_insights: Optional[AIInsightsSection] = None

    class Config:
        extra = "forbid"


class ImageAnalysisRequest(BaseModel):
    base64_image: str
    job_description: Optional[str] = None

    @validator('base64_image')
    def validate_base64(cls, v):
        if 'base64,' in v:
            v = v.split('base64,')[1]
        v = re.sub(r'\s+', '', v)
        try:
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError("Invalid base64 string")

    class Config:
        extra = "allow"


# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------

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


def prepare_base64_image(file_path: str) -> str:
    try:
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        raise ValueError(f"Error reading image file: {str(e)}")


@app.post("/analyze_image", response_model=ImageAnalysisResponse)
async def analyze_image(request: ImageAnalysisRequest):
    try:
        # Validate and clean the base64 string
        base64_image = request.base64_image
        
        # Ensure proper padding
        padding = len(base64_image) % 4
        if padding:
            base64_image += "=" * (4 - padding)

        # System prompt with detailed instructions
        system_prompt = """
        You are an expert resume analyzer. Analyze the resume image and provide a detailed structured response including:
        - Name and overall match scores
        - Education details with match score
        - Experience details including total years, relevant years, and job hopping risk
        - Skills analysis with required, present, missing, and transferable skills
        - Certifications status
        - Work preferences including notice period and location preferences
        - Compensation details if available
        - Soft skills assessment
        - Projects evaluation
        - Professional summary
        - AI-driven insights and recommendations
        
        Compare against the provided job description if available and provide match percentages.
        
        JSON format response. 
        """

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Analyze this resume image against the following job description in JSON structured response: {request.job_description if request.job_description else 'No job description provided and provide the JSON structured response.'}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        # Call OpenAI API with structured output approach
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",  # Example versioned model
            messages=messages,
            response_format=ImageAnalysisResponse,
            max_tokens=4096
        )

        # Get the parsed object directly from the response
        response_data = completion.choices[0].message.parsed
        return response_data

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
