from pydantic import BaseModel, Field, EmailStr

class ResponseText(BaseModel):
    response: str

class UserCreate(BaseModel):
    email: EmailStr = Field(...)
    password: str = Field(..., min_length=8)

# User model for database
class UserInDB(BaseModel):
    email: EmailStr
    password: str