from typing import Optional
from datetime import datetime, timedelta
import settings
from jose import jwt
import os

# Helper function to create access token
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, os.getenv("TOKEN_SECRET_KEY"), algorithm=settings.ALGORITHM)
    return encoded_jwt