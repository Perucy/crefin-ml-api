"""
    Configuration module for the Crefin ML API service
"""
from pydantic_settings import BaseSettings
from typing import Optional, List

class Settings(BaseSettings):
    """ Loading app settings from .env file """

    # API settings
    API_TITLE: str
    API_VERSION: str
    API_DESCRIPTION: str

    # Server settings
    HOST: str
    PORT: int

    # CORS settings (parsed as comma separated string)
    CORS_ORIGINS: str

    # Model Paths
    PAYMENT_PREDICTOR_PATH: str
    PAYMENT_PREDICTOR_METADATA_PATH: str
    RISK_SCORER_PATH: str
    CLIENT_SEGMENTER_PATH: str
    REVENUE_FORECASTER_PATH: str

    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Environment
    ENVIRONMENT: str = "development"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS_ORIGINS string into list"""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]


settings = Settings()