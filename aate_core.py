#!/usr/bin/env python3
"""
Autonomous Adaptive Trading Ecosystem (AATE) - Core Module
Fixed version with comprehensive error handling and resilience patterns
"""

import os
import sys
import json
import time
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Third-party imports with proper error handling
try:
    import pandas as pd
    import numpy as np
    HAS_DATA_LIBS = True
except ImportError as e:
    logging.warning(f"Data libraries not available: {e}")
    HAS_DATA_LIBS = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logging.error("Requests library not installed")

# Firebase for state management
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, initialize_app
    HAS_FIREBASE = True
except ImportError:
    HAS_FIREBASE = False
    logging.warning("Firebase-admin not installed, using local state only")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('aate_system.log')
    ]
)
logger = logging.getLogger(__name__)


class TradingSignal(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    EXIT = "EXIT"


@dataclass
class MarketData:
    """Market data structure with validation"""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    indicators: Dict[str, float]
    
    def validate(self) -> bool:
        """Validate market data"""
        if not self.symbol or self.price <= 0 or self.volume < 0:
            logger.error(f"Invalid market data: {self}")
            return False
        return True


@dataclass
class TradingDecision:
    """Trading decision with risk assessment"""
    signal: TradingSignal
    confidence: float  # 0-1 scale
    timestamp: datetime
    market_data: MarketData
    reasoning: str
    risk_score: float  # 0-1 scale, higher = riskier
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['signal'] = self.signal.value
        data['timestamp'] = self.timestamp.isoformat()
        data['market_data'] = asdict(self.market_data)
        data['market_data']['timestamp'] = self.market_data.timestamp.isoformat()
        return data


class OllamaModelClient:
    """Robust Ollama model client with comprehensive error handling"""
    
    def __init__(self, model_name: str = "llama2", base_url: str = "http://localhost:11434", 
                 timeout: int = 30, max_retries: int = 3):
        """
        Initialize Ollama client with proper error handling
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Ollama server URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.model_name =