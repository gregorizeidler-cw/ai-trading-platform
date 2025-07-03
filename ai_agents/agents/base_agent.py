"""
Base agent class for all trading agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger
from ai_agents.llm.openai_client import OpenAIClient


class BaseAgent(ABC):
    """Base class for all trading agents"""
    
    def __init__(
        self,
        name: str,
        description: str,
        llm_client: Optional[OpenAIClient] = None
    ):
        self.name = name
        self.description = description
        self.llm_client = llm_client or OpenAIClient()
        self.active = True
        self.last_analysis = None
        self.performance_metrics = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "average_response_time": 0.0,
            "accuracy_score": 0.0
        }
        
    @abstractmethod
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data and provide insights"""
        pass
    
    @abstractmethod
    async def get_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get recommendations based on analysis"""
        pass
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request and return response"""
        start_time = datetime.utcnow()
        
        try:
            # Perform analysis
            analysis = await self.analyze(request.get("data", {}))
            
            # Get recommendations
            recommendations = await self.get_recommendations(analysis)
            
            # Update performance metrics
            self._update_performance_metrics(start_time, success=True)
            
            response = {
                "agent_name": self.name,
                "timestamp": datetime.utcnow().isoformat(),
                "analysis": analysis,
                "recommendations": recommendations,
                "confidence_score": analysis.get("confidence_score", 0.0),
                "status": "success"
            }
            
            self.last_analysis = response
            logger.info(f"{self.name} completed analysis successfully")
            
            return response
            
        except Exception as e:
            self._update_performance_metrics(start_time, success=False)
            logger.error(f"Error in {self.name} analysis: {e}")
            
            return {
                "agent_name": self.name,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "status": "error"
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status and performance metrics"""
        return {
            "name": self.name,
            "description": self.description,
            "active": self.active,
            "performance_metrics": self.performance_metrics,
            "last_analysis_time": (
                self.last_analysis.get("timestamp") if self.last_analysis else None
            )
        }
    
    def _update_performance_metrics(self, start_time: datetime, success: bool):
        """Update performance metrics"""
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        self.performance_metrics["total_analyses"] += 1
        
        if success:
            self.performance_metrics["successful_analyses"] += 1
        
        # Update average response time
        total = self.performance_metrics["total_analyses"]
        current_avg = self.performance_metrics["average_response_time"]
        self.performance_metrics["average_response_time"] = (
            (current_avg * (total - 1) + response_time) / total
        )
        
        # Update accuracy score
        if total > 0:
            self.performance_metrics["accuracy_score"] = (
                self.performance_metrics["successful_analyses"] / total
            )
    
    def set_active(self, active: bool):
        """Set agent active status"""
        self.active = active
        logger.info(f"{self.name} {'activated' if active else 'deactivated'}")
        
    def reset_metrics(self):
        """Reset performance metrics"""
        self.performance_metrics = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "average_response_time": 0.0,
            "accuracy_score": 0.0
        }
        logger.info(f"{self.name} metrics reset") 