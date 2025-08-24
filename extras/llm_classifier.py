#!/usr/bin/env python3
"""
Mock LLM Classifier for Denial Reason Classification

This module simulates an LLM service that classifies ambiguous denial reasons
as either retryable or non-retryable for EMR claim resubmission analysis.
"""

import json
import logging
import time
import random
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re


class ClassificationConfidence(Enum):
    """Confidence levels for LLM classification."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ClassificationResult(Enum):
    """Possible classification results."""
    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"
    AMBIGUOUS = "ambiguous"


@dataclass
class LLMClassification:
    """Result of LLM classification."""
    classification: ClassificationResult
    confidence: ClassificationConfidence
    reasoning: str
    suggested_actions: List[str]
    processing_time: float
    model_version: str


class MockLLMClassifier:
    """
    Mock LLM classifier that simulates intelligent denial reason classification.
    
    In production, this would be replaced with actual LLM API calls to services
    like OpenAI GPT, Anthropic Claude, or similar.
    """
    
    def __init__(self, model_version: str = "gpt-4-simulated-v1.0"):
        """Initialize the mock LLM classifier."""
        self.model_version = model_version
        self.logger = logging.getLogger(__name__)
        
        # Training data patterns for classification
        self.retryable_patterns = {
            "missing": ["missing", "absent", "not provided", "omitted", "empty"],
            "incorrect": ["incorrect", "wrong", "invalid", "bad", "erroneous"],
            "incomplete": ["incomplete", "partial", "unfinished", "half"],
            "required": ["required", "needed", "mandatory", "essential"],
            "expired": ["expired", "outdated", "stale", "old"],
            "format": ["format", "structure", "layout", "template"],
            "documentation": ["documentation", "docs", "records", "files"]
        }
        
        self.non_retryable_patterns = {
            "expired": ["expired", "expiration", "past due", "overdue"],
            "invalid": ["invalid", "not valid", "unacceptable", "rejected"],
            "not_covered": ["not covered", "excluded", "denied", "ineligible"],
            "policy": ["policy", "guideline", "rule", "regulation"],
            "contract": ["contract", "agreement", "terms", "conditions"]
        }
        
        # Context-specific rules
        self.context_rules = {
            "authorization": {
                "expired": ClassificationResult.NON_RETRYABLE,
                "missing": ClassificationResult.RETRYABLE,
                "incorrect": ClassificationResult.RETRYABLE
            },
            "provider": {
                "npi": ClassificationResult.RETRYABLE,
                "type": ClassificationResult.NON_RETRYABLE,
                "credentials": ClassificationResult.RETRYABLE
            },
            "procedure": {
                "code": ClassificationResult.RETRYABLE,
                "modifier": ClassificationResult.RETRYABLE,
                "not_covered": ClassificationResult.NON_RETRYABLE
            }
        }
        
        # Historical classification examples for learning
        self.classification_history: List[Dict[str, Any]] = []
    
    def classify_denial_reason(self, denial_reason: str, 
                             claim_context: Optional[Dict[str, Any]] = None) -> LLMClassification:
        """
        Classify a denial reason using simulated LLM intelligence.
        
        Args:
            denial_reason: The denial reason text to classify
            claim_context: Optional context about the claim (procedure type, provider info, etc.)
            
        Returns:
            LLMClassification object with results
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Classifying denial reason: '{denial_reason}'")
            
            # Preprocess the denial reason
            processed_reason = self._preprocess_text(denial_reason)
            
            # Apply context-aware classification
            classification, confidence, reasoning = self._context_aware_classification(
                processed_reason, claim_context
            )
            
            # Generate suggested actions
            suggested_actions = self._generate_suggested_actions(
                classification, processed_reason, claim_context
            )
            
            # Record classification for learning
            self._record_classification(denial_reason, classification, confidence, reasoning)
            
            processing_time = time.time() - start_time
            
            result = LLMClassification(
                classification=classification,
                confidence=confidence,
                reasoning=reasoning,
                suggested_actions=suggested_actions,
                processing_time=processing_time,
                model_version=self.model_version
            )
            
            self.logger.info(f"Classification result: {classification.value} (confidence: {confidence.value})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in LLM classification: {str(e)}")
            # Return default classification on error
            return LLMClassification(
                classification=ClassificationResult.AMBIGUOUS,
                confidence=ClassificationConfidence.LOW,
                reasoning=f"Classification failed due to error: {str(e)}",
                suggested_actions=["Manual review required"],
                processing_time=time.time() - start_time,
                model_version=self.model_version
            )
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for classification."""
        if not text:
            return ""
        
        # Normalize text
        normalized = text.lower().strip()
        
        # Remove common punctuation and extra whitespace
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized
    
    def _context_aware_classification(self, processed_reason: str, 
                                    claim_context: Optional[Dict[str, Any]]) -> Tuple[ClassificationResult, ClassificationConfidence, str]:
        """Perform context-aware classification."""
        
        # Check for exact matches in known patterns
        exact_match = self._check_exact_matches(processed_reason)
        if exact_match:
            return exact_match[0], ClassificationConfidence.HIGH, exact_match[1]
        
        # Apply context-specific rules
        if claim_context:
            context_result = self._apply_context_rules(processed_reason, claim_context)
            if context_result:
                return context_result[0], ClassificationConfidence.MEDIUM, context_result[1]
        
        # Pattern-based classification
        pattern_result = self._pattern_based_classification(processed_reason)
        if pattern_result:
            return pattern_result[0], ClassificationConfidence.MEDIUM, pattern_result[1]
        
        # Semantic similarity analysis (simulated)
        semantic_result = self._semantic_similarity_analysis(processed_reason)
        if semantic_result:
            return semantic_result[0], ClassificationConfidence.LOW, semantic_result[1]
        
        # Default to ambiguous if no clear classification
        return ClassificationResult.AMBIGUOUS, ClassificationConfidence.LOW, "Unable to determine classification"
    
    def _check_exact_matches(self, processed_reason: str) -> Optional[Tuple[ClassificationResult, str]]:
        """Check for exact matches in known denial reason patterns."""
        
        # Known retryable reasons
        known_retryable = {
            "missing modifier": "Known retryable reason - modifier codes can be added",
            "incorrect npi": "Known retryable reason - NPI numbers can be corrected",
            "prior auth required": "Known retryable reason - prior authorization can be obtained",
            "form incomplete": "Known retryable reason - missing fields can be completed",
            "missing documentation": "Known retryable reason - documentation can be provided"
        }
        
        # Known non-retryable reasons
        known_non_retryable = {
            "authorization expired": "Known non-retryable reason - expired authorizations cannot be renewed retroactively",
            "incorrect provider type": "Known non-retryable reason - provider type is a fundamental characteristic",
            "not covered by policy": "Known non-retryable reason - coverage is determined by policy terms",
            "contract expired": "Known non-retryable reason - expired contracts require renewal"
        }
        
        # Check exact matches
        if processed_reason in known_retryable:
            return ClassificationResult.RETRYABLE, known_retryable[processed_reason]
        
        if processed_reason in known_non_retryable:
            return ClassificationResult.NON_RETRYABLE, known_non_retryable[processed_reason]
        
        return None
    
    def _apply_context_rules(self, processed_reason: str, 
                           claim_context: Dict[str, Any]) -> Optional[Tuple[ClassificationResult, str]]:
        """Apply context-specific classification rules."""
        
        # Check provider context
        if "provider" in claim_context:
            provider_info = claim_context["provider"]
            
            if "npi" in processed_reason and "npi" in provider_info:
                return ClassificationResult.RETRYABLE, "NPI number can be corrected and resubmitted"
            
            if "type" in processed_reason and "type" in provider_info:
                return ClassificationResult.NON_RETRYABLE, "Provider type is fundamental and cannot be changed"
        
        # Check procedure context
        if "procedure" in claim_context:
            procedure_info = claim_context["procedure"]
            
            if "code" in processed_reason:
                return ClassificationResult.RETRYABLE, "Procedure codes can be corrected and resubmitted"
            
            if "modifier" in processed_reason:
                return ClassificationResult.RETRYABLE, "Modifier codes can be added and resubmitted"
        
        # Check authorization context
        if "authorization" in claim_context:
            auth_info = claim_context["authorization"]
            
            if "expired" in processed_reason:
                return ClassificationResult.NON_RETRYABLE, "Expired authorizations cannot be renewed retroactively"
            
            if "missing" in processed_reason:
                return ClassificationResult.RETRYABLE, "Missing authorizations can be obtained and resubmitted"
        
        return None
    
    def _pattern_based_classification(self, processed_reason: str) -> Optional[Tuple[ClassificationResult, str]]:
        """Classify based on pattern matching and keyword analysis."""
        
        # Calculate scores for different categories
        retryable_score = 0
        non_retryable_score = 0
        
        # Check retryable patterns
        for category, patterns in self.retryable_patterns.items():
            for pattern in patterns:
                if pattern in processed_reason:
                    retryable_score += 1
        
        # Check non-retryable patterns
        for category, patterns in self.non_retryable_patterns.items():
            for pattern in patterns:
                if pattern in processed_reason:
                    non_retryable_score += 1
        
        # Determine classification based on scores
        if retryable_score > non_retryable_score:
            return ClassificationResult.RETRYABLE, f"Pattern analysis suggests retryable (score: {retryable_score} vs {non_retryable_score})"
        elif non_retryable_score > retryable_score:
            return ClassificationResult.NON_RETRYABLE, f"Pattern analysis suggests non-retryable (score: {non_retryable_score} vs {retryable_score})"
        
        return None
    
    def _semantic_similarity_analysis(self, processed_reason: str) -> Optional[Tuple[ClassificationResult, str]]:
        """Simulate semantic similarity analysis using historical data."""
        
        if not self.classification_history:
            return None
        
        # Find similar historical classifications
        similar_classifications = []
        
        for historical in self.classification_history:
            historical_reason = historical.get("denial_reason", "").lower()
            
            # Simple similarity calculation (in production, use proper NLP)
            similarity = self._calculate_similarity(processed_reason, historical_reason)
            
            if similarity > 0.6:  # 60% similarity threshold
                similar_classifications.append({
                    "classification": historical["classification"],
                    "similarity": similarity,
                    "reasoning": historical["reasoning"]
                })
        
        if similar_classifications:
            # Use most similar classification
            best_match = max(similar_classifications, key=lambda x: x["similarity"])
            
            if best_match["classification"] == ClassificationResult.RETRYABLE.value:
                return ClassificationResult.RETRYABLE, f"Similar to historical retryable case (similarity: {best_match['similarity']:.2f})"
            else:
                return ClassificationResult.NON_RETRYABLE, f"Similar to historical non-retryable case (similarity: {best_match['similarity']:.2f})"
        
        return None
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (simplified version)."""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _generate_suggested_actions(self, classification: ClassificationResult, 
                                  processed_reason: str, 
                                  claim_context: Optional[Dict[str, Any]]) -> List[str]:
        """Generate suggested actions based on classification."""
        
        if classification == ClassificationResult.RETRYABLE:
            actions = [
                "Review and correct the identified issue",
                "Gather required documentation or information",
                "Resubmit claim with corrections",
                "Monitor resubmission for approval"
            ]
            
            # Add specific actions based on reason
            if "modifier" in processed_reason:
                actions.append("Verify correct modifier codes for procedure")
            elif "npi" in processed_reason:
                actions.append("Confirm provider NPI number with provider")
            elif "auth" in processed_reason:
                actions.append("Obtain prior authorization before resubmission")
                
        elif classification == ClassificationResult.NON_RETRYABLE:
            actions = [
                "Review claim for fundamental issues",
                "Consider appeal process if applicable",
                "Document reason for non-retryable status",
                "Escalate to manual review team"
            ]
        else:  # AMBIGUOUS
            actions = [
                "Perform manual review of denial reason",
                "Gather additional context information",
                "Consult with clinical team if needed",
                "Document decision and reasoning"
            ]
        
        return actions
    
    def _record_classification(self, original_reason: str, classification: ClassificationResult, 
                             confidence: ClassificationConfidence, reasoning: str) -> None:
        """Record classification for learning and improvement."""
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "denial_reason": original_reason,
            "classification": classification.value,
            "confidence": confidence.value,
            "reasoning": reasoning,
            "model_version": self.model_version
        }
        
        self.classification_history.append(record)
        
        # Keep only last 1000 records to prevent memory issues
        if len(self.classification_history) > 1000:
            self.classification_history = self.classification_history[-1000:]
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get statistics about classification performance."""
        if not self.classification_history:
            return {"total_classifications": 0}
        
        total = len(self.classification_history)
        classifications = [record["classification"] for record in self.classification_history]
        confidences = [record["confidence"] for record in self.classification_history]
        
        return {
            "total_classifications": total,
            "classification_breakdown": {
                "retryable": classifications.count("retryable"),
                "non_retryable": classifications.count("non_retryable"),
                "ambiguous": classifications.count("ambiguous")
            },
            "confidence_breakdown": {
                "high": confidences.count("high"),
                "medium": confidences.count("medium"),
                "low": confidences.count("low")
            },
            "success_rate": (total - classifications.count("ambiguous")) / total if total > 0 else 0
        }


def main():
    """Demonstrate the LLM classifier functionality."""
    print("🤖 Mock LLM Classifier for Denial Reason Classification")
    print("=" * 60)
    
    # Initialize classifier
    classifier = MockLLMClassifier()
    
    # Test cases
    test_cases = [
        "Missing modifier code",
        "Authorization expired",
        "Incorrect NPI number",
        "Form incomplete",
        "Provider type not covered",
        "Documentation missing",
        "Procedure code invalid",
        "Prior auth required"
    ]
    
    print("\n📋 Classification Results:")
    for test_case in test_cases:
        print(f"\n🔍 Analyzing: '{test_case}'")
        
        # Simulate claim context
        context = {
            "provider": {"type": "physician", "npi": "1234567890"},
            "procedure": {"code": "99213", "type": "office_visit"},
            "authorization": {"status": "active", "expires": "2025-12-31"}
        }
        
        result = classifier.classify_denial_reason(test_case, context)
        
        print(f"   Classification: {result.classification.value}")
        print(f"   Confidence: {result.confidence.value}")
        print(f"   Reasoning: {result.reasoning}")
        print(f"   Actions: {', '.join(result.suggested_actions[:2])}...")
    
    # Display statistics
    print(f"\n📊 Classification Statistics:")
    stats = classifier.get_classification_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                print(f"     {sub_key}: {sub_value}")
        else:
            print(f"   {key}: {value}")


if __name__ == "__main__":
    main()
