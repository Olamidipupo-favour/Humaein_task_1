#!/usr/bin/env python3
"""
EMR Data Processing and Resubmission Eligibility Analysis

This script performs the complete EMR data processing pipeline:
1. Schema normalization from multiple EMR sources
2. Resubmission eligibility analysis based on business rules
3. Output generation with comprehensive logging and metrics
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from data_cleaner import EMRDataCleaner, NormalizedClaim


@dataclass
class ResubmissionCandidate:
    """Represents a claim eligible for resubmission."""
    claim_id: str
    resubmission_reason: str
    source_system: str
    recommended_changes: str


class ResubmissionAnalyzer:
    """Analyzes normalized claims for resubmission eligibility."""
    
    def __init__(self, reference_date: str = "2025-07-30") -> None:
        """
        Initialize the resubmission analyzer.
        
        Args:
            reference_date: Reference date for calculating claim age (YYYY-MM-DD)
        """
        self.reference_date = datetime.strptime(reference_date, "%Y-%m-%d")
        
        # Define known retryable and non-retryable denial reasons
        self.retryable_reasons = {
            "missing modifier",
            "incorrect npi", 
            "prior auth required"
        }
        
        self.non_retryable_reasons = {
            "authorization expired",
            "incorrect provider type"
        }
        
        # Hardcoded mapping for ambiguous reasons
        self.ambiguous_reason_mapping = {
            "incorrect procedure": "retryable",
            "form incomplete": "retryable", 
            "not billable": "non_retryable",
            "": "non_retryable"  # null/empty
        }
        
        self.logger = logging.getLogger(__name__)
    
    def _is_claim_old_enough(self, submitted_date: str) -> bool:
        """
        Check if claim is older than 7 days from reference date.
        
        Args:
            submitted_date: ISO formatted date string
            
        Returns:
            True if claim is older than 7 days
        """
        try:
            claim_date = datetime.fromisoformat(submitted_date)
            days_old = (self.reference_date - claim_date).days
            return days_old > 7
        except Exception as e:
            self.logger.warning(f"Error parsing date {submitted_date}: {str(e)}")
            return False
    
    def _classify_denial_reason(self, denial_reason: Optional[str]) -> Tuple[str, str]:
        """
        Classify denial reason as retryable, non-retryable, or ambiguous.
        
        Args:
            denial_reason: Denial reason string or None
            
        Returns:
            Tuple of (classification, explanation)
        """
        if denial_reason is None:
            return "non_retryable", "No denial reason provided"
        
        normalized_reason = denial_reason.lower().strip()
        
        # Check known retryable reasons
        if normalized_reason in self.retryable_reasons:
            return "retryable", f"Known retryable reason: {denial_reason}"
        
        # Check known non-retryable reasons
        if normalized_reason in self.non_retryable_reasons:
            return "non_retryable", f"Known non-retryable reason: {denial_reason}"
        
        # Check hardcoded ambiguous reason mapping
        if normalized_reason in self.ambiguous_reason_mapping:
            classification = self.ambiguous_reason_mapping[normalized_reason]
            return classification, f"Hardcoded classification: {denial_reason}"
        
        # For truly ambiguous reasons, use LLM-like classification
        return self._llm_classify_reason(normalized_reason)
    
    def _llm_classify_reason(self, reason: str) -> Tuple[str, str]:
        """
        Mock LLM classification for ambiguous denial reasons.
        In production, this would call an actual LLM service.
        
        Args:
            reason: Denial reason to classify
            
        Returns:
            Tuple of (classification, explanation)
        """
        # Simple heuristic-based classification
        retryable_keywords = {
            "missing", "incorrect", "incomplete", "required", "needed"
        }
        
        non_retryable_keywords = {
            "expired", "invalid", "not", "unable", "cannot"
        }
        
        reason_lower = reason.lower()
        
        # Count keyword matches
        retryable_score = sum(1 for keyword in retryable_keywords if keyword in reason_lower)
        non_retryable_score = sum(1 for keyword in non_retryable_keywords if keyword in reason_lower)
        
        if retryable_score > non_retryable_score:
            return "retryable", f"LLM classification (retryable): {reason}"
        elif non_retryable_score > retryable_score:
            return "non_retryable", f"LLM classification (non-retryable): {reason}"
        else:
            # Default to retryable for ambiguous cases
            return "retryable", f"LLM classification (default retryable): {reason}"
    
    def _generate_recommendations(self, denial_reason: Optional[str], classification: str) -> str:
        """
        Generate recommended changes based on denial reason and classification.
        
        Args:
            denial_reason: Original denial reason
            classification: Classification result
            
        Returns:
            Recommendation string
        """
        if classification == "non_retryable":
            return "Claim not eligible for resubmission - requires manual review"
        
        if not denial_reason:
            return "Review claim details and resubmit with complete information"
        
        # Generate specific recommendations based on denial reason
        reason_lower = denial_reason.lower()
        
        if "missing modifier" in reason_lower:
            return "Add appropriate modifier code and resubmit"
        elif "incorrect npi" in reason_lower:
            return "Review NPI number and resubmit with correct provider information"
        elif "prior auth required" in reason_lower:
            return "Obtain prior authorization and resubmit with auth number"
        elif "incorrect procedure" in reason_lower:
            return "Review procedure code and resubmit with correct code"
        elif "form incomplete" in reason_lower:
            return "Complete all required fields and resubmit"
        else:
            return "Review denial reason and resubmit with corrected information"
    
    def analyze_claim(self, claim: NormalizedClaim) -> Optional[ResubmissionCandidate]:
        """
        Analyze a single claim for resubmission eligibility.
        
        Args:
            claim: Normalized claim record
            
        Returns:
            ResubmissionCandidate if eligible, None otherwise
        """
        try:
            # Check basic eligibility criteria
            if claim.status != "denied":
                return None
            
            if not claim.patient_id or claim.patient_id.strip() == "":
                return None
            
            if not self._is_claim_old_enough(claim.submitted_at):
                return None
            
            # Classify denial reason
            classification, explanation = self._classify_denial_reason(claim.denial_reason)
            
            if classification != "retryable":
                return None
            
            # Generate recommendations
            recommended_changes = self._generate_recommendations(
                claim.denial_reason, classification
            )
            
            return ResubmissionCandidate(
                claim_id=claim.claim_id,
                resubmission_reason=claim.denial_reason or "No specific reason provided",
                source_system=claim.source_system,
                recommended_changes=recommended_changes
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing claim {claim.claim_id}: {str(e)}")
            return None
    
    def analyze_all_claims(self, claims: List[NormalizedClaim]) -> Tuple[List[ResubmissionCandidate], Dict[str, Any]]:
        """
        Analyze all claims for resubmission eligibility.
        
        Args:
            claims: List of normalized claim records
            
        Returns:
            Tuple of (eligible_claims, analysis_metrics)
        """
        eligible_claims = []
        analysis_metrics = {
            "total_claims": len(claims),
            "denied_claims": 0,
            "claims_without_patient_id": 0,
            "claims_too_recent": 0,
            "non_retryable_reasons": 0,
            "eligible_for_resubmission": 0,
            "classification_breakdown": {
                "retryable": 0,
                "non_retryable": 0,
                "ambiguous": 0
            }
        }
        
        for claim in claims:
            try:
                # Count denied claims
                if claim.status == "denied":
                    analysis_metrics["denied_claims"] += 1
                    
                    # Check patient ID
                    if not claim.patient_id or claim.patient_id.strip() == "":
                        analysis_metrics["claims_without_patient_id"] += 1
                        continue
                    
                    # Check claim age
                    if not self._is_claim_old_enough(claim.submitted_at):
                        analysis_metrics["claims_too_recent"] += 1
                        continue
                    
                    # Classify denial reason
                    classification, _ = self._classify_denial_reason(claim.denial_reason)
                    analysis_metrics["classification_breakdown"][classification] += 1
                    
                    if classification == "non_retryable":
                        analysis_metrics["non_retryable_reasons"] += 1
                        continue
                    
                    # If we get here, claim is eligible
                    analysis_metrics["eligible_for_resubmission"] += 1
                    candidate = self.analyze_claim(claim)
                    if candidate:
                        eligible_claims.append(candidate)
                
            except Exception as e:
                self.logger.error(f"Error processing claim {claim.claim_id}: {str(e)}")
                continue
        
        return eligible_claims, analysis_metrics


class EMRDataProcessor:
    """Main processor that orchestrates the complete EMR data pipeline."""
    
    def __init__(self, log_level: str = "INFO") -> None:
        """Initialize the data processor."""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('emr_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.data_cleaner = EMRDataCleaner(log_level=log_level)
        self.resubmission_analyzer = ResubmissionAnalyzer()
    
    def process_pipeline(self, data_dir: Path, output_dir: Path) -> None:
        """
        Execute the complete EMR data processing pipeline.
        
        Args:
            data_dir: Directory containing EMR data files
            output_dir: Directory for output files
        """
        try:
            self.logger.info("Starting EMR data processing pipeline")
            
            # Step 1: Schema Normalization
            self.logger.info("Step 1: Processing and normalizing EMR data sources")
            normalized_claims = self.data_cleaner.process_all_sources(data_dir)
            
            if not normalized_claims:
                self.logger.error("No claims were processed successfully")
                return
            
            # Export normalized data
            normalized_output = output_dir / "normalized_claims.json"
            self.data_cleaner.export_normalized_data(normalized_claims, normalized_output)
            
            # Step 2: Resubmission Eligibility Analysis
            self.logger.info("Step 2: Analyzing claims for resubmission eligibility")
            eligible_claims, analysis_metrics = self.resubmission_analyzer.analyze_all_claims(normalized_claims)
            
            # Step 3: Output Generation
            self.logger.info("Step 3: Generating output files and reports")
            
            # Export resubmission candidates
            resubmission_output = output_dir / "resubmission_candidates.json"
            self._export_resubmission_candidates(eligible_claims, resubmission_output)
            
            # Generate comprehensive report
            report_output = output_dir / "processing_report.json"
            self._generate_processing_report(normalized_claims, eligible_claims, analysis_metrics, report_output)
            
            # Display summary
            self._display_summary(normalized_claims, eligible_claims, analysis_metrics)
            
            self.logger.info("EMR data processing pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            raise
    
    def _export_resubmission_candidates(self, candidates: List[ResubmissionCandidate], output_path: Path) -> None:
        """Export resubmission candidates to JSON file."""
        try:
            export_data = [asdict(candidate) for candidate in candidates]
            
            with open(output_path, 'w', encoding='utf-8') as output_file:
                json.dump(export_data, output_file, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Exported {len(candidates)} resubmission candidates to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting resubmission candidates: {str(e)}")
            raise
    
    def _generate_processing_report(self, all_claims: List[NormalizedClaim], 
                                  eligible_claims: List[ResubmissionCandidate],
                                  analysis_metrics: Dict[str, Any], output_path: Path) -> None:
        """Generate comprehensive processing report."""
        try:
            # Get data summary from cleaner
            data_summary = self.data_cleaner.get_data_summary(all_claims)
            
            report = {
                "processing_timestamp": datetime.now().isoformat(),
                "data_summary": data_summary,
                "resubmission_analysis": analysis_metrics,
                "resubmission_candidates_count": len(eligible_claims),
                "processing_summary": {
                    "total_claims_processed": len(all_claims),
                    "claims_eligible_for_resubmission": len(eligible_claims),
                    "success_rate": f"{(len(eligible_claims) / len(all_claims) * 100):.2f}%" if all_claims else "0%"
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as output_file:
                json.dump(report, output_file, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Generated processing report: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating processing report: {str(e)}")
            raise
    
    def _display_summary(self, all_claims: List[NormalizedClaim], 
                        eligible_claims: List[ResubmissionCandidate],
                        analysis_metrics: Dict[str, Any]) -> None:
        """Display comprehensive summary to console."""
        print("\n" + "="*80)
        print("EMR DATA PROCESSING PIPELINE - COMPLETION SUMMARY")
        print("="*80)
        
        # Data processing summary
        print(f"\n📊 DATA PROCESSING SUMMARY:")
        print(f"   Total claims processed: {len(all_claims)}")
        
        data_summary = self.data_cleaner.get_data_summary(all_claims)
        for source, count in data_summary.get("source_system_counts", {}).items():
            print(f"   Claims from {source.upper()}: {count}")
        
        for status, count in data_summary.get("status_counts", {}).items():
            print(f"   Claims with status '{status}': {count}")
        
        # Resubmission analysis summary
        print(f"\n RESUBMISSION ELIGIBILITY ANALYSIS:")
        print(f"   Denied claims analyzed: {analysis_metrics['denied_claims']}")
        print(f"   Claims excluded (no patient ID): {analysis_metrics['claims_without_patient_id']}")
        print(f"   Claims excluded (too recent): {analysis_metrics['claims_too_recent']}")
        print(f"   Claims excluded (non-retryable): {analysis_metrics['non_retryable_reasons']}")
        print(f"   Claims eligible for resubmission: {analysis_metrics['eligible_for_resubmission']}")
        
        # Classification breakdown
        print(f"\n📋 DENIAL REASON CLASSIFICATION:")
        for classification, count in analysis_metrics['classification_breakdown'].items():
            print(f"   {classification.title()}: {count}")
        
        # Success metrics
        if all_claims:
            success_rate = (len(eligible_claims) / len(all_claims)) * 100
            print(f"\n SUCCESS METRICS:")
            print(f"   Overall success rate: {success_rate:.2f}%")
            print(f"   Resubmission candidates: {len(eligible_claims)}")
        
        print("\n" + "="*80)
        
        # Sample of eligible claims
        if eligible_claims:
            print(f"\n SAMPLE RESUBMISSION CANDIDATES:")
            for i, candidate in enumerate(eligible_claims[:3], 1):
                print(f"\n   Candidate {i}:")
                print(f"     Claim ID: {candidate.claim_id}")
                print(f"     Source: {candidate.source_system}")
                print(f"     Reason: {candidate.resubmission_reason}")
                print(f"     Recommendation: {candidate.recommended_changes}")
        
        print("\n" + "="*80)


def main() -> None:
    """Main execution function."""
    try:
        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Initialize processor
        processor = EMRDataProcessor(log_level="INFO")
        
        # Execute pipeline
        data_dir = Path("sample_data")
        processor.process_pipeline(data_dir, output_dir)
        
        print(f"\n Processing complete! Check the 'output' directory for results.")
        print(f"   - normalized_claims.json: All normalized claims")
        print(f"   - resubmission_candidates.json: Eligible claims for resubmission")
        print(f"   - processing_report.json: Comprehensive analysis report")
        print(f"   - emr_processing.log: Detailed processing log")
        
    except Exception as e:
        print(f" Pipeline execution failed: {str(e)}")
        logging.error(f"Main execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
