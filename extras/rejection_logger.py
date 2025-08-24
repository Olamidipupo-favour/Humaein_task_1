#!/usr/bin/env python3
"""
Rejection Logger for Failed EMR Records

This module handles logging and exporting of failed/ rejected records
during EMR data processing, providing detailed error information
and tracking for quality assurance.
"""

import json
import logging
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum


class RejectionReason(Enum):
    """Reasons for record rejection."""
    MISSING_REQUIRED_FIELD = "missing_required_field"
    INVALID_DATE_FORMAT = "invalid_date_format"
    DATA_VALIDATION_FAILED = "data_validation_failed"
    SCHEMA_MISMATCH = "schema_mismatch"
    PROCESSING_ERROR = "processing_error"
    UNKNOWN_ERROR = "unknown_error"


class RejectionSeverity(Enum):
    """Severity levels for rejections."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RejectedRecord:
    """Represents a rejected/failed record."""
    record_id: str
    source_system: str
    original_data: Dict[str, Any]
    rejection_reason: RejectionReason
    rejection_details: str
    severity: RejectionSeverity
    timestamp: str
    processing_stage: str
    suggested_fix: Optional[str] = None
    retry_count: int = 0


class RejectionLogger:
    """Handles logging and exporting of rejected records."""
    
    def __init__(self, log_dir: Path = Path("logs")):
        """Initialize the rejection logger."""
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Rejected records storage
        self.rejected_records: List[RejectedRecord] = []
        
        # Rejection statistics
        self.rejection_stats = {
            "total_rejections": 0,
            "rejections_by_reason": {},
            "rejections_by_severity": {},
            "rejections_by_source": {},
            "rejections_by_stage": {}
        }
    
    def log_rejection(self, record_id: str, source_system: str, 
                     original_data: Dict[str, Any], rejection_reason: RejectionReason,
                     rejection_details: str, severity: RejectionSeverity = RejectionSeverity.MEDIUM,
                     processing_stage: str = "unknown", suggested_fix: Optional[str] = None) -> None:
        """
        Log a rejected record.
        
        Args:
            record_id: Unique identifier for the record
            source_system: Source system identifier
            original_data: Original record data
            rejection_reason: Reason for rejection
            rejection_details: Detailed explanation
            severity: Severity level
            processing_stage: Processing stage where rejection occurred
            suggested_fix: Suggested fix for the issue
        """
        try:
            # Create rejected record
            rejected_record = RejectedRecord(
                record_id=record_id,
                source_system=source_system,
                original_data=original_data,
                rejection_reason=rejection_reason,
                rejection_details=rejection_details,
                severity=severity,
                timestamp=datetime.now().isoformat(),
                processing_stage=processing_stage,
                suggested_fix=suggested_fix
            )
            
            # Add to storage
            self.rejected_records.append(rejected_record)
            
            # Update statistics
            self._update_rejection_stats(rejected_record)
            
            # Log to console
            self.logger.warning(
                f"Record {record_id} rejected: {rejection_reason.value} - {rejection_details}"
            )
            
        except Exception as e:
            self.logger.error(f"Error logging rejection for record {record_id}: {str(e)}")
    
    def _update_rejection_stats(self, rejected_record: RejectedRecord) -> None:
        """Update rejection statistics."""
        self.rejection_stats["total_rejections"] += 1
        
        # Update reason statistics
        reason = rejected_record.rejection_reason.value
        self.rejection_stats["rejections_by_reason"][reason] = \
            self.rejection_stats["rejections_by_reason"].get(reason, 0) + 1
        
        # Update severity statistics
        severity = rejected_record.severity.value
        self.rejection_stats["rejections_by_severity"][severity] = \
            self.rejection_stats["rejections_by_severity"].get(severity, 0) + 1
        
        # Update source statistics
        source = rejected_record.source_system
        self.rejection_stats["rejections_by_source"][source] = \
            self.rejection_stats["rejections_by_source"].get(source, 0) + 1
        
        # Update stage statistics
        stage = rejected_record.processing_stage
        self.rejection_stats["rejections_by_stage"][stage] = \
            self.rejection_stats["rejections_by_stage"].get(stage, 0) + 1
    
    def export_rejection_log(self, output_path: Optional[Path] = None) -> Path:
        """
        Export rejected records to a comprehensive log file.
        
        Args:
            output_path: Optional custom output path
            
        Returns:
            Path to the exported file
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.log_dir / f"rejection_log_{timestamp}.json"
        
        try:
            # Prepare export data with enum values converted to strings
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_rejected_records": len(self.rejected_records),
                "rejection_statistics": self.rejection_stats,
                "rejected_records": [self._record_to_dict(record) for record in self.rejected_records]
            }
            
            # Export to JSON
            with open(output_path, 'w', encoding='utf-8') as output_file:
                json.dump(export_data, output_file, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Exported rejection log to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error exporting rejection log: {str(e)}")
            raise
    
    def _record_to_dict(self, record: RejectedRecord) -> Dict[str, Any]:
        """Convert a RejectedRecord to a JSON-serializable dictionary."""
        record_dict = asdict(record)
        # Convert enum values to strings for JSON serialization
        record_dict["rejection_reason"] = record.rejection_reason.value
        record_dict["severity"] = record.severity.value
        return record_dict
    
    def export_rejection_csv(self, output_path: Optional[Path] = None) -> Path:
        """
        Export rejected records to CSV format for easy analysis.
        
        Args:
            output_path: Optional custom output path
            
        Returns:
            Path to the exported CSV file
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.log_dir / f"rejection_log_{timestamp}.csv"
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                if not self.rejected_records:
                    # Write empty CSV with headers
                    fieldnames = [
                        "record_id", "source_system", "rejection_reason", 
                        "rejection_details", "severity", "timestamp", 
                        "processing_stage", "suggested_fix", "retry_count"
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                else:
                    # Write records
                    fieldnames = list(asdict(self.rejected_records[0]).keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for record in self.rejected_records:
                        # Convert enum values to strings for CSV
                        record_dict = self._record_to_dict(record)
                        writer.writerow(record_dict)
            
            self.logger.info(f"Exported rejection CSV to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error exporting rejection CSV: {str(e)}")
            raise
    
    def get_rejection_summary(self) -> Dict[str, Any]:
        """Get a summary of rejection statistics."""
        return {
            "total_rejections": len(self.rejected_records),
            "rejection_rate": self._calculate_rejection_rate(),
            "statistics": self.rejection_stats.copy(),
            "recent_rejections": [self._record_to_dict(record) for record in self._get_recent_rejections(10)]
        }
    
    def _calculate_rejection_rate(self) -> float:
        """Calculate rejection rate (placeholder for total processed count)."""
        # In a real implementation, you'd track total processed records
        # For now, return a placeholder
        return 0.0
    
    def _get_recent_rejections(self, count: int) -> List[RejectedRecord]:
        """Get the most recent rejections."""
        recent = sorted(self.rejected_records, key=lambda x: x.timestamp, reverse=True)[:count]
        return recent
    
    def get_rejections_by_reason(self, reason: RejectionReason) -> List[RejectedRecord]:
        """Get all rejections for a specific reason."""
        return [record for record in self.rejected_records if record.rejection_reason == reason]
    
    def get_rejections_by_severity(self, severity: RejectionSeverity) -> List[RejectedRecord]:
        """Get all rejections for a specific severity level."""
        return [record for record in self.rejected_records if record.severity == severity]
    
    def get_rejections_by_source(self, source: str) -> List[RejectedRecord]:
        """Get all rejections from a specific source system."""
        return [record for record in self.rejected_records if record.source_system == source]
    
    def clear_rejected_records(self) -> None:
        """Clear all rejected records (use with caution)."""
        self.rejected_records.clear()
        self.rejection_stats = {
            "total_rejections": 0,
            "rejections_by_reason": {},
            "rejections_by_severity": {},
            "rejections_by_source": {},
            "rejections_by_stage": {}
        }
        self.logger.info("Cleared all rejected records")


class DataQualityAnalyzer:
    """Analyzes data quality issues and generates recommendations."""
    
    def __init__(self, rejection_logger: RejectionLogger):
        """Initialize the data quality analyzer."""
        self.rejection_logger = rejection_logger
        self.logger = logging.getLogger(__name__)
    
    def analyze_data_quality(self) -> Dict[str, Any]:
        """Analyze data quality based on rejection patterns."""
        try:
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "overall_quality_score": self._calculate_quality_score(),
                "common_issues": self._identify_common_issues(),
                "source_system_quality": self._analyze_source_quality(),
                "recommendations": self._generate_recommendations()
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing data quality: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)."""
        if not self.rejection_logger.rejected_records:
            return 100.0
        
        # Simple scoring based on rejection count and severity
        total_rejections = len(self.rejection_logger.rejected_records)
        critical_rejections = len(self.rejection_logger.get_rejections_by_severity(RejectionSeverity.CRITICAL))
        high_rejections = len(self.rejection_logger.get_rejections_by_severity(RejectionSeverity.HIGH))
        
        # Penalty calculation
        penalty = (critical_rejections * 10) + (high_rejections * 5) + (total_rejections * 1)
        score = max(0, 100 - penalty)
        
        return round(score, 2)
    
    def _identify_common_issues(self) -> List[Dict[str, Any]]:
        """Identify the most common data quality issues."""
        if not self.rejection_logger.rejected_records:
            return []
        
        # Count issues by reason
        issue_counts = {}
        for record in self.rejection_logger.rejected_records:
            reason = record.rejection_reason.value
            issue_counts[reason] = issue_counts.get(reason, 0) + 1
        
        # Sort by frequency
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"issue": issue, "count": count, "percentage": round(count / len(self.rejection_logger.rejected_records) * 100, 2)}
            for issue, count in sorted_issues[:5]  # Top 5 issues
        ]
    
    def _analyze_source_quality(self) -> Dict[str, Dict[str, Any]]:
        """Analyze data quality by source system."""
        source_quality = {}
        
        for source in self.rejection_logger.rejection_stats["rejections_by_source"]:
            source_rejections = self.rejection_logger.get_rejections_by_source(source)
            total_rejections = len(source_rejections)
            
            # Calculate source-specific metrics
            critical_count = len([r for r in source_rejections if r.severity == RejectionSeverity.CRITICAL])
            high_count = len([r for r in source_rejections if r.severity == RejectionSeverity.HIGH])
            
            source_quality[source] = {
                "total_rejections": total_rejections,
                "critical_rejections": critical_count,
                "high_rejections": high_count,
                "quality_score": max(0, 100 - (critical_count * 10) - (high_count * 5) - total_rejections)
            }
        
        return source_quality
    
    def _generate_recommendations(self) -> List[str]:
        """Generate data quality improvement recommendations."""
        recommendations = []
        
        if not self.rejection_logger.rejected_records:
            recommendations.append("No data quality issues detected. Continue current processes.")
            return recommendations
        
        # Analyze rejection patterns
        missing_field_count = len(self.rejection_logger.get_rejections_by_reason(RejectionReason.MISSING_REQUIRED_FIELD))
        date_format_count = len(self.rejection_logger.get_rejections_by_reason(RejectionReason.INVALID_DATE_FORMAT))
        validation_count = len(self.rejection_logger.get_rejections_by_reason(RejectionReason.DATA_VALIDATION_FAILED))
        
        if missing_field_count > 0:
            recommendations.append(f"Implement data validation for required fields. {missing_field_count} records rejected due to missing data.")
        
        if date_format_count > 0:
            recommendations.append(f"Standardize date formats across source systems. {date_format_count} records rejected due to date issues.")
        
        if validation_count > 0:
            recommendations.append(f"Enhance data validation rules. {validation_count} records failed validation checks.")
        
        # Source-specific recommendations
        for source, quality in self._analyze_source_quality().items():
            if quality["quality_score"] < 70:
                recommendations.append(f"Review data quality processes for {source} system. Quality score: {quality['quality_score']}")
        
        if not recommendations:
            recommendations.append("Monitor data quality metrics and address issues as they arise.")
        
        return recommendations


def main():
    """Demonstrate the rejection logger functionality."""
    print("Rejection Logger for EMR Data Processing")
    print("=" * 50)
    
    # Initialize rejection logger
    rejection_logger = RejectionLogger()
    
    # Simulate some rejections
    print("\nLogging sample rejections...")
    
    # Sample rejection 1
    rejection_logger.log_rejection(
        record_id="A123",
        source_system="alpha",
        original_data={"claim_id": "A123", "patient_id": "", "status": "denied"},
        rejection_reason=RejectionReason.MISSING_REQUIRED_FIELD,
        rejection_details="Patient ID is missing",
        severity=RejectionSeverity.HIGH,
        processing_stage="validation",
        suggested_fix="Provide valid patient ID"
    )
    
    # Sample rejection 2
    rejection_logger.log_rejection(
        record_id="B987",
        source_system="beta",
        original_data={"id": "B987", "date": "invalid-date", "status": "denied"},
        rejection_reason=RejectionReason.INVALID_DATE_FORMAT,
        rejection_details="Date format is invalid",
        severity=RejectionSeverity.MEDIUM,
        processing_stage="normalization",
        suggested_fix="Use ISO date format (YYYY-MM-DD)"
    )
    
    # Sample rejection 3
    rejection_logger.log_rejection(
        record_id="C456",
        source_system="alpha",
        original_data={"claim_id": "C456", "patient_id": "P999", "status": "unknown"},
        rejection_reason=RejectionReason.DATA_VALIDATION_FAILED,
        rejection_details="Status value 'unknown' is not valid",
        severity=RejectionSeverity.MEDIUM,
        processing_stage="validation",
        suggested_fix="Use valid status values: 'approved' or 'denied'"
    )
    
    # Display rejection summary
    print("\nRejection Summary:")
    summary = rejection_logger.get_rejection_summary()
    print(f"   Total rejections: {summary['total_rejections']}")
    
    print("\n   Rejections by reason:")
    for reason, count in summary['statistics']['rejections_by_reason'].items():
        print(f"     {reason}: {count}")
    
    print("\n   Rejections by severity:")
    for severity, count in summary['statistics']['rejections_by_severity'].items():
        print(f"     {severity}: {count}")
    
    # Export rejection logs
    print("\nExporting rejection logs...")
    json_path = rejection_logger.export_rejection_log()
    csv_path = rejection_logger.export_rejection_csv()
    
    print(f"   JSON log: {json_path}")
    print(f"   CSV log: {csv_path}")
    
    # Data quality analysis
    print("\nData Quality Analysis:")
    analyzer = DataQualityAnalyzer(rejection_logger)
    quality_analysis = analyzer.analyze_data_quality()
    
    print(f"   Overall quality score: {quality_analysis['overall_quality_score']}/100")
    
    print("\n   Common issues:")
    for issue in quality_analysis['common_issues']:
        print(f"     {issue['issue']}: {issue['count']} ({issue['percentage']}%)")
    
    print("\n   Recommendations:")
    for i, rec in enumerate(quality_analysis['recommendations'], 1):
        print(f"     {i}. {rec}")


if __name__ == "__main__":
    main()
