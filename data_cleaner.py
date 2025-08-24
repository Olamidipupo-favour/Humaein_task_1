#!/usr/bin/env python3
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class NormalizedClaim:
    """Normalized claim data structure."""
    claim_id: str
    patient_id: str
    procedure_code: str
    denial_reason: Optional[str]
    status: str
    submitted_at: str
    source_system: str


class EMRDataCleaner:
    """Handles ingestion and normalization of EMR data from multiple sources."""
    
    def __init__(self, log_level: str = "INFO") -> None:
        """Initialize the data cleaner with logging configuration."""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Define field mappings for different EMR sources
        self.field_mappings = {
            'alpha': {
                'claim_id': 'claim_id',
                'patient_id': 'patient_id', 
                'procedure_code': 'procedure_code',
                'denial_reason': 'denial_reason',
                'submitted_at': 'submitted_at',
                'status': 'status'
            },
            'beta': {
                'claim_id': 'id',
                'patient_id': 'member',
                'procedure_code': 'code',
                'denial_reason': 'error_msg',
                'submitted_at': 'date',
                'status': 'status'
            }
        }
    
    def _normalize_date(self, date_str: str) -> str:
        """
        Normalize date strings to ISO format.
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            ISO formatted date string
            
        Raises:
            ValueError: If date cannot be parsed
        """
        if not date_str or pd.isna(date_str):
            raise ValueError("Date string is empty or null")
        
        # Handle different date formats
        date_formats = [
            '%Y-%m-%d',           # 2025-07-01
            '%Y-%m-%dT%H:%M:%S',  # 2025-07-03T00:00:00
            '%Y-%m-%d %H:%M:%S',  # 2025-07-03 00:00:00
            '%m/%d/%Y',           # 07/01/2025
            '%d/%m/%Y'            # 01/07/2025
        ]
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(str(date_str).strip(), fmt)
                return parsed_date.isoformat()
            except ValueError:
                continue
        
        raise ValueError(f"Unable to parse date: {date_str}")
    
    def _normalize_string(self, value: Any) -> str:
        """
        Normalize string values with consistent casing and formatting.
        
        Args:
            value: Input value to normalize
            
        Returns:
            Normalized string value
        """
        if value is None or pd.isna(value):
            return ""
        
        # Convert to string and normalize
        normalized = str(value).strip()
        
        # Handle special cases for status field
        if normalized.lower() in ['approved', 'denied']:
            return normalized.lower()
        
        return normalized
    
    def _normalize_denial_reason(self, value: Any) -> Optional[str]:
        """
        Normalize denial reason field, handling null values explicitly.
        
        Args:
            value: Input denial reason value
            
        Returns:
            Normalized denial reason or None if empty/null
        """
        if value is None or pd.isna(value) or str(value).strip() == "":
            return None
        
        normalized = str(value).strip()
        
        # Handle special cases
        if normalized.lower() in ['none', 'null', 'n/a', '']:
            return None
            
        return normalized
    
    def _validate_required_fields(self, data: Dict[str, Any], source: str) -> bool:
        """
        Validate that required fields are present and non-empty.
        
        Args:
            data: Data dictionary to validate
            source: Source system identifier
            
        Returns:
            True if validation passes, False otherwise
        """
        required_fields = ['claim_id', 'patient_id', 'procedure_code', 'status', 'submitted_at']
        
        for field in required_fields:
            if field not in data or not data[field] or pd.isna(data[field]):
                self.logger.warning(f"Missing required field '{field}' in {source} data: {data}")
                return False
        
        return True
    
    def _normalize_record(self, record: Dict[str, Any], source: str) -> Optional[NormalizedClaim]:
        """
        Normalize a single record from any EMR source.
        
        Args:
            record: Raw record data
            source: Source system identifier
            
        Returns:
            Normalized claim record or None if validation fails
        """
        try:
            # Map fields according to source schema
            mapping = self.field_mappings[source]
            mapped_data = {}
            
            for target_field, source_field in mapping.items():
                if source_field in record:
                    mapped_data[target_field] = record[source_field]
                else:
                    self.logger.warning(f"Missing field '{source_field}' in {source} record: {record}")
                    return None
            
            # Validate required fields
            if not self._validate_required_fields(mapped_data, source):
                return None
            
            # Normalize individual fields
            normalized_claim = NormalizedClaim(
                claim_id=self._normalize_string(mapped_data['claim_id']),
                patient_id=self._normalize_string(mapped_data['patient_id']),
                procedure_code=self._normalize_string(mapped_data['procedure_code']),
                denial_reason=self._normalize_denial_reason(mapped_data['denial_reason']),
                status=self._normalize_string(mapped_data['status']),
                submitted_at=self._normalize_date(mapped_data['submitted_at']),
                source_system=source
            )
            
            return normalized_claim
            
        except Exception as e:
            self.logger.error(f"Error normalizing {source} record {record}: {str(e)}")
            return None
    
    def ingest_csv_data(self, file_path: Path) -> List[NormalizedClaim]:
        """
        Ingest and normalize CSV data from EMR Alpha.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of normalized claim records
        """
        normalized_records = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row_num, row in enumerate(reader, start=2):  # Start at 2 to account for header
                    try:
                        normalized_record = self._normalize_record(row, 'alpha')
                        if normalized_record:
                            normalized_records.append(normalized_record)
                        else:
                            self.logger.warning(f"Skipping invalid record at row {row_num}")
                    except Exception as e:
                        self.logger.error(f"Error processing row {row_num}: {str(e)}")
                        continue
            
            self.logger.info(f"Successfully processed {len(normalized_records)} records from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error reading CSV file {file_path}: {str(e)}")
            raise
        
        return normalized_records
    
    def ingest_json_data(self, file_path: Path) -> List[NormalizedClaim]:
        """
        Ingest and normalize JSON data from EMR Beta.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of normalized claim records
        """
        normalized_records = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as jsonfile:
                data = json.load(jsonfile)
                
                if not isinstance(data, list):
                    self.logger.error(f"Expected JSON array, got {type(data)}")
                    raise ValueError("Invalid JSON format: expected array")
                
                for record_num, record in enumerate(data, start=1):
                    try:
                        if not isinstance(record, dict):
                            self.logger.warning(f"Skipping non-dict record at position {record_num}")
                            continue
                            
                        normalized_record = self._normalize_record(record, 'beta')
                        if normalized_record:
                            normalized_records.append(normalized_record)
                        else:
                            self.logger.warning(f"Skipping invalid record at position {record_num}")
                    except Exception as e:
                        self.logger.error(f"Error processing record {record_num}: {str(e)}")
                        continue
            
            self.logger.info(f"Successfully processed {len(normalized_records)} records from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error reading JSON file {file_path}: {str(e)}")
            raise
        
        return normalized_records
    
    def process_all_sources(self, data_dir: Path) -> List[NormalizedClaim]:
        """
        Process all EMR data sources in the specified directory.
        
        Args:
            data_dir: Directory containing EMR data files
            
        Returns:
            Combined list of all normalized claim records
        """
        all_records = []
        
        try:
            # Process CSV files (EMR Alpha)
            csv_files = list(data_dir.glob("*.csv"))
            for csv_file in csv_files:
                try:
                    records = self.ingest_csv_data(csv_file)
                    all_records.extend(records)
                except Exception as e:
                    self.logger.error(f"Failed to process {csv_file}: {str(e)}")
                    continue
            
            # Process JSON files (EMR Beta)
            json_files = list(data_dir.glob("*.json"))
            for json_file in json_files:
                try:
                    records = self.ingest_json_data(json_file)
                    all_records.extend(records)
                except Exception as e:
                    self.logger.error(f"Failed to process {json_file}: {str(e)}")
                    continue
            
            self.logger.info(f"Total records processed: {len(all_records)}")
            
        except Exception as e:
            self.logger.error(f"Error processing data sources: {str(e)}")
            raise
        
        return all_records
    
    def export_normalized_data(self, records: List[NormalizedClaim], output_path: Path) -> None:
        """
        Export normalized data to JSON file.
        
        Args:
            records: List of normalized claim records
            output_path: Output file path
        """
        try:
            # Convert dataclass objects to dictionaries
            export_data = [asdict(record) for record in records]
            
            with open(output_path, 'w', encoding='utf-8') as output_file:
                json.dump(export_data, output_file, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Exported {len(records)} records to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {str(e)}")
            raise
    
    def get_data_summary(self, records: List[NormalizedClaim]) -> Dict[str, Any]:
        """
        Generate summary statistics for the normalized data.
        
        Args:
            records: List of normalized claim records
            
        Returns:
            Dictionary containing summary statistics
        """
        if not records:
            return {"total_records": 0}
        
        # Count by source system
        source_counts = {}
        status_counts = {}
        
        for record in records:
            # Source system counts
            source = record.source_system
            source_counts[source] = source_counts.get(source, 0) + 1
            
            # Status counts
            status = record.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_records": len(records),
            "source_system_counts": source_counts,
            "status_counts": status_counts,
            "records_with_denial_reason": len([r for r in records if r.denial_reason is not None]),
            "records_without_denial_reason": len([r for r in records if r.denial_reason is None])
        }


def main() -> None:
    """Main execution function."""
    try:
        # Initialize data cleaner
        cleaner = EMRDataCleaner(log_level="INFO")
        
        # Define paths
        data_dir = Path("sample_data")
        output_file = Path("normalized_claims.json")
        
        # Process all EMR sources
        normalized_records = cleaner.process_all_sources(data_dir)
        
        if normalized_records:
            # Export normalized data
            cleaner.export_normalized_data(normalized_records, output_file)
            
            # Generate and display summary
            summary = cleaner.get_data_summary(normalized_records)
            print("\n=== Data Processing Summary ===")
            print(f"Total records processed: {summary['total_records']}")
            print(f"Source system breakdown: {summary['source_system_counts']}")
            print(f"Status breakdown: {summary['status_counts']}")
            print(f"Records with denial reason: {summary['records_with_denial_reason']}")
            print(f"Records without denial reason: {summary['records_without_denial_reason']}")
            
            # Display sample of normalized data
            print(f"\n=== Sample Normalized Records ===")
            for i, record in enumerate(normalized_records[:3], 1):
                print(f"Record {i}:")
                for field, value in asdict(record).items():
                    print(f"  {field}: {value}")
                print()
        else:
            print("No valid records were processed.")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        logging.error(f"Main execution failed: {str(e)}")


if __name__ == "__main__":
    main()
