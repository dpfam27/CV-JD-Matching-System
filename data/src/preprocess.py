import os
from pathlib import Path
import fitz  # PyMuPDF
import pandas as pd
import logging
from datetime import datetime

class PDFProcessor:
    def __init__(self):
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Setup paths
        current_file = Path(__file__)
        self.base_dir = current_file.parent.parent.parent
        self.input_dir = self.base_dir / 'data' / 'raw' / 'cv' / 'test'
        self.output_dir = self.base_dir / 'data' / 'processed' / 'cv'
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from a PDF file"""
        try:
            text = ""
            with fitz.open(str(pdf_path)) as doc:
                for page in doc:
                    text += page.get_text()
            return text
        except Exception as e:
            self.logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def process_pdf(self, pdf_path: Path) -> dict:
        """Process a single PDF file"""
        try:
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            
            if not text:
                self.logger.warning(f"No text extracted from {pdf_path.name}")
                return {}

            # Create result dictionary
            result = {
                "filename": pdf_path.name,
                "text": text,
                "word_count": len(text.split()),
                "processed_date": datetime.now().isoformat(),
                "file_size_kb": os.path.getsize(pdf_path) / 1024
            }
            
            return result

        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {e}")
            return {}

    def process_all_pdfs(self) -> list:
        """Process all PDFs in the input directory"""
        results = []
        try:
            # Get all PDF files
            pdf_files = list(self.input_dir.glob('*.pdf'))
            self.logger.info(f"Found {len(pdf_files)} PDF files")

            # Process each PDF
            for pdf_path in pdf_files:
                self.logger.info(f"Processing {pdf_path.name}")
                result = self.process_pdf(pdf_path)
                if result:
                    results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"Error processing PDFs: {e}")
            return results

    def save_results(self, results: list):
        """Save processing results in both JSON and CSV formats"""
    try:
        if not results:
            self.logger.warning("No results to save")
            return

        # Create DataFrame
        df = pd.DataFrame(results)

        # Save as JSON
        json_path = self.output_dir / 'processed_results.json'
        df.to_json(json_path, orient='records', indent=2)
        self.logger.info(f"Results saved to {json_path}")

        # Save as CSV
        csv_path = self.output_dir / 'cv_texts.csv'
        df.to_csv(csv_path, index=False, columns=['filename', 'text'])  # Save 'filename' and 'text' columns
        self.logger.info(f"Results saved to {csv_path}")

    except Exception as e:
        self.logger.error(f"Error saving results: {e}")


def main():
    try:
        # Initialize processor
        processor = PDFProcessor()

        # Process PDFs
        results = processor.process_all_pdfs()

        # Save results
        processor.save_results(results)

        # Print summary
        print(f"\nProcessing Summary:")
        print(f"Total PDFs processed: {len(results)}")
        if results:
            print(f"Results saved to: {processor.output_dir / 'processed_results.json'}")
        else:
            print("No documents were processed successfully")

    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()