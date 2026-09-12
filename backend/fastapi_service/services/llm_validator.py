import logging
import json
import os
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger("gst2_fastapi.llm_validator")


class LlamaGSTValidator:
    """
    Runs LLaMA 3 8B GGUF on CPU RAM.
    
    Purpose:
    - Validate extracted GST field values (is this a valid GSTIN format?)
    - Clean and normalize values (fix common OCR errors)
    - Fill in missing fields using context from surrounding text
    - Ensure amounts add up correctly
    """
    
    def __init__(self, model_path: str, n_threads: int = 8):
        """
        Args:
            model_path: Path to .gguf file
            n_threads: Number of CPU threads to use (use half your CPU cores)
        """
        self.model_path = model_path
        self.n_threads = n_threads
        self.llm = None
        self._loaded = False
    
    def load(self):
        """Load LLaMA model (runs on CPU using llama-cpp-python)."""
        if self._loaded:
            return
            
        logger.info(f"Loading LLaMA 3 8B GGUF on CPU from: {self.model_path}")
        logger.info("This uses CPU RAM (~5GB). First load takes 20-30 seconds...")
        
        if not Path(self.model_path).exists():
            logger.error(f"Model file not found: {self.model_path}")
            logger.error("Download from: https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF")
            logger.error("Get the Q4_K_M version (about 4.9GB)")
            return
        
        try:
            from llama_cpp import Llama
            
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=4096,          # Context window
                n_threads=self.n_threads,  # CPU threads
                n_gpu_layers=0,      # 0 = run entirely on CPU
                verbose=False
            )
            self._loaded = True
            logger.info("LLaMA 3 8B loaded on CPU successfully!")
        except Exception as e:
            logger.error(f"Failed to load LLaMA: {e}")
    
    def validate_and_clean(self, extracted_fields: Dict[str, Any], full_text: str) -> Dict[str, Any]:
        """
        Validate and clean extracted GST fields using LLaMA.
        
        Args:
            extracted_fields: Fields extracted by LayoutLMv3
            full_text: Full OCR text for context
            
        Returns:
            Cleaned and validated fields
        """
        if not self._loaded:
            self.load()
        
        if not self._loaded:
            logger.warning("LLaMA not available, skipping validation")
            return extracted_fields
        
        prompt = self._build_validation_prompt(extracted_fields, full_text)
        
        try:
            response = self.llm(
                prompt,
                max_tokens=1000,
                temperature=0.1,  # Low temperature = more deterministic
                stop=["</json>", "```"]
            )
            
            response_text = response["choices"][0]["text"].strip()
            
            # Parse JSON response
            validated = self._parse_llm_response(response_text, extracted_fields)
            logger.info(f"LLaMA validation complete. Corrected {self._count_corrections(extracted_fields, validated)} fields.")
            return validated
            
        except Exception as e:
            logger.error(f"LLaMA validation failed: {e}")
            return extracted_fields
    
    def _build_validation_prompt(self, fields: Dict, full_text: str) -> str:
        """Build a structured prompt for GST field validation."""
        
        fields_json = json.dumps(fields, indent=2)
        
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a GST (Goods and Services Tax) expert specializing in Indian GST 2A statements.
Your job is to validate and correct OCR-extracted fields from GST 2A statements.

GST 2A Statement Rules:
- GSTIN format: 2 digits + 5 letters + 4 digits + 1 letter + 1 alphanumeric + Z + 1 alphanumeric (e.g., 29ABCDE1234F1Z5)
- Invoice dates are in DD/MM/YYYY or DD-MM-YYYY format
- Amounts are in Indian Rupees, typically formatted with commas (e.g., 1,23,456.78)
- IGST = CGST + SGST (for inter-state) OR just IGST (for intra-state)
- Return period format: MM/YYYY (e.g., 03/2024)

Fix common OCR errors:
- '0' and 'O' confusion in GSTINs
- '1' and 'I' confusion
- Missing decimal points in amounts
- Extra spaces in numbers
<|eot_id|><|start_header_id|>user<|end_header_id|>

Extracted fields from OCR:
{fields_json}

Context (full OCR text for reference):
{full_text[:2000]}

Please validate each field, fix any OCR errors, and return ONLY a valid JSON object with the corrected values. 
Keep the same field names. If a value looks correct, keep it as-is.
Return format: {{"GSTIN": ["29XXXXX..."], "INVOICE_NO": ["INV/001"], ...}}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    
    def _parse_llm_response(self, response_text: str, original_fields: Dict) -> Dict:
        """Parse JSON from LLaMA response."""
        try:
            # Find JSON in response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return original_fields
    
    def _count_corrections(self, original: Dict, corrected: Dict) -> int:
        """Count how many fields were changed."""
        changes = 0
        for key in corrected:
            if key in original and str(original[key]) != str(corrected[key]):
                changes += 1
        return changes


# Singleton
_llama_instance = None

def get_llama_validator(model_path: str) -> LlamaGSTValidator:
    global _llama_instance
    if _llama_instance is None:
        _llama_instance = LlamaGSTValidator(model_path)
    return _llama_instance