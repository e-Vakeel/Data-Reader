from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any
import csv
import json
from pathlib import Path

class BaseReader(ABC):
    @abstractmethod
    def read(self, file_path: str) -> List[Union[str, Dict[str, Any]]]:
        pass

class CSVReader(BaseReader):
    def read(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            return list(reader)

class JSONReader(BaseReader):
    def read(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data if isinstance(data, list) else [data]

class TextReader(BaseReader):
    def read(self, file_path: str) -> List[str]:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.readlines()
        

class UnifiedReader:
    def __init__(self):
        self.readers = {
            '.csv': CSVReader(),
            '.json': JSONReader(),
            '.txt': TextReader()
        }

    def read_file(self, file_path: str) -> List[Dict[str, Any]]:
        ext = Path(file_path).suffix
        if ext not in self.readers:
            raise ValueError(f"Unsupported file type: {ext}")
        
        raw_data = self.readers[ext].read(file_path)
        
        # Normalize the data structure
        normalized_data = []
        for item in raw_data:
            if isinstance(item, str):
                normalized_data.append({'text': item.strip(), 'file': file_path})
            elif isinstance(item, dict):
                item['file'] = file_path
                normalized_data.append(item)
            else:
                raise ValueError(f"Unexpected data type: {type(item)}")
        
        return normalized_data

    def read_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        all_data = []
        for file_path in Path(dir_path).rglob('*'):
            if file_path.suffix in self.readers:
                all_data.extend(self.read_file(str(file_path)))
        return all_data
    
from transformers import AutoTokenizer

class LLMDataPreparer:
    def __init__(self, model_name: str, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def prepare_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prepared_data = []
        for item in data:
            # Assume 'text' is the key for the main content, adjust if needed
            text = item.get('text', '')
            if not isinstance(text, str):
                text = str(text)  # Convert to string if it's not

            encoded = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True
            )
            prepared_item = {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'file': item.get('file', ''),  # Include source file information
            }
            # Include any other metadata from the original item
            prepared_item.update({k: v for k, v in item.items() if k not in prepared_item})
            prepared_data.append(prepared_item)
        return prepared_data
    
    def parser(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prepared_data = []
        for item in data:
            # Assume 'text' is the key for the main content, adjust if needed
            text = item.get('text', '')
            if not isinstance(text, str):
                text = str(text)

def main():
    reader = UnifiedReader()
    data_dir = 'data'
    all_data = reader.read_directory(data_dir)
    
    print("Total number of items:", len(all_data))
    print("Sample items:")
    for i in range(min(3, len(all_data))):
        print(f"Item {i + 1}:", all_data[i])
    
    preparer = LLMDataPreparer('bert-base-uncased')
    prepared_data = preparer.prepare_data(all_data)
    
    print(f"\nTotal number of prepared samples: {len(prepared_data)}")
    print(f"Sample prepared data point:")
    print(prepared_data)

if __name__ == "__main__":
    main()