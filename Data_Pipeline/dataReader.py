import csv
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Union
import random
from transformers import AutoTokenizer

class BaseReader:
    def read(self, file_path: str) -> List[str]:
        raise NotImplementedError

class CSVReader(BaseReader):
    def read(self, file_path: str) -> List[str]:
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            # Assuming the text is in the first column. Adjust if needed.
            return [row[0] for row in reader]

class JSONReader(BaseReader):
    def read(self, file_path: str) -> List[str]:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # Assuming the JSON structure has a 'text' field. Adjust if needed.
            return [item for item in data]

class TextReader(BaseReader):
    def read(self, file_path: str) -> List[str]:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Assuming each line is a separate text. Adjust if needed.
            return file.readlines()

class YAMLReader(BaseReader):
    def read(self, file_path: str) -> List[str]:
        with open(file_path , 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            # Assuming the YAML structure has a 'text' field. Adjust if needed.
            return [item['text'] for item in data]

class UnifiedReader:
    def __init__(self):
        self.readers = {
            '.csv': CSVReader(),
            '.json': JSONReader(),
            '.txt': TextReader()
        }

    def read_file(self, file_path: str) -> List[str]:
        ext = Path(file_path).suffix
        if ext not in self.readers:
            raise ValueError(f"Unsupported file type: {ext}")
        return self.readers[ext].read(file_path)

    def read_directory(self, dir_path: str) -> List[str]:
        all_texts = []
        for file_path in Path(dir_path).rglob('*'):
            if file_path.suffix in self.readers:
                all_texts.extend(self.read_file(str(file_path)))
        return all_texts

class LLMDataPreparer:
    def __init__(self, model_name: str, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def prepare_data(self, texts: Union[List[str], List[List[str]]]) -> List[Dict[str, Any]]:
        prepared_data = []
        for text in texts:
            # If text is a list, join it into a single string
            if isinstance(text, list):
                text = ' '.join(text)
            
            # Ensure text is a string
            if not isinstance(text, str):
                raise ValueError(f"Expected string or list of strings, got {type(text)}")

            encoded = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True
            )
            prepared_data.append({
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask']
            })
        return prepared_data



def main():
    reader = UnifiedReader()
    data_dir = 'data'
    all_texts = reader.read_directory(data_dir)
    print(f"Total number of texts: {len(all_texts)}")
    print(f"Total number of texts: {all_texts}")
    # Shuffle the data
    flattened_texts = [item for sublist in all_texts for item in (sublist if isinstance(sublist, list) else [sublist])]
    
    preparer = LLMDataPreparer('bert-base-uncased')
    prepared_data = preparer.prepare_data(flattened_texts)
    
    print(f"Total number of samples: {len(prepared_data)}")
    print(f"Sample data point: {prepared_data[0]}")

if __name__ == "__main__":
    main()