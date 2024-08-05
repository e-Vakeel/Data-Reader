from torch.utils.data import Dataset, DataLoader
from Data_Pipeline.dataReader import UnifiedReader, LLMDataPreparer

class LargeDataset(Dataset):
    def __init__(self, data_dir, model_name, max_length=512):
        self.reader = UnifiedReader()
        self.preparer = LLMDataPreparer(model_name, max_length)
        self.data = self.reader.read_directory(data_dir)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.preparer.prepare_data([self.data[idx]])[0]

# In main function:
data_dir = 'data'

dataset = LargeDataset(data_dir, 'bert-base-uncased')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Now you can iterate over dataloader in your training loop
for batch in dataloader:
    # Your training code here
    pass