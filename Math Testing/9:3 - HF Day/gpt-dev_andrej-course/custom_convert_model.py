import torch
from transformers import PreTrainedModel, PretrainedConfig
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from model_definition import Head, MultiHeadAttention, FeedFoward, Block, GPTLanguageModel
from model_definition import GPTLanguageModel, vocab_size, n_embd, n_head, n_layer, block_size

class CustomGPTConfig(PretrainedConfig):
    model_type = "custom_gpt"
    def __init__(self, vocab_size=65, n_positions=256, n_embd=384, n_layer=6, n_head=6, **kwargs):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        super().__init__(**kwargs)

class CustomGPTLMHeadModel(PreTrainedModel):
    config_class = CustomGPTConfig
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPTLanguageModel()
        self.lm_head = self.transformer.lm_head

    def forward(self, input_ids, labels=None):
        transformer_outputs = self.transformer(input_ids)
        lm_logits = transformer_outputs[0]
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        return {'loss': loss, 'logits': lm_logits}

    def generate(self, input_ids, max_length):
        return self.transformer.generate(input_ids, max_length)

# Load the trained model
trained_model = torch.load('./full_model.pth', map_location=torch.device('cpu'))

# Create config
config = CustomGPTConfig(
    vocab_size=vocab_size,
    n_positions=block_size,
    n_embd=n_embd,
    n_layer=n_layer,
    n_head=n_head
)

# Create and initialize the custom model
custom_model = CustomGPTLMHeadModel(config)
custom_model.transformer.load_state_dict(trained_model.state_dict())

# Create a character-level tokenizer
def create_char_tokenizer(text_file_path):
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.pre_tokenizer = pre_tokenizers.CharBPETokenizer()
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.UnigramTrainer(
        special_tokens=["<unk>"],
        unk_token="<unk>",
    )

    tokenizer.train(files=[text_file_path], trainer=trainer)
    return tokenizer

# Assuming you have a text file with your training data
char_tokenizer = create_char_tokenizer("path/to/your/input.txt")

# Wrap the tokenizer for use with transformers
wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=char_tokenizer,
    unk_token="<unk>",
)

# Save the model and tokenizer
custom_model.save_pretrained("./custom_hf_model")
wrapped_tokenizer.save_pretrained("./custom_hf_model")

print("Custom model and tokenizer converted and saved successfully!")

# Test the model locally
from transformers import AutoModelForCausalLM, AutoTokenizer

loaded_model = AutoModelForCausalLM.from_pretrained("./custom_hf_model")
loaded_tokenizer = AutoTokenizer.from_pretrained("./custom_hf_model")

# Test generation
input_text = "Hello, world!"
input_ids = loaded_tokenizer.encode(input_text, return_tensors="pt")
output_ids = loaded_model.generate(input_ids, max_length=50)
output_text = loaded_tokenizer.decode(output_ids[0])

print(f"Input: {input_text}")
print(f"Output: {output_text}")