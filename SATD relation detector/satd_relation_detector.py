import argparse
import torch
from torch import nn
from transformers import BertModel, BertConfig
from transformers import BertTokenizer

DEF_LABELS = ['No-relation', 'SATD-duplication', 'SATD-repayment']


# Define the custom model class that extends nn.Module
class SATDRelationDetector(nn.Module):

    def __init__(self, cfg):
        super(SATDRelationDetector, self).__init__()
        # Initialize the BERT model with the default configuration
        self.bert = BertModel(BertConfig())
        # Add a dropout layer
        self.dropout = nn.Dropout(cfg.dropout)
        # Add a linear layer for classification
        self.linear = nn.Linear(BertConfig().hidden_size, 3)

    # Define the forward pass of the model
    def forward(self, input_ids, attention_mask, token_type_ids):
        # Pass inputs through the BERT model
        pooled_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        ).pooler_output
        # Apply dropout to the output
        dropout_output = self.dropout(pooled_output)
        # Pass the output through the linear layer for classification
        return self.linear(dropout_output)


# Test the loaded model with a few examples
def test_model(cfg, model, tokenizer, satd_a, satd_b):
    # Preprocess the input texts
    inputs = tokenizer(
        satd_a.replace('//', '').replace('/*', '').replace('*/', ''),
        satd_b.replace('//', '').replace('/*', '').replace('*/', ''),
        padding='max_length',
        truncation=True,
        max_length=cfg.max_length,
        return_tensors='pt',
    )

    # Evaluate the model on the preprocessed inputs
    with torch.no_grad():
        logits = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids'],
        )
        # Get the predicted class
        predictions = torch.argmax(logits, dim=-1)
        # Print the results
        print('\nSATD Text A: {}\nSATD Text B: {}'.format(satd_a, satd_b))
        print('-' * 5)
        print('Predicted result: {}\n'.format(DEF_LABELS[predictions.item()]))


# Load the trained model from a checkpoint file
def load_model(cfg):
    model = SATDRelationDetector(cfg)
    model.load_state_dict(torch.load(cfg.snapshot))
    model.eval()
    return model


# Run a few simple tests on the model
def simple_test(cfg, model):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Call the test_model function with various input pairs
    test_model(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        satd_a='Enable periodic rebalance as a temporary work-around for the Helix issue.',
        satd_b='TODO: Enable periodic rebalance per 10 seconds as a temporary work-around for the Helix issue.'
    )
    test_model(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        satd_a='TODO: Remove the legacy delimiter after releasing 0.6.0',
        satd_b='Same as before. We should try to enhance the same code.'
    )
    test_model(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        satd_a='Servlet injection does not always work for servlet container. We use a hacking here to initialize '
               'static variables at Spring wiring time.',
        satd_b='Remove temporary hacking and use Official way to wire-up servlet with injection under Spring.'
    )
    test_model(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        satd_a='I added this TODO. Kept the old config names from BoundedBBPool for BC.',
        satd_b='// TODO better config names?'
    )
    test_model(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        satd_a='Since the state machine is not implemented yet, we should get the configured dummy message from Ratis',
        satd_b='Please take care of the checkStyle issues and ASF licensing issue while committing.'
    )
    test_model(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        satd_a='// TODO: Include the generated file name in the response to the server',
        satd_b='// Add a TODO to include the generated file name in the response to server'
    )


# Parse command-line arguments
def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=str, default=None,
                        help='filename of model checkpoint [default: None]')
    parser.add_argument('--max_length', type=int, default=128,
                        help='max length of input text [default: 128]')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='the probability for dropout [default: 0.1]')
    args = parser.parse_args()
    return args


def main():
    cfg = get_params()
    model = load_model(cfg)
    simple_test(cfg, model)


if __name__ == '__main__':
    main()
