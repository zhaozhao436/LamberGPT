import os
import argparse
import sentencepiece as spm


parser = argparse.ArgumentParser()
parser.add_argument('--corpus_name_or_path', type=str, required=True)
parser.add_argument('--output_path', type=str, default='./custom')
parser.add_argument('--model_name', type=str, default='tokenizer', help='model name to be saved')
parser.add_argument('--vocab_size', type=int, default=30000)
parser.add_argument('--character_coverage', type=float, default=0.9995)
parser.add_argument('--model_type', type=str, default='bpe')
parser.add_argument('--user_defined_symbols', type=str, default='<sepra>')
args = parser.parse_args()


# 语料
if os.path.isdir(args.corpus_name_or_path):
    corpus_list = []
    for file in os.listdir(args.corpus_name_or_path):
        if file.endswith('.txt'):
            corpus_list.append(os.path.join(args.corpus_name_or_path, file))
else:
    corpus_list = args.corpus_name_or_path.split(',')
print(f'train {len(corpus_list)} corpus')


spm.SentencePieceTrainer.train(
    input=corpus_list,
    model_prefix=os.path.join(args.output_path, args.model_name),
    vocab_size=args.vocab_size,
    user_defined_symbols=args.user_defined_symbols.split(','), 
    character_coverage=args.character_coverage,
    model_type=args.model_type,
)

