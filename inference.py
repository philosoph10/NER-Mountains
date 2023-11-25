from transformers import pipeline
from transformers import AutoModelForTokenClassification, AutoTokenizer
import argparse
import os


# noinspection PyShadowingNames
def parse_arguments():
    # a helper function to parse arguments
    parser = argparse.ArgumentParser(description='Get the directory to save the model.')

    # the arguments' purposes are described in help
    parser.add_argument('--model-path', type=str, default='huggingsaurusRex/bert-base-uncased-for-mountain-ner',
                        help='Specify the directory from where to load the model')
    parser.add_argument('--tokenizer-path', type=str, default='huggingsaurusRex/bert-base-uncased-for-mountain-ner',
                        help='Specify the directory from where to load the tokenizer')
    parser.add_argument('--input-path', type=str,
                        help='Specify the path to txt file for inference')
    parser.add_argument('--output-path', type=str,
                        help='Specify the directory in which to store the output file')

    args = parser.parse_args()

    return args


def get_filename_without_extension(file_path):
    """
    a helper function to get the file name out of the directory (excluding the extension)
    for example: path/to/some_file.ext -> some_file
    :param file_path: the full path
    :return: the raw name
    """
    base_name = os.path.basename(file_path)
    filename_without_extension, _ = os.path.splitext(base_name)
    return filename_without_extension


# noinspection PyShadowingNames
def highlight_words(content, output_file, highlights):
    """
    highlight the mountains in the file using <mountain> tag, in xml style
    :param content: the content of the input file
    :param output_file: the path to the output file
    :param highlights: a list of pairs (start_mountain_index, end_mountain_index)
    """
    highlighted_content = content
    for i, (start, end) in enumerate(highlights):
        # insert the tags counting for the shift
        highlighted_content = (
            highlighted_content[:start + i*21] +
            '<mountain>' + content[start:end] + '</mountain>' +
            highlighted_content[end + i*21:]
        )

    # save the tagged file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(highlighted_content)


# noinspection PyShadowingNames
def infer(model, tokenizer, input_file):
    """
    perform inference
    :param model: the model
    :param tokenizer: the tokenizer for the model
    :param input_file: the input file, containing the text
    :return: the content of the file for further usage; the dictionary of raw model outputs;
             a list of pairs (start_mountain_index, end_mountain_index)
    """
    # read the file
    with open(input_file, 'r', encoding='utf-8') as infile:
        content = infile.read()

    # create a ner pipeline
    model.to('cpu')
    ner = pipeline('ner', model=model, tokenizer=tokenizer)

    # get the answer for tokens
    ans_for_tokens = ner(content)

    # get the answer for words
    last = -1
    ans_for_words = []
    for ent in ans_for_tokens:
        # the entity is not a beginning of a word
        if ent['word'][:2] == '##':
            last['word'] += ent['word'][2:]
            last['end'] = ent['end']
        # the entity is a beginning of a word
        else:
            if last != -1:
                ans_for_words.append(last)
            last = ent
    ans_for_words.append(last)

    # save the (start_mountain_index, end_mountain_index) pairs
    highlights = [(word_label['start'], word_label['end']) for word_label in ans_for_words
                  if word_label['entity'] == 'LABEL_1']

    return content, ans_for_words, highlights


if __name__ == '__main__':
    # parse cmd arguments
    args = parse_arguments()

    # load the model and the tokenizer
    loaded_model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    loaded_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    content, word_labels, highlights = infer(loaded_model, loaded_tokenizer, args.input_path)
    print(f"Model raw output:\n{word_labels}")

    # define the path to output file
    output_file = os.path.join(args.output_path, get_filename_without_extension(args.input_path) + '_pred.txt')

    highlight_words(content, output_file, highlights)
