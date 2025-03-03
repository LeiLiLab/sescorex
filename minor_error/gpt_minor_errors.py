import openai
import time
import random
import re
from tqdm.auto import tqdm
import asyncio
import argparse
from typing import Any
import glob
from openai.error import RateLimitError, Timeout, APIError, APIConnectionError, ServiceUnavailableError
import backoff
import Levenshtein

def calculate_edit_distance(line1, line2, line3):
    # Extract content from the lines
    # Calculate edit distances
    distance_line1_line2 = Levenshtein.distance(line1, line2)
    distance_line1_line3 = Levenshtein.distance(line1, line3)

    return distance_line1_line2, distance_line1_line3

def extract_content(lines):
    pattern = r'Result:\s*"(.*?)"'
    match = re.match(pattern, lines)
    if match:
        return match.group(1)
    else:
        return None

def backoff_hdlr(details):
    print("backoff_hdlr", details)
    time.sleep(15)

def batchify(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

@backoff.on_exception(backoff.expo, (RateLimitError, Timeout, APIError, APIConnectionError, ServiceUnavailableError), max_tries=5, on_backoff=backoff_hdlr)
async def dispatch_openai_requests(messages_list, model, temperature, request_timeout, max_tokens, top_p):
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
            top_p=top_p,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

def main(input_file, output_file, api_key):
    openai.api_key = api_key

    prompts = [
        """
        Your task is to introduce specific error type "Style/Unnatural" to the **Translation**, not to the **Reference**.
        You must make sure the introduced error is a minor error, which don't lead to change of loss of meaning but will be noticed.

        Style/Unnatural: Style involving excessive wordiness or overly embedded clauses, often due to inappropriate retention of source text style in the target text.

        Below are examples of input and expect output format:
        -----------
        Example 1:
        Input:
        Reference: This man page's substance, which is confined to characterizing the alternatives, is extricated from the GNU C Compiler's comprehensive documentation.
        Translation: This manual page is taken from the comprehensive documentation of the GNU C Compiler and is limited to explaining the meaning of the options.

        Output:
        Result: This manual page is taken from the full documentation of the GNU C Compiler and is limited to explaining the meaning of the options.

        Example 2:
        Input:
        Reference: The EURo record and the Information record are definitive unless it is effectively kept up.
        Translation: Unless effectively maintained, EURo files, Info files are authoritative documents.

        Output:
        Result: Unless voluntarily maintained, EURo files, Info files are authoritative documents.

        Example 3:
        Input:
        Reference: Check the Data area in the event that you find a contrast between the man page and the program.
        Translation: If you find a contrast between the manual page and the software, please check Info

        Output:
        Result: If you find a contradiction between the manual page and the software, please check Info
        -----------

        Input:
        Reference: "{reference_sentence}"
        Translation: "{translation_sentence}"

        Output:
        """,
        """
        Your task is to introduce specific error type "Accuracy/Mistranslation" to the **Translation**, not to the **Reference**.
        You must make sure the introduced error is a minor error, which don't lead to change of loss of meaning but will be noticed.

        Accuracy/Mistranslation: Error occuring when the target content that does not accurately represent the source content.

        Below are examples of input and expect output format:
        -----------
        Example 1:
        Input:
        Reference: The EURo record and the Information record are definitive unless it is effectively kept up.
        Translation: Unless someone volunteers to maintain the EURo record and the Information record, it is authoritative.

        Output:
        Result: Unless someone volunteers to maintain it, the EURo and Info documents are authoritative.

        Example 2:
        Input:
        Reference: Table 3 Course of action of code focuses within the four-character news portion
        Translation: Table 3 Course of action of code of the four-character hearing section

        Output:
        Result: Table 3 Code position arrangement of the four-character hearing section

        Example 3:
        Input:
        Reference: Appendix Diao, "Table of General Normative Chinese Characters; Code Positions of Discussed Characters," has been added as an educational supplement.
        Translation: Added an informational appendix "General specification character list; the code position of the Discussed Characters" See Appendix Diao.


        Output:
        Result: Added an informational appendix "General specification character list; the code position of the discussion characters" See Appendix Diao.
        -----------

        Input:
        Reference: "{reference_sentence}"
        Translation: "{translation_sentence}"

        Output:
        """,
        """
        Your task is to introduce specific error type "Terminology/Inappropriate for context" to the **Translation**, not to the **Reference**.
        You must make sure the introduced error is a minor error, which don't lead to change of loss of meaning but will be noticed.

        Accuracy/Mistranslation: Errors that occur when the terminology used in a translation or content does not fit the specific context in which it is used.

        Below are examples of input and expect output format:
        -----------
        Example 1:
        Input:
        Reference: Demonstrate clearly that dialect is the dialect of any extra input records (rather than the default choice from the conclusion of the filename).
        Translation: Demonstrate clearly that the language of the next input file is language (not the default choice obtained from the file name).

        Output:
        Result: Make it clear that the language of the next input file is language (not the default choice obtained from the file name).

        Example 2:
        Input:
        Reference: Sections = 25200 bits, 0xFD308130-0xFE39FE39.
        Translation: There are 25,200 bits of user-defined area from 0xFD308130 to 0xFE39FE39.

        Output:
        Result: There are 25,200 bytes of user-defined area from 0xFD308130 to 0xFE39FE39.

        Example 3:
        Input:
        Reference: COVID-19's impacts on the pharmaceutical showcase
        Translation: The impact of COVID-19 on the drug market


        Output:
        Result: The impact of coronavirus pneumonia on the drug market
        -----------

        Input:
        Reference: "{reference_sentence}"
        Translation: "{translation_sentence}"

        Output:
        """
    ]

    gpt_mode = "gpt-3.5-turbo"
    temperature = 0
    max_tokens = 512

    responses_list = []

    data_lines = open(input_file, 'r', encoding='utf-8-sig').readlines()
    ref_ls, mt_ls, score_ls = [], [], []
    data_lines = [ele[:-1] for ele in data_lines]
    for line in data_lines:
        line_ls = line.split('\t')
        ref, mt, score = line_ls[0], line_ls[1], line_ls[2]
        ref_ls += [ref]
        mt_ls += [mt]
        score_ls += [float(score)]
    common_input_list = ref_ls

    counter = 0
    file = open(output_file, "a", encoding='utf-8-sig')
    batch_messages_ls = []
    for common_input, mt in zip(common_input_list, mt_ls):
        prompt = random.choice(prompts)
        full_prompt = prompt.format(reference_sentence=common_input, translation_sentence=mt)
        #print(full_prompt)
        batch_messages_ls += [[{"role": "user", "content": full_prompt}]]

    batch_size = 14
    batches = list(batchify(batch_messages_ls, batch_size))
    ref_ls = list(batchify(ref_ls, batch_size))
    mt_ls = list(batchify(mt_ls, batch_size))
    score_ls = list(batchify(score_ls, batch_size))
    with tqdm(total=len(batches)) as pbar:
        counter = 0
        for batch, ref, mt, sc in zip(batches, ref_ls, mt_ls, score_ls):
            responses = asyncio.run(
                dispatch_openai_requests(
                    messages_list=batch,
                    model=gpt_mode,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    request_timeout=60,
                    top_p=1.0,
                )
            )
            for response, r, m, s in zip(responses, ref, mt, sc):
                result_response = response["choices"][0]['message']['content'].strip()
                responses_list.append((result_response, ref, m, s))
                lines = result_response
                content = extract_content(lines)
                if content != None:
                    content = str(content)
                    r = str(r)
                    m = str(m)
                    distance_line1_line2, distance_line1_line3 = calculate_edit_distance(content, r, m)
                    if content == r:
                        s = float(0)
                    elif content == m:
                        s = float(s)
                    elif distance_line1_line2 < distance_line1_line3:
                        s = float(-1)
                    else:
                        s = float(s-1)
                    item = (r, content, s)
                    lines = '<ENDSENTENCE>'.join(map(str, item))
                    file.write(f"{lines}\n")
                else:
                    item = (r, m, s)
                    lines = '<ENDSENTENCE>'.join(map(str, item))
                    file.write(f"{lines}\n")
            pbar.update(1)

    file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some files using OpenAI API.")
    parser.add_argument('--input_file', type=str, required=True, help="The path to the input file.")
    parser.add_argument('--output_file', type=str, required=True, help="The path to the output file.")
    parser.add_argument('--api_key', type=str, required=True, help="The OpenAI API key.")
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.api_key)
