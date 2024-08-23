import json
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from openai import OpenAI
import re
import os

NVIDAI_API_KEY = "---key---"

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDAI_API_KEY
)

# Function to count the number of tokens in a sentence
def count_tokens(sentence):
    return len(sentence.split())

# Function to create batches of sentences with a maximum token limit
def create_batches(sentences, max_tokens=242):
    batches = []
    current_batch = []
    current_tokens = 0

    for idx, sentence in enumerate(sentences):
        tokens = count_tokens(sentence)
        if current_tokens + tokens > max_tokens:
            batches.append(current_batch)
            current_batch = [sentence]
            current_tokens = tokens
        else:
            current_batch.append(sentence)
            current_tokens += tokens

    if current_batch:
        batches.append(current_batch)

    return batches

# Load list of words from words.json
with open('words.json', 'r', encoding='utf-8') as file:
    words_list = json.load(file)
words_set = set(words_list)  # Convert list to set for faster lookups

# Load sentences from JSON file
with open('scrap_sentences.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Define your desired data structure.
class StructOBJ(BaseModel):
    sentence: str = Field(description="The sentence to be translated")
    translation: str = Field(description="The translation of the sentence")
    words: list = Field(description="The extra words that the model preserves because it feels natural in the context.")

_LANG_MAP = {
    'as': 'Assamese',
    'bn': 'Bengali',
    'hi': 'Hindi',
    'gu': 'Gujarati',
    'ka': 'Kannada',
    'ml': 'Malayalam',
    'mr': 'Marathi',
    'or': 'Odia',
    'pa': 'Punjabi',
    'ta': 'Tamil',
    'te': 'Telugu',
}

_EXAMPLE_LANG = {
    'as': ['''  Input: "The boy goes to school every day to play cricket."

                Output: "ছোৱালীয়ে প্ৰতিদিনে schoolলৈ যায় cricket খেলিবলৈ।"

                Input: "She doesn't like to play football but loves to cheer for her team."

                Output: "তেওঁক football খেলিবলৈ ভাল নালাগে but তেওঁৰ teamক cheer কৰিবলৈ ভাল লাগে।"'''],

    'bn': ['''  Input: "The boy goes to school every day to play cricket."

                Output: "ছেলেটা প্রতিদিন school যায় cricket খেলতে।"

                Input: "She doesn't like to play football but loves to cheer for her team."

                Output: "তেওঁক football খেলিবলৈ ভাল নালাগে but তেওঁৰ teamক cheer কৰিবলৈ ভাল লাগে।"'''],

    'hi': ['''  Input: "The boy goes to school every day to play cricket."

                Output: "लड़का हर दिन school जाता है cricket खेलने।", ["school", "cricket"]

                Input: "She doesn't like to play football but loves to cheer for her team."

                Output: "उसे football खेलना पसंद नहीं है but अपनी team के लिए cheer करना पसंद है।", ["football", "team", "cheer"]'''],

    'gu': ['''  Input: "The boy goes to school every day to play cricket."

                Output: "છોકરો દરરોજ school જાય છે cricket રમવા."

                Input: "She doesn't like to play football but loves to cheer for her team."

                Output: "તેણે football રમવું ગમતું નથી but તેની teamને cheer કરવું ગમે છે."'''],

    'ka': ['''  Input: "The boy goes to school every day to play cricket."

                Output: "ಬಾಯ್ ಪ್ರತಿದಿನವೂ school ಗೆ cricket ಆಟವಾಡಲು ಹೋಗುತ್ತಾನೆ."

                Input: "She doesn't like to play football but loves to cheer for her team."

                Output: "ಅವಳಿಗೆ football ಆಟವಾಡಲು ಇಷ್ಟವಿಲ್ಲ but ತನ್ನ team ಗೆ cheer ಮಾಡಲು ಇಷ್ಟವಿದೆ."'''],

    'ml': ['''  Input: "The boy goes to school every day to play cricket."

                Output: "അച്ചന്‍ പ്രതിദിനം school-ലേക്ക് cricket കളിക്കാന്‍ പോകുന്നു."

                Input: "She doesn't like to play football but loves to cheer for her team."

                Output: "അവള്‍ക്ക് football കളിക്കാന്‍ ഇഷ്ടമില്ല but അവളുടെ team-നായി cheer ചെയ്യാൻ ഇഷ്ടം."'''],

    'mr': ['''  Input: "The boy goes to school every day to play cricket."

                Output: "मुलगा रोज school ला cricket खेळण्यासाठी जातो."

                Input: "She doesn't like to play football but loves to cheer for her team."

                Output: "तिला football खेळायला आवडत नाही but तिच्या team साठी cheer करायला आवडतं."'''],

    'or': ['''  Input: "The boy goes to school every day to play cricket."

                Output: "ଲାଡ୍କା ପ୍ରତିଦିନ school କୁ cricket ଖେଳିବାକୁ ଯାଏ।"

                Input: "She doesn't like to play football but loves to cheer for her team."

                Output: "ତାଙ୍କୁ football ଖେଳିବାକୁ ଭଲ ଲାଗେନାହିଁ but ତାଙ୍କର team ପାଇଁ cheer କରିବାକୁ ଭଲ ଲାଗେ।"'''],

    'pa': ['''  Input: "The boy goes to school every day to play cricket."

                Output: "ਲੜਕਾ ਹਰ ਰੋਜ਼ school ਜਾਂਦਾ ਹੈ cricket ਖੇਡਣ ਲਈ।"

                Input: "She doesn't like to play football but loves to cheer for her team."

                Output: "ਉਹਨੂੰ football ਖੇਡਣਾ ਪਸੰਦ ਨਹੀਂ ਹੈ but ਆਪਣੇ team ਲਈ cheer ਕਰਨਾ ਪਸੰਦ ਹੈ।"'''],

    'ta': ['''  Input: "The boy goes to school every day to play cricket."

                Output: "The boy தினமும் schoolக்கு போகிறான் cricket விளையாட."

                Input: "She doesn't like to play football but loves to cheer for her team."

                Output: "அவளுக்கு football விளையாட விருப்பம் இல்லை but தன் teamக்கு cheer செய்ய விருப்பம்."'''],

    'te': ['''  Input: "The boy goes to school every day to play cricket."

                Output: "Tఅబ్బాయ్ ప్రతి రోజు schoolకి cricket ఆడటానికి వెళ్ళిపోతాడు."

                Input: "She doesn't like to play football but loves to cheer for her team."

                Output: "ఆమెకు football ఆడటం ఇష్టం లేదు but తన team కోసం cheer చేయడం ఇష్టం."'''],
}

for code, language in _LANG_MAP.items():
    print("-----------------Working on ", language)

    responses = []
    cnt = 0
    for category, sentences in data.items():
        batches = create_batches(sentences)
        category_responses = []

        print(f"# of batches {len(batches)}\n")

        for batch in batches:
            
            cnt += 1
            print(f"Batch {cnt}\n")
            print(f"# of sentences {len(batch)}\n")

            en_set = set()
            for sentence in batch:
                words_in_sentence = set(sentence.split())
                for word in words_in_sentence:
                    if word.lower() in words_set:
                        en_set.add(word.lower())

            # Convert the set to a sorted list for use in the prompt
            en_list = sorted(en_set)

            # print(en_list)

            # Set up a parser + inject instructions into the prompt template.
            parser = JsonOutputParser(pydantic_object=StructOBJ)

            # Create prompt with the en_list
            prompt = f'''
                Hi,

                You are a translator tasked with the following:\n

                Task: Translate Sentences\n

                You will receive a list of sentences in English. Your goal is to translate these sentences into a blend of {language} and English, using a colloquial or code-mix style.\n

                Instructions:\n

                Translation Style:\n

                Translate the sentences into a mixture of {language} and English.\n
                Use common expressions and terms that people might use in casual conversation.\n
                
                Preserve Specific Words:\n

                Some words from the list below should remain in English in the translation.\n
                You may also keep additional words in English if it feels natural in the context, also let me know the extra words which you kept in English\n
                Words to Preserve: {en_list}

                \nFormatting:\n

                English words in the translated sentences should be written in English script.\n
                Only {language} words should be in {language} script.\n
                Example:\n

                {_EXAMPLE_LANG[code]}

                \nPlease ensure that the translated sentences retain their original meaning while incorporating the provided words in English.\n

                Here are the list of sentences:\n{batch} \n

                {parser.get_format_instructions()}
            '''

            # print(prompt)

            # Make the API call
            completion = client.chat.completions.create(
                model="meta/llama-3.1-405b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                top_p=0.7,
                max_tokens=4096,
                stream=False
            )

            response_buffer = completion.choices[0].message.content

            # Parse the response buffer using the parser
            try:
                parsed_response = parser.parse(response_buffer)

                # Iterate through each item in the parsed_response
                for response in parsed_response:
                    words_set.update(response["words"])

                category_responses.append(parsed_response)
                # resp_file.write(json.dumps(parsed_response, ensure_ascii=False) + "\n")
            
            except Exception as e:
                try:
                    match = re.search(r'\[.*\]', response_buffer, re.DOTALL)
                    
                    if match:
                        # Extracted content (without the outer square brackets)
                        extracted_content = match.group(0)
                        response_buffer = extracted_content
                        parsed_response = parser.parse(response_buffer)

                        for response in parsed_response:
                            words_set.update(response["words"])

                        category_responses.append(parsed_response)
                
                except Exception as e:
                    print(f"Error parsing response: {e}")

        responses.extend(category_responses)

    # Define the directory path
    directory_path = "translations"
    # Ensure the directory exists
    os.makedirs(directory_path, exist_ok=True)
    # Define the file path
    file_path = os.path.join(directory_path, f"{code}_translations.json")

    with open(file_path, 'w', encoding='utf-8') as resp_file:
        json.dump(responses, resp_file, ensure_ascii=False, indent=2)

# Convert the set back to a list
updated_words_list = list(words_set)

# Save the updated words list back to the JSON file
with open('words.json', 'w', encoding='utf-8') as file:
    json.dump(updated_words_list, file, ensure_ascii=False, indent=2)

print("Processing complete. Responses saved to translations.json.")

# 4655 --- API
