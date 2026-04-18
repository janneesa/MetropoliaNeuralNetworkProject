# Instructions for Willow chatbot

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
2. Download the models you want from GitHub release page.
- .keras files contain the model weights and configuration files.
- You do not need to unzip them!

3. Place the downloaded .keras files in the `checkpoints` directory.
- If the `checkpoints` directory does not exist, create it in the root of the willow directory.

4. Download the willow_tokenizer.model and place it in the root of willow directory.
- Tokenizer should be in the same directory as the willow.py file.

5. Check willow.py to ensure the model paths are correct. Update the paths if necessary.
- Willow.py has few configurations that have been set to default values. You can change the file paths and other settings as needed.
- MAX_SEQ_LEN is set to 256 and MAX_CONTEXT_TURNS is set to 4. These values should not be changed since they are the values used during training. Changing them may lead to unexpected behavior.
- GEN_MAX_NEW_TOKENS is set to 80. This is the maximum number of tokens that the model will generate in response to a user input. You can adjust this value based on your needs, but keep in mind that setting it too high may lead to longer response times.
- TEMPERATURE is set to 0.8. This controls the randomness of the model's responses. A higher temperature will result in more diverse responses, while a lower temperature will make the responses more focused and deterministic. You can experiment with this value to find the right balance for your use case.
- TOP_K is set to 40. This limits the number of tokens to consider when generating a response. A lower value will make the model more focused, while a higher value will allow for more diverse responses. You can adjust this value based on your needs.
- SEED is set to 42. This is used to ensure reproducibility of the model's responses. You can change this value if you want to generate different responses, but keep in mind that using the same seed will produce the same responses for the same inputs.
- - SEED is used when generating responses to ensure that the same input will produce the same output. This is useful for testing and debugging purposes. If you want to generate different responses for the same input, you can change the seed value or set it to None to disable seeding.