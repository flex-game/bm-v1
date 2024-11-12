import openai
import base64
import os

def load_system_prompt(prompt_file_path):
    """Load the system prompt from a text file."""
    with open(prompt_file_path, 'r') as file:
        return file.read()

def generate_frame_description(image_path, prompt_file_path):
    """Generate a description for a frame using OpenAI."""
    openai.api_key = os.getenv('OPENAI_API_KEY')
    system_prompt = load_system_prompt(prompt_file_path)
    
    with open(image_path, 'rb') as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyse this Civ VI screenshot as instructed in your system prompt. Return the analysis as a text file."
                    },
                    {
                        "type": "image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            }
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content

def generate_action_description(image_path1, image_path2, prompt_file_path):
    """Generate a description of the differences between two frames using OpenAI."""
    openai.api_key = os.getenv('OPENAI_API_KEY')
    system_prompt = load_system_prompt(prompt_file_path)
    
    description1 = generate_frame_description(image_path1, prompt_file_path)
    description2 = generate_frame_description(image_path2, prompt_file_path)
    
    with open(image_path1, 'rb') as image_file1, open(image_path2, 'rb') as image_file2:
        base64_image1 = base64.b64encode(image_file1.read()).decode('utf-8')
        base64_image2 = base64.b64encode(image_file2.read()).decode('utf-8')

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"I am going to give you two screenshots and a text description of each. Where you can see a difference between the two images, or where there's an indication in the text of something that has changed, list the changes as bullet points. Specifically try and make each change an action that the player must have taken in order to effect that change. Do your best on order of operations if there are multiple actions the user must have taken.\n\nDescription of Frame 1:\n{description1}\n\nDescription of Frame 2:\n{description2}\n\nProvide the analysis as a new text file."
                    },
                    {
                        "type": "image",
                        "image_url": f"data:image/jpeg;base64,{base64_image1}"
                    },
                    {
                        "type": "image",
                        "image_url": f"data:image/jpeg;base64,{base64_image2}"
                    }
                ]
            }
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content 